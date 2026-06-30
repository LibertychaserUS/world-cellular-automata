import type { ControlManifestInspectionResult } from "./contracts.ts";

const requiredManifestKeys = [
  "schema_version",
  "control_plane_version",
  "experiment_id",
  "protocol",
  "submission_policy",
  "git",
  "eval",
  "checkpoint",
  "runner",
  "artifacts",
  "model_matrix",
  "jobs",
  "report_contract",
] as const;

const queueProvenanceFileNames = ["manifest.json", "remote_submission.json", "status.jsonl", "queue_summary.json"];
const rawFieldExternalFamilies = new Set(["convnet", "fno", "unet"]);
const rawFieldExternalRoleMarkers = ["raw_field", "not_token_equivalent"] as const;
const tokenLevelRoleMarkers = ["wca", "token_equivalent", "learnable_interface", "decoder_capacity"] as const;
const exploratoryComparisonStatuses = new Set(["exploratory", "diagnostic", "external_anchor", "not_in_strict_matched_eval"]);

export function inspectControlManifest(payload: unknown): ControlManifestInspectionResult {
  const errors: string[] = [];
  const warnings: string[] = [];
  if (!isRecord(payload)) {
    return emptyResult(["control manifest must be a JSON object"], warnings);
  }

  for (const key of requiredManifestKeys) {
    if (!(key in payload)) {
      errors.push(`missing required manifest key: ${key}`);
    }
  }

  const protocol = asRecord(payload.protocol);
  const submissionPolicy = asRecord(payload.submission_policy);
  const git = asRecord(payload.git);
  const artifacts = asRecord(payload.artifacts);
  const planner = asRecord(payload.planner);
  const jobs = Array.isArray(payload.jobs) ? payload.jobs : [];
  const modelMatrix = Array.isArray(payload.model_matrix) ? payload.model_matrix : [];
  const requiredPaths = stringArray(artifacts?.required_paths);
  const queueDir = typeof artifacts?.queue_dir === "string" ? artifacts.queue_dir : null;
  const reportDirs = stringArray(artifacts?.report_dirs);
  const runDirs = stringArray(artifacts?.run_dirs);
  const requiresCompleteManifests = stringArray(payload.requires_complete_manifests);
  const phase = typeof protocol?.phase === "string" ? protocol.phase : null;
  const strict = typeof protocol?.strict === "boolean" ? protocol.strict : null;
  const allowSubmit = typeof submissionPolicy?.allow_submit === "boolean" ? submissionPolicy.allow_submit : null;
  const forbidDirectQueueSubmit =
    typeof submissionPolicy?.forbid_direct_queue_submit === "boolean"
      ? submissionPolicy.forbid_direct_queue_submit
      : null;
  const requireCleanGit = typeof submissionPolicy?.require_clean_git === "boolean" ? submissionPolicy.require_clean_git : null;

  const experimentId = typeof payload.experiment_id === "string" && payload.experiment_id.length > 0 ? payload.experiment_id : null;
  if (!experimentId) {
    errors.push("experiment_id must be a non-empty string");
  }
  if (!protocol) {
    errors.push("protocol must be an object");
  } else {
    if (!phase) {
      errors.push("protocol.phase must be a non-empty string");
    }
    if (strict !== true) {
      errors.push("protocol.strict must be true for formal control manifests");
    }
  }
  if (!submissionPolicy) {
    errors.push("submission_policy must be an object");
  }
  if (!artifacts) {
    errors.push("artifacts must be an object");
  }
  if (!Array.isArray(payload.model_matrix) || modelMatrix.length === 0) {
    errors.push("model_matrix must be a non-empty list");
  }
  if (!Array.isArray(payload.jobs) || jobs.length === 0) {
    errors.push("jobs must be a non-empty list");
  }

  validateArtifactPaths({ queueDir, reportDirs, runDirs, requiredPaths, errors });
  const { jobIds, duplicateJobIds } = validateJobs(jobs, errors);
  validateModelMatrix(modelMatrix, errors);
  validateStrictModelMatrixScope({ modelMatrix, strict, allowSubmit, errors });

  if (strict === true && allowSubmit === true) {
    if (forbidDirectQueueSubmit !== true) {
      errors.push("strict submit manifests must set submission_policy.forbid_direct_queue_submit=true");
    }
    if (requireCleanGit !== true) {
      errors.push("strict submit manifests must set submission_policy.require_clean_git=true");
    }
    const phaseNumber = parsePhaseNumber(phase);
    if (phaseNumber !== null && phaseNumber >= 21) {
      validateStrictV21Contract({ queueDir, requiredPaths, requiresCompleteManifests, errors });
    }
  } else if (allowSubmit === false) {
    warnings.push("allow_submit=false: manifest is draft/validation-only");
  }

  return {
    schema_version: 1,
    ok: errors.length === 0,
    experiment_id: experimentId,
    phase,
    strict,
    allow_submit: allowSubmit,
    forbid_direct_queue_submit: forbidDirectQueueSubmit,
    require_clean_git: requireCleanGit,
    queue_dir: queueDir,
    report_dirs: reportDirs,
    run_dirs: runDirs,
    required_paths: requiredPaths,
    job_count: jobs.length,
    job_ids: jobIds,
    duplicate_job_ids: duplicateJobIds,
    model_matrix_count: modelMatrix.length,
    requires_complete_manifests: requiresCompleteManifests,
    planner_enabled: planner?.enabled === true,
    errors,
    warnings,
  };
}

function emptyResult(errors: string[], warnings: string[]): ControlManifestInspectionResult {
  return {
    schema_version: 1,
    ok: false,
    experiment_id: null,
    phase: null,
    strict: null,
    allow_submit: null,
    forbid_direct_queue_submit: null,
    require_clean_git: null,
    queue_dir: null,
    report_dirs: [],
    run_dirs: [],
    required_paths: [],
    job_count: 0,
    job_ids: [],
    duplicate_job_ids: [],
    model_matrix_count: 0,
    requires_complete_manifests: [],
    planner_enabled: false,
    errors,
    warnings,
  };
}

function validateArtifactPaths(input: {
  queueDir: string | null;
  reportDirs: string[];
  runDirs: string[];
  requiredPaths: string[];
  errors: string[];
}): void {
  if (!input.queueDir) {
    input.errors.push("artifacts.queue_dir must be a non-empty string");
  } else if (!isSafeRelativePath(input.queueDir)) {
    input.errors.push(`artifacts.queue_dir must be a safe repository-relative path: ${input.queueDir}`);
  }
  for (const [label, values] of [
    ["artifacts.report_dirs", input.reportDirs],
    ["artifacts.run_dirs", input.runDirs],
    ["artifacts.required_paths", input.requiredPaths],
  ] as const) {
    for (const value of values) {
      if (!isSafeRelativePath(value)) {
        input.errors.push(`${label} contains unsafe path: ${value}`);
      }
    }
  }
}

function validateJobs(jobs: unknown[], errors: string[]): { jobIds: string[]; duplicateJobIds: string[] } {
  const seen = new Set<string>();
  const duplicateJobIds = new Set<string>();
  const jobIds: string[] = [];
  jobs.forEach((job, index) => {
    if (!isRecord(job)) {
      errors.push(`jobs[${index}] must be an object`);
      return;
    }
    if (typeof job.id !== "string" || job.id.length === 0) {
      errors.push(`jobs[${index}].id must be a non-empty string`);
      return;
    }
    jobIds.push(job.id);
    if (seen.has(job.id)) {
      duplicateJobIds.add(job.id);
    }
    seen.add(job.id);
    if (!Array.isArray(job.argv) || job.argv.length === 0 || !job.argv.every((value) => typeof value === "string")) {
      errors.push(`jobs[${index}].argv must be a non-empty string list`);
    }
  });
  for (const duplicateJobId of duplicateJobIds) {
    errors.push(`duplicate job id: ${duplicateJobId}`);
  }
  return { jobIds, duplicateJobIds: [...duplicateJobIds].sort() };
}

function validateModelMatrix(modelMatrix: unknown[], errors: string[]): void {
  modelMatrix.forEach((model, index) => {
    if (!isRecord(model)) {
      errors.push(`model_matrix[${index}] must be an object`);
      return;
    }
    if (typeof model.id !== "string" || model.id.length === 0) {
      errors.push(`model_matrix[${index}].id must be a non-empty string`);
    }
    if (typeof model.run_dir === "string" && !isSafeRelativePath(model.run_dir)) {
      errors.push(`model_matrix[${index}].run_dir must be repository-relative: ${model.run_dir}`);
    }
    if (typeof model.config_path === "string" && !isSafeRelativePath(model.config_path)) {
      errors.push(`model_matrix[${index}].config_path must be repository-relative: ${model.config_path}`);
    }
  });
}

function validateStrictModelMatrixScope(input: {
  modelMatrix: unknown[];
  strict: boolean | null;
  allowSubmit: boolean | null;
  errors: string[];
}): void {
  if (input.strict !== true || input.allowSubmit !== true) {
    return;
  }

  const grouped = new Map<string, Map<string, string[]>>();
  input.modelMatrix.forEach((entry, index) => {
    if (!isRecord(entry) || isExploratoryModelMatrixEntry(entry)) {
      return;
    }
    const group = modelMatrixComparisonGroup(entry);
    const kind = modelMatrixScopeKind(entry);
    const label = typeof entry.id === "string" && entry.id.length > 0
      ? entry.id
      : typeof entry.run_dir === "string" && entry.run_dir.length > 0
        ? entry.run_dir
        : `model_matrix[${index}]`;
    if (!grouped.has(group)) {
      grouped.set(group, new Map());
    }
    const byKind = grouped.get(group)!;
    byKind.set(kind, [...(byKind.get(kind) ?? []), label]);
  });

  for (const [group, byKind] of [...grouped.entries()].sort(([left], [right]) => left.localeCompare(right))) {
    const raw = byKind.get("raw_field_external_anchor_not_token_equivalent") ?? [];
    const token = byKind.get("token_level_formal") ?? [];
    if (raw.length > 0 && token.length > 0) {
      input.errors.push(
        "strict model_matrix comparison group mixes token-level formal entries with " +
        "raw_field_external_anchor_not_token_equivalent entries; mark raw anchors exploratory " +
        `or split comparison_group. group=${group} raw=${JSON.stringify(raw.slice(0, 5))} token=${JSON.stringify(token.slice(0, 5))}`,
      );
    }
  }
}

function modelMatrixComparisonGroup(entry: Record<string, unknown>): string {
  for (const key of ["comparison_group", "strict_comparison_group", "comparison_table"]) {
    const value = entry[key];
    if (typeof value === "string" && value.length > 0) {
      return value;
    }
  }
  return "strict_matched";
}

function isExploratoryModelMatrixEntry(entry: Record<string, unknown>): boolean {
  if (entry.exploratory === true) {
    return true;
  }
  for (const key of ["comparison_status", "formal_status", "evidence_status"]) {
    const value = entry[key];
    if (typeof value === "string" && exploratoryComparisonStatuses.has(value)) {
      return true;
    }
  }
  return false;
}

function modelMatrixScopeKind(entry: Record<string, unknown>): string {
  const family = typeof entry.family === "string" ? entry.family.toLowerCase() : "";
  const role = typeof entry.role === "string" ? entry.role.toLowerCase() : "";
  const text = `${family} ${role}`;
  if (rawFieldExternalFamilies.has(family) || rawFieldExternalRoleMarkers.every((marker) => text.includes(marker))) {
    return "raw_field_external_anchor_not_token_equivalent";
  }
  if (tokenLevelRoleMarkers.some((marker) => text.includes(marker))) {
    return "token_level_formal";
  }
  return "other";
}

function validateStrictV21Contract(input: {
  queueDir: string | null;
  requiredPaths: string[];
  requiresCompleteManifests: string[];
  errors: string[];
}): void {
  if (!input.queueDir) {
    return;
  }
  if (input.requiredPaths.length === 0) {
    input.errors.push("V21+ strict submit manifests must define artifacts.required_paths");
  }
  const requiredPathSet = new Set(input.requiredPaths.map(normalizeRelativePath));
  for (const name of queueProvenanceFileNames) {
    const expected = `${normalizeRelativePath(input.queueDir)}/${name}`;
    if (!requiredPathSet.has(expected)) {
      input.errors.push(`V21+ strict submit manifests must require queue provenance path: ${expected}`);
    }
  }
  if (input.requiresCompleteManifests.length === 0) {
    input.errors.push("V21+ strict submit manifests must define requires_complete_manifests");
  }
}

function parsePhaseNumber(phase: string | null): number | null {
  if (!phase) {
    return null;
  }
  const match = phase.trim().toUpperCase().match(/^V(\d+)/);
  return match ? Number(match[1]) : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return isRecord(value) ? value : null;
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string" && item.length > 0) : [];
}

function isSafeRelativePath(value: string): boolean {
  if (value.length === 0 || value.startsWith("/") || value.includes("\0")) {
    return false;
  }
  return !value.split(/[\\/]+/).includes("..");
}

function normalizeRelativePath(value: string): string {
  return value.replaceAll("\\", "/").replace(/^\/+/, "").replace(/\/+$/, "");
}
