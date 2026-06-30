import { createHash } from "node:crypto";
import { readdir, readFile, stat } from "node:fs/promises";
import { relative, resolve } from "node:path";
import path from "node:path/posix";

import type { ArtifactInventory } from "./contracts.ts";

const queueFileNames = new Set(["manifest.json", "status.jsonl", "queue_summary.json", "launcher.log", "runner.pid", "remote_submission.json"]);
const runFileNames = new Set(["config.json", "resolved_config.json", "summary.json", "train_log.csv", "model.pt", "model_hash.json"]);
const reportFileNames = new Set([
  "results.md",
  "results.csv",
  "formal_evidence.md",
  "results_by_horizon.md",
  "results_by_horizon.csv",
  "summary.json",
  "per_sample_rows.csv",
  "eval_plan.json",
]);

export async function buildArtifactInventory(input: {
  manifest: Record<string, unknown>;
  repoRoot: string;
}): Promise<ArtifactInventory> {
  const repoRoot = resolve(input.repoRoot);
  const experimentId = requiredString(input.manifest.experiment_id, "experiment_id");
  const { artifacts, source, errors } = await submittedArtifactContract(input.manifest, repoRoot, experimentId);
  const requiredFiles = errors.length === 0 ? requiredArtifactPathsFromArtifacts(artifacts) : [];
  const presentFiles: string[] = [];
  const missingFiles: string[] = [];

  for (const requiredFile of requiredFiles) {
    const resolved = await resolveRequiredArtifactPath(repoRoot, requiredFile);
    if (resolved.length > 0) {
      presentFiles.push(...resolved);
    } else {
      missingFiles.push(requiredFile);
    }
  }

  const provenance = await validateArtifactProvenance(input.manifest, repoRoot, artifacts);
  const staleFetchErrors = await validateFetchMissingSources(repoRoot, artifacts);
  const validationErrors = [...errors, ...provenance.errors, ...staleFetchErrors];
  const uniquePresent = [...new Set(presentFiles)].sort();
  return {
    schema_version: 1,
    experiment_id: experimentId,
    artifact_contract: artifacts,
    artifact_contract_source: source,
    required_files: requiredFiles,
    present_files: uniquePresent,
    missing_files: missingFiles,
    validation_errors: validationErrors,
    provenance_warnings: provenance.warnings,
    status: missingFiles.length === 0 && validationErrors.length === 0 ? "complete" : "incomplete",
  };
}

async function submittedArtifactContract(
  manifest: Record<string, unknown>,
  repoRoot: string,
  experimentId: string,
): Promise<{ artifacts: Record<string, unknown>; source: string; errors: string[] }> {
  const currentArtifacts = asRecord(manifest.artifacts) ?? {};
  const candidates: Array<[string, string, "queue" | "plan"]> = [
    ["submitted_generated_queue_recovered", `artifacts/control/${experimentId}/submitted_generated_queue_recovered.json`, "queue"],
    ["submitted_plan_recovered", `artifacts/control/${experimentId}/submitted_plan_recovered.json`, "plan"],
    ["generated_queue", `artifacts/control/${experimentId}/generated_queue.json`, "queue"],
    ["plan", `artifacts/control/${experimentId}/plan.json`, "plan"],
  ];

  for (const [source, relativePath, kind] of candidates) {
    const payload = await readJsonObject(resolve(repoRoot, relativePath));
    if (!payload) {
      continue;
    }
    const artifacts = kind === "queue" ? artifactContractFromGeneratedQueue(payload) : asRecord(payload.artifacts);
    const contract = artifacts ?? currentArtifacts;
    return {
      artifacts: contract,
      source,
      errors: artifactContractErrors(contract, source),
    };
  }

  return {
    artifacts: currentArtifacts,
    source: "current_manifest",
    errors: artifactContractErrors(currentArtifacts, "current_manifest"),
  };
}

function artifactContractFromGeneratedQueue(generatedQueue: Record<string, unknown>): Record<string, unknown> {
  const contract = { ...(asRecord(generatedQueue.artifacts) ?? {}) };
  if (typeof contract.queue_dir !== "string" && typeof generatedQueue.output_dir === "string") {
    contract.queue_dir = generatedQueue.output_dir;
  }
  return contract;
}

function artifactContractErrors(artifacts: Record<string, unknown> | null, source: string): string[] {
  if (!artifacts) {
    return [`${source} artifact contract is missing or not an object`];
  }
  const errors: string[] = [];
  const queueDir = artifacts.queue_dir;
  if (typeof queueDir !== "string" || queueDir.length === 0) {
    errors.push(`${source} artifact contract is missing queue_dir`);
  } else if (!isSafeRelativePath(queueDir)) {
    errors.push(`${source} artifact contract queue_dir is unsafe: ${queueDir}`);
  }
  for (const key of ["report_dirs", "run_dirs"] as const) {
    for (const value of stringArray(artifacts[key])) {
      if (!isSafeRelativePath(value)) {
        errors.push(`${source} artifact contract ${key} contains unsafe path: ${value}`);
      }
    }
  }
  for (const value of stringArray(artifacts.required_paths)) {
    if (!isSafeRelativePath(value)) {
      errors.push(`${source} artifact contract required_paths contains unsafe path: ${value}`);
    }
  }
  return errors;
}

function requiredArtifactPathsFromArtifacts(artifacts: Record<string, unknown>): string[] {
  const exactPaths = stringArray(artifacts.required_paths);
  if (exactPaths.length > 0) {
    return [...new Set(exactPaths.map(normalizeRelativePath))].sort();
  }
  const required = stringArray(artifacts.required_files);
  const queueDir = normalizeRelativePath(requiredString(artifacts.queue_dir, "artifacts.queue_dir"));
  const reportDirs = stringArray(artifacts.report_dirs).map(normalizeRelativePath);
  const runDirs = stringArray(artifacts.run_dirs).map(normalizeRelativePath);
  const paths: string[] = [];
  for (const name of required) {
    if (queueFileNames.has(name)) {
      paths.push(`${queueDir}/${name}`);
    }
    if (runFileNames.has(name)) {
      for (const runDir of runDirs) {
        paths.push(`${runDir}/${name}`);
      }
    }
    if (reportFileNames.has(name)) {
      for (const reportDir of reportDirs) {
        paths.push(`${reportDir}/${name}`);
      }
    }
  }
  if (required.length === 0) {
    paths.push(`${queueDir}/manifest.json`, `${queueDir}/status.jsonl`, `${queueDir}/queue_summary.json`);
  }
  return [...new Set(paths)].sort();
}

async function resolveRequiredArtifactPath(repoRoot: string, relativePath: string): Promise<string[]> {
  const normalized = normalizeRelativePath(relativePath);
  if (await pathExists(resolve(repoRoot, normalized))) {
    return [normalized];
  }

  const parentText = normalized.slice(0, normalized.lastIndexOf("/"));
  const fileName = normalized.slice(normalized.lastIndexOf("/") + 1);
  if (!runFileNames.has(fileName) || !parentText.startsWith("runs/")) {
    return [];
  }
  const parent = resolve(repoRoot, parentText);
  if (!(await pathExists(parent))) {
    return [];
  }
  const matches = await findFilesByName(parent, fileName);
  return matches.map((match) => normalizeRelativePath(relative(repoRoot, match))).sort();
}

async function validateArtifactProvenance(
  manifest: Record<string, unknown>,
  repoRoot: string,
  artifacts: Record<string, unknown>,
): Promise<{ errors: string[]; warnings: string[] }> {
  const protocol = asRecord(manifest.protocol);
  const policy = asRecord(manifest.submission_policy);
  if (protocol?.strict !== true || policy?.allow_submit !== true) {
    return { errors: [], warnings: [] };
  }
  const queueDir = typeof artifacts.queue_dir === "string" ? artifacts.queue_dir : "";
  const queueManifestPath = resolve(repoRoot, normalizeRelativePath(queueDir), "manifest.json");
  if (!(await pathExists(queueManifestPath))) {
    return { errors: [], warnings: [] };
  }

  const experimentId = requiredString(manifest.experiment_id, "experiment_id");
  const remoteManifest = await readJsonObject(queueManifestPath);
  const remoteQueue = asRecord(remoteManifest?.queue_spec);
  if (!remoteQueue) {
    return { errors: [`queue manifest is missing queue_spec: ${normalizeRelativePath(relative(repoRoot, queueManifestPath))}`], warnings: [] };
  }
  const remoteSha = typeof remoteQueue.control_manifest_sha256 === "string" ? remoteQueue.control_manifest_sha256 : null;
  const currentSha = sha256Json(manifest);
  const plan = await readFirstJsonObject([
    `artifacts/control/${experimentId}/submitted_plan_recovered.json`,
    `artifacts/control/${experimentId}/plan.json`,
  ], repoRoot);
  const generatedQueue = await readFirstJsonObject([
    `artifacts/control/${experimentId}/submitted_generated_queue_recovered.json`,
    `artifacts/control/${experimentId}/generated_queue.json`,
  ], repoRoot);

  const errors: string[] = [];
  const warnings: string[] = [];
  const planSha = typeof plan?.control_manifest_sha256 === "string" ? plan.control_manifest_sha256 : null;
  if (planSha !== remoteSha) {
    errors.push(`control plan sha does not match fetched queue manifest for ${experimentId}: plan=${planSha} fetched=${remoteSha}`);
  }
  if (planSha !== currentSha) {
    warnings.push(`current control manifest sha does not match submitted plan for ${experimentId}: plan=${planSha} current=${currentSha}`);
  }
  const generatedSha = typeof generatedQueue?.control_manifest_sha256 === "string" ? generatedQueue.control_manifest_sha256 : null;
  if (generatedSha !== remoteSha) {
    errors.push(`generated queue sha does not match fetched queue manifest for ${experimentId}: generated=${generatedSha} fetched=${remoteSha}`);
  }
  if (generatedQueue?.generated_from_manifest !== true) {
    errors.push(`generated queue is missing generated_from_manifest=true for ${experimentId}`);
  }
  if (generatedSha !== currentSha) {
    warnings.push(`current control manifest sha does not match submitted generated queue for ${experimentId}: generated=${generatedSha} current=${currentSha}`);
  }
  return { errors, warnings };
}

async function validateFetchMissingSources(repoRoot: string, artifacts: Record<string, unknown>): Promise<string[]> {
  if (typeof artifacts.queue_dir !== "string") {
    return [];
  }
  const missingPath = resolve(repoRoot, normalizeRelativePath(artifacts.queue_dir), "fetch_missing_sources.json");
  if (!(await pathExists(missingPath))) {
    return [];
  }
  const text = await readFile(missingPath, "utf8");
  if (text.trim().length === 0) {
    return [];
  }
  const payload = await readJsonObject(missingPath);
  if (!payload) {
    return [`fetch_missing_sources.json is non-empty but not valid JSON: ${normalizeRelativePath(relative(repoRoot, missingPath))}`];
  }
  const missing = payload.missing_sources;
  if (Array.isArray(missing) && missing.length > 0) {
    return [`fetch_missing_sources.json still lists missing sources: ${normalizeRelativePath(relative(repoRoot, missingPath))}`];
  }
  return [];
}

async function readFirstJsonObject(relativePaths: string[], repoRoot: string): Promise<Record<string, unknown> | null> {
  for (const relativePath of relativePaths) {
    const payload = await readJsonObject(resolve(repoRoot, relativePath));
    if (payload) {
      return payload;
    }
  }
  return null;
}

async function readJsonObject(filePath: string): Promise<Record<string, unknown> | null> {
  try {
    const payload = JSON.parse(await readFile(filePath, "utf8")) as unknown;
    return asRecord(payload);
  } catch {
    return null;
  }
}

async function pathExists(filePath: string): Promise<boolean> {
  try {
    await stat(filePath);
    return true;
  } catch {
    return false;
  }
}

async function findFilesByName(root: string, fileName: string): Promise<string[]> {
  const results: string[] = [];
  for (const entry of await readdir(root, { withFileTypes: true })) {
    const child = resolve(root, entry.name);
    if (entry.isDirectory()) {
      results.push(...(await findFilesByName(child, fileName)));
    } else if (entry.isFile() && entry.name === fileName) {
      results.push(child);
    }
  }
  return results;
}

function sha256Json(payload: unknown): string {
  return createHash("sha256").update(JSON.stringify(sortJson(payload))).digest("hex");
}

function sortJson(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(sortJson);
  }
  if (asRecord(value)) {
    return Object.fromEntries(Object.entries(value).sort(([left], [right]) => left.localeCompare(right)).map(([key, child]) => [key, sortJson(child)]));
  }
  return value;
}

function requiredString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${label} must be a non-empty string`);
  }
  return value;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? value as Record<string, unknown> : null;
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
  return path.normalize(value.replaceAll("\\", "/")).replace(/^\/+/, "").replace(/\/+$/, "");
}
