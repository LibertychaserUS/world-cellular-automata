import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

import type {
  MockActiveControlJobSpec,
  MockActiveControlReport,
  MockActiveControlSpec,
  QueueStatusState,
} from "./contracts.ts";
import { prepareShadowOutputDirectory, prepareShadowOutputFile } from "./shadow_paths.ts";

const MODE = "local_mock_active_control";

export async function runMockActiveControlFile(specPath: string): Promise<MockActiveControlReport> {
  const payload = JSON.parse(await readFile(specPath, "utf8")) as unknown;
  return runMockActiveControl(payload);
}

export async function runMockActiveControl(payload: unknown): Promise<MockActiveControlReport> {
  const validation = validateSpec(payload);
  if (validation.errors.length > 0 || !validation.spec) {
    return emptyReport(validation.errors);
  }

  const spec = validation.spec;
  let outputDir: string;
  let queueDir: string;
  try {
    outputDir = await prepareShadowOutputDirectory(spec.output_dir, "output_dir");
    queueDir = await prepareShadowOutputDirectory(resolve(outputDir, "queue"), "output_dir");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return reportForSpec(spec, null, null, [message], validation.warnings);
  }

  const artifacts = buildArtifacts(spec, queueDir);
  await mkdir(queueDir, { recursive: true });
  await writeShadowFile(resolve(queueDir, "manifest.json"), stableJson(artifacts.manifest));
  await writeShadowFile(resolve(queueDir, "remote_submission.json"), stableJson(artifacts.remoteSubmission));
  await writeShadowFile(resolve(queueDir, "status.jsonl"), artifacts.statusJsonl);
  await writeShadowFile(resolve(queueDir, "queue_summary.json"), stableJson(artifacts.queueSummary));

  return reportForSpec(spec, queueDir, artifacts.state, artifacts.ok ? [] : ["mock queue outcome requested failure"], validation.warnings);
}

function validateSpec(payload: unknown): {
  spec: MockActiveControlSpec | null;
  errors: string[];
  warnings: string[];
} {
  const errors: string[] = [];
  const warnings: string[] = [];
  if (!isRecord(payload)) {
    return { spec: null, errors: ["mock spec must be a JSON object"], warnings };
  }
  if (payload.mode !== MODE) {
    errors.push(`mode must be '${MODE}'`);
  }
  if (payload.allow_remote !== false) {
    errors.push("allow_remote must be false");
  }
  const experimentId = requiredString(payload, "experiment_id", errors);
  const queueName = requiredString(payload, "queue_name", errors);
  const runId = requiredString(payload, "run_id", errors);
  const outputDir = requiredString(payload, "output_dir", errors);
  const baseTime = requiredString(payload, "base_time", errors);
  const outcome = payload.outcome === "success" || payload.outcome === "failure" ? payload.outcome : null;
  if (!outcome) {
    errors.push("outcome must be 'success' or 'failure'");
  }
  const jobs = validateJobs(payload.jobs, errors);
  if (baseTime && Number.isNaN(Date.parse(baseTime))) {
    errors.push("base_time must be an ISO timestamp");
  }
  if (jobs.length === 0) {
    errors.push("jobs must contain at least one job");
  }
  if (errors.length > 0 || !experimentId || !queueName || !runId || !outputDir || !baseTime || !outcome) {
    return { spec: null, errors, warnings };
  }
  return {
    spec: {
      schema_version: 1,
      mode: MODE,
      allow_remote: false,
      experiment_id: experimentId,
      queue_name: queueName,
      run_id: runId,
      output_dir: outputDir,
      outcome,
      base_time: baseTime,
      jobs,
    },
    errors,
    warnings,
  };
}

function validateJobs(value: unknown, errors: string[]): MockActiveControlJobSpec[] {
  if (!Array.isArray(value)) {
    errors.push("jobs must be an array");
    return [];
  }
  const seen = new Set<string>();
  const jobs: MockActiveControlJobSpec[] = [];
  for (const [index, item] of value.entries()) {
    if (!isRecord(item)) {
      errors.push(`jobs[${index}] must be an object`);
      continue;
    }
    const jobId = requiredString(item, "job_id", errors, `jobs[${index}].`);
    if (!jobId) {
      continue;
    }
    if (seen.has(jobId)) {
      errors.push(`jobs[${index}].job_id duplicates '${jobId}'`);
      continue;
    }
    seen.add(jobId);
    jobs.push({
      job_id: jobId,
      exit_code: optionalInteger(item.exit_code, 0),
      duration_seconds: optionalNonNegativeNumber(item.duration_seconds, 1),
      timed_out: typeof item.timed_out === "boolean" ? item.timed_out : false,
    });
  }
  return jobs;
}

function buildArtifacts(spec: MockActiveControlSpec, queueDir: string): {
  ok: boolean;
  state: QueueStatusState;
  manifest: Record<string, unknown>;
  remoteSubmission: Record<string, unknown>;
  statusJsonl: string;
  queueSummary: Record<string, unknown>;
} {
  const jobSpecs = spec.jobs.map((job, index) => {
    const forcedFailure = spec.outcome === "failure" && index === spec.jobs.length - 1;
    return {
      ...job,
      exit_code: forcedFailure ? (job.exit_code && job.exit_code !== 0 ? job.exit_code : 1) : (job.exit_code ?? 0),
      duration_seconds: job.duration_seconds ?? 1,
      timed_out: job.timed_out ?? false,
    };
  });
  const queueExitCode = jobSpecs.some((job) => job.exit_code !== 0 || job.timed_out) ? 1 : 0;
  const state: QueueStatusState = queueExitCode === 0 ? "complete" : "failed";
  const startedAt = addSeconds(spec.base_time, 0);
  const finishedAt = addSeconds(spec.base_time, jobSpecs.length * 2 + 1);
  const events: Record<string, unknown>[] = [
    {
      at: startedAt,
      event: "queue_started",
      job_count: jobSpecs.length,
      queue_name: spec.queue_name,
      run_id: spec.run_id,
    },
  ];
  const summaryJobs = jobSpecs.map((job, index) => {
    const jobStartedAt = addSeconds(spec.base_time, index * 2 + 1);
    const jobFinishedAt = addSeconds(spec.base_time, index * 2 + 2);
    events.push({ at: jobStartedAt, event: "job_started", job_id: job.job_id, run_id: spec.run_id });
    events.push({
      at: jobFinishedAt,
      duration_seconds: job.duration_seconds,
      event: "job_finished",
      exit_code: job.exit_code,
      job_id: job.job_id,
      run_id: spec.run_id,
      timed_out: job.timed_out,
    });
    return {
      duration_seconds: job.duration_seconds,
      exit_code: job.exit_code,
      finished_at: jobFinishedAt,
      id: job.job_id,
      started_at: jobStartedAt,
      timed_out: job.timed_out,
    };
  });
  events.push({
    at: finishedAt,
    event: "queue_finished",
    exit_code: queueExitCode,
    queue_name: spec.queue_name,
    run_id: spec.run_id,
  });

  return {
    ok: queueExitCode === 0,
    state,
    manifest: {
      schema_version: 1,
      control_plane: "wca-exp-control-ts",
      experiment_id: spec.experiment_id,
      mode: MODE,
      queue_dir: queueDir,
      queue_name: spec.queue_name,
      run_id: spec.run_id,
      shadow_only: true,
    },
    remoteSubmission: {
      schema_version: 1,
      active_takeover_allowed: false,
      allow_remote: false,
      dry_run: true,
      experiment_id: spec.experiment_id,
      mode: MODE,
      queue_dir: queueDir,
      queue_name: spec.queue_name,
      remote_submission: "not_attempted",
      run_id: spec.run_id,
    },
    statusJsonl: `${events.map((event) => JSON.stringify(sortJson(event))).join("\n")}\n`,
    queueSummary: {
      queue_name: spec.queue_name,
      run_id: spec.run_id,
      started_at: startedAt,
      finished_at: finishedAt,
      exit_code: queueExitCode,
      jobs: summaryJobs,
    },
  };
}

function reportForSpec(
  spec: MockActiveControlSpec,
  queueDir: string | null,
  state: QueueStatusState | null,
  errors: string[],
  warnings: string[],
): MockActiveControlReport {
  const ok = errors.length === 0;
  return {
    schema_version: 1,
    ok,
    mode: MODE,
    active_takeover_allowed: false,
    allow_remote: false,
    experiment_id: spec.experiment_id,
    queue_name: spec.queue_name,
    run_id: spec.run_id,
    queue_dir: queueDir,
    state,
    exit_code: ok ? 0 : 1,
    artifact_paths: artifactPaths(queueDir),
    errors,
    warnings,
    notes: [
      "Local mock active-control lifecycle only. It does not SSH, rsync, scp, tar, submit, fetch, train, or evaluate.",
    ],
  };
}

function emptyReport(errors: string[]): MockActiveControlReport {
  return {
    schema_version: 1,
    ok: false,
    mode: MODE,
    active_takeover_allowed: false,
    allow_remote: false,
    experiment_id: null,
    queue_name: null,
    run_id: null,
    queue_dir: null,
    state: null,
    exit_code: 1,
    artifact_paths: artifactPaths(null),
    errors,
    warnings: [],
    notes: [
      "Local mock active-control lifecycle only. It does not SSH, rsync, scp, tar, submit, fetch, train, or evaluate.",
    ],
  };
}

function artifactPaths(queueDir: string | null): MockActiveControlReport["artifact_paths"] {
  return {
    manifest_json: queueDir ? resolve(queueDir, "manifest.json") : null,
    remote_submission_json: queueDir ? resolve(queueDir, "remote_submission.json") : null,
    status_jsonl: queueDir ? resolve(queueDir, "status.jsonl") : null,
    queue_summary_json: queueDir ? resolve(queueDir, "queue_summary.json") : null,
  };
}

async function writeShadowFile(path: string, contents: string): Promise<void> {
  await writeFile(await prepareShadowOutputFile(path, "output_dir"), contents, "utf8");
}

function requiredString(
  payload: Record<string, unknown>,
  key: string,
  errors: string[],
  prefix = "",
): string | null {
  const value = payload[key];
  if (typeof value !== "string" || value.trim() === "") {
    errors.push(`${prefix}${key} must be a non-empty string`);
    return null;
  }
  return value;
}

function optionalInteger(value: unknown, fallback: number): number {
  return Number.isInteger(value) ? Number(value) : fallback;
}

function optionalNonNegativeNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : fallback;
}

function stableJson(value: unknown): string {
  return `${JSON.stringify(sortJson(value), null, 2)}\n`;
}

function sortJson(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(sortJson);
  }
  if (!isRecord(value)) {
    return value;
  }
  return Object.fromEntries(
    Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, nested]) => [key, sortJson(nested)]),
  );
}

function addSeconds(baseTime: string, seconds: number): string {
  const date = new Date(Date.parse(baseTime) + seconds * 1000);
  return date.toISOString().replace(".000Z", "Z");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
