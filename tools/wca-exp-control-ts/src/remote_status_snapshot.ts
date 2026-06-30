import { readFile } from "node:fs/promises";

import type {
  QueueStatusReport,
  QueueStatusState,
  RemoteStatusSnapshotValidationReport,
} from "./contracts.ts";

const queueStates = new Set(["unknown", "waiting", "running", "complete", "failed"]);
const forbiddenKeyPattern = /(password|passwd|secret|token|api[_-]?key|ssh_command|private_key)/iu;

export async function validateRemoteStatusSnapshotFile(path: string): Promise<RemoteStatusSnapshotValidationReport> {
  try {
    const payload = JSON.parse(await readFile(path, "utf8")) as unknown;
    return validateRemoteStatusSnapshot(payload);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return invalidRemoteStatusSnapshotReport([`failed to read or parse remote status snapshot: ${message}`]);
  }
}

export function validateRemoteStatusSnapshot(payload: unknown): RemoteStatusSnapshotValidationReport {
  const errors: string[] = [];
  const warnings: string[] = [];
  if (!isRecord(payload)) {
    return invalidRemoteStatusSnapshotReport(["remote status snapshot must be an object"]);
  }

  if (payload.schema_version !== 1) {
    errors.push("schema_version must be 1");
  }
  if (payload.mode !== "shadow_remote_status_snapshot") {
    errors.push("mode must be shadow_remote_status_snapshot");
  }
  const snapshotId = requireString(payload.snapshot_id, "snapshot_id", errors);
  requireNonNegativeNumber(payload.generated_at_epoch_seconds, "generated_at_epoch_seconds", errors);
  if (payload.source !== "remote_read_only_probe_fixture") {
    errors.push("source must be remote_read_only_probe_fixture");
  }
  requireSafeLabel(payload.remote_host_label, "remote_host_label", errors);
  requireSafeRemotePath(payload.remote_root, "remote_root", errors);
  const forbiddenKeys = findForbiddenKeys(payload);
  if (forbiddenKeys.length > 0) {
    errors.push(`snapshot contains forbidden secret/command-like keys: ${forbiddenKeys.join(", ")}`);
  }

  const queueStatus = validateQueueStatus(payload.queue_status, errors, warnings);

  return {
    schema_version: 1,
    ok: errors.length === 0,
    mode: "shadow_remote_status_snapshot_validation",
    snapshot_id: snapshotId,
    queue_name: queueStatus?.queue_name ?? null,
    queue_state: queueStatus?.state ?? null,
    current_job_id: queueStatus?.current_job_id ?? null,
    error_count: errors.length,
    warning_count: warnings.length,
    errors,
    warnings,
  };
}

function validateQueueStatus(
  value: unknown,
  errors: string[],
  warnings: string[],
): QueueStatusReport | null {
  if (!isRecord(value)) {
    errors.push("queue_status must be an object");
    return null;
  }
  if (value.schema_version !== 1) {
    errors.push("queue_status.schema_version must be 1");
  }
  if (value.mode !== "shadow_queue_status") {
    errors.push("queue_status.mode must be shadow_queue_status");
  }
  if (value.source !== "remote_read_only_snapshot") {
    errors.push("queue_status.source must be remote_read_only_snapshot");
  }
  if (value.ok !== true) {
    errors.push("queue_status.ok must be true");
  }
  requireSafeRemotePath(value.queue_dir, "queue_status.queue_dir", errors);
  requireNullableString(value.queue_name, "queue_status.queue_name", errors);
  requireNullableString(value.run_id, "queue_status.run_id", errors);
  if (typeof value.state !== "string" || !queueStates.has(value.state)) {
    errors.push("queue_status.state is invalid");
  } else if (value.state === "unknown") {
    errors.push("queue_status.state must not be unknown");
  }
  requireNullableString(value.current_job_id, "queue_status.current_job_id", errors);
  validateStringArray(value.current_job_ids, "queue_status.current_job_ids", errors);
  requireNullableString(value.started_at, "queue_status.started_at", errors);
  requireNullableString(value.finished_at, "queue_status.finished_at", errors);
  requireNullableNumber(value.exit_code, "queue_status.exit_code", errors);
  requireNullableNumber(value.job_count, "queue_status.job_count", errors);
  requireNonNegativeInteger(value.started_job_count, "queue_status.started_job_count", errors);
  requireNonNegativeInteger(value.finished_job_count, "queue_status.finished_job_count", errors);
  requireNonNegativeInteger(value.failed_job_count, "queue_status.failed_job_count", errors);
  requireNonNegativeInteger(value.status_event_count, "queue_status.status_event_count", errors);
  requireNonNegativeInteger(value.summary_job_count, "queue_status.summary_job_count", errors);
  validateJobs(value.jobs, errors);
  const missingFiles = validateStringArray(value.missing_files, "queue_status.missing_files", errors);
  const parseErrors = validateStringArray(value.parse_errors, "queue_status.parse_errors", errors);
  validateStringArray(value.notes, "queue_status.notes", errors);
  if ((missingFiles?.length ?? 0) > 0) {
    errors.push("queue_status.missing_files must be empty for remote read-only status evidence");
  }
  if ((parseErrors?.length ?? 0) > 0) {
    errors.push("queue_status.parse_errors must be empty for remote read-only status evidence");
  }
  if (value.state === "running" && (!Array.isArray(value.current_job_ids) || value.current_job_ids.length === 0)) {
    errors.push("queue_status.current_job_ids must be non-empty when state is running");
  }
  if (value.state === "complete" && value.exit_code !== 0) {
    errors.push("queue_status.exit_code must be 0 when state is complete");
  }
  if (value.state === "failed" && value.exit_code === 0) {
    errors.push("queue_status.exit_code must be non-zero or null when state is failed");
  }
  if (Array.isArray(value.notes) && !value.notes.some((note) => typeof note === "string" && note.includes("read-only"))) {
    warnings.push("queue_status.notes should describe the read-only boundary");
  }

  return value as unknown as QueueStatusReport;
}

function validateJobs(value: unknown, errors: string[]): void {
  if (!Array.isArray(value)) {
    errors.push("queue_status.jobs must be an array");
    return;
  }
  value.forEach((job, index) => {
    if (!isRecord(job)) {
      errors.push(`queue_status.jobs[${index}] must be an object`);
      return;
    }
    requireString(job.job_id, `queue_status.jobs[${index}].job_id`, errors);
    requireNullableString(job.started_at, `queue_status.jobs[${index}].started_at`, errors);
    requireNullableString(job.finished_at, `queue_status.jobs[${index}].finished_at`, errors);
    requireNullableNumber(job.exit_code, `queue_status.jobs[${index}].exit_code`, errors);
    requireNullableNumber(job.duration_seconds, `queue_status.jobs[${index}].duration_seconds`, errors);
    if (typeof job.timed_out !== "boolean" && job.timed_out !== null) {
      errors.push(`queue_status.jobs[${index}].timed_out must be boolean or null`);
    }
  });
}

function validateStringArray(value: unknown, path: string, errors: string[]): string[] | null {
  if (!Array.isArray(value)) {
    errors.push(`${path} must be an array`);
    return null;
  }
  const result: string[] = [];
  value.forEach((item, index) => {
    if (typeof item !== "string") {
      errors.push(`${path}[${index}] must be a string`);
    } else {
      result.push(item);
    }
  });
  return result;
}

function requireString(value: unknown, path: string, errors: string[]): string | null {
  if (typeof value !== "string" || value.length === 0) {
    errors.push(`${path} must be a non-empty string`);
    return null;
  }
  return value;
}

function requireNullableString(value: unknown, path: string, errors: string[]): void {
  if (typeof value !== "string" && value !== null) {
    errors.push(`${path} must be a string or null`);
  }
}

function requireNullableNumber(value: unknown, path: string, errors: string[]): void {
  if ((typeof value !== "number" || !Number.isFinite(value)) && value !== null) {
    errors.push(`${path} must be a finite number or null`);
  }
}

function requireNonNegativeInteger(value: unknown, path: string, errors: string[]): void {
  if (!Number.isInteger(value) || typeof value !== "number" || value < 0) {
    errors.push(`${path} must be a non-negative integer`);
  }
}

function requireNonNegativeNumber(value: unknown, path: string, errors: string[]): void {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    errors.push(`${path} must be a non-negative finite number`);
  }
}

function requireSafeLabel(value: unknown, path: string, errors: string[]): void {
  if (typeof value !== "string" || value.length === 0 || value.includes("\0") || value.includes("@")) {
    errors.push(`${path} must be a sanitized non-empty label`);
  }
}

function requireSafeRemotePath(value: unknown, path: string, errors: string[]): void {
  if (typeof value !== "string" || value.length === 0 || value.includes("\0") || value.includes("@")) {
    errors.push(`${path} must be a safe remote path string`);
  }
}

function findForbiddenKeys(value: unknown): string[] {
  const hits = new Set<string>();
  visitKeys(value, "", hits);
  return [...hits].sort();
}

function visitKeys(value: unknown, prefix: string, hits: Set<string>): void {
  if (!isRecord(value) && !Array.isArray(value)) {
    return;
  }
  if (Array.isArray(value)) {
    value.forEach((item, index) => visitKeys(item, `${prefix}[${index}]`, hits));
    return;
  }
  for (const [key, child] of Object.entries(value)) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (forbiddenKeyPattern.test(key)) {
      hits.add(path);
    }
    visitKeys(child, path, hits);
  }
}

function invalidRemoteStatusSnapshotReport(errors: string[]): RemoteStatusSnapshotValidationReport {
  return {
    schema_version: 1,
    ok: false,
    mode: "shadow_remote_status_snapshot_validation",
    snapshot_id: null,
    queue_name: null,
    queue_state: null,
    current_job_id: null,
    error_count: errors.length,
    warning_count: 0,
    errors,
    warnings: [],
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
