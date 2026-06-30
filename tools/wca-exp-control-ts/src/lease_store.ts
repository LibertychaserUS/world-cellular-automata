import { readFile } from "node:fs/promises";

import type {
  ResourceLease,
  RuntimeLeaseRecord,
  RuntimeLeaseStoreValidationReport,
} from "./contracts.ts";

const runtimeLeaseStatuses = new Set(["active", "released", "stale"]);
const resourceTypes = new Set(["gpu_group", "writable_path", "cache_path", "cache_write_path"]);
const lockModes = new Set(["exclusive_gpu", "exclusive_write", "shared_read"]);

export async function validateRuntimeLeaseStoreFile(path: string): Promise<RuntimeLeaseStoreValidationReport> {
  try {
    const payload = JSON.parse(await readFile(path, "utf8")) as unknown;
    return validateRuntimeLeaseStoreSnapshot(payload);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return invalidStoreReport([`failed to read or parse lease store: ${message}`]);
  }
}

export function validateRuntimeLeaseStoreSnapshot(payload: unknown): RuntimeLeaseStoreValidationReport {
  const errors: string[] = [];
  const warnings: string[] = [];
  if (!isRecord(payload)) {
    return invalidStoreReport(["lease store must be an object"]);
  }

  if (payload.schema_version !== 1) {
    errors.push("schema_version must be 1");
  }
  if (payload.mode !== "shadow_runtime_lease_store") {
    errors.push("mode must be shadow_runtime_lease_store");
  }
  const leaseStoreId = typeof payload.lease_store_id === "string" && payload.lease_store_id.length > 0
    ? payload.lease_store_id
    : null;
  if (!leaseStoreId) {
    errors.push("lease_store_id must be a non-empty string");
  }
  requireNonNegativeNumber(payload.generated_at_epoch_seconds, "generated_at_epoch_seconds", errors);
  const leases = Array.isArray(payload.leases) ? payload.leases : [];
  if (!Array.isArray(payload.leases)) {
    errors.push("leases must be an array");
  }

  const validLeaseRecords: RuntimeLeaseRecord[] = [];
  leases.forEach((lease, index) => {
    const leaseErrors = validateRuntimeLeaseRecord(lease, `leases[${index}]`);
    if (leaseErrors.length > 0) {
      errors.push(...leaseErrors);
    } else {
      validLeaseRecords.push(lease as RuntimeLeaseRecord);
    }
  });

  const duplicateRuntimeLeaseIds = duplicateIds(leases.flatMap((lease) => collectRuntimeLeaseIds(lease)));
  const duplicateResourceLeaseIds = duplicateIds(leases.flatMap((lease) => collectResourceLeaseIds(lease)));
  for (const id of duplicateRuntimeLeaseIds) {
    errors.push(`duplicate runtime_lease_id: ${id}`);
  }
  for (const id of duplicateResourceLeaseIds) {
    errors.push(`duplicate resource lease_id: ${id}`);
  }

  if (validLeaseRecords.some((lease) => lease.status === "active" && lease.resource_leases.length === 0)) {
    warnings.push("active lease records without resource_leases do not protect any resource");
  }

  return {
    schema_version: 1,
    ok: errors.length === 0,
    mode: "shadow_runtime_lease_store_validation",
    lease_store_id: leaseStoreId,
    lease_count: leases.length,
    active_lease_count: validLeaseRecords.filter((lease) => lease.status === "active").length,
    duplicate_runtime_lease_id_count: duplicateRuntimeLeaseIds.length,
    duplicate_resource_lease_id_count: duplicateResourceLeaseIds.length,
    invalid_lease_count: leases.length - validLeaseRecords.length,
    duplicate_runtime_lease_ids: duplicateRuntimeLeaseIds,
    duplicate_resource_lease_ids: duplicateResourceLeaseIds,
    errors,
    warnings,
  };
}

function validateRuntimeLeaseRecord(value: unknown, path: string): string[] {
  const errors: string[] = [];
  if (!isRecord(value)) {
    return [`${path} must be an object`];
  }
  if (value.schema_version !== 1) {
    errors.push(`${path}.schema_version must be 1`);
  }
  requireNonEmptyString(value.runtime_lease_id, `${path}.runtime_lease_id`, errors);
  requireNonEmptyString(value.queue_id, `${path}.queue_id`, errors);
  requireNonEmptyString(value.job_id, `${path}.job_id`, errors);
  if (typeof value.status !== "string" || !runtimeLeaseStatuses.has(value.status)) {
    errors.push(`${path}.status must be active, released, or stale`);
  }
  requirePositiveNumber(value.ttl_seconds, `${path}.ttl_seconds`, errors);
  validateOwner(value.owner, `${path}.owner`, errors);
  if (!Array.isArray(value.resource_leases)) {
    errors.push(`${path}.resource_leases must be an array`);
    return errors;
  }
  value.resource_leases.forEach((lease, index) => {
    errors.push(...validateResourceLease(lease, `${path}.resource_leases[${index}]`, value.job_id));
  });
  return errors;
}

function validateOwner(value: unknown, path: string, errors: string[]): void {
  if (!isRecord(value)) {
    errors.push(`${path} must be an object`);
    return;
  }
  requireNonEmptyString(value.hostname, `${path}.hostname`, errors);
  requirePositiveInteger(value.pid, `${path}.pid`, errors);
  requireNonNegativeNumber(value.started_at_epoch_seconds, `${path}.started_at_epoch_seconds`, errors);
  requireNonNegativeNumber(value.last_heartbeat_epoch_seconds, `${path}.last_heartbeat_epoch_seconds`, errors);
  if (
    typeof value.started_at_epoch_seconds === "number" &&
    typeof value.last_heartbeat_epoch_seconds === "number" &&
    value.last_heartbeat_epoch_seconds < value.started_at_epoch_seconds
  ) {
    errors.push(`${path}.last_heartbeat_epoch_seconds must be >= started_at_epoch_seconds`);
  }
}

function validateResourceLease(value: unknown, path: string, ownerJobId: unknown): string[] {
  const errors: string[] = [];
  if (!isRecord(value)) {
    return [`${path} must be an object`];
  }
  requireNonEmptyString(value.lease_id, `${path}.lease_id`, errors);
  requireNonEmptyString(value.request_id, `${path}.request_id`, errors);
  requireNonEmptyString(value.job_id, `${path}.job_id`, errors);
  if (typeof ownerJobId === "string" && typeof value.job_id === "string" && value.job_id !== ownerJobId) {
    errors.push(`${path}.job_id must match parent job_id`);
  }
  if (typeof value.resource_type !== "string" || !resourceTypes.has(value.resource_type)) {
    errors.push(`${path}.resource_type is invalid`);
  }
  if (typeof value.mode !== "string" || !lockModes.has(value.mode)) {
    errors.push(`${path}.mode is invalid`);
  }
  requireNonNegativeNumber(value.starts_at_seconds, `${path}.starts_at_seconds`, errors);
  requireNonNegativeNumber(value.ends_at_seconds, `${path}.ends_at_seconds`, errors);
  if (
    typeof value.starts_at_seconds === "number" &&
    typeof value.ends_at_seconds === "number" &&
    value.ends_at_seconds < value.starts_at_seconds
  ) {
    errors.push(`${path}.ends_at_seconds must be >= starts_at_seconds`);
  }
  if (typeof value.atomic !== "boolean") {
    errors.push(`${path}.atomic must be a boolean`);
  }
  if (value.enforcement !== "shadow_only") {
    errors.push(`${path}.enforcement must be shadow_only`);
  }
  validateLeaseResourceShape(value as Partial<ResourceLease>, path, errors);
  return errors;
}

function validateLeaseResourceShape(value: Partial<ResourceLease>, path: string, errors: string[]): void {
  if (value.resource_type === "gpu_group") {
    if (value.mode !== "exclusive_gpu") {
      errors.push(`${path}.mode must be exclusive_gpu for gpu_group`);
    }
    if (!Array.isArray(value.gpu_ids) || value.gpu_ids.length === 0) {
      errors.push(`${path}.gpu_ids must be a non-empty array for gpu_group`);
    } else {
      const invalidGpuIds = value.gpu_ids.filter((gpuId) => !Number.isInteger(gpuId) || gpuId < 0);
      if (invalidGpuIds.length > 0) {
        errors.push(`${path}.gpu_ids must contain non-negative integers`);
      }
      if (new Set(value.gpu_ids).size !== value.gpu_ids.length) {
        errors.push(`${path}.gpu_ids must not contain duplicates`);
      }
    }
  } else if (value.resource_type && value.mode === "exclusive_gpu") {
    errors.push(`${path}.mode exclusive_gpu is only valid for gpu_group`);
  }

  if (value.resource_type === "writable_path" || value.resource_type === "cache_path" || value.resource_type === "cache_write_path") {
    if (typeof value.path !== "string" || value.path.length === 0) {
      errors.push(`${path}.path must be a non-empty string for path leases`);
    } else if (!isSafeRelativePath(value.path)) {
      errors.push(`${path}.path must be a safe relative path for path leases`);
    }
    if (value.resource_type === "cache_path" && value.mode !== "shared_read") {
      errors.push(`${path}.mode must be shared_read for cache_path`);
    }
    if ((value.resource_type === "writable_path" || value.resource_type === "cache_write_path") && value.mode !== "exclusive_write") {
      errors.push(`${path}.mode must be exclusive_write for writable path leases`);
    }
  }
}

function invalidStoreReport(errors: string[]): RuntimeLeaseStoreValidationReport {
  return {
    schema_version: 1,
    ok: false,
    mode: "shadow_runtime_lease_store_validation",
    lease_store_id: null,
    lease_count: 0,
    active_lease_count: 0,
    duplicate_runtime_lease_id_count: 0,
    duplicate_resource_lease_id_count: 0,
    invalid_lease_count: 0,
    duplicate_runtime_lease_ids: [],
    duplicate_resource_lease_ids: [],
    errors,
    warnings: [],
  };
}

function collectRuntimeLeaseIds(value: unknown): string[] {
  if (!isRecord(value) || typeof value.runtime_lease_id !== "string" || value.runtime_lease_id.length === 0) {
    return [];
  }
  return [value.runtime_lease_id];
}

function collectResourceLeaseIds(value: unknown): string[] {
  if (!isRecord(value) || !Array.isArray(value.resource_leases)) {
    return [];
  }
  return value.resource_leases.flatMap((lease) => {
    if (!isRecord(lease) || typeof lease.lease_id !== "string" || lease.lease_id.length === 0) {
      return [];
    }
    return [lease.lease_id];
  });
}

function isSafeRelativePath(value: string): boolean {
  if (
    value.length === 0 ||
    value.startsWith("/") ||
    value.startsWith("\\") ||
    /^[A-Za-z]:[\\/]/u.test(value) ||
    value.includes("\0")
  ) {
    return false;
  }
  return !value.split(/[\\/]+/u).includes("..");
}

function duplicateIds(values: string[]): string[] {
  const seen = new Set<string>();
  const duplicates = new Set<string>();
  for (const value of values) {
    if (seen.has(value)) {
      duplicates.add(value);
    }
    seen.add(value);
  }
  return [...duplicates].sort();
}

function requireNonEmptyString(value: unknown, path: string, errors: string[]): void {
  if (typeof value !== "string" || value.length === 0) {
    errors.push(`${path} must be a non-empty string`);
  }
}

function requirePositiveInteger(value: unknown, path: string, errors: string[]): void {
  if (!Number.isInteger(value) || typeof value !== "number" || value <= 0) {
    errors.push(`${path} must be a positive integer`);
  }
}

function requirePositiveNumber(value: unknown, path: string, errors: string[]): void {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    errors.push(`${path} must be a positive finite number`);
  }
}

function requireNonNegativeNumber(value: unknown, path: string, errors: string[]): void {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    errors.push(`${path} must be a non-negative finite number`);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
