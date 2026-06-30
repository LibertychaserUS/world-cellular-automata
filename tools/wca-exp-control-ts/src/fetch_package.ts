import { createReadStream } from "node:fs";
import { readFile, stat } from "node:fs/promises";
import { createHash } from "node:crypto";
import { dirname, resolve } from "node:path";

import type {
  FetchPackageArchiveClaim,
  FetchPackageArchiveIntegrityReport,
  FetchPackageManifest,
  FetchPackageMode,
  FetchPackageValidationReport,
} from "./contracts.ts";

const packageModes = new Set(["report_first", "raw_complete"]);
const sha256Pattern = /^[a-f0-9]{64}$/u;

export async function validateFetchPackageFile(path: string): Promise<FetchPackageValidationReport> {
  try {
    const payload = JSON.parse(await readFile(path, "utf8")) as unknown;
    return await validateFetchPackageManifest(payload, dirname(path));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return invalidFetchPackageReport([`failed to read or parse fetch package: ${message}`]);
  }
}

export async function validateFetchPackageManifest(
  payload: unknown,
  manifestDirectory: string,
): Promise<FetchPackageValidationReport> {
  const errors: string[] = [];
  const warnings: string[] = [];
  if (!isRecord(payload)) {
    return invalidFetchPackageReport(["fetch package must be an object"]);
  }

  if (payload.schema_version !== 1) {
    errors.push("schema_version must be 1");
  }
  const fetchMode = typeof payload.mode === "string" && packageModes.has(payload.mode)
    ? payload.mode as FetchPackageMode
    : null;
  if (!fetchMode) {
    errors.push("mode must be report_first or raw_complete");
  }
  const packageId = typeof payload.package_id === "string" && payload.package_id.length > 0
    ? payload.package_id
    : null;
  if (!packageId) {
    errors.push("package_id must be a non-empty string");
  }
  requireNonNegativeNumber(payload.generated_at_epoch_seconds, "generated_at_epoch_seconds", errors);

  const packageRootValue = typeof payload.package_root === "string" ? payload.package_root : ".";
  const packageRootSafe = isSafeRelativePath(packageRootValue);
  if (!packageRootSafe) {
    errors.push("package_root must be a safe relative path");
  }
  const packageRoot = packageRootSafe ? resolve(manifestDirectory, packageRootValue) : manifestDirectory;

  const requestedArtifactPaths = stringArray(payload.requested_artifact_paths, "requested_artifact_paths", errors);
  const existingArtifactPaths = stringArray(payload.existing_artifact_paths, "existing_artifact_paths", errors);
  const missingArtifactPaths = stringArray(payload.missing_artifact_paths, "missing_artifact_paths", errors);
  const compactReportPaths = optionalStringArray(payload.compact_report_paths, "compact_report_paths", errors);

  validatePathArray(requestedArtifactPaths, "requested_artifact_paths", errors);
  validatePathArray(existingArtifactPaths, "existing_artifact_paths", errors);
  validatePathArray(missingArtifactPaths, "missing_artifact_paths", errors);
  validatePathArray(compactReportPaths, "compact_report_paths", errors);

  if (requestedArtifactPaths.length === 0) {
    errors.push("requested_artifact_paths must contain at least one path");
  }
  if (missingArtifactPaths.length > 0) {
    errors.push("missing_artifact_paths must be empty for a valid fetched package");
  }
  if (fetchMode === "report_first" && compactReportPaths.length === 0) {
    errors.push("compact_report_paths must contain at least one path for report_first packages");
  }

  const existingSet = new Set(existingArtifactPaths);
  for (const compactPath of compactReportPaths) {
    if (!existingSet.has(compactPath)) {
      errors.push(`compact_report_paths entry is not listed in existing_artifact_paths: ${compactPath}`);
    }
  }

  const archiveClaim = validateArchiveClaim(payload.archive, fetchMode, errors);
  const localFilePaths = fetchMode === "raw_complete"
    ? [...new Set(compactReportPaths)].filter(isSafeRelativePath)
    : [...new Set([...existingArtifactPaths, ...compactReportPaths])].filter(isSafeRelativePath);
  let missingLocalFileCount = 0;
  for (const localPath of localFilePaths) {
    const filePath = resolve(packageRoot, localPath);
    if (!(await isFile(filePath))) {
      missingLocalFileCount += 1;
      errors.push(`local fetched artifact is missing or not a file: ${localPath}`);
    }
  }

  let archiveIntegrity: FetchPackageArchiveIntegrityReport | null = null;
  if (archiveClaim) {
    archiveIntegrity = await inspectArchiveIntegrity(archiveClaim, packageRoot, missingArtifactPaths.length, errors);
  }

  return {
    schema_version: 1,
    ok: errors.length === 0,
    mode: "shadow_fetch_package_validation",
    package_id: packageId,
    fetch_mode: fetchMode,
    package_root: packageRoot,
    requested_count: requestedArtifactPaths.length,
    existing_count: existingArtifactPaths.length,
    missing_count: missingArtifactPaths.length,
    compact_report_count: compactReportPaths.length,
    checked_local_file_count: localFilePaths.length,
    missing_local_file_count: missingLocalFileCount,
    archive_integrity: archiveIntegrity,
    errors,
    warnings,
  };
}

function validateArchiveClaim(value: unknown, fetchMode: FetchPackageMode | null, errors: string[]): FetchPackageArchiveClaim | null {
  if (fetchMode === "raw_complete" && !isRecord(value)) {
    errors.push("archive must be an object for raw_complete packages");
    return null;
  }
  if (value === undefined) {
    return null;
  }
  if (!isRecord(value)) {
    errors.push("archive must be an object");
    return null;
  }

  const remoteArchivePath = stringField(value.remote_archive_path, "archive.remote_archive_path", errors);
  const localArchivePath = stringField(value.local_archive_path, "archive.local_archive_path", errors);
  requireNonNegativeNumber(value.remote_byte_size, "archive.remote_byte_size", errors);
  const remoteSha256 = stringField(value.remote_sha256, "archive.remote_sha256", errors);

  if (remoteArchivePath && !isSafeRemoteArchivePath(remoteArchivePath)) {
    errors.push("archive.remote_archive_path must be a safe path string");
  }
  const remoteArchivePathSafe = remoteArchivePath ? isSafeRemoteArchivePath(remoteArchivePath) : false;
  if (localArchivePath && !isSafeRelativePath(localArchivePath)) {
    errors.push("archive.local_archive_path must be a safe relative path");
  }
  const localArchivePathSafe = localArchivePath ? isSafeRelativePath(localArchivePath) : false;
  if (remoteSha256 && !sha256Pattern.test(remoteSha256)) {
    errors.push("archive.remote_sha256 must be a lowercase hex sha256");
  }
  if (
    typeof value.remote_byte_size === "number" &&
    Number.isFinite(value.remote_byte_size) &&
    value.remote_byte_size === 0
  ) {
    errors.push("archive.remote_byte_size must be greater than zero");
  }

  if (
    !remoteArchivePath ||
    !localArchivePath ||
    typeof value.remote_byte_size !== "number" ||
    !Number.isFinite(value.remote_byte_size) ||
    value.remote_byte_size <= 0 ||
    !remoteSha256 ||
    !sha256Pattern.test(remoteSha256) ||
    !remoteArchivePathSafe ||
    !localArchivePathSafe
  ) {
    return null;
  }

  return {
    remote_archive_path: remoteArchivePath,
    local_archive_path: localArchivePath,
    remote_byte_size: value.remote_byte_size,
    remote_sha256: remoteSha256,
  };
}

async function inspectArchiveIntegrity(
  archive: FetchPackageArchiveClaim,
  packageRoot: string,
  missingArtifactCount: number,
  errors: string[],
): Promise<FetchPackageArchiveIntegrityReport> {
  const archivePath = resolve(packageRoot, archive.local_archive_path);
  let localByteSize: number | null = null;
  let localSha256: string | null = null;
  if (await isFile(archivePath)) {
    const fileStat = await stat(archivePath);
    localByteSize = fileStat.size;
    localSha256 = await sha256File(archivePath);
  } else {
    errors.push(`local archive is missing or not a file: ${archive.local_archive_path}`);
  }

  const sizeMatch = localByteSize === archive.remote_byte_size;
  const sha256Match = localSha256 === archive.remote_sha256;
  if (localByteSize !== null && !sizeMatch) {
    errors.push(`archive byte size mismatch: expected ${archive.remote_byte_size}, got ${localByteSize}`);
  }
  if (localSha256 !== null && !sha256Match) {
    errors.push("archive sha256 mismatch");
  }

  return {
    remote_archive_path: archive.remote_archive_path,
    local_archive_path: archive.local_archive_path,
    remote_byte_size: archive.remote_byte_size,
    local_byte_size: localByteSize,
    size_match: sizeMatch,
    remote_sha256: archive.remote_sha256,
    local_sha256: localSha256,
    sha256_match: sha256Match,
    extraction_allowed: sizeMatch && sha256Match && missingArtifactCount === 0,
  };
}

function stringArray(value: unknown, label: string, errors: string[]): string[] {
  if (!Array.isArray(value)) {
    errors.push(`${label} must be an array`);
    return [];
  }
  const result: string[] = [];
  value.forEach((item, index) => {
    if (typeof item !== "string" || item.length === 0) {
      errors.push(`${label}[${index}] must be a non-empty string`);
    } else {
      result.push(item);
    }
  });
  return result;
}

function optionalStringArray(value: unknown, label: string, errors: string[]): string[] {
  if (value === undefined) {
    return [];
  }
  return stringArray(value, label, errors);
}

function validatePathArray(values: string[], label: string, errors: string[]): void {
  const duplicates = duplicateValues(values);
  for (const duplicate of duplicates) {
    errors.push(`${label} contains duplicate path: ${duplicate}`);
  }
  for (const value of values) {
    if (!isSafeRelativePath(value)) {
      errors.push(`${label} contains unsafe path: ${value}`);
    }
  }
}

function stringField(value: unknown, label: string, errors: string[]): string | null {
  if (typeof value !== "string" || value.length === 0) {
    errors.push(`${label} must be a non-empty string`);
    return null;
  }
  return value;
}

async function isFile(path: string): Promise<boolean> {
  try {
    return (await stat(path)).isFile();
  } catch {
    return false;
  }
}

async function sha256File(path: string): Promise<string> {
  const hash = createHash("sha256");
  await new Promise<void>((resolvePromise, reject) => {
    const stream = createReadStream(path);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("error", reject);
    stream.on("end", resolvePromise);
  });
  return hash.digest("hex");
}

function duplicateValues(values: string[]): string[] {
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

function invalidFetchPackageReport(errors: string[]): FetchPackageValidationReport {
  return {
    schema_version: 1,
    ok: false,
    mode: "shadow_fetch_package_validation",
    package_id: null,
    fetch_mode: null,
    package_root: null,
    requested_count: 0,
    existing_count: 0,
    missing_count: 0,
    compact_report_count: 0,
    checked_local_file_count: 0,
    missing_local_file_count: 0,
    archive_integrity: null,
    errors,
    warnings: [],
  };
}

function requireNonNegativeNumber(value: unknown, path: string, errors: string[]): void {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    errors.push(`${path} must be a non-negative finite number`);
  }
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

function isSafeRemoteArchivePath(value: string): boolean {
  return value.length > 0 && !value.includes("\0");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
