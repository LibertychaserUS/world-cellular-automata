import type {
  JobProfileEstimate,
  ResourceProfileCoverageRequirement,
  ResourceProfileEstimateFilter,
  ResourceProfileLibraryAuditReport,
  ResourceProfileRequirementAudit,
  ResourceProfileRow,
  ResourceProfileValidationResult,
} from "./contracts.ts";

const defaultMemorySafetyFactor = 1.15;
const defaultWallClockSafetyFactor = 1.1;
const defaultGpuUtilization = 0.5;

export function validateResourceProfileRow(row: unknown): ResourceProfileValidationResult {
  const errors: string[] = [];
  if (!isRecord(row)) {
    return { ok: false, errors: ["row must be an object"] };
  }
  requireLiteral(row.schema_version, 1, "schema_version", errors);
  requireNonEmptyString(row.profile_id, "profile_id", errors);
  requireNonEmptyString(row.source_run_id, "source_run_id", errors);
  requireOptionalNonNegative(row.observed_at_epoch_seconds, "observed_at_epoch_seconds", errors);
  requireNonEmptyString(row.recipe_id, "recipe_id", errors);
  requireNonEmptyString(row.model_family, "model_family", errors);
  requireNonEmptyString(row.dataset_id, "dataset_id", errors);
  requireNonEmptyString(row.split_id, "split_id", errors);
  requireNonEmptyString(row.gpu_model, "gpu_model", errors);
  requirePositiveNumber(row.gpu_count, "gpu_count", errors);
  requirePositiveNumber(row.gpu_slots, "gpu_slots", errors);
  requirePositiveNumber(row.nodes_or_tokens, "nodes_or_tokens", errors);
  requirePositiveNumber(row.batch_size, "batch_size", errors);
  requireNonEmptyString(row.precision, "precision", errors);
  requirePositiveNumber(row.peak_memory_mb, "peak_memory_mb", errors);
  requirePositiveNumber(row.wall_clock_seconds, "wall_clock_seconds", errors);
  requireBoolean(row.oom, "oom", errors);
  requireOptionalNonNegative(row.steady_memory_mb, "steady_memory_mb", errors);
  requireOptionalNonNegative(row.train_step_count, "train_step_count", errors);
  requireOptionalNonNegative(row.step_time_ms, "step_time_ms", errors);
  requireOptionalNonNegative(row.eval_time_ms, "eval_time_ms", errors);
  requireOptionalNonNegative(row.samples_per_second, "samples_per_second", errors);
  requireOptionalFraction(row.gpu_utilization_mean, "gpu_utilization_mean", errors);
  requireOptionalFraction(row.gpu_utilization_p10, "gpu_utilization_p10", errors);
  requireOptionalFraction(row.gpu_utilization_p90, "gpu_utilization_p90", errors);
  requireOptionalNonNegative(row.host_ram_mb, "host_ram_mb", errors);
  requireOptionalNonNegative(row.cache_read_mb_s, "cache_read_mb_s", errors);
  requireOptionalNonNegative(row.cache_wait_seconds, "cache_wait_seconds", errors);

  if (row.gpu_slots > row.gpu_count) {
    errors.push("gpu_slots must be <= gpu_count");
  }
  if (
    typeof row.gpu_utilization_p10 === "number" &&
    typeof row.gpu_utilization_p90 === "number" &&
    row.gpu_utilization_p10 > row.gpu_utilization_p90
  ) {
    errors.push("gpu_utilization_p10 must be <= gpu_utilization_p90");
  }

  return { ok: errors.length === 0, errors };
}

export function validateResourceProfileRows(rows: unknown[]): ResourceProfileValidationResult {
  const errors: string[] = [];
  rows.forEach((row, index) => {
    const result = validateResourceProfileRow(row);
    for (const error of result.errors) {
      errors.push(`rows[${index}].${error}`);
    }
  });
  return { ok: errors.length === 0, errors };
}

export function auditResourceProfileLibrary(
  rows: unknown[],
  requirements: ResourceProfileCoverageRequirement[] = [],
  options: { now_epoch_seconds?: number } = {},
): ResourceProfileLibraryAuditReport {
  const rowsValidation = validateResourceProfileRows(rows);
  const validRows = rows.filter((row): row is ResourceProfileRow => validateResourceProfileRow(row).ok);
  const duplicateProfileIds = duplicateIds(validRows.map((row) => row.profile_id));
  const requirementAudits = requirements.map((requirement) =>
    auditCoverageRequirement(validRows, requirement, options.now_epoch_seconds),
  );
  const failingRequirementCount = requirementAudits.filter((audit) => audit.status === "fail").length;
  const duplicateErrors = duplicateProfileIds.map((profileId) => `duplicate profile_id: ${profileId}`);

  return {
    schema_version: 1,
    ok: rowsValidation.ok && duplicateProfileIds.length === 0 && failingRequirementCount === 0,
    mode: "shadow_resource_profile_library_audit",
    row_count: rows.length,
    valid_row_count: validRows.length,
    invalid_row_count: rows.length - validRows.length,
    duplicate_profile_id_count: duplicateProfileIds.length,
    requirement_count: requirements.length,
    passing_requirement_count: requirementAudits.length - failingRequirementCount,
    failing_requirement_count: failingRequirementCount,
    rows_validation_errors: [...rowsValidation.errors, ...duplicateErrors],
    duplicate_profile_ids: duplicateProfileIds,
    requirements: requirementAudits,
    notes: [
      "Profile rows are unit-bearing resource estimates for planner audit, not proof of future runtime behavior.",
      "Freshness checks require observed_at_epoch_seconds; rows without it fail freshness-gated requirements.",
    ],
  };
}

export function estimateJobProfileFromRows(
  rows: ResourceProfileRow[],
  options: {
    recipe_id: string;
    match?: ResourceProfileEstimateFilter;
    memory_safety_factor?: number;
    wall_clock_safety_factor?: number;
  },
): JobProfileEstimate {
  const memorySafetyFactor = options.memory_safety_factor ?? defaultMemorySafetyFactor;
  const wallClockSafetyFactor = options.wall_clock_safety_factor ?? defaultWallClockSafetyFactor;
  requirePositiveFinite(memorySafetyFactor, "memory_safety_factor");
  requirePositiveFinite(wallClockSafetyFactor, "wall_clock_safety_factor");

  const matchingRows = rows.filter((row) => row.recipe_id === options.recipe_id && !row.oom && rowMatchesFilter(row, options.match ?? {}));
  if (matchingRows.length === 0) {
    throw new Error(`no non-OOM profile rows for recipe_id: ${options.recipe_id}`);
  }
  rejectAmbiguousProfileEstimate(matchingRows, options.match ?? {});

  return {
    recipe_id: options.recipe_id,
    expected_peak_memory_mb: maxOf(matchingRows, (row) => row.peak_memory_mb) * memorySafetyFactor,
    expected_wall_clock_seconds: maxOf(matchingRows, (row) => row.wall_clock_seconds) * wallClockSafetyFactor,
    expected_gpu_utilization: averageOf(
      matchingRows,
      (row) => row.gpu_utilization_mean ?? defaultGpuUtilization,
    ),
    sample_count: matchingRows.length,
    memory_safety_factor: memorySafetyFactor,
    wall_clock_safety_factor: wallClockSafetyFactor,
  };
}

export function rowMatchesFilter(row: ResourceProfileRow, filter: ResourceProfileEstimateFilter): boolean {
  return (Object.entries(filter) as Array<[keyof ResourceProfileEstimateFilter, string | number | undefined]>).every(
    ([key, expected]) => expected === undefined || row[key] === expected,
  );
}

function auditCoverageRequirement(
  rows: ResourceProfileRow[],
  requirement: ResourceProfileCoverageRequirement,
  nowEpochSeconds: number | undefined,
): ResourceProfileRequirementAudit {
  const errors: string[] = [];
  const warnings: string[] = [];
  requirePositiveIntegerForAudit(requirement.min_non_oom_samples, "min_non_oom_samples", errors);
  requireOptionalFractionForAudit(requirement.max_oom_rate, "max_oom_rate", errors);
  requireOptionalPositiveForAudit(requirement.max_stale_age_seconds, "max_stale_age_seconds", errors);

  const matchingRows = rows.filter((row) => row.recipe_id === requirement.recipe_id && rowMatchesFilter(row, requirement.match ?? {}));
  const nonOomRows = matchingRows.filter((row) => !row.oom);
  const oomRows = matchingRows.filter((row) => row.oom);
  const freshness = classifyFreshness(nonOomRows, requirement.max_stale_age_seconds, nowEpochSeconds);
  const oomRate = matchingRows.length > 0 ? oomRows.length / matchingRows.length : null;

  if (matchingRows.length === 0) {
    errors.push(`no matching profile rows for recipe_id=${requirement.recipe_id}`);
  }
  if (freshness.usableRows.length < requirement.min_non_oom_samples) {
    errors.push(
      `usable non-OOM sample count ${freshness.usableRows.length} is below min_non_oom_samples ${requirement.min_non_oom_samples}`,
    );
  }
  if (typeof requirement.max_oom_rate === "number" && oomRate !== null && oomRate > requirement.max_oom_rate) {
    errors.push(`oom_rate ${oomRate.toFixed(4)} exceeds max_oom_rate ${requirement.max_oom_rate}`);
  }
  if (requirement.max_stale_age_seconds !== undefined && nowEpochSeconds === undefined) {
    errors.push("now_epoch_seconds is required when max_stale_age_seconds is set");
  }
  if (freshness.unknownRows.length > 0) {
    warnings.push(`${freshness.unknownRows.length} non-OOM rows have unknown freshness`);
  }
  if (freshness.staleRows.length > 0) {
    warnings.push(`${freshness.staleRows.length} non-OOM rows are stale for this requirement`);
  }
  if (freshness.futureRows.length > 0) {
    errors.push(`${freshness.futureRows.length} non-OOM rows are future-dated relative to now_epoch_seconds`);
  }

  return {
    requirement_id: requirement.requirement_id,
    recipe_id: requirement.recipe_id,
    status: errors.length === 0 ? "pass" : "fail",
    required_for_formal_plan: Boolean(requirement.required_for_formal_plan),
    total_matching_count: matchingRows.length,
    non_oom_count: nonOomRows.length,
    oom_count: oomRows.length,
    stale_count: freshness.staleRows.length,
    future_count: freshness.futureRows.length,
    unknown_freshness_count: freshness.unknownRows.length,
    usable_non_oom_count: freshness.usableRows.length,
    oom_rate: oomRate,
    matched_profile_ids: matchingRows.map((row) => row.profile_id).sort(),
    errors,
    warnings,
  };
}

function classifyFreshness(
  rows: ResourceProfileRow[],
  maxStaleAgeSeconds: number | undefined,
  nowEpochSeconds: number | undefined,
): {
  usableRows: ResourceProfileRow[];
  staleRows: ResourceProfileRow[];
  futureRows: ResourceProfileRow[];
  unknownRows: ResourceProfileRow[];
} {
  if (maxStaleAgeSeconds === undefined) {
    return { usableRows: rows, staleRows: [], futureRows: [], unknownRows: [] };
  }
  const usableRows: ResourceProfileRow[] = [];
  const staleRows: ResourceProfileRow[] = [];
  const futureRows: ResourceProfileRow[] = [];
  const unknownRows: ResourceProfileRow[] = [];
  for (const row of rows) {
    if (row.observed_at_epoch_seconds === undefined || nowEpochSeconds === undefined) {
      unknownRows.push(row);
    } else if (row.observed_at_epoch_seconds > nowEpochSeconds) {
      futureRows.push(row);
    } else if (nowEpochSeconds - row.observed_at_epoch_seconds > maxStaleAgeSeconds) {
      staleRows.push(row);
    } else {
      usableRows.push(row);
    }
  }
  return { usableRows, staleRows, futureRows, unknownRows };
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

function requirePositiveIntegerForAudit(value: unknown, label: string, errors: string[]): void {
  if (!Number.isInteger(value) || typeof value !== "number" || value <= 0) {
    errors.push(`${label} must be a positive integer`);
  }
}

function requireOptionalPositiveForAudit(value: unknown, label: string, errors: string[]): void {
  if (value === undefined) {
    return;
  }
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    errors.push(`${label} must be a positive finite number when provided`);
  }
}

function requireOptionalFractionForAudit(value: unknown, label: string, errors: string[]): void {
  if (value === undefined) {
    return;
  }
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
    errors.push(`${label} must be in [0, 1] when provided`);
  }
}

function rejectAmbiguousProfileEstimate(rows: ResourceProfileRow[], filter: ResourceProfileEstimateFilter): void {
  const requiredFilterFields: Array<keyof ResourceProfileEstimateFilter> = [
    "model_family",
    "model_variant",
    "dataset_id",
    "split_id",
    "gpu_model",
    "gpu_count",
    "gpu_slots",
    "nodes_or_tokens",
    "batch_size",
    "precision",
    "support_mode",
    "pair_kernel",
  ];

  for (const field of requiredFilterFields) {
    if (filter[field] !== undefined) {
      continue;
    }
    const values = new Set(rows.map((row) => row[field]).filter((value) => value !== undefined));
    if (values.size > 1) {
      throw new Error(`ambiguous profile estimate for recipe_id: filter by ${field}`);
    }
  }
}

function maxOf(rows: ResourceProfileRow[], getter: (row: ResourceProfileRow) => number): number {
  return rows.reduce((maxValue, row) => Math.max(maxValue, getter(row)), Number.NEGATIVE_INFINITY);
}

function averageOf(rows: ResourceProfileRow[], getter: (row: ResourceProfileRow) => number): number {
  return rows.reduce((sum, row) => sum + getter(row), 0) / rows.length;
}

function requireLiteral(value: unknown, expected: unknown, label: string, errors: string[]): void {
  if (value !== expected) {
    errors.push(`${label} must be ${String(expected)}`);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requireNonEmptyString(value: unknown, label: string, errors: string[]): void {
  if (typeof value !== "string" || value.length === 0) {
    errors.push(`${label} must be a non-empty string`);
  }
}

function requireBoolean(value: unknown, label: string, errors: string[]): void {
  if (typeof value !== "boolean") {
    errors.push(`${label} must be a boolean`);
  }
}

function requirePositiveNumber(value: unknown, label: string, errors: string[]): void {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    errors.push(`${label} must be a positive finite number`);
  }
}

function requireOptionalNonNegative(value: unknown, label: string, errors: string[]): void {
  if (value === undefined) {
    return;
  }
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    errors.push(`${label} must be a non-negative finite number when provided`);
  }
}

function requireOptionalFraction(value: unknown, label: string, errors: string[]): void {
  if (value === undefined) {
    return;
  }
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
    errors.push(`${label} must be in [0, 1] when provided`);
  }
}

function requirePositiveFinite(value: number, label: string): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${label} must be a positive finite number`);
  }
}
