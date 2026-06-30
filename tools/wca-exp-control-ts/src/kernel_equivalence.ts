import { readFile } from "node:fs/promises";

import type {
  KernelBackendKind,
  KernelEquivalenceCase,
  KernelEquivalenceStatus,
  KernelEquivalenceTargetCoverage,
  KernelEquivalenceValidationReport,
  KernelEvidenceStatus,
  KernelOptimizationRole,
} from "./contracts.ts";

const validBackendKinds = new Set<KernelBackendKind>([
  "pytorch_cpp_cuda_extension",
  "triton_prototype",
  "python_checkpointing",
  "python_chunking",
]);
const validOptimizationRoles = new Set<KernelOptimizationRole>([
  "semantic_preserving",
  "architecture_variant",
  "provisional_system",
]);
const validEquivalenceStatuses = new Set<KernelEquivalenceStatus>(["pass", "fail", "not_applicable"]);
const requiredPromotionTargets = [
  "H_final",
  "prediction",
  "scalar_loss",
  "input_gradients",
  "parameter_gradients",
];
const lowPrecisionDtypes = new Set(["bf16", "bfloat16", "fp16", "float16"]);

export async function validateKernelEquivalenceGateFile(path: string): Promise<KernelEquivalenceValidationReport> {
  try {
    const payload = JSON.parse(await readFile(path, "utf8")) as unknown;
    return validateKernelEquivalenceGate(payload);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return invalidKernelEquivalenceReport([`failed to read or parse kernel equivalence gate: ${message}`]);
  }
}

export function validateKernelEquivalenceGate(payload: unknown): KernelEquivalenceValidationReport {
  const errors: string[] = [];
  const warnings: string[] = [];
  if (!isRecord(payload)) {
    return invalidKernelEquivalenceReport(["kernel equivalence gate must be an object"]);
  }

  if (payload.schema_version !== 1) {
    errors.push("schema_version must be 1");
  }
  if (payload.mode !== "wca_kernel_equivalence_gate") {
    errors.push("mode must be wca_kernel_equivalence_gate");
  }
  const gateId = requireString(payload.gate_id, "gate_id", errors);
  requireNonNegativeNumber(payload.generated_at_epoch_seconds, "generated_at_epoch_seconds", errors);
  const backendName = requireString(payload.backend_name, "backend_name", errors);
  const backendKind = requireEnum(payload.backend_kind, "backend_kind", validBackendKinds, errors);
  requireString(payload.optimized_backend_flag, "optimized_backend_flag", errors);
  if (payload.default_backend_unchanged !== true) {
    errors.push("default_backend_unchanged must be true");
  }
  if (payload.baseline_reference !== "FullRecursiveWorldStateNCA") {
    errors.push("baseline_reference must be FullRecursiveWorldStateNCA");
  }
  const optimizationRole = requireEnum(payload.optimization_role, "optimization_role", validOptimizationRoles, errors);
  if (typeof payload.equivalence_required !== "boolean") {
    errors.push("equivalence_required must be boolean");
  }
  const equivalenceStatus = requireEnum(
    payload.equivalence_status,
    "equivalence_status",
    validEquivalenceStatuses,
    errors,
  );
  const promotionRequested = payload.promotion_requested === true;
  if (typeof payload.promotion_requested !== "boolean") {
    errors.push("promotion_requested must be boolean");
  }
  const mockBackend = payload.mock_backend === true;
  if (typeof payload.mock_backend !== "boolean" && payload.mock_backend !== undefined) {
    errors.push("mock_backend must be boolean when present");
  }
  if (typeof payload.formal_claim_eligible !== "boolean" && payload.formal_claim_eligible !== undefined) {
    errors.push("formal_claim_eligible must be boolean when present");
  }
  const formalClaimExplicitlyRejected = payload.formal_claim_eligible === false;
  validateStringArrayOptional(payload.notes, "notes", errors);

  const cases = validateCases(payload.cases, errors, warnings);
  const requiredTargets = requiredTargetsForCases(cases);
  const coverage = targetCoverage(cases, requiredTargets);
  const hasLowPrecisionCases = cases.some((testCase) => lowPrecisionDtypes.has(testCase.dtype.toLowerCase()));
  const hasCpuCases = cases.some((testCase) => testCase.device.toLowerCase() === "cpu");

  if (promotionRequested) {
    if (mockBackend) {
      errors.push("mock_backend may not request formal promotion");
    }
    if (payload.formal_claim_eligible !== true) {
      errors.push("formal promotion requires formal_claim_eligible=true");
    }
    if (hasCpuCases) {
      errors.push("CPU evidence may not request formal promotion");
    }
    if (backendKind === "triton_prototype") {
      errors.push("triton_prototype may not request formal promotion");
    }
    if (optimizationRole !== "semantic_preserving") {
      errors.push("promotion requires optimization_role=semantic_preserving");
    }
    if (payload.equivalence_required !== true) {
      errors.push("promotion requires equivalence_required=true");
    }
    if (equivalenceStatus !== "pass") {
      errors.push("promotion requires equivalence_status=pass");
    }
    for (const row of coverage) {
      if (!row.covered) {
        errors.push(`promotion target coverage missing: ${row.target}`);
      }
    }
    if (hasLowPrecisionCases && payload.low_precision_quality_gate_status !== "pass") {
      errors.push("low precision cases require low_precision_quality_gate_status=pass before promotion");
    }
  }

  if (equivalenceStatus === "pass" && cases.some((testCase) => !testCase.passed)) {
    errors.push("equivalence_status=pass conflicts with failing cases");
  }
  if (equivalenceStatus === "not_applicable" && optimizationRole === "semantic_preserving") {
    errors.push("semantic_preserving optimization cannot use equivalence_status=not_applicable");
  }
  if (optimizationRole !== "semantic_preserving" && promotionRequested) {
    warnings.push("non-semantic optimization requested promotion; report remains rejected");
  }
  if (!promotionRequested && (mockBackend || hasCpuCases || formalClaimExplicitlyRejected)) {
    warnings.push("non-promotional kernel evidence accepted for guardrail validation only");
  }

  const ok = errors.length === 0;
  const promotionAllowed = ok && promotionRequested;
  const evidenceStatus: KernelEvidenceStatus = promotionAllowed ? "promotion_allowed" : ok ? "guardrail_only" : "invalid";
  return {
    schema_version: 1,
    ok,
    mode: "wca_kernel_equivalence_validation",
    gate_id: gateId,
    backend_name: backendName,
    backend_kind: backendKind,
    promotion_requested: promotionRequested,
    promotion_allowed: promotionAllowed,
    evidence_status: evidenceStatus,
    case_count: cases.length,
    passing_case_count: cases.filter((testCase) => testCase.passed).length,
    failing_case_count: cases.filter((testCase) => !testCase.passed).length,
    dtype_count: new Set(cases.map((testCase) => testCase.dtype)).size,
    required_target_coverage: coverage,
    errors,
    warnings,
  };
}

function validateCases(value: unknown, errors: string[], warnings: string[]): KernelEquivalenceCase[] {
  if (!Array.isArray(value) || value.length === 0) {
    errors.push("cases must be a non-empty array");
    return [];
  }
  const cases: KernelEquivalenceCase[] = [];
  const seenIds = new Set<string>();
  value.forEach((item, index) => {
    if (!isRecord(item)) {
      errors.push(`cases[${index}] must be an object`);
      return;
    }
    const caseId = requireString(item.case_id, `cases[${index}].case_id`, errors);
    if (caseId && seenIds.has(caseId)) {
      errors.push(`duplicate case_id: ${caseId}`);
    }
    if (caseId) {
      seenIds.add(caseId);
    }
    const dtype = requireString(item.dtype, `cases[${index}].dtype`, errors);
    const device = requireString(item.device, `cases[${index}].device`, errors);
    requireNonNegativeInteger(item.seed, `cases[${index}].seed`, errors);
    validateShape(item.shape, `cases[${index}].shape`, errors);
    validateTolerance(item.tolerance, `cases[${index}].tolerance`, errors);
    const checkedTargets = validateStringArray(item.checked_targets, `cases[${index}].checked_targets`, errors) ?? [];
    requireNonNegativeNumber(item.max_abs_error, `cases[${index}].max_abs_error`, errors);
    requireNonNegativeNumber(item.max_rel_error, `cases[${index}].max_rel_error`, errors);
    if (typeof item.passed !== "boolean") {
      errors.push(`cases[${index}].passed must be boolean`);
    }
    validatePassedCaseWithinTolerance(item, index, errors);
    if (typeof item.diagnostics_requested !== "boolean" && item.diagnostics_requested !== undefined) {
      errors.push(`cases[${index}].diagnostics_requested must be boolean when present`);
    }
    if (device && device !== "cuda") {
      warnings.push(`cases[${index}] uses device=${device}; CUDA kernel promotion evidence should include cuda cases`);
    }
    if (caseId && dtype && device && typeof item.passed === "boolean") {
      cases.push({
        case_id: caseId,
        dtype,
        device,
        seed: typeof item.seed === "number" ? item.seed : -1,
        shape: isRecord(item.shape)
          ? {
              batch_size: typeof item.shape.batch_size === "number" ? item.shape.batch_size : -1,
              node_count: typeof item.shape.node_count === "number" ? item.shape.node_count : -1,
              hidden_dim: typeof item.shape.hidden_dim === "number" ? item.shape.hidden_dim : -1,
              ...(typeof item.shape.center_count === "number" ? { center_count: item.shape.center_count } : {}),
              ...(typeof item.shape.receiver_count === "number" ? { receiver_count: item.shape.receiver_count } : {}),
              ...(typeof item.shape.sender_count === "number" ? { sender_count: item.shape.sender_count } : {}),
            }
          : { batch_size: -1, node_count: -1, hidden_dim: -1 },
        tolerance: isRecord(item.tolerance)
          ? {
              atol: typeof item.tolerance.atol === "number" ? item.tolerance.atol : -1,
              rtol: typeof item.tolerance.rtol === "number" ? item.tolerance.rtol : -1,
            }
          : { atol: -1, rtol: -1 },
        checked_targets: checkedTargets,
        max_abs_error: typeof item.max_abs_error === "number" ? item.max_abs_error : Number.NaN,
        max_rel_error: typeof item.max_rel_error === "number" ? item.max_rel_error : Number.NaN,
        passed: item.passed,
        ...(item.diagnostics_requested === true ? { diagnostics_requested: true } : {}),
      });
    }
  });
  return cases;
}

function validatePassedCaseWithinTolerance(
  item: Record<string, unknown>,
  index: number,
  errors: string[],
): void {
  if (item.passed !== true || !isRecord(item.tolerance)) {
    return;
  }
  if (
    typeof item.max_abs_error === "number" &&
    Number.isFinite(item.max_abs_error) &&
    typeof item.tolerance.atol === "number" &&
    Number.isFinite(item.tolerance.atol) &&
    item.max_abs_error > item.tolerance.atol
  ) {
    errors.push(`cases[${index}].max_abs_error exceeds tolerance.atol while passed=true`);
  }
  if (
    typeof item.max_rel_error === "number" &&
    Number.isFinite(item.max_rel_error) &&
    typeof item.tolerance.rtol === "number" &&
    Number.isFinite(item.tolerance.rtol) &&
    item.max_rel_error > item.tolerance.rtol
  ) {
    errors.push(`cases[${index}].max_rel_error exceeds tolerance.rtol while passed=true`);
  }
}

function validateShape(value: unknown, path: string, errors: string[]): void {
  if (!isRecord(value)) {
    errors.push(`${path} must be an object`);
    return;
  }
  requirePositiveInteger(value.batch_size, `${path}.batch_size`, errors);
  requirePositiveInteger(value.node_count, `${path}.node_count`, errors);
  requirePositiveInteger(value.hidden_dim, `${path}.hidden_dim`, errors);
  requirePositiveIntegerOptional(value.center_count, `${path}.center_count`, errors);
  requirePositiveIntegerOptional(value.receiver_count, `${path}.receiver_count`, errors);
  requirePositiveIntegerOptional(value.sender_count, `${path}.sender_count`, errors);
}

function validateTolerance(value: unknown, path: string, errors: string[]): void {
  if (!isRecord(value)) {
    errors.push(`${path} must be an object`);
    return;
  }
  requireNonNegativeNumber(value.atol, `${path}.atol`, errors);
  requireNonNegativeNumber(value.rtol, `${path}.rtol`, errors);
}

function requiredTargetsForCases(cases: KernelEquivalenceCase[]): string[] {
  const targets = new Set(requiredPromotionTargets);
  if (cases.some((testCase) => testCase.diagnostics_requested)) {
    targets.add("last_local_worlds");
  }
  return [...targets].sort();
}

function targetCoverage(cases: KernelEquivalenceCase[], targets: string[]): KernelEquivalenceTargetCoverage[] {
  return targets.map((target) => {
    const coveredCaseCount = cases.filter((testCase) => testCase.checked_targets.includes(target)).length;
    return {
      target,
      covered_case_count: coveredCaseCount,
      required_case_count: cases.length,
      covered: cases.length > 0 && coveredCaseCount === cases.length,
    };
  });
}

function validateStringArray(value: unknown, path: string, errors: string[]): string[] | null {
  if (!Array.isArray(value)) {
    errors.push(`${path} must be an array`);
    return null;
  }
  const result: string[] = [];
  value.forEach((item, index) => {
    if (typeof item !== "string" || item.length === 0) {
      errors.push(`${path}[${index}] must be a non-empty string`);
    } else {
      result.push(item);
    }
  });
  return result;
}

function validateStringArrayOptional(value: unknown, path: string, errors: string[]): void {
  if (value === undefined) {
    return;
  }
  validateStringArray(value, path, errors);
}

function requireString(value: unknown, path: string, errors: string[]): string | null {
  if (typeof value !== "string" || value.length === 0) {
    errors.push(`${path} must be a non-empty string`);
    return null;
  }
  return value;
}

function requireEnum<T extends string>(
  value: unknown,
  path: string,
  allowed: Set<T>,
  errors: string[],
): T | null {
  if (typeof value !== "string" || !allowed.has(value as T)) {
    errors.push(`${path} is invalid`);
    return null;
  }
  return value as T;
}

function requirePositiveInteger(value: unknown, path: string, errors: string[]): void {
  if (!Number.isInteger(value) || typeof value !== "number" || value <= 0) {
    errors.push(`${path} must be a positive integer`);
  }
}

function requirePositiveIntegerOptional(value: unknown, path: string, errors: string[]): void {
  if (value !== undefined) {
    requirePositiveInteger(value, path, errors);
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

function invalidKernelEquivalenceReport(errors: string[]): KernelEquivalenceValidationReport {
  return {
    schema_version: 1,
    ok: false,
    mode: "wca_kernel_equivalence_validation",
    gate_id: null,
    backend_name: null,
    backend_kind: null,
    promotion_requested: false,
    promotion_allowed: false,
    evidence_status: "invalid",
    case_count: 0,
    passing_case_count: 0,
    failing_case_count: 0,
    dtype_count: 0,
    required_target_coverage: [],
    errors,
    warnings: [],
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
