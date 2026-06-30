export type SchedulingStrategy = "priority_lpt" | "resource_best_fit_v1";

export type ExecutionState = "shadow_only" | "blocked" | "eligible_for_human_review" | "executable";

export type ResourceLockResourceType = "gpu_group" | "writable_path" | "cache_path" | "cache_write_path";

export type ResourceLockMode = "exclusive_gpu" | "exclusive_write" | "shared_read";

export type JobStage = "prepare" | "train" | "eval" | "report" | "fetch" | "aggregate" | "custom";

export type ResourceLockConflictType =
  | "gpu_window_overlap"
  | "writable_path_collision"
  | "cache_write_collision"
  | "cache_read_write_collision";

export type EvidenceClass =
  | "formal_candidate"
  | "provisional_capacity_rerun_candidate"
  | "provisional_capacity";

export interface Manifest {
  schema_version: number;
  control_plane_version?: string;
  experiment_id: string;
  protocol?: {
    family?: string;
    phase?: string;
    strict?: boolean;
  };
  submission_policy?: {
    allow_submit?: boolean;
    requires_complete_manifests?: boolean;
  };
  planner?: PlannerSpec;
  jobs?: JobProfile[];
}

export interface PlannerSpec {
  machine?: {
    gpu_count: number;
    memory_total_mb: number;
    memory_fraction_limit?: number;
    start_gpu_id?: number;
  };
  slots?: GpuSlot[];
  policy?: Partial<PlannerPolicy>;
}

export interface PlanSpec extends PlannerSpec {
  manifest?: Manifest;
  jobs: JobProfile[];
  resource_profile_rows?: ResourceProfileRow[];
}

export interface GpuSlot {
  gpu_id: number;
  memory_total_mb: number;
  memory_fraction_limit?: number;
}

export interface NormalizedGpuSlot {
  gpu_id: number;
  memory_total_mb: number;
  memory_fraction_limit: number;
  memory_limit_mb: number;
}

export interface JobProfile {
  job_id: string;
  role: string;
  stage?: JobStage;
  dependencies?: string[];
  gpu_slots: number;
  expected_peak_memory_mb: number;
  expected_wall_clock_seconds: number;
  expected_gpu_utilization: number;
  run_dir: string;
  report_dir: string;
  log_dir: string;
  queue_dir: string;
  tmp_dir: string;
  cache_write_dir: string;
  model_family?: string;
  model_variant?: string;
  resource_profile_recipe_id?: string;
  resource_profile_match?: ResourceProfileEstimateFilter;
  require_resource_profile?: boolean;
  resource_profile_error?: string;
  resource_profile_note?: string;
  requires_cache_paths?: string[];
  formal_rerun_candidate?: boolean;
  allow_multiprocess_gpu?: boolean;
}

export interface ResourceProfileRow {
  schema_version: 1;
  profile_id: string;
  source_run_id: string;
  observed_at_epoch_seconds?: number;
  recipe_id: string;
  model_family: string;
  model_variant: string;
  dataset_id: string;
  split_id: string;
  gpu_model: string;
  gpu_count: number;
  gpu_slots: number;
  nodes_or_tokens: number;
  hidden_dim?: number;
  outer_steps?: number;
  inner_steps?: number;
  batch_size: number;
  precision: "fp32" | "bf16" | "fp16" | string;
  support_mode?: string;
  pair_kernel?: string;
  peak_memory_mb: number;
  steady_memory_mb?: number;
  wall_clock_seconds: number;
  train_step_count?: number;
  step_time_ms?: number;
  eval_time_ms?: number;
  samples_per_second?: number;
  gpu_utilization_mean?: number;
  gpu_utilization_p10?: number;
  gpu_utilization_p90?: number;
  host_ram_mb?: number;
  cache_read_mb_s?: number;
  cache_wait_seconds?: number;
  oom: boolean;
  notes?: string;
}

export interface ResourceProfileValidationResult {
  ok: boolean;
  errors: string[];
}

export interface ResourceProfileExtractionSkip {
  summary_path: string;
  reason: string;
}

export interface ResourceProfileExtractionReport {
  schema_version: 1;
  ok: boolean;
  source_roots: string[];
  scanned_summary_count: number;
  extracted_count: number;
  skipped_count: number;
  rows: ResourceProfileRow[];
  skipped: ResourceProfileExtractionSkip[];
}

export interface JobProfileEstimate {
  recipe_id: string;
  expected_peak_memory_mb: number;
  expected_wall_clock_seconds: number;
  expected_gpu_utilization: number;
  sample_count: number;
  memory_safety_factor: number;
  wall_clock_safety_factor: number;
}

export interface ResourceProfileEstimateFilter {
  model_family?: string;
  model_variant?: string;
  dataset_id?: string;
  split_id?: string;
  gpu_model?: string;
  gpu_count?: number;
  gpu_slots?: number;
  nodes_or_tokens?: number;
  hidden_dim?: number;
  outer_steps?: number;
  inner_steps?: number;
  batch_size?: number;
  precision?: string;
  support_mode?: string;
  pair_kernel?: string;
}

export interface ResourceProfileCoverageRequirement {
  requirement_id: string;
  recipe_id: string;
  match?: ResourceProfileEstimateFilter;
  min_non_oom_samples: number;
  max_oom_rate?: number;
  max_stale_age_seconds?: number;
  required_for_formal_plan?: boolean;
}

export interface ResourceProfileRequirementAudit {
  requirement_id: string;
  recipe_id: string;
  status: "pass" | "fail";
  required_for_formal_plan: boolean;
  total_matching_count: number;
  non_oom_count: number;
  oom_count: number;
  stale_count: number;
  future_count: number;
  unknown_freshness_count: number;
  usable_non_oom_count: number;
  oom_rate: number | null;
  matched_profile_ids: string[];
  errors: string[];
  warnings: string[];
}

export interface ResourceProfileLibraryAuditReport {
  schema_version: 1;
  ok: boolean;
  mode: "shadow_resource_profile_library_audit";
  row_count: number;
  valid_row_count: number;
  invalid_row_count: number;
  duplicate_profile_id_count: number;
  requirement_count: number;
  passing_requirement_count: number;
  failing_requirement_count: number;
  rows_validation_errors: string[];
  duplicate_profile_ids: string[];
  requirements: ResourceProfileRequirementAudit[];
  notes: string[];
}

export interface PlannerPolicy {
  allow_multiprocess_per_gpu: boolean;
  frozen_formal_manifest: boolean;
  keep_eval_slot_free: boolean;
  require_known_cache_paths: boolean;
  require_profile_estimates_for_roles: string[];
  available_cache_paths: string[];
  scheduling_strategy: SchedulingStrategy | string;
}

export interface PlannedJob {
  job_id: string;
  role: string;
  stage?: JobStage;
  gpu_ids: number[];
  start_after_seconds: number;
  expected_finish_seconds: number;
  expected_peak_memory_mb: number;
  expected_wall_clock_seconds: number;
  expected_gpu_utilization: number;
  memory_headroom_mb: number;
  dependency_ready_at_seconds: number;
  slot_ready_at_seconds: number;
  resource_profile_note?: string;
  evidence_class: EvidenceClass;
  decision: "scheduled";
  reason: string;
}

export interface ResourceLockRequest {
  request_id: string;
  job_id: string;
  resource_type: ResourceLockResourceType;
  mode: ResourceLockMode;
  starts_at_seconds: number;
  ends_at_seconds: number;
  gpu_ids?: number[];
  path?: string;
  path_field?: keyof Pick<JobProfile, "run_dir" | "report_dir" | "log_dir" | "queue_dir" | "tmp_dir" | "cache_write_dir">;
  atomic: boolean;
  source: "shadow_planner";
}

export interface ResourceLease {
  lease_id: string;
  request_id: string;
  job_id: string;
  resource_type: ResourceLockResourceType;
  mode: ResourceLockMode;
  starts_at_seconds: number;
  ends_at_seconds: number;
  gpu_ids?: number[];
  path?: string;
  path_field?: ResourceLockRequest["path_field"];
  atomic: boolean;
  enforcement: "shadow_only";
}

export interface ResourceLockGrant {
  grant_id: string;
  request_id: string;
  job_id: string;
  granted: true;
  lease: ResourceLease;
  source: "shadow_planner";
}

export type RuntimeLeaseStatus = "active" | "released" | "stale";

export type StaleLeaseProcessState = "unknown" | "live" | "dead";

export interface RuntimeLeaseOwner {
  hostname: string;
  pid: number;
  started_at_epoch_seconds: number;
  last_heartbeat_epoch_seconds: number;
}

export interface RuntimeLeaseRecord {
  schema_version: 1;
  runtime_lease_id: string;
  queue_id: string;
  job_id: string;
  status: RuntimeLeaseStatus;
  ttl_seconds: number;
  owner: RuntimeLeaseOwner;
  resource_leases: ResourceLease[];
}

export interface RuntimeLeaseStoreSnapshot {
  schema_version: 1;
  mode: "shadow_runtime_lease_store";
  lease_store_id: string;
  generated_at_epoch_seconds: number;
  leases: RuntimeLeaseRecord[];
  notes?: string[];
}

export interface RuntimeLeaseStoreValidationReport {
  schema_version: 1;
  ok: boolean;
  mode: "shadow_runtime_lease_store_validation";
  lease_store_id: string | null;
  lease_count: number;
  active_lease_count: number;
  duplicate_runtime_lease_id_count: number;
  duplicate_resource_lease_id_count: number;
  invalid_lease_count: number;
  duplicate_runtime_lease_ids: string[];
  duplicate_resource_lease_ids: string[];
  errors: string[];
  warnings: string[];
}

export type FetchPackageMode = "report_first" | "raw_complete";

export interface FetchPackageArchiveClaim {
  remote_archive_path: string;
  local_archive_path: string;
  remote_byte_size: number;
  remote_sha256: string;
}

export interface FetchPackageManifest {
  schema_version: 1;
  mode: FetchPackageMode;
  package_id: string;
  generated_at_epoch_seconds: number;
  package_root?: string;
  requested_artifact_paths: string[];
  existing_artifact_paths: string[];
  missing_artifact_paths: string[];
  compact_report_paths?: string[];
  archive?: FetchPackageArchiveClaim;
  notes?: string[];
}

export interface FetchPackageArchiveIntegrityReport {
  remote_archive_path: string;
  local_archive_path: string;
  remote_byte_size: number;
  local_byte_size: number | null;
  size_match: boolean;
  remote_sha256: string;
  local_sha256: string | null;
  sha256_match: boolean;
  extraction_allowed: boolean;
}

export interface FetchPackageValidationReport {
  schema_version: 1;
  ok: boolean;
  mode: "shadow_fetch_package_validation";
  package_id: string | null;
  fetch_mode: FetchPackageMode | null;
  package_root: string | null;
  requested_count: number;
  existing_count: number;
  missing_count: number;
  compact_report_count: number;
  checked_local_file_count: number;
  missing_local_file_count: number;
  archive_integrity: FetchPackageArchiveIntegrityReport | null;
  errors: string[];
  warnings: string[];
}

export type KernelBackendKind =
  | "pytorch_cpp_cuda_extension"
  | "triton_prototype"
  | "python_checkpointing"
  | "python_chunking";

export type KernelOptimizationRole =
  | "semantic_preserving"
  | "architecture_variant"
  | "provisional_system";

export type KernelEquivalenceStatus = "pass" | "fail" | "not_applicable";
export type KernelEvidenceStatus = "promotion_allowed" | "guardrail_only" | "invalid";

export interface KernelEquivalenceTolerance {
  atol: number;
  rtol: number;
}

export interface KernelEquivalenceShapeCase {
  batch_size: number;
  node_count: number;
  hidden_dim: number;
  center_count?: number;
  receiver_count?: number;
  sender_count?: number;
}

export interface KernelEquivalenceCase {
  case_id: string;
  dtype: string;
  device: "cuda" | "cpu" | string;
  seed: number;
  shape: KernelEquivalenceShapeCase;
  tolerance: KernelEquivalenceTolerance;
  checked_targets: string[];
  max_abs_error: number;
  max_rel_error: number;
  passed: boolean;
  diagnostics_requested?: boolean;
}

export interface KernelEquivalenceGateManifest {
  schema_version: 1;
  mode: "wca_kernel_equivalence_gate";
  gate_id: string;
  generated_at_epoch_seconds: number;
  backend_name: string;
  backend_kind: KernelBackendKind;
  optimized_backend_flag: string;
  default_backend_unchanged: boolean;
  baseline_reference: "FullRecursiveWorldStateNCA" | string;
  optimization_role: KernelOptimizationRole;
  equivalence_required: boolean;
  equivalence_status: KernelEquivalenceStatus;
  promotion_requested: boolean;
  mock_backend?: boolean;
  formal_claim_eligible?: boolean;
  evidence_status?: KernelEvidenceStatus;
  low_precision_quality_gate_status?: "pass" | "fail" | "not_applicable";
  cases: KernelEquivalenceCase[];
  notes?: string[];
}

export interface KernelEquivalenceTargetCoverage {
  target: string;
  covered_case_count: number;
  required_case_count: number;
  covered: boolean;
}

export interface KernelEquivalenceValidationReport {
  schema_version: 1;
  ok: boolean;
  mode: "wca_kernel_equivalence_validation";
  gate_id: string | null;
  backend_name: string | null;
  backend_kind: KernelBackendKind | string | null;
  promotion_requested: boolean;
  promotion_allowed: boolean;
  evidence_status: KernelEvidenceStatus;
  case_count: number;
  passing_case_count: number;
  failing_case_count: number;
  dtype_count: number;
  required_target_coverage: KernelEquivalenceTargetCoverage[];
  errors: string[];
  warnings: string[];
}

export interface StaleLeaseDecision {
  runtime_lease_id: string;
  job_id: string;
  stale_for_seconds: number;
  process_state: StaleLeaseProcessState;
  decision: "keep_active" | "mark_stale" | "manual_review";
  reason: string;
}

export interface RuntimeLeaseConflict {
  conflict_id: string;
  request_id: string;
  request_job_id: string;
  holder_runtime_lease_id: string;
  holder_job_id: string;
  resource_type: ResourceLockResourceType;
  conflict_type: ResourceLockConflictType;
  gpu_ids?: number[];
  path?: string;
  message: string;
}

export interface RuntimeLeaseEvaluation {
  schema_version: 1;
  ok: boolean;
  now_epoch_seconds: number;
  requested_count: number;
  active_lease_count: number;
  stale_lease_count: number;
  conflict_count: number;
  stale_decisions: StaleLeaseDecision[];
  conflicts: RuntimeLeaseConflict[];
}

export interface ResourceLockConflict {
  conflict_id: string;
  conflict_type: ResourceLockConflictType;
  resource_type: ResourceLockResourceType;
  request_ids: [string, string];
  job_ids: [string, string];
  starts_at_seconds: number;
  ends_at_seconds: number;
  gpu_ids?: number[];
  path?: string;
  path_fields?: [ResourceLockRequest["path_field"], ResourceLockRequest["path_field"]];
  message: string;
}

export interface RejectedJob {
  job_id: string;
  reason: string;
}

export type PlannerObjectiveDirection = "minimize" | "maximize" | "guardrail";

export interface PlannerObjectiveMetric {
  name: string;
  value: number | null;
  unit: string;
  direction: PlannerObjectiveDirection;
  source: string;
}

export interface PlannerObjectiveBreakdown {
  objective_id: "priority_lpt_audit_v1" | "resource_best_fit_shadow_v1";
  used_for_scheduling: boolean;
  global_optimality_claim: boolean;
  job_order_key: string[];
  gpu_group_selection_key: string[];
  metrics: PlannerObjectiveMetric[];
  notes: string[];
}

export interface SchedulerDecision {
  ok: boolean;
  execution_state: ExecutionState;
  evidence_default: "provisional_capacity";
  frozen_formal_manifest: boolean;
  allow_multiprocess_per_gpu: boolean;
  keep_eval_slot_free: boolean;
  require_known_cache_paths: boolean;
  available_cache_path_count: number;
  scheduling_strategy: string;
  usable_gpu_count: number;
  planned_job_count: number;
  rejected_job_count: number;
  collision_error_count: number;
  resource_lock_request_count: number;
  resource_lock_grant_count: number;
  resource_lock_conflict_count: number;
  expected_makespan_seconds: number;
  expected_total_slot_seconds: number;
  expected_busy_slot_seconds: number;
  expected_idle_slot_seconds: number;
  expected_idle_fraction: number;
  expected_utilization_weighted_slot_seconds: number;
  expected_mean_gpu_utilization: number;
  expected_min_memory_headroom_mb: number | null;
  planner_objective: PlannerObjectiveBreakdown;
  reason?: string;
}

export interface PlannerResult {
  schema_version: 1;
  scheduler_decision: SchedulerDecision;
  slots: NormalizedGpuSlot[];
  job_profiles: JobProfile[];
  planned_jobs: PlannedJob[];
  rejected_jobs: RejectedJob[];
  collision_errors: string[];
  resource_lock_requests: ResourceLockRequest[];
  resource_lock_grants: ResourceLockGrant[];
  resource_lock_conflicts: ResourceLockConflict[];
}

export interface PlannerStrategyComparisonRow {
  strategy: SchedulingStrategy;
  ok: boolean;
  planned_job_count: number;
  rejected_job_count: number;
  resource_lock_conflict_count: number;
  expected_makespan_seconds: number;
  expected_idle_slot_seconds: number;
  expected_idle_fraction: number;
  expected_mean_gpu_utilization: number;
  expected_min_memory_headroom_mb: number | null;
  expected_memory_waste_mb_slot_seconds: number | null;
  objective_id: PlannerObjectiveBreakdown["objective_id"];
  reason?: string;
}

export interface PlannerStrategyComparisonReport {
  schema_version: 1;
  ok: boolean;
  mode: "shadow_planner_strategy_comparison";
  compared_strategy_count: number;
  baseline_strategy: SchedulingStrategy;
  best_by_makespan: SchedulingStrategy | null;
  best_by_memory_waste: SchedulingStrategy | null;
  rows: PlannerStrategyComparisonRow[];
  warnings: string[];
}

export interface ControlManifestInspectionResult {
  schema_version: 1;
  ok: boolean;
  experiment_id: string | null;
  phase: string | null;
  strict: boolean | null;
  allow_submit: boolean | null;
  forbid_direct_queue_submit: boolean | null;
  require_clean_git: boolean | null;
  queue_dir: string | null;
  report_dirs: string[];
  run_dirs: string[];
  required_paths: string[];
  job_count: number;
  job_ids: string[];
  duplicate_job_ids: string[];
  model_matrix_count: number;
  requires_complete_manifests: string[];
  planner_enabled: boolean;
  errors: string[];
  warnings: string[];
}

export interface PythonManifestValidationResult {
  ok: boolean;
  experiment_id: string | null;
  errors: string[];
  warnings: string[];
}

export interface ManifestParityRow {
  manifest_path: string;
  experiment_id: string | null;
  ts_ok: boolean;
  python_ok: boolean | null;
  status: "match" | "difference" | "python_error";
  ts_error_count: number;
  python_error_count: number | null;
  ts_warning_count: number;
  python_warning_count: number | null;
  ts_errors: string[];
  python_errors: string[];
  ts_warnings: string[];
  python_warnings: string[];
}

export interface ManifestParityReport {
  schema_version: 1;
  ok: boolean;
  mode: "shadow_manifest_parity";
  python_validation: "validate_manifest_no_clean_git_no_prerequisites";
  manifest_count: number;
  match_count: number;
  difference_count: number;
  python_error_count: number;
  rows: ManifestParityRow[];
}

export interface ArtifactInventory {
  schema_version: 1;
  experiment_id: string;
  artifact_contract?: Record<string, unknown>;
  artifact_contract_source: string;
  required_files: string[];
  present_files: string[];
  missing_files: string[];
  validation_errors: string[];
  provenance_warnings: string[];
  status: "complete" | "incomplete";
}

export interface ArtifactInventoryParityRow {
  manifest_path: string;
  experiment_id: string | null;
  ts_status: "complete" | "incomplete" | null;
  python_status: "complete" | "incomplete" | null;
  status: "match" | "difference" | "python_error";
  ts_required_count: number | null;
  python_required_count: number | null;
  ts_present_count: number | null;
  python_present_count: number | null;
  ts_missing_count: number | null;
  python_missing_count: number | null;
  ts_validation_error_count: number | null;
  python_validation_error_count: number | null;
  missing_diff: string[];
  validation_error_diff: string[];
  python_errors: string[];
}

export interface ArtifactInventoryParityReport {
  schema_version: 1;
  ok: boolean;
  mode: "shadow_artifact_inventory_parity";
  python_inventory: "build_artifact_inventory_read_only";
  manifest_count: number;
  match_count: number;
  difference_count: number;
  python_error_count: number;
  rows: ArtifactInventoryParityRow[];
}

export interface FormalEvidenceStub {
  schema_version: 1;
  experiment_id: string;
  strict_gate: "complete" | "incomplete";
  report_relative_path: string;
  report_text: string;
  report_sha256: string;
}

export interface ReportGateParityRow {
  manifest_path: string;
  experiment_id: string | null;
  ts_strict_gate: "complete" | "incomplete" | null;
  python_strict_gate: "complete" | "incomplete" | null;
  status: "match" | "difference" | "python_error";
  ts_report_relative_path: string | null;
  python_report_relative_path: string | null;
  ts_report_sha256: string | null;
  python_report_sha256: string | null;
  differences: string[];
  python_errors: string[];
}

export interface ReportGateParityReport {
  schema_version: 1;
  ok: boolean;
  mode: "shadow_report_gate_parity";
  python_report_gate: "write_formal_evidence_stub_tempdir";
  manifest_count: number;
  match_count: number;
  difference_count: number;
  python_error_count: number;
  rows: ReportGateParityRow[];
}

export type QueueStatusState = "unknown" | "waiting" | "running" | "complete" | "failed";

export interface QueueStatusJobSummary {
  job_id: string;
  started_at: string | null;
  finished_at: string | null;
  exit_code: number | null;
  duration_seconds: number | null;
  timed_out: boolean | null;
}

export interface QueueStatusReport {
  schema_version: 1;
  ok: boolean;
  mode: "shadow_queue_status";
  source: "local_queue_artifacts" | "remote_read_only_snapshot";
  queue_dir: string;
  queue_name: string | null;
  run_id: string | null;
  state: QueueStatusState;
  current_job_id: string | null;
  current_job_ids: string[];
  started_at: string | null;
  finished_at: string | null;
  exit_code: number | null;
  job_count: number | null;
  started_job_count: number;
  finished_job_count: number;
  failed_job_count: number;
  status_event_count: number;
  summary_job_count: number;
  jobs: QueueStatusJobSummary[];
  missing_files: string[];
  parse_errors: string[];
  notes: string[];
}

export interface RemoteStatusSnapshot {
  schema_version: 1;
  mode: "shadow_remote_status_snapshot";
  snapshot_id: string;
  generated_at_epoch_seconds: number;
  source: "remote_read_only_probe_fixture";
  remote_host_label: string;
  remote_root: string;
  queue_status: QueueStatusReport;
  notes?: string[];
}

export interface RemoteStatusSnapshotValidationReport {
  schema_version: 1;
  ok: boolean;
  mode: "shadow_remote_status_snapshot_validation";
  snapshot_id: string | null;
  queue_name: string | null;
  queue_state: QueueStatusState | null;
  current_job_id: string | null;
  error_count: number;
  warning_count: number;
  errors: string[];
  warnings: string[];
}

export interface QueueStatusParityRow {
  queue_dir: string;
  queue_name: string | null;
  ts_state: QueueStatusState | null;
  python_state: QueueStatusState | null;
  status: "match" | "difference" | "python_error";
  differences: string[];
  python_errors: string[];
}

export interface QueueStatusParityReport {
  schema_version: 1;
  ok: boolean;
  mode: "shadow_queue_status_parity";
  python_status: "local_queue_status_read_only";
  queue_count: number;
  match_count: number;
  difference_count: number;
  python_error_count: number;
  rows: QueueStatusParityRow[];
}

export type MockActiveControlOutcome = "success" | "failure";

export interface MockActiveControlJobSpec {
  job_id: string;
  exit_code?: number;
  duration_seconds?: number;
  timed_out?: boolean;
}

export interface MockActiveControlSpec {
  schema_version: 1;
  mode: "local_mock_active_control";
  allow_remote: false;
  experiment_id: string;
  queue_name: string;
  run_id: string;
  output_dir: string;
  outcome: MockActiveControlOutcome;
  base_time: string;
  jobs: MockActiveControlJobSpec[];
}

export interface MockActiveControlReport {
  schema_version: 1;
  ok: boolean;
  mode: "local_mock_active_control";
  active_takeover_allowed: false;
  allow_remote: false;
  experiment_id: string | null;
  queue_name: string | null;
  run_id: string | null;
  queue_dir: string | null;
  state: QueueStatusState | null;
  exit_code: number | null;
  artifact_paths: {
    manifest_json: string | null;
    remote_submission_json: string | null;
    status_jsonl: string | null;
    queue_summary_json: string | null;
  };
  errors: string[];
  warnings: string[];
  notes: string[];
}

export type MigrationReadinessGateStatus = "pass" | "fail" | "not_applicable";

export interface MigrationReadinessGate {
  gate_id: string;
  status: MigrationReadinessGateStatus;
  required_for_active_takeover: boolean;
  reason: string;
  evidence: Record<string, unknown>;
}

export interface MigrationReadinessReport {
  schema_version: 1;
  ok: boolean;
  mode: "shadow_migration_readiness";
  execution_state: ExecutionState;
  active_takeover_allowed: boolean;
  recommendation: "keep_python_active" | "eligible_for_human_review";
  manifest_count: number;
  queue_count: number;
  gate_count: number;
  passing_gate_count: number;
  blocking_gate_count: number;
  gates: MigrationReadinessGate[];
  blockers: string[];
  notes: string[];
}
