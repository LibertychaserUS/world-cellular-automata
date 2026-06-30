import type {
  EvidenceClass,
  GpuSlot,
  JobStage,
  JobProfile,
  NormalizedGpuSlot,
  PlanSpec,
  PlannerPolicy,
  PlannerResult,
  PlannedJob,
  RejectedJob,
  ResourceLockConflict,
  ResourceLockRequest,
  PlannerObjectiveBreakdown,
} from "./contracts.ts";
import { estimateJobProfileFromRows } from "./profiles.ts";

import path from "node:path/posix";

const rolePriority: Record<string, number> = {
  formal_gate: 0,
  matched_baseline: 10,
  mechanism_control: 20,
  candidate_search: 30,
  capacity_scaling: 40,
  exploratory: 50,
};

const writableFields = [
  "run_dir",
  "report_dir",
  "log_dir",
  "queue_dir",
  "tmp_dir",
  "cache_write_dir",
] as const;

const artifactWritableFields = ["run_dir", "report_dir", "log_dir", "queue_dir", "tmp_dir"] as const;

const validJobStages = new Set<JobStage>(["prepare", "train", "eval", "report", "fetch", "aggregate", "custom"]);

export function defaultPlannerPolicy(policy: Partial<PlannerPolicy> = {}): PlannerPolicy {
  return {
    allow_multiprocess_per_gpu: Boolean(policy.allow_multiprocess_per_gpu),
    frozen_formal_manifest: Boolean(policy.frozen_formal_manifest),
    keep_eval_slot_free: Boolean(policy.keep_eval_slot_free),
    require_known_cache_paths: Boolean(policy.require_known_cache_paths),
    require_profile_estimates_for_roles: Array.isArray(policy.require_profile_estimates_for_roles)
      ? [...policy.require_profile_estimates_for_roles]
      : [],
    available_cache_paths: Array.isArray(policy.available_cache_paths) ? [...policy.available_cache_paths] : [],
    scheduling_strategy: policy.scheduling_strategy ?? "priority_lpt",
  };
}

export function buildGpuSlots(input: {
  gpu_count: number;
  memory_total_mb: number;
  memory_fraction_limit?: number;
  start_gpu_id?: number;
}): NormalizedGpuSlot[] {
  const gpuCount = requirePositiveInteger(input.gpu_count, "machine.gpu_count");
  const memoryTotalMb = requirePositiveNumber(input.memory_total_mb, "machine.memory_total_mb");
  const memoryFractionLimit = input.memory_fraction_limit ?? 0.86;
  validateMemoryFraction(memoryFractionLimit, "machine.memory_fraction_limit");
  const startGpuId = input.start_gpu_id ?? 0;

  return Array.from({ length: gpuCount }, (_, index) =>
    normalizeGpuSlot({
      gpu_id: startGpuId + index,
      memory_total_mb: memoryTotalMb,
      memory_fraction_limit: memoryFractionLimit,
    }),
  );
}

export function normalizeGpuSlot(slot: GpuSlot): NormalizedGpuSlot {
  const gpuId = requireInteger(slot.gpu_id, "slot.gpu_id");
  const memoryTotalMb = requirePositiveNumber(slot.memory_total_mb, "slot.memory_total_mb");
  const memoryFractionLimit = slot.memory_fraction_limit ?? 0.86;
  validateMemoryFraction(memoryFractionLimit, "slot.memory_fraction_limit");
  return {
    gpu_id: gpuId,
    memory_total_mb: memoryTotalMb,
    memory_fraction_limit: memoryFractionLimit,
    memory_limit_mb: memoryTotalMb * memoryFractionLimit,
  };
}

export function normalizePlanSpec(spec: PlanSpec): {
  slots: NormalizedGpuSlot[];
  jobs: JobProfile[];
  policy: PlannerPolicy;
} {
  const plannerFromManifest = spec.manifest?.planner ?? {};
  const machine = spec.machine ?? plannerFromManifest.machine;
  const rawSlots = spec.slots ?? plannerFromManifest.slots;
  const jobs = spec.jobs ?? spec.manifest?.jobs;

  if (!Array.isArray(jobs) || jobs.length === 0) {
    throw new Error("plan spec must define non-empty jobs list");
  }

  let slots: NormalizedGpuSlot[];
  if (machine) {
    slots = buildGpuSlots(machine);
  } else if (Array.isArray(rawSlots) && rawSlots.length > 0) {
    slots = rawSlots.map(normalizeGpuSlot);
  } else {
    throw new Error("plan spec must define non-empty slots list or machine gpu_count/memory_total_mb");
  }

  const policy = defaultPlannerPolicy({
    ...(plannerFromManifest.policy ?? {}),
    ...(spec.policy ?? {}),
  });
  const normalizedJobs = jobs.map(normalizeJobProfile);

  return {
    slots,
    jobs: applyProfileEstimates(normalizedJobs, spec.resource_profile_rows ?? [], policy),
    policy,
  };
}

export function planFromSpec(spec: PlanSpec): PlannerResult {
  const { slots, jobs, policy } = normalizePlanSpec(spec);
  return planJobs(slots, jobs, policy);
}

function applyProfileEstimates(
  jobs: JobProfile[],
  profileRows: PlanSpec["resource_profile_rows"],
  policy: PlannerPolicy,
): JobProfile[] {
  const requiredRoles = new Set(policy.require_profile_estimates_for_roles);
  return jobs.map((job) => {
    const requiresProfile = Boolean(job.require_resource_profile) || requiredRoles.has(job.role);
    const recipeId = job.resource_profile_recipe_id;
    if (!requiresProfile && !recipeId) {
      return job;
    }
    if (!recipeId) {
      return {
        ...job,
        resource_profile_error: `resource profile required for role ${job.role} but resource_profile_recipe_id is missing`,
      };
    }
    try {
      const estimate = estimateJobProfileFromRows(profileRows ?? [], {
        recipe_id: recipeId,
        match: job.resource_profile_match,
      });
      return {
        ...job,
        expected_peak_memory_mb: estimate.expected_peak_memory_mb,
        expected_wall_clock_seconds: estimate.expected_wall_clock_seconds,
        expected_gpu_utilization: estimate.expected_gpu_utilization,
        resource_profile_note:
          `profile_estimate recipe=${recipeId} samples=${estimate.sample_count} ` +
          `memory_safety=${estimate.memory_safety_factor} wall_clock_safety=${estimate.wall_clock_safety_factor}`,
      };
    } catch (error) {
      if (!requiresProfile) {
        return job;
      }
      const message = error instanceof Error ? error.message : String(error);
      return {
        ...job,
        resource_profile_error: `resource profile estimate failed for ${recipeId}: ${message}`,
      };
    }
  });
}

export function planJobs(
  slots: NormalizedGpuSlot[],
  jobs: JobProfile[],
  policyInput: Partial<PlannerPolicy> = {},
): PlannerResult {
  const policy = defaultPlannerPolicy(policyInput);
  if (!isSupportedSchedulingStrategy(policy.scheduling_strategy)) {
    const rejectedJobs = jobs.map((job) => ({
      job_id: job.job_id,
      reason: `unsupported scheduling_strategy: ${policy.scheduling_strategy}`,
    }));
    return planningResult(slots, jobs, [], rejectedJobs, [], policy, {
      ok: false,
      reason: `unsupported scheduling_strategy: ${policy.scheduling_strategy}`,
    });
  }

  const collisions = collisionErrors(jobs);
  if (policy.allow_multiprocess_per_gpu) {
    return planningResult(
      slots,
      jobs,
      [],
      jobs.map((job) => ({
        job_id: job.job_id,
        reason:
          "allow_multiprocess_per_gpu is not implemented; current planner emits exclusive_gpu leases only",
      })),
      collisions,
      policy,
      {
        ok: false,
        reason: "allow_multiprocess_per_gpu is not implemented; exclusive GPU leasing remains required",
        usable_gpu_count: 0,
      },
    );
  }
  const usableSlots =
    policy.keep_eval_slot_free && slots.length > 1
      ? [...slots].sort((a, b) => a.gpu_id - b.gpu_id).slice(0, -1)
      : [...slots];

  if (usableSlots.length === 0) {
    return planningResult(
      slots,
      jobs,
      [],
      jobs.map((job) => ({ job_id: job.job_id, reason: "no usable GPU slots" })),
      collisions,
      policy,
      { ok: false, reason: "no usable GPU slots", usable_gpu_count: 0 },
    );
  }

  const knownCachePaths = new Set(policy.available_cache_paths.map(normalizePathKey));
  const availableAt = new Map<number, number>(usableSlots.map((slot) => [slot.gpu_id, 0]));
  const plannedJobs: PlannedJob[] = [];
  const rejectedJobs: RejectedJob[] = [];
  const rejectedJobIds = new Set<string>();
  const plannedById = new Map<string, PlannedJob>();
  const jobsById = new Map<string, JobProfile>();
  for (const job of jobs) {
    if (jobsById.has(job.job_id)) {
      rejectedJobs.push({ job_id: job.job_id, reason: `duplicate job_id: ${job.job_id}` });
      rejectedJobIds.add(job.job_id);
      continue;
    }
    if (job.resource_profile_error) {
      rejectedJobs.push({ job_id: job.job_id, reason: job.resource_profile_error });
      rejectedJobIds.add(job.job_id);
      jobsById.set(job.job_id, job);
      continue;
    }
    jobsById.set(job.job_id, job);
  }
  const pending = new Set(jobs.filter((job) => !rejectedJobIds.has(job.job_id)).map((job) => job.job_id));

  let progress = true;
  while (pending.size > 0 && progress) {
    progress = false;
    const readyJobs = [...pending]
      .map((jobId) => jobsById.get(jobId))
      .filter((job): job is JobProfile => Boolean(job))
      .filter((job) => dependenciesReady(job, jobsById, plannedById, rejectedJobIds, rejectedJobs));

    for (const job of readyJobs.sort(jobSortKey)) {
      const scheduled = tryScheduleJob(job, usableSlots, availableAt, plannedById, policy, knownCachePaths, rejectedJobs);
      pending.delete(job.job_id);
      if (scheduled) {
        plannedJobs.push(scheduled);
        plannedById.set(job.job_id, scheduled);
      } else {
        rejectedJobIds.add(job.job_id);
      }
      progress = true;
    }
  }

  for (const jobId of [...pending].sort()) {
    const job = jobsById.get(jobId);
    if (!job) {
      continue;
    }
    const unresolved = (job.dependencies ?? []).filter((dependency) => !plannedById.has(dependency));
    rejectedJobs.push({
      job_id: job.job_id,
      reason: `unresolved or cyclic dependencies: ${unresolved.length > 0 ? unresolved.join(",") : job.dependencies?.join(",") ?? ""}`,
    });
  }

  return planningResult(slots, jobs, plannedJobs, rejectedJobs, collisions, policy, {
    usable_gpu_count: usableSlots.length,
  });
}

function dependenciesReady(
  job: JobProfile,
  jobsById: Map<string, JobProfile>,
  plannedById: Map<string, PlannedJob>,
  rejectedJobIds: Set<string>,
  rejectedJobs: RejectedJob[],
): boolean {
  for (const dependency of job.dependencies ?? []) {
    if (!jobsById.has(dependency)) {
      rejectedJobs.push({
        job_id: job.job_id,
        reason: `missing dependency: ${dependency}`,
      });
      rejectedJobIds.add(job.job_id);
      return false;
    }
    if (rejectedJobIds.has(dependency)) {
      rejectedJobs.push({
        job_id: job.job_id,
        reason: `dependency was rejected: ${dependency}`,
      });
      rejectedJobIds.add(job.job_id);
      return false;
    }
    if (!plannedById.has(dependency)) {
      return false;
    }
  }
  return true;
}

function tryScheduleJob(
  job: JobProfile,
  usableSlots: NormalizedGpuSlot[],
  availableAt: Map<number, number>,
  plannedById: Map<string, PlannedJob>,
  policy: PlannerPolicy,
  knownCachePaths: Set<string>,
  rejectedJobs: RejectedJob[],
): PlannedJob | null {
  if (job.allow_multiprocess_gpu && !policy.allow_multiprocess_per_gpu) {
    rejectedJobs.push({
      job_id: job.job_id,
      reason: "job requests multiprocess GPU sharing but policy forbids it",
    });
    return null;
  }

  if (policy.require_known_cache_paths) {
    const missing = (job.requires_cache_paths ?? []).filter((cachePath) => !knownCachePaths.has(normalizePathKey(cachePath)));
    if (missing.length > 0) {
      rejectedJobs.push({
        job_id: job.job_id,
        reason: `requires cache paths not listed in planner policy: ${missing.join(",")}`,
      });
      return null;
    }
  }

  if (job.gpu_slots > usableSlots.length) {
    rejectedJobs.push({
      job_id: job.job_id,
      reason: `requires ${job.gpu_slots} GPU slots, only ${usableSlots.length} usable`,
    });
    return null;
  }

  const dependencyReadyAt = Math.max(
    0,
    ...(job.dependencies ?? []).map((dependency) => plannedById.get(dependency)?.expected_finish_seconds ?? 0),
  );
  const feasibleGroups = candidateGpuGroups(usableSlots, job.gpu_slots, availableAt, policy.scheduling_strategy)
    .filter((group) => group.every((slot) => job.expected_peak_memory_mb <= slot.memory_limit_mb))
    .map((group) => ({
      start: Math.max(dependencyReadyAt, ...group.map((slot) => availableAt.get(slot.gpu_id) ?? 0)),
      slotReadyAt: Math.max(0, ...group.map((slot) => availableAt.get(slot.gpu_id) ?? 0)),
      memoryHeadroomMb: Math.min(...group.map((slot) => slot.memory_limit_mb - job.expected_peak_memory_mb)),
      memoryWasteMb: sumNumbers(group.map((slot) => slot.memory_limit_mb - job.expected_peak_memory_mb)),
      group,
    }));

  if (feasibleGroups.length === 0) {
    const maxLimit = Math.max(...usableSlots.map((slot) => slot.memory_limit_mb));
    rejectedJobs.push({
      job_id: job.job_id,
      reason: `expected_peak_memory_mb=${job.expected_peak_memory_mb} exceeds usable slot limit ${maxLimit.toFixed(1)}`,
    });
    return null;
  }

  feasibleGroups.sort((a, b) => compareGpuGroupCandidates(a, b, job, policy.scheduling_strategy));

  const selected = feasibleGroups[0];
  const finish = selected.start + job.expected_wall_clock_seconds;
  const selectedGpuIds = selected.group.map((slot) => slot.gpu_id);
  for (const slot of selected.group) {
    availableAt.set(slot.gpu_id, finish);
  }
  return {
    job_id: job.job_id,
    role: job.role,
    ...(job.stage ? { stage: job.stage } : {}),
    gpu_ids: selectedGpuIds,
    start_after_seconds: selected.start,
    expected_finish_seconds: finish,
    expected_peak_memory_mb: job.expected_peak_memory_mb,
    expected_wall_clock_seconds: job.expected_wall_clock_seconds,
    expected_gpu_utilization: job.expected_gpu_utilization,
    memory_headroom_mb: selected.memoryHeadroomMb,
    dependency_ready_at_seconds: dependencyReadyAt,
    slot_ready_at_seconds: selected.slotReadyAt,
    ...(job.resource_profile_note ? { resource_profile_note: job.resource_profile_note } : {}),
    evidence_class: evidenceClass(job, policy),
    decision: "scheduled",
    reason:
      `${policy.scheduling_strategy} selected gpu_ids=${selectedGpuIds.join(",")} ` +
      `start=${selected.start} finish=${finish} ` +
      `dependency_ready=${dependencyReadyAt} slot_ready=${selected.slotReadyAt} ` +
      `memory_headroom_mb=${selected.memoryHeadroomMb.toFixed(1)} ` +
      `memory_waste_mb=${selected.memoryWasteMb.toFixed(1)}`,
  };
}

export function collisionErrors(jobs: JobProfile[]): string[] {
  const seen = new Map<string, { jobId: string; fieldName: string; path: string }>();
  const errors: string[] = [];

  for (const job of jobs) {
    for (const fieldName of writableFields) {
      const pathValue = job[fieldName];
      if (!pathValue) {
        errors.push(`${job.job_id}.${fieldName} is empty`);
        continue;
      }
      const pathKey = normalizePathKey(pathValue);
      const previous = findOverlappingPath(seen, pathKey);
      if (previous) {
        errors.push(
          `writable path collision: ${job.job_id}.${fieldName} and ${previous.jobId}.${previous.fieldName} both write overlapping paths ${pathValue} and ${previous.path}`,
        );
      }
      seen.set(pathKey, { jobId: job.job_id, fieldName, path: pathValue });
    }
  }

  return errors;
}

export function buildResourceLockRequests(plannedJobs: PlannedJob[], jobs: JobProfile[]): ResourceLockRequest[] {
  const jobsById = new Map(jobs.map((job) => [job.job_id, job]));
  const requests: ResourceLockRequest[] = [];

  for (const plannedJob of plannedJobs) {
    const job = jobsById.get(plannedJob.job_id);
    if (!job) {
      continue;
    }

    requests.push({
      request_id: `${plannedJob.job_id}:gpu_group`,
      job_id: plannedJob.job_id,
      resource_type: "gpu_group",
      mode: "exclusive_gpu",
      starts_at_seconds: plannedJob.start_after_seconds,
      ends_at_seconds: plannedJob.expected_finish_seconds,
      gpu_ids: [...plannedJob.gpu_ids].sort((a, b) => a - b),
      atomic: plannedJob.gpu_ids.length > 1,
      source: "shadow_planner",
    });

    for (const fieldName of artifactWritableFields) {
      requests.push({
        request_id: `${plannedJob.job_id}:writable_path:${fieldName}`,
        job_id: plannedJob.job_id,
        resource_type: "writable_path",
        mode: "exclusive_write",
        starts_at_seconds: plannedJob.start_after_seconds,
        ends_at_seconds: plannedJob.expected_finish_seconds,
        path: job[fieldName],
        path_field: fieldName,
        atomic: false,
        source: "shadow_planner",
      });
    }

    requests.push({
      request_id: `${plannedJob.job_id}:cache_write_path`,
      job_id: plannedJob.job_id,
      resource_type: "cache_write_path",
      mode: "exclusive_write",
      starts_at_seconds: plannedJob.start_after_seconds,
      ends_at_seconds: plannedJob.expected_finish_seconds,
      path: job.cache_write_dir,
      path_field: "cache_write_dir",
      atomic: false,
      source: "shadow_planner",
    });

    for (const cachePath of job.requires_cache_paths ?? []) {
      requests.push({
        request_id: `${plannedJob.job_id}:cache_read:${cachePath}`,
        job_id: plannedJob.job_id,
        resource_type: "cache_path",
        mode: "shared_read",
        starts_at_seconds: plannedJob.start_after_seconds,
        ends_at_seconds: plannedJob.expected_finish_seconds,
        path: cachePath,
        atomic: false,
        source: "shadow_planner",
      });
    }
  }

  return requests;
}

export function detectResourceLockConflicts(requests: ResourceLockRequest[]): ResourceLockConflict[] {
  const conflicts: ResourceLockConflict[] = [];

  for (let leftIndex = 0; leftIndex < requests.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < requests.length; rightIndex += 1) {
      const left = requests[leftIndex];
      const right = requests[rightIndex];
      if (left.job_id === right.job_id || !timeWindowsOverlap(left, right)) {
        continue;
      }

      const gpuIds = overlappingGpuIds(left, right);
      if (gpuIds.length > 0) {
        conflicts.push({
          conflict_id: `gpu_window_overlap:${left.request_id}:${right.request_id}`,
          conflict_type: "gpu_window_overlap",
          resource_type: "gpu_group",
          request_ids: [left.request_id, right.request_id],
          job_ids: [left.job_id, right.job_id],
          starts_at_seconds: Math.max(left.starts_at_seconds, right.starts_at_seconds),
          ends_at_seconds: Math.min(left.ends_at_seconds, right.ends_at_seconds),
          gpu_ids: gpuIds,
          message: `${left.job_id} and ${right.job_id} request GPU(s) ${gpuIds.join(",")} during overlapping windows`,
        });
        continue;
      }

      if (left.path && right.path && pathsOverlap(left.path, right.path)) {
        if (left.resource_type === "cache_write_path" && right.resource_type === "cache_write_path") {
          conflicts.push({
            conflict_id: `cache_write_collision:${left.request_id}:${right.request_id}`,
            conflict_type: "cache_write_collision",
            resource_type: "cache_write_path",
            request_ids: [left.request_id, right.request_id],
            job_ids: [left.job_id, right.job_id],
            starts_at_seconds: Math.max(left.starts_at_seconds, right.starts_at_seconds),
            ends_at_seconds: Math.min(left.ends_at_seconds, right.ends_at_seconds),
            path: normalizePathKey(left.path),
            path_fields: [left.path_field, right.path_field],
            message: `${left.job_id} and ${right.job_id} both request cache write path ${left.path}`,
          });
        } else if (isCacheReadWriteConflict(left, right)) {
          conflicts.push({
            conflict_id: `cache_read_write_collision:${left.request_id}:${right.request_id}`,
            conflict_type: "cache_read_write_collision",
            resource_type: "cache_path",
            request_ids: [left.request_id, right.request_id],
            job_ids: [left.job_id, right.job_id],
            starts_at_seconds: Math.max(left.starts_at_seconds, right.starts_at_seconds),
            ends_at_seconds: Math.min(left.ends_at_seconds, right.ends_at_seconds),
            path: normalizePathKey(left.path),
            path_fields: [left.path_field, right.path_field],
            message: `${left.job_id} and ${right.job_id} request incompatible cache read/write access to ${left.path}`,
          });
        } else if (left.resource_type === "writable_path" && right.resource_type === "writable_path") {
          conflicts.push({
            conflict_id: `writable_path_collision:${left.request_id}:${right.request_id}`,
            conflict_type: "writable_path_collision",
            resource_type: "writable_path",
            request_ids: [left.request_id, right.request_id],
            job_ids: [left.job_id, right.job_id],
            starts_at_seconds: Math.max(left.starts_at_seconds, right.starts_at_seconds),
            ends_at_seconds: Math.min(left.ends_at_seconds, right.ends_at_seconds),
            path: normalizePathKey(left.path),
            path_fields: [left.path_field, right.path_field],
            message: `${left.job_id} and ${right.job_id} both request writable path ${left.path}`,
          });
        }
      }
    }
  }

  return conflicts;
}

function planningResult(
  slots: NormalizedGpuSlot[],
  jobs: JobProfile[],
  plannedJobs: PlannedJob[],
  rejectedJobs: RejectedJob[],
  collisionErrorsValue: string[],
  policy: PlannerPolicy,
  decisionOverrides: Partial<PlannerResult["scheduler_decision"]>,
): PlannerResult {
  const makespan = plannedJobs.reduce((max, job) => Math.max(max, job.expected_finish_seconds), 0);
  const resourceLockRequests = buildResourceLockRequests(plannedJobs, jobs);
  const resourceLockConflicts = detectResourceLockConflicts(resourceLockRequests);
  const resourceLockGrants = grantResourceLocks(resourceLockRequests, resourceLockConflicts);
  const usableGpuCount = decisionOverrides.usable_gpu_count ?? slots.length;
  const planSummary = summarizePlannedResources(plannedJobs, jobs, slots, usableGpuCount, makespan);
  return {
    schema_version: 1,
    scheduler_decision: {
      ok:
        decisionOverrides.ok ??
        (collisionErrorsValue.length === 0 && rejectedJobs.length === 0 && resourceLockConflicts.length === 0),
      execution_state: "shadow_only",
      evidence_default: "provisional_capacity",
      frozen_formal_manifest: policy.frozen_formal_manifest,
      allow_multiprocess_per_gpu: policy.allow_multiprocess_per_gpu,
      keep_eval_slot_free: policy.keep_eval_slot_free,
      require_known_cache_paths: policy.require_known_cache_paths,
      available_cache_path_count: policy.available_cache_paths.length,
      scheduling_strategy: policy.scheduling_strategy,
      usable_gpu_count: decisionOverrides.usable_gpu_count ?? slots.length,
      planned_job_count: plannedJobs.length,
      rejected_job_count: rejectedJobs.length,
      collision_error_count: collisionErrorsValue.length,
      resource_lock_request_count: resourceLockRequests.length,
      resource_lock_grant_count: resourceLockGrants.length,
      resource_lock_conflict_count: resourceLockConflicts.length,
      expected_makespan_seconds: makespan,
      expected_total_slot_seconds: planSummary.totalSlotSeconds,
      expected_busy_slot_seconds: planSummary.busySlotSeconds,
      expected_idle_slot_seconds: planSummary.idleSlotSeconds,
      expected_idle_fraction: planSummary.idleFraction,
      expected_utilization_weighted_slot_seconds: planSummary.utilizationWeightedSlotSeconds,
      expected_mean_gpu_utilization: planSummary.meanGpuUtilization,
      expected_min_memory_headroom_mb: planSummary.minMemoryHeadroomMb,
      planner_objective: buildPlannerObjective({
        strategy: policy.scheduling_strategy,
        planSummary,
        makespan,
        rejectedJobCount: rejectedJobs.length,
        collisionErrorCount: collisionErrorsValue.length,
        resourceLockConflictCount: resourceLockConflicts.length,
      }),
      ...(decisionOverrides.reason ? { reason: decisionOverrides.reason } : {}),
    },
    slots,
    job_profiles: jobs,
    planned_jobs: plannedJobs,
    rejected_jobs: rejectedJobs,
    collision_errors: collisionErrorsValue,
    resource_lock_requests: resourceLockRequests,
    resource_lock_grants: resourceLockGrants,
    resource_lock_conflicts: resourceLockConflicts,
  };
}

function buildPlannerObjective(input: {
  strategy: string;
  planSummary: ReturnType<typeof summarizePlannedResources>;
  makespan: number;
  rejectedJobCount: number;
  collisionErrorCount: number;
  resourceLockConflictCount: number;
}): PlannerObjectiveBreakdown {
  const usesResourceBestFit = input.strategy === "resource_best_fit_v1";
  return {
    objective_id: usesResourceBestFit ? "resource_best_fit_shadow_v1" : "priority_lpt_audit_v1",
    used_for_scheduling: true,
    global_optimality_claim: false,
    job_order_key: [
      "role_priority_ascending",
      "expected_wall_clock_seconds_descending",
      "job_id_ascending",
    ],
    gpu_group_selection_key: usesResourceBestFit
      ? [
          "projected_finish_seconds_ascending",
          "memory_waste_mb_ascending",
          "slot_ready_at_seconds_ascending",
          "gpu_id_list_lexicographic",
        ]
      : [
          "earliest_start_seconds_ascending",
          "gpu_id_list_lexicographic",
        ],
    metrics: [
      {
        name: "expected_makespan_seconds",
        value: input.makespan,
        unit: "seconds",
        direction: "minimize",
        source: "planned_jobs.expected_finish_seconds",
      },
      {
        name: "expected_idle_slot_seconds",
        value: input.planSummary.idleSlotSeconds,
        unit: "gpu_slot_seconds",
        direction: "minimize",
        source: "makespan * usable_gpu_count - busy_slot_seconds",
      },
      {
        name: "expected_busy_slot_seconds",
        value: input.planSummary.busySlotSeconds,
        unit: "gpu_slot_seconds",
        direction: "maximize",
        source: "sum(job_duration_seconds * gpu_slots)",
      },
      {
        name: "expected_mean_gpu_utilization",
        value: input.planSummary.meanGpuUtilization,
        unit: "fraction",
        direction: "maximize",
        source: "sum(job_duration_seconds * gpu_slots * expected_gpu_utilization) / total_slot_seconds",
      },
      {
        name: "expected_min_memory_headroom_mb",
        value: input.planSummary.minMemoryHeadroomMb,
        unit: "megabytes",
        direction: "maximize",
        source: "min(slot.memory_limit_mb - job.expected_peak_memory_mb)",
      },
      {
        name: "expected_memory_waste_mb_slot_seconds",
        value: input.planSummary.memoryWasteMbSlotSeconds,
        unit: "megabyte_gpu_slot_seconds",
        direction: "minimize",
        source: "sum(duration_seconds * sum(slot.memory_limit_mb - job.expected_peak_memory_mb))",
      },
      {
        name: "rejected_job_count",
        value: input.rejectedJobCount,
        unit: "count",
        direction: "guardrail",
        source: "planner rejection list",
      },
      {
        name: "collision_error_count",
        value: input.collisionErrorCount,
        unit: "count",
        direction: "guardrail",
        source: "static path collision detector",
      },
      {
        name: "resource_lock_conflict_count",
        value: input.resourceLockConflictCount,
        unit: "count",
        direction: "guardrail",
        source: "shadow resource lock conflict detector",
      },
    ],
    notes: [
      usesResourceBestFit
        ? "resource_best_fit_v1 is deterministic multi-resource list scheduling, not a CP-SAT/MILP global optimum proof."
        : "priority_lpt is deterministic list scheduling, not a CP-SAT/MILP global optimum proof.",
      ...(usesResourceBestFit
        ? ["resource_best_fit_v1 uses explicit memory and time units but still requires profile rows and telemetry before formal claims."]
        : []),
      "Metrics are audit variables; do not compare plans without matching manifest, profiles, and policy.",
    ],
  };
}

function summarizePlannedResources(
  plannedJobs: PlannedJob[],
  jobs: JobProfile[],
  slots: NormalizedGpuSlot[],
  usableGpuCount: number,
  makespan: number,
): {
  totalSlotSeconds: number;
  busySlotSeconds: number;
  idleSlotSeconds: number;
  idleFraction: number;
  utilizationWeightedSlotSeconds: number;
  meanGpuUtilization: number;
  minMemoryHeadroomMb: number | null;
  memoryWasteMbSlotSeconds: number;
} {
  const jobsById = new Map(jobs.map((job) => [job.job_id, job]));
  const slotsById = new Map(slots.map((slot) => [slot.gpu_id, slot]));
  let busySlotSeconds = 0;
  let utilizationWeightedSlotSeconds = 0;
  let minMemoryHeadroomMb: number | null = null;
  let memoryWasteMbSlotSeconds = 0;

  for (const plannedJob of plannedJobs) {
    const job = jobsById.get(plannedJob.job_id);
    if (!job) {
      continue;
    }
    const duration = plannedJob.expected_finish_seconds - plannedJob.start_after_seconds;
    const slotCount = plannedJob.gpu_ids.length;
    busySlotSeconds += duration * slotCount;
    utilizationWeightedSlotSeconds += duration * slotCount * job.expected_gpu_utilization;

    for (const gpuId of plannedJob.gpu_ids) {
      const slot = slotsById.get(gpuId);
      if (!slot) {
        continue;
      }
      const headroom = slot.memory_limit_mb - job.expected_peak_memory_mb;
      minMemoryHeadroomMb = minMemoryHeadroomMb === null ? headroom : Math.min(minMemoryHeadroomMb, headroom);
      memoryWasteMbSlotSeconds += duration * headroom;
    }
  }

  const totalSlotSeconds = makespan * usableGpuCount;
  const idleSlotSeconds = Math.max(0, totalSlotSeconds - busySlotSeconds);
  const idleFraction = totalSlotSeconds > 0 ? idleSlotSeconds / totalSlotSeconds : 0;
  const meanGpuUtilization = totalSlotSeconds > 0 ? utilizationWeightedSlotSeconds / totalSlotSeconds : 0;

  return {
    totalSlotSeconds,
    busySlotSeconds,
    idleSlotSeconds,
    idleFraction,
    utilizationWeightedSlotSeconds,
    meanGpuUtilization,
    minMemoryHeadroomMb,
    memoryWasteMbSlotSeconds,
  };
}

function normalizeJobProfile(job: JobProfile): JobProfile {
  const normalized: JobProfile = {
    ...job,
    job_id: requireString(job.job_id, "job.job_id"),
    role: requireString(job.role, "job.role"),
    ...(job.stage ? { stage: requireJobStage(job.stage, "job.stage") } : {}),
    dependencies: Array.isArray(job.dependencies)
      ? job.dependencies.map((dependency, index) => requireString(dependency, `job.dependencies[${index}]`))
      : [],
    gpu_slots: requirePositiveInteger(job.gpu_slots, "job.gpu_slots"),
    expected_peak_memory_mb: requirePositiveNumber(job.expected_peak_memory_mb, "job.expected_peak_memory_mb"),
    expected_wall_clock_seconds: requireNonNegativeNumber(
      job.expected_wall_clock_seconds,
      "job.expected_wall_clock_seconds",
    ),
    expected_gpu_utilization: requireNonNegativeNumber(job.expected_gpu_utilization, "job.expected_gpu_utilization"),
    run_dir: requireString(job.run_dir, "job.run_dir"),
    report_dir: requireString(job.report_dir, "job.report_dir"),
    log_dir: requireString(job.log_dir, "job.log_dir"),
    queue_dir: requireString(job.queue_dir, "job.queue_dir"),
    tmp_dir: requireString(job.tmp_dir, "job.tmp_dir"),
    cache_write_dir: requireString(job.cache_write_dir, "job.cache_write_dir"),
    ...(job.resource_profile_recipe_id ? { resource_profile_recipe_id: requireString(job.resource_profile_recipe_id, "job.resource_profile_recipe_id") } : {}),
    ...(job.resource_profile_match ? { resource_profile_match: { ...job.resource_profile_match } } : {}),
    require_resource_profile: Boolean(job.require_resource_profile),
    requires_cache_paths: Array.isArray(job.requires_cache_paths) ? [...job.requires_cache_paths] : [],
    formal_rerun_candidate: Boolean(job.formal_rerun_candidate),
    allow_multiprocess_gpu: Boolean(job.allow_multiprocess_gpu),
  };
  return normalized;
}

function jobSortKey(a: JobProfile, b: JobProfile): number {
  const priorityDelta = priority(a) - priority(b);
  if (priorityDelta !== 0) {
    return priorityDelta;
  }
  const durationDelta = b.expected_wall_clock_seconds - a.expected_wall_clock_seconds;
  if (durationDelta !== 0) {
    return durationDelta;
  }
  return a.job_id.localeCompare(b.job_id);
}

function priority(job: JobProfile): number {
  return rolePriority[job.role] ?? rolePriority.exploratory;
}

function candidateGpuGroups(
  slots: NormalizedGpuSlot[],
  gpuSlots: number,
  availableAt: Map<number, number>,
  strategy: string = "priority_lpt",
): NormalizedGpuSlot[][] {
  if (gpuSlots <= 0) {
    return [];
  }
  const ordered = [...slots].sort((a, b) => {
    const availableDelta = (availableAt.get(a.gpu_id) ?? 0) - (availableAt.get(b.gpu_id) ?? 0);
    if (availableDelta !== 0) {
      return availableDelta;
    }
    const memoryDelta = b.memory_limit_mb - a.memory_limit_mb;
    if (memoryDelta !== 0) {
      return memoryDelta;
    }
    return a.gpu_id - b.gpu_id;
  });
  if (gpuSlots === 1) {
    return ordered.map((slot) => [slot]);
  }
  if (strategy === "resource_best_fit_v1") {
    return combinations(ordered, gpuSlots);
  }
  const groups: NormalizedGpuSlot[][] = [];
  for (let index = 0; index <= ordered.length - gpuSlots; index += 1) {
    groups.push(ordered.slice(index, index + gpuSlots));
  }
  return groups;
}

function compareGpuGroupCandidates(
  a: {
    start: number;
    slotReadyAt: number;
    memoryHeadroomMb: number;
    memoryWasteMb: number;
    group: NormalizedGpuSlot[];
  },
  b: {
    start: number;
    slotReadyAt: number;
    memoryHeadroomMb: number;
    memoryWasteMb: number;
    group: NormalizedGpuSlot[];
  },
  job: JobProfile,
  strategy: string,
): number {
  if (strategy === "resource_best_fit_v1") {
    const finishDelta = (a.start + job.expected_wall_clock_seconds) - (b.start + job.expected_wall_clock_seconds);
    if (finishDelta !== 0) {
      return finishDelta;
    }
    const memoryWasteDelta = a.memoryWasteMb - b.memoryWasteMb;
    if (memoryWasteDelta !== 0) {
      return memoryWasteDelta;
    }
    if (a.slotReadyAt !== b.slotReadyAt) {
      return a.slotReadyAt - b.slotReadyAt;
    }
  } else if (a.start !== b.start) {
    return a.start - b.start;
  }
  return compareNumberArrays(
    a.group.map((slot) => slot.gpu_id),
    b.group.map((slot) => slot.gpu_id),
  );
}

function combinations<T>(items: T[], size: number): T[][] {
  if (size === 0) {
    return [[]];
  }
  if (size > items.length) {
    return [];
  }
  const result: T[][] = [];
  function visit(startIndex: number, chosen: T[]): void {
    if (chosen.length === size) {
      result.push([...chosen]);
      return;
    }
    const remainingNeeded = size - chosen.length;
    for (let index = startIndex; index <= items.length - remainingNeeded; index += 1) {
      chosen.push(items[index]);
      visit(index + 1, chosen);
      chosen.pop();
    }
  }
  visit(0, []);
  return result;
}

function isSupportedSchedulingStrategy(strategy: string): boolean {
  return strategy === "priority_lpt" || strategy === "resource_best_fit_v1";
}

function sumNumbers(values: number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function evidenceClass(job: JobProfile, policy: PlannerPolicy): EvidenceClass {
  if (
    policy.frozen_formal_manifest &&
    ["formal_gate", "matched_baseline", "mechanism_control"].includes(job.role)
  ) {
    return "formal_candidate";
  }
  if (job.formal_rerun_candidate) {
    return "provisional_capacity_rerun_candidate";
  }
  return "provisional_capacity";
}

function compareNumberArrays(a: number[], b: number[]): number {
  for (let index = 0; index < Math.min(a.length, b.length); index += 1) {
    if (a[index] !== b[index]) {
      return a[index] - b[index];
    }
  }
  return a.length - b.length;
}

function grantResourceLocks(requests: ResourceLockRequest[], conflicts: ResourceLockConflict[]) {
  const conflictedJobIds = new Set(conflicts.flatMap((conflict) => conflict.job_ids));
  return requests
    .filter((request) => !conflictedJobIds.has(request.job_id))
    .map((request) => ({
      grant_id: `${request.request_id}:grant`,
      request_id: request.request_id,
      job_id: request.job_id,
      granted: true as const,
      lease: {
        lease_id: `${request.request_id}:lease`,
        request_id: request.request_id,
        job_id: request.job_id,
        resource_type: request.resource_type,
        mode: request.mode,
        starts_at_seconds: request.starts_at_seconds,
        ends_at_seconds: request.ends_at_seconds,
        ...(request.gpu_ids ? { gpu_ids: [...request.gpu_ids] } : {}),
        ...(request.path ? { path: request.path } : {}),
        ...(request.path_field ? { path_field: request.path_field } : {}),
        atomic: request.atomic,
        enforcement: "shadow_only" as const,
      },
      source: "shadow_planner" as const,
    }));
}

function timeWindowsOverlap(left: ResourceLockRequest, right: ResourceLockRequest): boolean {
  return left.starts_at_seconds < right.ends_at_seconds && right.starts_at_seconds < left.ends_at_seconds;
}

function overlappingGpuIds(left: ResourceLockRequest, right: ResourceLockRequest): number[] {
  if (left.resource_type !== "gpu_group" || right.resource_type !== "gpu_group") {
    return [];
  }
  const rightGpuIds = new Set(right.gpu_ids ?? []);
  return (left.gpu_ids ?? []).filter((gpuId) => rightGpuIds.has(gpuId)).sort((a, b) => a - b);
}

function isCacheReadWriteConflict(left: ResourceLockRequest, right: ResourceLockRequest): boolean {
  const leftIsCache = left.resource_type === "cache_path" || left.resource_type === "cache_write_path";
  const rightIsCache = right.resource_type === "cache_path" || right.resource_type === "cache_write_path";
  if (!leftIsCache || !rightIsCache) {
    return false;
  }
  return left.mode === "exclusive_write" || right.mode === "exclusive_write";
}

function findOverlappingPath<T extends { path: string }>(seen: Map<string, T>, candidate: string): T | undefined {
  for (const [seenPath, value] of seen.entries()) {
    if (normalizedPathsOverlap(seenPath, candidate)) {
      return value;
    }
  }
  return undefined;
}

function pathsOverlap(left: string, right: string): boolean {
  return normalizedPathsOverlap(normalizePathKey(left), normalizePathKey(right));
}

function normalizedPathsOverlap(left: string, right: string): boolean {
  return left === right || isPathAncestor(left, right) || isPathAncestor(right, left);
}

function isPathAncestor(parent: string, child: string): boolean {
  if (parent === "." || child === ".") {
    return false;
  }
  const relativePath = path.relative(parent, child);
  return relativePath.length > 0 && !relativePath.startsWith("..") && !path.isAbsolute(relativePath);
}

function normalizePathKey(value: string): string {
  const normalized = path.normalize(value.replaceAll("\\", "/"));
  return normalized.length > 1 && normalized.endsWith("/") ? normalized.slice(0, -1) : normalized;
}

function requireJobStage(value: unknown, label: string): JobStage {
  if (typeof value !== "string" || !validJobStages.has(value as JobStage)) {
    throw new Error(`${label} must be one of: ${[...validJobStages].join(", ")}`);
  }
  return value as JobStage;
}

function requireString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${label} must be a non-empty string`);
  }
  return value;
}

function requireInteger(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isInteger(value)) {
    throw new Error(`${label} must be an integer`);
  }
  return value;
}

function requirePositiveInteger(value: unknown, label: string): number {
  const integer = requireInteger(value, label);
  if (integer <= 0) {
    throw new Error(`${label} must be positive`);
  }
  return integer;
}

function requirePositiveNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new Error(`${label} must be a positive number`);
  }
  return value;
}

function requireNonNegativeNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new Error(`${label} must be a non-negative number`);
  }
  return value;
}

function validateMemoryFraction(value: number, label: string): void {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0 || value > 1) {
    throw new Error(`${label} must be in (0, 1]`);
  }
}
