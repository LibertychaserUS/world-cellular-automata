import path from "node:path/posix";

import type {
  ResourceLease,
  ResourceLockConflictType,
  ResourceLockRequest,
  RuntimeLeaseConflict,
  RuntimeLeaseEvaluation,
  RuntimeLeaseRecord,
  StaleLeaseDecision,
  StaleLeaseProcessState,
} from "./contracts.ts";

export function evaluateRuntimeLeases(input: {
  requests: ResourceLockRequest[];
  existingLeases: RuntimeLeaseRecord[];
  nowEpochSeconds: number;
  processStates?: Record<string, StaleLeaseProcessState>;
}): RuntimeLeaseEvaluation {
  const staleDecisions = input.existingLeases.flatMap((lease) =>
    staleDecision(lease, input.nowEpochSeconds, input.processStates?.[lease.runtime_lease_id]),
  );
  const blockingLeases = input.existingLeases.filter((lease) =>
    leaseBlocksNewRequests(lease, input.nowEpochSeconds, input.processStates?.[lease.runtime_lease_id]),
  );
  const conflicts = runtimeConflicts(input.requests, blockingLeases);

  return {
    schema_version: 1,
    ok: conflicts.length === 0 && staleDecisions.every((decision) => decision.decision !== "manual_review"),
    now_epoch_seconds: input.nowEpochSeconds,
    requested_count: input.requests.length,
    active_lease_count: blockingLeases.length,
    stale_lease_count: staleDecisions.length,
    conflict_count: conflicts.length,
    stale_decisions: staleDecisions,
    conflicts,
  };
}

export function leaseIsExpired(lease: RuntimeLeaseRecord, nowEpochSeconds: number): boolean {
  if (lease.status !== "active") {
    return false;
  }
  return lease.owner.last_heartbeat_epoch_seconds + lease.ttl_seconds < nowEpochSeconds;
}

function staleDecision(
  lease: RuntimeLeaseRecord,
  nowEpochSeconds: number,
  processState: StaleLeaseProcessState = "unknown",
): StaleLeaseDecision[] {
  if (!leaseIsExpired(lease, nowEpochSeconds)) {
    return [];
  }
  const staleFor = nowEpochSeconds - (lease.owner.last_heartbeat_epoch_seconds + lease.ttl_seconds);
  if (processState === "live") {
    return [{
      runtime_lease_id: lease.runtime_lease_id,
      job_id: lease.job_id,
      stale_for_seconds: staleFor,
      process_state: processState,
      decision: "keep_active",
      reason: "lease heartbeat expired but owner process is still live; keep blocking until heartbeat/recovery is resolved",
    }];
  }
  if (processState === "dead") {
    return [{
      runtime_lease_id: lease.runtime_lease_id,
      job_id: lease.job_id,
      stale_for_seconds: staleFor,
      process_state: processState,
      decision: "mark_stale",
      reason: "lease heartbeat expired and owner process is dead; lease is recoverable after evidence is preserved",
    }];
  }
  return [{
    runtime_lease_id: lease.runtime_lease_id,
    job_id: lease.job_id,
    stale_for_seconds: staleFor,
    process_state: processState,
    decision: "manual_review",
    reason: "lease heartbeat expired but owner process state is unknown; fail closed",
  }];
}

function leaseBlocksNewRequests(
  lease: RuntimeLeaseRecord,
  nowEpochSeconds: number,
  processState: StaleLeaseProcessState = "unknown",
): boolean {
  if (lease.status !== "active") {
    return false;
  }
  if (!leaseIsExpired(lease, nowEpochSeconds)) {
    return true;
  }
  return processState !== "dead";
}

function runtimeConflicts(requests: ResourceLockRequest[], existingLeases: RuntimeLeaseRecord[]): RuntimeLeaseConflict[] {
  const conflicts: RuntimeLeaseConflict[] = [];
  for (const request of requests) {
    for (const holder of existingLeases) {
      for (const lease of holder.resource_leases) {
        const conflict = runtimeConflict(request, holder, lease);
        if (conflict) {
          conflicts.push(conflict);
        }
      }
    }
  }
  return conflicts;
}

function runtimeConflict(
  request: ResourceLockRequest,
  holder: RuntimeLeaseRecord,
  lease: ResourceLease,
): RuntimeLeaseConflict | null {
  const gpuIds = overlappingGpuIds(request.gpu_ids, lease.gpu_ids);
  if (request.resource_type === "gpu_group" && lease.resource_type === "gpu_group" && gpuIds.length > 0) {
    return {
      conflict_id: `runtime_gpu:${request.request_id}:${holder.runtime_lease_id}:${lease.lease_id}`,
      request_id: request.request_id,
      request_job_id: request.job_id,
      holder_runtime_lease_id: holder.runtime_lease_id,
      holder_job_id: holder.job_id,
      resource_type: "gpu_group",
      conflict_type: "gpu_window_overlap",
      gpu_ids: gpuIds,
      message: `${request.job_id} requests GPU(s) ${gpuIds.join(",")} held by active lease ${holder.runtime_lease_id}`,
    };
  }

  if (!request.path || !lease.path || !pathsOverlap(request.path, lease.path) || !modesConflict(request.mode, lease.mode)) {
    return null;
  }

  const conflictType = pathConflictType(request.resource_type, lease.resource_type);
  return {
    conflict_id: `runtime_path:${request.request_id}:${holder.runtime_lease_id}:${lease.lease_id}`,
    request_id: request.request_id,
    request_job_id: request.job_id,
    holder_runtime_lease_id: holder.runtime_lease_id,
    holder_job_id: holder.job_id,
    resource_type: request.resource_type,
    conflict_type: conflictType,
    path: normalizePathKey(request.path),
    message: `${request.job_id} requests ${request.path} but active lease ${holder.runtime_lease_id} holds ${lease.path}`,
  };
}

function overlappingGpuIds(left?: number[], right?: number[]): number[] {
  if (!left || !right) {
    return [];
  }
  const rightIds = new Set(right);
  return left.filter((gpuId) => rightIds.has(gpuId)).sort((a, b) => a - b);
}

function modesConflict(left: string, right: string): boolean {
  return left === "exclusive_write" || right === "exclusive_write";
}

function pathConflictType(left: string, right: string): ResourceLockConflictType {
  if (left === "cache_write_path" && right === "cache_write_path") {
    return "cache_write_collision";
  }
  if (left.startsWith("cache") || right.startsWith("cache")) {
    return "cache_read_write_collision";
  }
  return "writable_path_collision";
}

function pathsOverlap(left: string, right: string): boolean {
  const normalizedLeft = normalizePathKey(left);
  const normalizedRight = normalizePathKey(right);
  return normalizedLeft === normalizedRight ||
    normalizedLeft.startsWith(`${normalizedRight}/`) ||
    normalizedRight.startsWith(`${normalizedLeft}/`);
}

function normalizePathKey(value: string): string {
  return path.normalize(value).replace(/^\/+/u, "").replace(/\/+$/u, "");
}
