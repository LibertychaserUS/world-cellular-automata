"""Deterministic resource planner for WCA experiment queues.

The planner is intentionally conservative. It does not try to maximize GPU
busy time at the expense of evidence quality. Formal claims still require a
frozen manifest and strict report gate; planner-selected runs are provisional
unless the caller explicitly marks the plan as a frozen formal rerun.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


WRITABLE_FIELDS = (
    "run_dir",
    "report_dir",
    "log_dir",
    "queue_dir",
    "tmp_dir",
    "cache_write_dir",
)

ROLE_PRIORITY = {
    "formal_gate": 0,
    "matched_baseline": 10,
    "mechanism_control": 20,
    "candidate_search": 30,
    "capacity_scaling": 40,
    "exploratory": 50,
}


@dataclass(frozen=True)
class GpuSlot:
    gpu_id: int
    memory_total_mb: float
    memory_fraction_limit: float = 0.86

    @property
    def memory_limit_mb(self) -> float:
        return self.memory_total_mb * self.memory_fraction_limit


@dataclass(frozen=True)
class JobProfile:
    job_id: str
    role: str
    gpu_slots: int
    expected_peak_memory_mb: float
    expected_wall_clock_seconds: float
    expected_gpu_utilization: float
    run_dir: str
    report_dir: str
    log_dir: str
    queue_dir: str
    tmp_dir: str
    cache_write_dir: str
    model_family: str = ""
    model_variant: str = ""
    requires_cache_paths: tuple[str, ...] = ()
    formal_rerun_candidate: bool = False
    allow_multiprocess_gpu: bool = False

    @property
    def priority(self) -> int:
        return ROLE_PRIORITY.get(self.role, ROLE_PRIORITY["exploratory"])

    @property
    def writable_paths(self) -> dict[str, str]:
        return {field_name: str(getattr(self, field_name)) for field_name in WRITABLE_FIELDS}


@dataclass(frozen=True)
class PlannerPolicy:
    allow_multiprocess_per_gpu: bool = False
    frozen_formal_manifest: bool = False
    keep_eval_slot_free: bool = False
    require_known_cache_paths: bool = False
    available_cache_paths: tuple[str, ...] = ()
    scheduling_strategy: str = "priority_lpt"


@dataclass(frozen=True)
class PlannedJob:
    job_id: str
    gpu_ids: tuple[int, ...]
    start_after_seconds: float
    expected_finish_seconds: float
    evidence_class: str
    decision: str
    reason: str = ""


@dataclass(frozen=True)
class RejectedJob:
    job_id: str
    reason: str


@dataclass(frozen=True)
class PlanningResult:
    planned_jobs: tuple[PlannedJob, ...]
    rejected_jobs: tuple[RejectedJob, ...]
    collision_errors: tuple[str, ...]
    scheduler_decision: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.collision_errors and not self.rejected_jobs


def collision_errors(jobs: Sequence[JobProfile]) -> list[str]:
    seen: dict[str, tuple[str, str]] = {}
    errors: list[str] = []
    for job in jobs:
        for field_name, path in job.writable_paths.items():
            if not path:
                errors.append(f"{job.job_id}.{field_name} is empty")
                continue
            previous = seen.get(path)
            if previous is not None:
                prev_job, prev_field = previous
                errors.append(
                    f"writable path collision: {job.job_id}.{field_name} and "
                    f"{prev_job}.{prev_field} both write {path}"
                )
            seen[path] = (job.job_id, field_name)
    return errors


def _job_sort_key(job: JobProfile) -> tuple[int, float, str]:
    # Higher duration first inside the same scientific priority reduces tail time
    # for independent single-GPU batches.
    return (job.priority, -job.expected_wall_clock_seconds, job.job_id)


def build_gpu_slots(
    *,
    gpu_count: int,
    memory_total_mb: float,
    memory_fraction_limit: float = 0.86,
    start_gpu_id: int = 0,
) -> list[GpuSlot]:
    if gpu_count <= 0:
        raise ValueError("gpu_count must be positive")
    if memory_total_mb <= 0:
        raise ValueError("memory_total_mb must be positive")
    if not 0 < memory_fraction_limit <= 1:
        raise ValueError("memory_fraction_limit must be in (0, 1]")
    return [
        GpuSlot(
            gpu_id=start_gpu_id + index,
            memory_total_mb=memory_total_mb,
            memory_fraction_limit=memory_fraction_limit,
        )
        for index in range(gpu_count)
    ]


def _candidate_gpu_groups(
    slots: Sequence[GpuSlot],
    gpu_slots: int,
    available_at: dict[int, float],
) -> Iterable[tuple[GpuSlot, ...]]:
    if gpu_slots <= 0:
        return []
    ordered = sorted(slots, key=lambda slot: (available_at[slot.gpu_id], -slot.memory_limit_mb, slot.gpu_id))
    if gpu_slots == 1:
        return ((slot,) for slot in ordered)
    groups: list[tuple[GpuSlot, ...]] = []
    for index in range(0, len(ordered) - gpu_slots + 1):
        groups.append(tuple(ordered[index : index + gpu_slots]))
    return groups


def _evidence_class(job: JobProfile, policy: PlannerPolicy) -> str:
    if policy.frozen_formal_manifest and job.role in {"formal_gate", "matched_baseline", "mechanism_control"}:
        return "formal_candidate"
    if job.formal_rerun_candidate:
        return "provisional_capacity_rerun_candidate"
    return "provisional_capacity"


def plan_jobs(slots: Sequence[GpuSlot], jobs: Sequence[JobProfile], policy: PlannerPolicy | None = None) -> PlanningResult:
    policy = policy or PlannerPolicy()
    if policy.scheduling_strategy not in {"priority_lpt"}:
        return PlanningResult(
            planned_jobs=(),
            rejected_jobs=tuple(RejectedJob(job.job_id, f"unsupported scheduling_strategy: {policy.scheduling_strategy}") for job in jobs),
            collision_errors=(),
            scheduler_decision={"ok": False, "reason": f"unsupported scheduling_strategy: {policy.scheduling_strategy}"},
        )
    collisions = collision_errors(jobs)
    known_cache_paths = set(policy.available_cache_paths)
    if policy.keep_eval_slot_free and len(slots) > 1:
        usable_slots = sorted(slots, key=lambda slot: slot.gpu_id)[:-1]
    else:
        usable_slots = list(slots)

    if not usable_slots:
        return PlanningResult(
            planned_jobs=(),
            rejected_jobs=tuple(RejectedJob(job.job_id, "no usable GPU slots") for job in jobs),
            collision_errors=tuple(collisions),
            scheduler_decision={"ok": False, "reason": "no usable GPU slots"},
        )

    available_at = {slot.gpu_id: 0.0 for slot in usable_slots}
    planned: list[PlannedJob] = []
    rejected: list[RejectedJob] = []

    for job in sorted(jobs, key=_job_sort_key):
        if job.allow_multiprocess_gpu and not policy.allow_multiprocess_per_gpu:
            rejected.append(RejectedJob(job.job_id, "job requests multiprocess GPU sharing but policy forbids it"))
            continue
        if policy.require_known_cache_paths:
            missing_cache_paths = [path for path in job.requires_cache_paths if path not in known_cache_paths]
            if missing_cache_paths:
                rejected.append(
                    RejectedJob(
                        job.job_id,
                        "requires cache paths not listed in planner policy: " + ",".join(missing_cache_paths),
                    )
                )
                continue
        if job.gpu_slots > len(usable_slots):
            rejected.append(RejectedJob(job.job_id, f"requires {job.gpu_slots} GPU slots, only {len(usable_slots)} usable"))
            continue

        candidates = list(_candidate_gpu_groups(usable_slots, job.gpu_slots, available_at))
        feasible: list[tuple[float, tuple[GpuSlot, ...]]] = []
        for group in candidates:
            if all(job.expected_peak_memory_mb <= slot.memory_limit_mb for slot in group):
                start = max(available_at[slot.gpu_id] for slot in group)
                feasible.append((start, group))
        if not feasible:
            max_limit = max(slot.memory_limit_mb for slot in usable_slots)
            rejected.append(
                RejectedJob(
                    job.job_id,
                    f"expected_peak_memory_mb={job.expected_peak_memory_mb} exceeds usable slot limit {max_limit:.1f}",
                )
            )
            continue

        start, group = min(feasible, key=lambda item: (item[0], tuple(slot.gpu_id for slot in item[1])))
        finish = start + job.expected_wall_clock_seconds
        for slot in group:
            available_at[slot.gpu_id] = finish
        planned.append(
            PlannedJob(
                job_id=job.job_id,
                gpu_ids=tuple(slot.gpu_id for slot in group),
                start_after_seconds=start,
                expected_finish_seconds=finish,
                evidence_class=_evidence_class(job, policy),
                decision="scheduled",
            )
        )

    makespan = max((job.expected_finish_seconds for job in planned), default=0.0)
    scheduler_decision = {
        "ok": not collisions and not rejected,
        "evidence_default": "provisional_capacity",
        "frozen_formal_manifest": policy.frozen_formal_manifest,
        "allow_multiprocess_per_gpu": policy.allow_multiprocess_per_gpu,
        "keep_eval_slot_free": policy.keep_eval_slot_free,
        "require_known_cache_paths": policy.require_known_cache_paths,
        "available_cache_path_count": len(policy.available_cache_paths),
        "scheduling_strategy": policy.scheduling_strategy,
        "usable_gpu_count": len(usable_slots),
        "planned_job_count": len(planned),
        "rejected_job_count": len(rejected),
        "collision_error_count": len(collisions),
        "expected_makespan_seconds": makespan,
    }
    return PlanningResult(
        planned_jobs=tuple(planned),
        rejected_jobs=tuple(rejected),
        collision_errors=tuple(collisions),
        scheduler_decision=scheduler_decision,
    )


def _require_number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    return float(value)


def gpu_slot_from_json(payload: dict[str, Any]) -> GpuSlot:
    return GpuSlot(
        gpu_id=int(_require_number(payload, "gpu_id")),
        memory_total_mb=_require_number(payload, "memory_total_mb"),
        memory_fraction_limit=float(payload.get("memory_fraction_limit", 0.86)),
    )


def job_profile_from_json(payload: dict[str, Any]) -> JobProfile:
    requires_cache_paths = payload.get("requires_cache_paths", [])
    if not isinstance(requires_cache_paths, list) or not all(isinstance(item, str) for item in requires_cache_paths):
        raise ValueError("requires_cache_paths must be a list of strings")
    kwargs = {
        "job_id": str(payload["job_id"]),
        "role": str(payload["role"]),
        "gpu_slots": int(_require_number(payload, "gpu_slots")),
        "expected_peak_memory_mb": _require_number(payload, "expected_peak_memory_mb"),
        "expected_wall_clock_seconds": _require_number(payload, "expected_wall_clock_seconds"),
        "expected_gpu_utilization": _require_number(payload, "expected_gpu_utilization"),
        "run_dir": str(payload["run_dir"]),
        "report_dir": str(payload["report_dir"]),
        "log_dir": str(payload["log_dir"]),
        "queue_dir": str(payload["queue_dir"]),
        "tmp_dir": str(payload["tmp_dir"]),
        "cache_write_dir": str(payload["cache_write_dir"]),
        "model_family": str(payload.get("model_family", "")),
        "model_variant": str(payload.get("model_variant", "")),
        "requires_cache_paths": tuple(requires_cache_paths),
        "formal_rerun_candidate": bool(payload.get("formal_rerun_candidate", False)),
        "allow_multiprocess_gpu": bool(payload.get("allow_multiprocess_gpu", False)),
    }
    return JobProfile(**kwargs)


def planning_payload(result: PlanningResult, slots: Sequence[GpuSlot], jobs: Sequence[JobProfile]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scheduler_decision": result.scheduler_decision,
        "slots": [asdict(slot) for slot in slots],
        "job_profiles": [asdict(job) for job in jobs],
        "planned_jobs": [asdict(job) for job in result.planned_jobs],
        "rejected_jobs": [asdict(job) for job in result.rejected_jobs],
        "collision_errors": list(result.collision_errors),
    }


def write_planning_artifacts(output_dir: Path, result: PlanningResult, slots: Sequence[GpuSlot], jobs: Sequence[JobProfile]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = planning_payload(result, slots, jobs)
    (output_dir / "scheduler_decision.json").write_text(
        json.dumps(payload["scheduler_decision"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "gpu_assignment.json").write_text(
        json.dumps(payload["planned_jobs"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "job_profiles.json").write_text(
        json.dumps(payload["job_profiles"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "planner_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "planner_decisions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "job_id",
                "gpu_ids",
                "start_after_seconds",
                "expected_finish_seconds",
                "evidence_class",
                "decision",
                "reason",
            ],
        )
        writer.writeheader()
        for job in result.planned_jobs:
            row = asdict(job)
            row["gpu_ids"] = ",".join(str(item) for item in job.gpu_ids)
            writer.writerow(row)
        for job in result.rejected_jobs:
            writer.writerow(
                {
                    "job_id": job.job_id,
                    "gpu_ids": "",
                    "start_after_seconds": "",
                    "expected_finish_seconds": "",
                    "evidence_class": "",
                    "decision": "rejected",
                    "reason": job.reason,
                }
            )
