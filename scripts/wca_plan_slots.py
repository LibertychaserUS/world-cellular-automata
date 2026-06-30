#!/usr/bin/env python3
"""Plan WCA experiment jobs onto GPU slots without launching training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wca.experiments.resource_planner import (  # noqa: E402
    PlannerPolicy,
    build_gpu_slots,
    gpu_slot_from_json,
    job_profile_from_json,
    plan_jobs,
    write_planning_artifacts,
)


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan_spec", type=Path, help="JSON file with slots, jobs, and optional policy")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-count", type=int, default=0, help="Override/generate this many single-GPU slots")
    parser.add_argument("--memory-total-mb", type=float, default=0.0, help="Memory per generated GPU slot")
    parser.add_argument("--memory-fraction-limit", type=float, default=0.86)
    parser.add_argument("--start-gpu-id", type=int, default=0)
    parser.add_argument("--strategy", default="", help="Override policy.scheduling_strategy")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = load_json_object(args.plan_spec)
    slots_raw = spec.get("slots")
    jobs_raw = spec.get("jobs")
    if not isinstance(jobs_raw, list) or not jobs_raw:
        raise ValueError("plan spec must define non-empty jobs list")
    policy_raw = spec.get("policy", {})
    if not isinstance(policy_raw, dict):
        raise ValueError("policy must be an object when provided")

    if args.gpu_count:
        if args.memory_total_mb <= 0:
            raise ValueError("--memory-total-mb is required when --gpu-count is used")
        slots = build_gpu_slots(
            gpu_count=args.gpu_count,
            memory_total_mb=args.memory_total_mb,
            memory_fraction_limit=args.memory_fraction_limit,
            start_gpu_id=args.start_gpu_id,
        )
    elif isinstance(spec.get("machine"), dict):
        machine = spec["machine"]
        slots = build_gpu_slots(
            gpu_count=int(machine["gpu_count"]),
            memory_total_mb=float(machine["memory_total_mb"]),
            memory_fraction_limit=float(machine.get("memory_fraction_limit", args.memory_fraction_limit)),
            start_gpu_id=int(machine.get("start_gpu_id", args.start_gpu_id)),
        )
    elif isinstance(slots_raw, list) and slots_raw:
        slots = [gpu_slot_from_json(item) for item in slots_raw if isinstance(item, dict)]
    else:
        raise ValueError("plan spec must define non-empty slots list or machine gpu_count/memory_total_mb")
    jobs = [job_profile_from_json(item) for item in jobs_raw if isinstance(item, dict)]
    scheduling_strategy = args.strategy or str(policy_raw.get("scheduling_strategy", "priority_lpt"))
    policy = PlannerPolicy(
        allow_multiprocess_per_gpu=bool(policy_raw.get("allow_multiprocess_per_gpu", False)),
        frozen_formal_manifest=bool(policy_raw.get("frozen_formal_manifest", False)),
        keep_eval_slot_free=bool(policy_raw.get("keep_eval_slot_free", False)),
        require_known_cache_paths=bool(policy_raw.get("require_known_cache_paths", False)),
        available_cache_paths=tuple(
            item for item in policy_raw.get("available_cache_paths", []) if isinstance(item, str)
        ),
        scheduling_strategy=scheduling_strategy,
    )
    result = plan_jobs(slots, jobs, policy)
    write_planning_artifacts(args.output_dir, result, slots, jobs)
    print(json.dumps(result.scheduler_decision, sort_keys=True))
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
