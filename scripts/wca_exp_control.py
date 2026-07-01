#!/usr/bin/env python3
"""Manifest-driven control plane for strict WCA experiment campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


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


REQUIRED_MANIFEST_KEYS = (
    "schema_version",
    "control_plane_version",
    "experiment_id",
    "protocol",
    "submission_policy",
    "git",
    "eval",
    "checkpoint",
    "runner",
    "artifacts",
    "model_matrix",
    "jobs",
    "report_contract",
)

QUEUE_FILE_NAMES = {
    "manifest.json",
    "status.jsonl",
    "queue_summary.json",
    "launcher.log",
    "runner.pid",
    "remote_submission.json",
}
RUN_FILE_NAMES = {
    "config.json",
    "resolved_config.json",
    "summary.json",
    "train_log.csv",
    "model.pt",
    "model_hash.json",
}
QUEUE_PROVENANCE_FILE_NAMES = {
    "manifest.json",
    "remote_submission.json",
    "status.jsonl",
    "queue_summary.json",
}
REPORT_FILE_NAMES = {
    "results.md",
    "results.csv",
    "formal_evidence.md",
    "results_by_horizon.md",
    "results_by_horizon.csv",
    "summary.json",
    "per_sample_rows.csv",
    "eval_plan.json",
    "sentinel_eval_plan.json",
    "sentinel_results.csv",
    "sentinel_summary.json",
}
KNOWN_ARTIFACT_FILE_NAMES = QUEUE_FILE_NAMES | RUN_FILE_NAMES | REPORT_FILE_NAMES | {
    "capacity_plan.json",
    "capacity_plan.md",
    "cache_equivalence.json",
    "cache_profile.csv",
    "evidence_index.json",
    "equivalence_results.csv",
    "gate_summary.json",
    "gpu_assignment.json",
    "gpu_utilization_trace.csv",
    "job_profiles.json",
    "machine_profile.json",
    "memory_long.csv",
    "optimization_gate.json",
    "performance_long.csv",
    "planner_decisions.csv",
    "profile_long.csv",
    "profile_metadata.json",
    "scheduler_decision.json",
    "system_profile.json",
}
RAW_FIELD_EXTERNAL_FAMILIES = {"convnet", "fno", "unet"}
RAW_FIELD_EXTERNAL_ROLE_MARKERS = ("raw_field", "not_token_equivalent")
TOKEN_LEVEL_ROLE_MARKERS = ("wca", "token_equivalent", "learnable_interface", "decoder_capacity")
EXPLORATORY_COMPARISON_STATUSES = {"exploratory", "diagnostic", "external_anchor", "not_in_strict_matched_eval"}
LOCAL_PRECLOUD_GATE_SUBSTRINGS = ("static_tests",)
REMOTE_PYTHON_NAMES = {"python", "python3", "python3.10", "python3.11", "python3.12"}
REMOTE_REPO_ROOT = "/root/wca"
TEXT_CAPTURE_LIMIT = 8000


def is_safe_relative_path(value: str) -> bool:
    path = Path(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def normalize_relative_path(value: str) -> str:
    return Path(value).as_posix().strip("/")


def phase_number(phase: object) -> int | None:
    text = str(phase or "").strip().upper()
    if text.startswith("V"):
        text = text[1:]
    digits = ""
    for char in text:
        if char.isdigit():
            digits += char
        else:
            break
    return int(digits) if digits else None


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_text(canonical_json(payload))


def load_manifest(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        payload = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on optional local env
            raise ValueError("YAML manifests require PyYAML; use JSON or install pyyaml") from exc
        payload = yaml.safe_load(text)
    else:
        raise ValueError(f"unsupported manifest extension: {path.suffix}")
    if not isinstance(payload, dict):
        raise ValueError("control manifest must be a JSON/YAML object")
    return payload


def validate_manifest_shape(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in REQUIRED_MANIFEST_KEYS:
        if key not in manifest:
            errors.append(f"missing required manifest key: {key}")

    if not errors:
        if not isinstance(manifest["experiment_id"], str) or not manifest["experiment_id"]:
            errors.append("experiment_id must be a non-empty string")
        if not isinstance(manifest["protocol"], dict):
            errors.append("protocol must be an object")
        if not isinstance(manifest["submission_policy"], dict):
            errors.append("submission_policy must be an object")
        if not isinstance(manifest["artifacts"], dict):
            errors.append("artifacts must be an object")
        if not isinstance(manifest["model_matrix"], list) or not manifest["model_matrix"]:
            errors.append("model_matrix must be a non-empty list")
        if not isinstance(manifest["jobs"], list) or not manifest["jobs"]:
            errors.append("jobs must be a non-empty list")

    if isinstance(manifest.get("protocol"), dict):
        protocol = manifest["protocol"]
        if not isinstance(protocol.get("phase"), str) or not protocol.get("phase"):
            errors.append("protocol.phase must be a non-empty string")
        if protocol.get("strict") is not True:
            errors.append("protocol.strict must be true for formal control manifests")

    if isinstance(manifest.get("artifacts"), dict):
        artifacts = manifest["artifacts"]
        if not isinstance(artifacts.get("queue_dir"), str) or not artifacts.get("queue_dir"):
            errors.append("artifacts.queue_dir must be a non-empty string")
        elif not is_safe_relative_path(artifacts["queue_dir"]):
            errors.append("artifacts.queue_dir must be a safe repository-relative path")
        for key in ("report_dirs", "run_dirs"):
            value = artifacts.get(key, [])
            if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
                errors.append(f"artifacts.{key} must be a list of non-empty strings")
            else:
                for item in value:
                    if not is_safe_relative_path(item):
                        errors.append(f"artifacts.{key} contains unsafe path: {item}")
        required_files = artifacts.get("required_files", [])
        if required_files and not isinstance(required_files, list):
            errors.append("artifacts.required_files must be a list when provided")
        elif isinstance(required_files, list):
            for item in required_files:
                if not isinstance(item, str) or not item:
                    errors.append("artifacts.required_files must contain non-empty strings")
                elif item not in KNOWN_ARTIFACT_FILE_NAMES:
                    errors.append(f"artifacts.required_files contains unknown artifact file name: {item}")
        required_paths = artifacts.get("required_paths", [])
        if required_paths and not isinstance(required_paths, list):
            errors.append("artifacts.required_paths must be a list when provided")
        elif isinstance(required_paths, list):
            for item in required_paths:
                if not isinstance(item, str) or not item:
                    errors.append("artifacts.required_paths must contain non-empty strings")
                elif not is_safe_relative_path(item):
                    errors.append(f"artifacts.required_paths contains unsafe path: {item}")

    for index, job in enumerate(manifest.get("jobs", [])):
        if not isinstance(job, dict):
            errors.append(f"jobs[{index}] must be an object")
            continue
        if not isinstance(job.get("id"), str) or not job.get("id"):
            errors.append(f"jobs[{index}].id must be a non-empty string")
        if not isinstance(job.get("argv"), list) or not job.get("argv"):
            errors.append(f"jobs[{index}].argv must be a non-empty list")
        elif not all(isinstance(arg, str) for arg in job["argv"]):
            errors.append(f"jobs[{index}].argv must contain strings only")

    return errors


def validate_manifest(
    manifest: dict[str, Any],
    *,
    local_root: Path,
    enforce_clean_git: bool = True,
    enforce_prerequisites: bool = True,
    enforce_submit_contract: bool = True,
) -> tuple[list[str], list[str]]:
    errors = validate_manifest_shape(manifest)
    warnings: list[str] = []
    policy = manifest.get("submission_policy")
    if isinstance(policy, dict) and policy.get("allow_submit") is False:
        warnings.append("allow_submit=false: manifest is draft/validation-only")

    if (
        not errors
        and enforce_clean_git
        and isinstance(policy, dict)
        and policy.get("allow_submit") is True
        and bool(policy.get("require_clean_git"))
    ):
        dirty = git_status_short(local_root)
        if dirty:
            errors.append("git working tree is dirty but submission_policy.require_clean_git=true")

    if not errors and enforce_submit_contract:
        errors.extend(validate_strict_contract(manifest))
    if not errors:
        errors.extend(validate_artifact_ownership_contract(manifest))
    if not errors:
        errors.extend(validate_strict_model_matrix_scope(manifest))
    if not errors:
        errors.extend(validate_referenced_repo_paths(manifest, local_root=local_root))
        errors.extend(validate_planner_contract(manifest))
        if enforce_prerequisites:
            errors.extend(validate_required_prerequisites(manifest, local_root=local_root))

    return errors, warnings


def _planner_enabled(manifest: dict[str, Any]) -> bool:
    planner = manifest.get("planner")
    return isinstance(planner, dict) and planner.get("enabled") is True


def _planner_policy(planner: dict[str, Any]) -> PlannerPolicy:
    policy = planner.get("policy", {})
    if not isinstance(policy, dict):
        policy = {}
    available_cache_paths = policy.get("available_cache_paths", [])
    if not isinstance(available_cache_paths, list):
        available_cache_paths = []
    return PlannerPolicy(
        allow_multiprocess_per_gpu=bool(policy.get("allow_multiprocess_per_gpu", False)),
        frozen_formal_manifest=bool(policy.get("frozen_formal_manifest", False)),
        keep_eval_slot_free=bool(policy.get("keep_eval_slot_free", False)),
        require_known_cache_paths=bool(policy.get("require_known_cache_paths", False)),
        available_cache_paths=tuple(item for item in available_cache_paths if isinstance(item, str)),
        scheduling_strategy=str(policy.get("scheduling_strategy", "priority_lpt")),
    )


def build_planner_inputs(manifest: dict[str, Any]):
    planner = manifest.get("planner")
    if not isinstance(planner, dict):
        raise ValueError("planner must be an object")
    machine_raw = planner.get("machine")
    slots_raw = planner.get("slots")
    profiles_raw = planner.get("job_profiles")
    if not isinstance(profiles_raw, list) or not profiles_raw:
        raise ValueError("planner.job_profiles must be a non-empty list")
    if not all(isinstance(item, dict) for item in profiles_raw):
        raise ValueError("planner.job_profiles must contain objects")
    if isinstance(machine_raw, dict):
        try:
            slots = build_gpu_slots(
                gpu_count=int(machine_raw["gpu_count"]),
                memory_total_mb=float(machine_raw["memory_total_mb"]),
                memory_fraction_limit=float(machine_raw.get("memory_fraction_limit", 0.86)),
                start_gpu_id=int(machine_raw.get("start_gpu_id", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"planner.machine invalid: {exc}") from exc
    elif isinstance(slots_raw, list) and slots_raw:
        if not all(isinstance(item, dict) for item in slots_raw):
            raise ValueError("planner.slots must contain objects")
        slots = [gpu_slot_from_json(item) for item in slots_raw]
    else:
        raise ValueError("planner must define non-empty slots list or machine gpu_count/memory_total_mb")
    return (
        slots,
        [job_profile_from_json(item) for item in profiles_raw],
        _planner_policy(planner),
    )


def validate_planner_contract(manifest: dict[str, Any]) -> list[str]:
    if not _planner_enabled(manifest):
        return []
    errors: list[str] = []
    try:
        slots, profiles, policy = build_planner_inputs(manifest)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"planner contract invalid: {exc}"]

    manifest_job_ids = {
        job.get("id")
        for job in manifest.get("jobs", [])
        if isinstance(job, dict) and isinstance(job.get("id"), str)
    }
    for profile in profiles:
        if profile.job_id not in manifest_job_ids:
            errors.append(f"planner.job_profiles contains job_id not present in jobs: {profile.job_id}")
    result = plan_jobs(slots, profiles, policy)
    errors.extend(f"planner collision: {item}" for item in result.collision_errors)
    errors.extend(f"planner rejected {item.job_id}: {item.reason}" for item in result.rejected_jobs)
    return errors


def validate_strict_contract(manifest: dict[str, Any]) -> list[str]:
    protocol = manifest.get("protocol")
    policy = manifest.get("submission_policy")
    artifacts = manifest.get("artifacts")
    if not (isinstance(protocol, dict) and isinstance(policy, dict) and isinstance(artifacts, dict)):
        return []
    if protocol.get("strict") is not True or policy.get("allow_submit") is not True:
        return []

    errors: list[str] = []
    phase = phase_number(protocol.get("phase"))
    if phase is not None and phase >= 21:
        required_paths = artifacts.get("required_paths", [])
        if not isinstance(required_paths, list) or not required_paths:
            errors.append("V21+ strict submit manifests must define artifacts.required_paths")
        else:
            queue_dir = normalize_relative_path(str(artifacts.get("queue_dir", "")))
            required_set = {normalize_relative_path(path) for path in required_paths if isinstance(path, str)}
            for name in QUEUE_PROVENANCE_FILE_NAMES:
                expected = f"{queue_dir}/{name}"
                if expected not in required_set:
                    errors.append(f"V21+ strict submit manifests must require queue provenance path: {expected}")
        prerequisites = manifest.get("requires_complete_manifests", [])
        if not isinstance(prerequisites, list) or not prerequisites:
            errors.append("V21+ strict submit manifests must define requires_complete_manifests")
    return errors


def validate_artifact_ownership_contract(manifest: dict[str, Any]) -> list[str]:
    protocol = manifest.get("protocol")
    policy = manifest.get("submission_policy")
    artifacts = manifest.get("artifacts")
    phase = phase_number(protocol.get("phase")) if isinstance(protocol, dict) else None
    strict_submit = (
        isinstance(protocol, dict)
        and isinstance(policy, dict)
        and protocol.get("strict") is True
        and policy.get("allow_submit") is True
    )
    requires_explicit_owners = strict_submit and phase is not None and phase >= 25

    report_contract = manifest.get("report_contract")
    if not isinstance(report_contract, dict):
        if requires_explicit_owners:
            return ["V25+ strict submit manifests must define report_contract"]
        return []

    def is_named_artifact(path: str, name: str) -> bool:
        return path == name or path.endswith(f"/{name}")

    required_outputs_raw = report_contract.get("required_outputs", [])
    if required_outputs_raw is None:
        required_outputs_raw = []
    if not isinstance(required_outputs_raw, list):
        return ["report_contract.required_outputs must be a list when provided"]

    errors: list[str] = []
    required_outputs: list[str] = []
    for output in required_outputs_raw:
        if not isinstance(output, str):
            errors.append("report_contract.required_outputs must contain strings only")
            continue
        normalized_output = normalize_relative_path(output)
        if not is_safe_relative_path(normalized_output):
            errors.append(f"report_contract.required_outputs contains unsafe path: {output}")
            continue
        required_outputs.append(normalized_output)

    raw_owners = report_contract.get("artifact_owners")
    if raw_owners is None:
        if requires_explicit_owners and not required_outputs:
            return ["V25+ strict submit manifests must define report_contract.required_outputs"]
        if requires_explicit_owners:
            return ["V25+ strict submit manifests with report_contract.required_outputs must define report_contract.artifact_owners"]
        return []
    if not isinstance(raw_owners, list) or not raw_owners:
        return ["report_contract.artifact_owners must be a non-empty list when provided"]

    job_ids = {
        str(job.get("id"))
        for job in manifest.get("jobs", [])
        if isinstance(job, dict) and isinstance(job.get("id"), str)
    }
    required_output_set = set(required_outputs)
    artifact_required_paths: set[str] = set()
    if isinstance(artifacts, dict):
        required_paths_raw = artifacts.get("required_paths", [])
        if isinstance(required_paths_raw, list):
            artifact_required_paths = {
                normalize_relative_path(path)
                for path in required_paths_raw
                if isinstance(path, str) and is_safe_relative_path(normalize_relative_path(path))
            }
    seen_paths: dict[str, str] = {}
    for index, item in enumerate(raw_owners):
        if not isinstance(item, dict):
            errors.append(f"report_contract.artifact_owners[{index}] must be an object")
            continue
        path = item.get("path")
        owner = item.get("owner")
        if not isinstance(path, str) or not path:
            errors.append(f"report_contract.artifact_owners[{index}].path must be a non-empty string")
            continue
        normalized_path = normalize_relative_path(path)
        if not is_safe_relative_path(normalized_path):
            errors.append(f"report_contract.artifact_owners[{index}].path is unsafe: {path}")
            continue
        if not isinstance(owner, str) or not owner:
            errors.append(f"report_contract.artifact_owners[{index}].owner must be a non-empty job id")
            continue
        if owner not in job_ids:
            errors.append(f"report_contract.artifact_owners[{index}].owner is not a job id: {owner}")
        previous_owner = seen_paths.get(normalized_path)
        if previous_owner is not None:
            errors.append(
                f"report_contract.artifact_owners duplicates artifact path {normalized_path}: "
                f"{previous_owner} and {owner}"
            )
        if requires_explicit_owners and normalized_path not in required_output_set:
            errors.append(f"artifact owner path must be listed in report_contract.required_outputs: {normalized_path}")
        if requires_explicit_owners and artifact_required_paths and normalized_path not in artifact_required_paths:
            errors.append(f"artifact owner path must be listed in artifacts.required_paths: {normalized_path}")
        seen_paths[normalized_path] = owner

    if requires_explicit_owners:
        if not required_outputs:
            errors.append("V25+ strict submit manifests must define report_contract.required_outputs")
        for normalized_output in required_outputs:
            if normalized_output not in seen_paths:
                errors.append(f"report required output must declare exactly one artifact owner: {normalized_output}")
            if artifact_required_paths and normalized_output not in artifact_required_paths:
                errors.append(f"report required output must be listed in artifacts.required_paths: {normalized_output}")
    for normalized_output in required_outputs:
        if is_named_artifact(normalized_output, "eval_plan.json") and normalized_output not in seen_paths:
            errors.append(f"formal eval plan must declare exactly one artifact owner: {normalized_output}")
        if is_named_artifact(normalized_output, "sentinel_eval_plan.json") and normalized_output not in seen_paths:
            errors.append(f"sentinel eval plan must declare exactly one artifact owner: {normalized_output}")

    formal_eval_plan_owners = [
        owner
        for path, owner in seen_paths.items()
        if is_named_artifact(path, "eval_plan.json")
    ]
    for owner in formal_eval_plan_owners:
        job = next((job for job in manifest.get("jobs", []) if isinstance(job, dict) and job.get("id") == owner), {})
        argv = " ".join(str(arg) for arg in job.get("argv", []) if isinstance(arg, str)) if isinstance(job, dict) else ""
        if "eval_field_sentinels.py" in argv:
            errors.append("eval_field_sentinels.py must not own formal eval_plan.json; use sentinel_eval_plan.json")
    return errors


def _model_matrix_comparison_group(entry: dict[str, Any]) -> str:
    for key in ("comparison_group", "strict_comparison_group", "comparison_table"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value
    return "strict_matched"


def _model_matrix_is_exploratory(entry: dict[str, Any]) -> bool:
    if entry.get("exploratory") is True:
        return True
    for key in ("comparison_status", "formal_status", "evidence_status"):
        value = entry.get(key)
        if isinstance(value, str) and value in EXPLORATORY_COMPARISON_STATUSES:
            return True
    return False


def _model_matrix_scope_kind(entry: dict[str, Any]) -> str:
    family = str(entry.get("family", "")).lower()
    role = str(entry.get("role", "")).lower()
    text = f"{family} {role}"
    if family in RAW_FIELD_EXTERNAL_FAMILIES or all(marker in text for marker in RAW_FIELD_EXTERNAL_ROLE_MARKERS):
        return "raw_field_external_anchor_not_token_equivalent"
    if any(marker in text for marker in TOKEN_LEVEL_ROLE_MARKERS):
        return "token_level_formal"
    return "other"


def validate_strict_model_matrix_scope(manifest: dict[str, Any]) -> list[str]:
    protocol = manifest.get("protocol")
    policy = manifest.get("submission_policy")
    if not (
        isinstance(protocol, dict)
        and isinstance(policy, dict)
        and protocol.get("strict") is True
        and policy.get("allow_submit") is True
    ):
        return []

    errors: list[str] = []
    grouped: dict[str, dict[str, list[str]]] = {}
    for index, entry in enumerate(manifest.get("model_matrix", [])):
        if not isinstance(entry, dict):
            continue
        if _model_matrix_is_exploratory(entry):
            continue
        kind = _model_matrix_scope_kind(entry)
        group = _model_matrix_comparison_group(entry)
        label = str(entry.get("id") or entry.get("run_dir") or f"model_matrix[{index}]")
        grouped.setdefault(group, {}).setdefault(kind, []).append(label)

    for group, by_kind in sorted(grouped.items()):
        raw = by_kind.get("raw_field_external_anchor_not_token_equivalent", [])
        token = by_kind.get("token_level_formal", [])
        if raw and token:
            errors.append(
                "strict model_matrix comparison group mixes token-level formal entries with "
                "raw_field_external_anchor_not_token_equivalent entries; mark raw anchors exploratory "
                f"or split comparison_group. group={group} raw={raw[:5]} token={token[:5]}"
            )
    return errors


def validate_referenced_repo_paths(manifest: dict[str, Any], *, local_root: Path) -> list[str]:
    errors: list[str] = []
    config_paths: set[str] = set()
    for index, model in enumerate(manifest.get("model_matrix", [])):
        if not isinstance(model, dict):
            continue
        config_path = model.get("config_path")
        if config_path is None:
            continue
        if not isinstance(config_path, str) or not config_path:
            errors.append(f"model_matrix[{index}].config_path must be a non-empty string when provided")
            continue
        if not is_safe_relative_path(config_path):
            errors.append(f"model_matrix[{index}].config_path must be repository-relative: {config_path}")
            continue
        config_paths.add(normalize_relative_path(config_path))

    for job_index, job in enumerate(manifest.get("jobs", [])):
        if not isinstance(job, dict) or not isinstance(job.get("argv"), list):
            continue
        argv = job["argv"]
        for index, arg in enumerate(argv[:-1]):
            if arg != "--config":
                continue
            config_path = argv[index + 1]
            if not isinstance(config_path, str) or not config_path:
                errors.append(f"jobs[{job_index}].argv --config value must be a non-empty string")
            elif not is_safe_relative_path(config_path):
                errors.append(f"jobs[{job_index}].argv --config must be repository-relative: {config_path}")
            else:
                config_paths.add(normalize_relative_path(config_path))

    for config_path in sorted(config_paths):
        if not (local_root / config_path).exists():
            errors.append(f"referenced config_path does not exist: {config_path}")
    return errors


def validate_required_prerequisites(manifest: dict[str, Any], *, local_root: Path) -> list[str]:
    policy = manifest.get("submission_policy")
    if not (isinstance(policy, dict) and policy.get("allow_submit") is True):
        return []

    errors: list[str] = []
    prerequisites = manifest.get("requires_complete_manifests", [])
    if prerequisites is None:
        return errors
    if not isinstance(prerequisites, list):
        return ["requires_complete_manifests must be a list when provided"]
    for item in prerequisites:
        if not isinstance(item, str) or not item:
            errors.append("requires_complete_manifests must contain non-empty strings")
            continue
        inventory_path = control_dir(local_root, item) / "artifact_inventory.json"
        if not inventory_path.exists():
            errors.append(f"required manifest inventory is missing: {item}")
            continue
        try:
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"required manifest inventory is unreadable: {item}: {exc}")
            continue
        if inventory.get("status") != "complete":
            errors.append(f"required manifest inventory is not complete: {item}")
    return errors


def git_status_short(local_root: Path) -> str:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=local_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return f"<git status failed: {result.stderr.strip()}>"
    return result.stdout.strip()


def _copy_job_for_queue(job: dict[str, Any]) -> dict[str, Any]:
    queue_job: dict[str, Any] = {
        "id": job["id"],
        "argv": list(job["argv"]),
    }
    for key in ("cwd", "env", "enabled", "timeout_seconds", "local_precloud_gate"):
        if key in job:
            queue_job[key] = job[key]
    return queue_job


def build_queue_from_manifest(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    manifest_sha256: str,
) -> dict[str, Any]:
    artifacts = manifest["artifacts"]
    runner = manifest.get("runner", {})
    queue = {
        "queue_name": manifest["experiment_id"],
        "output_dir": artifacts["queue_dir"],
        "generated_from_manifest": True,
        "control_manifest_path": manifest_path.as_posix(),
        "control_manifest_sha256": manifest_sha256,
        "protocol": manifest["protocol"],
        "submission_policy": manifest.get("submission_policy", {}),
        "report_contract": manifest.get("report_contract", {}),
        "requires_complete_manifests": list(manifest.get("requires_complete_manifests", [])),
        "git": manifest.get("git", {}),
        "strict_claim": bool(manifest["protocol"].get("strict")),
        "artifacts": {
            "run_dirs": list(artifacts.get("run_dirs", [])),
            "report_dirs": list(artifacts.get("report_dirs", [])),
            "required_files": list(artifacts.get("required_files", [])),
            "required_paths": list(artifacts.get("required_paths", [])),
        },
        "jobs": [_copy_job_for_queue(job) for job in manifest["jobs"]],
    }
    wait_for_idle_pattern = runner.get("wait_for_idle_pattern") if isinstance(runner, dict) else None
    if isinstance(wait_for_idle_pattern, str) and wait_for_idle_pattern:
        queue["wait_for_idle_pattern"] = wait_for_idle_pattern
    return queue


def control_dir(local_root: Path, experiment_id: str) -> Path:
    return local_root / "artifacts" / "control" / experiment_id


def recovered_plan_path(local_root: Path, experiment_id: str) -> Path:
    return control_dir(local_root, experiment_id) / "submitted_plan_recovered.json"


def recovered_queue_path(local_root: Path, experiment_id: str) -> Path:
    return control_dir(local_root, experiment_id) / "submitted_generated_queue_recovered.json"


def freeze_decision_path(local_root: Path, experiment_id: str) -> Path:
    return control_dir(local_root, experiment_id) / "freeze_decision.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _text_tail(text: str | bytes | None, *, limit: int = TEXT_CAPTURE_LIMIT) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    if len(text) <= limit:
        return text
    return text[-limit:]


def _is_remote_producer_job(job: dict[str, Any]) -> bool:
    job_id = str(job.get("id", "")).lower()
    return job_id.startswith(("train_", "eval_", "report_", "audit_", "profile_"))


def _is_local_precloud_gate_job(job: dict[str, Any], *, producer_seen: bool = False) -> bool:
    if job.get("enabled") is False:
        return False
    explicit = job.get("local_precloud_gate")
    if explicit is True:
        return True
    if explicit is False:
        return False
    job_id = str(job.get("id", "")).lower()
    return (
        job_id.startswith("strict_")
        or job_id.startswith("compile_")
        or job_id.startswith("preflight")
        or (job_id.startswith("contract_check") and not producer_seen)
        or any(marker in job_id for marker in LOCAL_PRECLOUD_GATE_SUBSTRINGS)
    )


def _localize_gate_cwd(cwd: Any, *, local_root: Path) -> Path:
    if cwd is None:
        return local_root
    if not isinstance(cwd, str) or not cwd:
        raise ValueError("local pre-cloud gate job cwd must be a non-empty string when provided")
    if cwd == REMOTE_REPO_ROOT:
        return local_root
    remote_prefix = f"{REMOTE_REPO_ROOT}/"
    if cwd.startswith(remote_prefix):
        return local_root / cwd[len(remote_prefix) :]
    path = Path(cwd)
    if path.is_absolute():
        raise ValueError(f"local pre-cloud gate job cwd is outside the local repository mapping: {cwd}")
    return local_root / path


def _localize_gate_argv(argv: Any) -> list[str]:
    if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) for arg in argv):
        raise ValueError("local pre-cloud gate job argv must be a non-empty list of strings")
    localized = list(argv)
    first = Path(localized[0]).name
    if localized[0].startswith("/root/") or first in REMOTE_PYTHON_NAMES:
        localized[0] = sys.executable
    return localized


def run_local_precloud_gate(queue_path: Path, *, local_root: Path) -> tuple[bool, Path, list[str]]:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    experiment_id = str(queue.get("queue_name") or queue.get("experiment_id") or "unknown_experiment")
    gate_path = control_dir(local_root, experiment_id) / "local_precloud_gate.json"
    jobs: list[dict[str, Any]] = []
    producer_seen = False
    for raw_job in queue.get("jobs", []):
        if not isinstance(raw_job, dict):
            continue
        if _is_local_precloud_gate_job(raw_job, producer_seen=producer_seen):
            jobs.append(raw_job)
        if _is_remote_producer_job(raw_job):
            producer_seen = True
    errors: list[str] = []
    results: list[dict[str, Any]] = []

    if not jobs:
        errors.append("no local pre-cloud gate jobs were found; mark safe gate jobs with local_precloud_gate=true or use strict/static/compile/preflight/contract_check ids")
        write_json(
            gate_path,
            {
                "schema_version": 1,
                "status": "failed",
                "queue_path": _relative_path(queue_path, local_root),
                "jobs_run": [],
                "errors": errors,
            },
        )
        return False, gate_path, errors

    for job in jobs:
        job_id = str(job.get("id", "<missing-id>"))
        try:
            argv = _localize_gate_argv(job.get("argv"))
            cwd = _localize_gate_cwd(job.get("cwd"), local_root=local_root)
        except ValueError as exc:
            errors.append(f"{job_id}: {exc}")
            results.append({"id": job_id, "status": "failed_to_prepare", "error": str(exc)})
            continue

        env = os.environ.copy()
        job_env = job.get("env")
        if isinstance(job_env, dict):
            env.update({str(key): str(value) for key, value in job_env.items()})
        timeout = job.get("timeout_seconds")
        timeout_seconds = int(timeout) if isinstance(timeout, int) and timeout > 0 else None
        try:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            errors.append(f"{job_id}: timed out after {timeout_seconds} seconds")
            results.append(
                {
                    "id": job_id,
                    "status": "timeout",
                    "argv": argv,
                    "cwd": _relative_path(cwd, local_root),
                    "timeout_seconds": timeout_seconds,
                    "stdout_tail": _text_tail(exc.stdout or ""),
                    "stderr_tail": _text_tail(exc.stderr or ""),
                }
            )
            continue

        status = "passed" if completed.returncode == 0 else "failed"
        if completed.returncode != 0:
            errors.append(f"{job_id}: exited {completed.returncode}")
        results.append(
            {
                "id": job_id,
                "status": status,
                "returncode": completed.returncode,
                "argv": argv,
                "cwd": _relative_path(cwd, local_root),
                "timeout_seconds": timeout_seconds,
                "stdout_tail": _text_tail(completed.stdout),
                "stderr_tail": _text_tail(completed.stderr),
            }
        )

    write_json(
        gate_path,
        {
            "schema_version": 1,
            "status": "passed" if not errors else "failed",
            "queue_path": _relative_path(queue_path, local_root),
            "jobs_run": [result["id"] for result in results],
            "results": results,
            "errors": errors,
        },
    )
    return not errors, gate_path, errors


def plan_manifest(
    manifest_path: Path,
    *,
    local_root: Path,
    enforce_clean_git: bool = True,
) -> tuple[dict[str, Any], Path, Path]:
    manifest = load_manifest(manifest_path)
    errors, _warnings = validate_manifest(manifest, local_root=local_root, enforce_clean_git=enforce_clean_git)
    if errors:
        raise ValueError("; ".join(errors))
    manifest_sha = sha256_json(manifest)
    experiment_id = manifest["experiment_id"]
    out_dir = control_dir(local_root, experiment_id)
    queue_path = out_dir / "generated_queue.json"
    plan_path = out_dir / "plan.json"
    queue = build_queue_from_manifest(manifest, manifest_path=manifest_path, manifest_sha256=manifest_sha)
    plan = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "control_manifest_path": manifest_path.as_posix(),
        "control_manifest_sha256": manifest_sha,
        "generated_queue_path": queue_path.as_posix(),
        "queue_name": queue["queue_name"],
        "job_ids": [job["id"] for job in queue["jobs"]],
        "artifacts": manifest["artifacts"],
        "protocol": manifest["protocol"],
    }
    write_json(queue_path, queue)
    write_json(plan_path, plan)
    if _planner_enabled(manifest):
        slots, profiles, policy = build_planner_inputs(manifest)
        planner_result = plan_jobs(slots, profiles, policy)
        slot_plan_dir = out_dir / "slot_plan"
        write_planning_artifacts(slot_plan_dir, planner_result, slots, profiles)
        plan["slot_plan_dir"] = slot_plan_dir.as_posix()
        plan["slot_plan_status"] = "complete" if planner_result.ok else "incomplete"
        write_json(plan_path, plan)
    return plan, plan_path, queue_path


def submitted_queue_path_for_remote(manifest: dict[str, Any], *, local_root: Path) -> Path:
    experiment_id = manifest["experiment_id"]
    recovered_queue = recovered_queue_path(local_root, experiment_id)
    if recovered_queue.exists():
        return recovered_queue
    queue_path = control_dir(local_root, experiment_id) / "generated_queue.json"
    if queue_path.exists():
        return queue_path
    raise FileNotFoundError(
        f"generated queue is missing for {experiment_id}; run plan or submit before status/fetch"
    )


def existing_queue_path(manifest: dict[str, Any], *, local_root: Path) -> Path:
    try:
        return submitted_queue_path_for_remote(manifest, local_root=local_root)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"generated queue is missing for {manifest['experiment_id']}; run plan or submit before status/fetch"
        ) from exc


def artifact_contract_errors(artifacts: Any, *, source: str) -> list[str]:
    if not isinstance(artifacts, dict):
        return [f"{source} artifact contract is missing or not an object"]
    errors: list[str] = []
    queue_dir = artifacts.get("queue_dir")
    if not isinstance(queue_dir, str) or not queue_dir:
        errors.append(f"{source} artifact contract is missing queue_dir")
    elif not is_safe_relative_path(queue_dir):
        errors.append(f"{source} artifact contract queue_dir is unsafe: {queue_dir}")
    for key in ("report_dirs", "run_dirs"):
        value = artifacts.get(key, [])
        if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
            errors.append(f"{source} artifact contract {key} must be a list of non-empty strings")
            continue
        for item in value:
            if not is_safe_relative_path(item):
                errors.append(f"{source} artifact contract {key} contains unsafe path: {item}")
    required_files = artifacts.get("required_files", [])
    if required_files and not isinstance(required_files, list):
        errors.append(f"{source} artifact contract required_files must be a list when provided")
    elif isinstance(required_files, list):
        for item in required_files:
            if not isinstance(item, str) or not item:
                errors.append(f"{source} artifact contract required_files must contain non-empty strings")
            elif item not in KNOWN_ARTIFACT_FILE_NAMES:
                errors.append(f"{source} artifact contract required_files contains unknown file name: {item}")
    required_paths = artifacts.get("required_paths", [])
    if required_paths and not isinstance(required_paths, list):
        errors.append(f"{source} artifact contract required_paths must be a list when provided")
    elif isinstance(required_paths, list):
        for item in required_paths:
            if not isinstance(item, str) or not item:
                errors.append(f"{source} artifact contract required_paths must contain non-empty strings")
            elif not is_safe_relative_path(item):
                errors.append(f"{source} artifact contract required_paths contains unsafe path: {item}")
    return errors


def artifact_contract_from_generated_queue(generated_queue: dict[str, Any]) -> dict[str, Any]:
    artifacts = generated_queue.get("artifacts")
    if not isinstance(artifacts, dict):
        return {}
    contract = dict(artifacts)
    if "queue_dir" not in contract and isinstance(generated_queue.get("output_dir"), str):
        contract["queue_dir"] = generated_queue["output_dir"]
    return contract


def submitted_artifact_contract(manifest: dict[str, Any], *, local_root: Path) -> tuple[dict[str, Any], str, list[str]]:
    current_artifacts = manifest["artifacts"]
    experiment_id = manifest["experiment_id"]
    recovered_queue = recovered_queue_path(local_root, experiment_id)
    recovered_plan = recovered_plan_path(local_root, experiment_id)
    queue_path = control_dir(local_root, experiment_id) / "generated_queue.json"
    plan_path = control_dir(local_root, experiment_id) / "plan.json"

    if recovered_queue.exists():
        generated_queue = _read_json_object(recovered_queue)
        if generated_queue is None:
            return current_artifacts, "submitted_generated_queue_recovered", [
                f"recovered generated queue artifact contract is unreadable: {recovered_queue.relative_to(local_root).as_posix()}"
            ]
        contract = artifact_contract_from_generated_queue(generated_queue)
        errors = artifact_contract_errors(contract, source="recovered generated queue")
        errors.extend(
            _strict_recovered_queue_contract_errors(
                manifest=manifest,
                remote_queue=generated_queue,
                artifacts=contract,
                source="recovered generated queue",
            )
        )
        return contract, "submitted_generated_queue_recovered", errors

    if recovered_plan.exists():
        plan = _read_json_object(recovered_plan)
        if plan is None:
            return current_artifacts, "submitted_plan_recovered", [
                f"recovered plan artifact contract is unreadable: {recovered_plan.relative_to(local_root).as_posix()}"
            ]
        artifacts = plan.get("artifacts")
        errors = artifact_contract_errors(artifacts, source="recovered plan")
        contract = artifacts if isinstance(artifacts, dict) else current_artifacts
        errors.extend(
            _strict_recovered_queue_contract_errors(
                manifest=manifest,
                remote_queue=plan,
                artifacts=contract,
                source="recovered plan",
            )
        )
        return contract, "submitted_plan_recovered", errors

    if queue_path.exists():
        generated_queue = _read_json_object(queue_path)
        if generated_queue is None:
            return current_artifacts, "generated_queue", [
                f"generated queue artifact contract is unreadable: {queue_path.relative_to(local_root).as_posix()}"
            ]
        contract = artifact_contract_from_generated_queue(generated_queue)
        errors = artifact_contract_errors(contract, source="generated queue")
        return contract, "generated_queue", errors

    if plan_path.exists():
        plan = _read_json_object(plan_path)
        if plan is None:
            return current_artifacts, "plan", [
                f"plan artifact contract is unreadable: {plan_path.relative_to(local_root).as_posix()}"
            ]
        artifacts = plan.get("artifacts")
        errors = artifact_contract_errors(artifacts, source="plan")
        return artifacts if isinstance(artifacts, dict) else current_artifacts, "plan", errors

    return current_artifacts, "current_manifest", []


def required_artifact_paths_from_artifacts(artifacts: dict[str, Any]) -> list[str]:
    exact_paths = artifacts.get("required_paths", [])
    if isinstance(exact_paths, list) and exact_paths:
        return sorted(dict.fromkeys(normalize_relative_path(path) for path in exact_paths if isinstance(path, str) and path))

    raw_required = artifacts.get("required_files", [])
    required = [item for item in raw_required if isinstance(item, str) and item]
    queue_dir = normalize_relative_path(artifacts["queue_dir"])
    report_dirs = [normalize_relative_path(item) for item in artifacts.get("report_dirs", [])]
    run_dirs = [normalize_relative_path(item) for item in artifacts.get("run_dirs", [])]
    paths: list[str] = []
    for name in required:
        if name in QUEUE_FILE_NAMES:
            paths.append(f"{queue_dir}/{name}")
        if name in RUN_FILE_NAMES:
            for run_dir in run_dirs:
                paths.append(f"{run_dir}/{name}")
        if name in REPORT_FILE_NAMES:
            for report_dir in report_dirs:
                paths.append(f"{report_dir}/{name}")
    if not required:
        paths.extend([f"{queue_dir}/manifest.json", f"{queue_dir}/status.jsonl", f"{queue_dir}/queue_summary.json"])
    return sorted(dict.fromkeys(paths))


def required_artifact_paths(manifest: dict[str, Any]) -> list[str]:
    return required_artifact_paths_from_artifacts(manifest["artifacts"])


def resolve_required_artifact_path(local_root: Path, relative: str) -> list[str]:
    normalized = normalize_relative_path(relative)
    exact = local_root / normalized
    if exact.exists():
        return [normalized]

    parent_text, _, file_name = normalized.rpartition("/")
    if file_name not in RUN_FILE_NAMES or not parent_text.startswith("runs/"):
        return []

    parent = local_root / parent_text
    if not parent.exists() or not parent.is_dir():
        return []

    matches = sorted(path for path in parent.rglob(file_name) if path.is_file())
    resolved: list[str] = []
    for path in matches:
        try:
            resolved.append(path.relative_to(local_root).as_posix())
        except ValueError:
            continue
    return resolved


def build_artifact_inventory(manifest: dict[str, Any], *, local_root: Path) -> dict[str, Any]:
    artifacts, artifact_contract_source, artifact_contract_validation_errors = submitted_artifact_contract(
        manifest,
        local_root=local_root,
    )
    required_paths = required_artifact_paths_from_artifacts(artifacts) if not artifact_contract_validation_errors else []
    present: list[str] = []
    missing: list[str] = []
    resolved_aliases: dict[str, list[str]] = {}
    for relative in required_paths:
        resolved = resolve_required_artifact_path(local_root, relative)
        if resolved:
            present.extend(resolved)
            if resolved != [relative]:
                resolved_aliases[relative] = resolved
        else:
            missing.append(relative)
    provenance = validate_artifact_provenance(manifest, local_root=local_root, artifacts=artifacts)
    provenance_errors = list(provenance.get("errors", []))
    stale_fetch_errors = validate_fetch_missing_sources(manifest, local_root=local_root, artifacts=artifacts)
    validation_errors = artifact_contract_validation_errors + provenance_errors + stale_fetch_errors
    return {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "artifact_contract": artifacts,
        "artifact_contract_source": artifact_contract_source,
        "required_files": required_paths,
        "present_files": sorted(dict.fromkeys(present)),
        "missing_files": missing,
        "resolved_aliases": resolved_aliases,
        "provenance": provenance,
        "provenance_warnings": list(provenance.get("warnings", [])),
        "validation_errors": validation_errors,
        "status": "complete" if not missing and not validation_errors else "incomplete",
    }


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def empty_artifact_provenance(*, current_manifest_sha256: str | None = None) -> dict[str, Any]:
    return {
        "checked": False,
        "current_manifest_sha256": current_manifest_sha256,
        "current_manifest_matches_submitted": None,
        "submitted_plan_sha256": None,
        "generated_queue_sha256": None,
        "fetched_queue_sha256": None,
        "errors": [],
        "warnings": [],
    }


def _relative_path(path: Path, local_root: Path) -> str:
    try:
        return path.relative_to(local_root).as_posix()
    except ValueError:
        return path.as_posix()


def _first_existing_path(candidates: Sequence[tuple[str, Path]]) -> tuple[str, Path]:
    for label, path in candidates:
        if path.exists():
            return label, path
    return candidates[-1]


def validate_artifact_provenance(
    manifest: dict[str, Any],
    *,
    local_root: Path,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current_manifest_sha = sha256_json(manifest)
    protocol = manifest.get("protocol")
    policy = manifest.get("submission_policy")
    if not (
        isinstance(protocol, dict)
        and protocol.get("strict") is True
        and isinstance(policy, dict)
        and policy.get("allow_submit") is True
    ):
        return empty_artifact_provenance(current_manifest_sha256=current_manifest_sha)

    artifacts = manifest.get("artifacts", {}) if artifacts is None else artifacts
    if not isinstance(artifacts, dict):
        return empty_artifact_provenance(current_manifest_sha256=current_manifest_sha)
    queue_dir = artifacts.get("queue_dir")
    if not isinstance(queue_dir, str) or not queue_dir:
        return empty_artifact_provenance(current_manifest_sha256=current_manifest_sha)
    queue_manifest_path = local_root / normalize_relative_path(queue_dir) / "manifest.json"
    if not queue_manifest_path.exists():
        return empty_artifact_provenance(current_manifest_sha256=current_manifest_sha)

    errors: list[str] = []
    warnings: list[str] = []
    experiment_id = manifest["experiment_id"]
    plan_label, plan_path = _first_existing_path(
        [
            ("recovered", recovered_plan_path(local_root, experiment_id)),
            ("local", control_dir(local_root, experiment_id) / "plan.json"),
        ]
    )
    generated_queue_label, generated_queue_path = _first_existing_path(
        [
            ("recovered", recovered_queue_path(local_root, experiment_id)),
            ("local", control_dir(local_root, experiment_id) / "generated_queue.json"),
        ]
    )
    plan = _read_json_object(plan_path)
    generated_queue = _read_json_object(generated_queue_path)
    remote_manifest = _read_json_object(queue_manifest_path)
    remote_queue = remote_manifest.get("queue_spec") if isinstance(remote_manifest, dict) else None
    provenance = {
        "checked": True,
        "current_manifest_sha256": current_manifest_sha,
        "current_manifest_matches_submitted": None,
        "submitted_plan_sha256": None,
        "generated_queue_sha256": None,
        "fetched_queue_sha256": None,
        "submitted_plan_source": plan_label,
        "generated_queue_source": generated_queue_label,
        "errors": errors,
        "warnings": warnings,
    }
    if not isinstance(remote_queue, dict):
        errors.append(f"queue manifest is missing queue_spec: {queue_manifest_path.relative_to(local_root).as_posix()}")
        return provenance

    remote_sha = remote_queue.get("control_manifest_sha256")
    provenance["fetched_queue_sha256"] = remote_sha

    if plan is None:
        errors.append(f"{plan_label} control plan is missing or unreadable: {_relative_path(plan_path, local_root)}")
    else:
        plan_sha = plan.get("control_manifest_sha256")
        provenance["submitted_plan_sha256"] = plan_sha
        provenance["current_manifest_matches_submitted"] = plan_sha == current_manifest_sha
        if plan_sha != remote_sha:
            errors.append(
                f"{plan_label} control plan sha does not match fetched queue manifest "
                f"for {experiment_id}: plan={plan_sha} fetched={remote_sha}"
            )
        if plan_sha != current_manifest_sha:
            warnings.append(
                "current control manifest sha does not match submitted plan "
                f"for {experiment_id}: plan={plan_sha} current={current_manifest_sha}"
            )

    if generated_queue is None:
        errors.append(f"{generated_queue_label} generated queue is missing or unreadable: {_relative_path(generated_queue_path, local_root)}")
    else:
        generated_sha = generated_queue.get("control_manifest_sha256")
        provenance["generated_queue_sha256"] = generated_sha
        if generated_sha != remote_sha:
            errors.append(
                f"{generated_queue_label} generated queue sha does not match fetched queue manifest "
                f"for {experiment_id}: generated={generated_sha} fetched={remote_sha}"
            )
        if generated_queue.get("generated_from_manifest") is not True:
            errors.append(f"{generated_queue_label} generated queue is missing generated_from_manifest=true for {experiment_id}")
        if generated_sha != current_manifest_sha:
            warnings.append(
                "current control manifest sha does not match submitted generated queue "
                f"for {experiment_id}: generated={generated_sha} current={current_manifest_sha}"
            )
    return provenance


def validate_fetch_missing_sources(
    manifest: dict[str, Any],
    *,
    local_root: Path,
    artifacts: dict[str, Any] | None = None,
) -> list[str]:
    artifacts = manifest.get("artifacts", {}) if artifacts is None else artifacts
    if not isinstance(artifacts, dict):
        return []
    queue_dir = artifacts.get("queue_dir")
    if not isinstance(queue_dir, str) or not queue_dir:
        return []
    missing_path = local_root / normalize_relative_path(queue_dir) / "fetch_missing_sources.json"
    if not missing_path.exists():
        return []
    try:
        text = missing_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"fetch_missing_sources.json is unreadable: {missing_path.relative_to(local_root).as_posix()}: {exc}"]
    if not text.strip():
        return []
    payload = _read_json_object(missing_path)
    if payload is None:
        return [f"fetch_missing_sources.json is non-empty but not valid JSON: {missing_path.relative_to(local_root).as_posix()}"]
    missing = payload.get("missing_sources") if isinstance(payload, dict) else None
    if isinstance(missing, list) and missing:
        return [f"fetch_missing_sources.json still lists missing sources: {missing_path.relative_to(local_root).as_posix()}"]
    return []


def load_freeze_decision(manifest: dict[str, Any], *, local_root: Path) -> dict[str, Any] | None:
    path = freeze_decision_path(local_root, manifest["experiment_id"])
    if not path.exists():
        return None
    return _read_json_object(path) or {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "decision": "freeze",
        "reason": f"freeze decision is unreadable: {_relative_path(path, local_root)}",
    }


def freeze_validation_errors(freeze_decision: dict[str, Any] | None) -> list[str]:
    if freeze_decision is None:
        return []
    reason = freeze_decision.get("reason")
    if not isinstance(reason, str) or not reason:
        reason = "no reason recorded"
    return [f"run is frozen by recovery decision: {reason}"]


def write_inventory(manifest: dict[str, Any], *, local_root: Path) -> dict[str, Any]:
    inventory = build_artifact_inventory(manifest, local_root=local_root)
    freeze_decision = load_freeze_decision(manifest, local_root=local_root)
    freeze_errors = freeze_validation_errors(freeze_decision)
    if freeze_errors:
        inventory["freeze_decision"] = freeze_decision
        inventory["validation_errors"] = list(inventory["validation_errors"]) + freeze_errors
        inventory["status"] = "incomplete"
    out_dir = control_dir(local_root, manifest["experiment_id"])
    write_json(out_dir / "artifact_inventory.json", inventory)
    invalid = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "status": inventory["status"],
        "missing_files": inventory["missing_files"],
        "validation_errors": inventory["validation_errors"],
        "provenance_warnings": inventory["provenance_warnings"],
    }
    write_json(out_dir / "invalid_runs.json", invalid)
    return inventory


def write_formal_evidence_stub(manifest: dict[str, Any], inventory: dict[str, Any], *, local_root: Path) -> Path:
    artifact_contract = inventory.get("artifact_contract")
    if not isinstance(artifact_contract, dict):
        artifact_contract = manifest.get("artifacts", {})
    report_dirs = artifact_contract.get("report_dirs") if isinstance(artifact_contract, dict) else None
    if not isinstance(report_dirs, list) or not report_dirs:
        report_dirs = [f"artifacts/reports/{manifest['experiment_id']}"]
    report_dir = local_root / str(report_dirs[0]).rstrip("/")
    report_path = report_dir / "formal_evidence.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {manifest['experiment_id']} Formal Evidence",
        "",
        f"strict_gate: {inventory['status']}",
        "",
        "This is an automatically generated control-plane stub.",
        "Final model capability analysis must wait for complete fetched artifacts and paired statistics.",
        "",
        "## Missing Files",
        "",
    ]
    if inventory["missing_files"]:
        lines.extend(f"- `{path}`" for path in inventory["missing_files"])
    else:
        lines.append("- none")
    lines.extend(["", "## Validation Errors", ""])
    if inventory["validation_errors"]:
        lines.extend(f"- {error}" for error in inventory["validation_errors"])
    else:
        lines.append("- none")
    lines.extend(["", "## Provenance Warnings", ""])
    if inventory["provenance_warnings"]:
        lines.extend(f"- {warning}" for warning in inventory["provenance_warnings"])
    else:
        lines.append("- none")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _queue_manifest_path_from_queue_dir(queue_dir: Any, *, local_root: Path) -> Path | None:
    if not isinstance(queue_dir, str) or not queue_dir:
        return None
    if not is_safe_relative_path(queue_dir):
        return None
    return local_root / normalize_relative_path(queue_dir) / "manifest.json"


def _stable_submitted_queue_dir(experiment_id: str) -> str:
    return f"artifacts/queues/{experiment_id}"


def _append_queue_manifest_candidate(
    candidates: list[Path],
    seen: set[str],
    queue_dir: Any,
    *,
    local_root: Path,
) -> None:
    path = _queue_manifest_path_from_queue_dir(queue_dir, local_root=local_root)
    if path is None:
        return
    key = path.as_posix()
    if key in seen:
        return
    seen.add(key)
    candidates.append(path)


def _append_recovered_queue_candidates(
    candidates: list[Path],
    seen: set[str],
    artifacts: dict[str, Any],
    *,
    local_root: Path,
) -> None:
    _append_queue_manifest_candidate(candidates, seen, artifacts.get("queue_dir"), local_root=local_root)


def _append_local_submitted_queue_candidate(
    candidates: list[Path],
    seen: set[str],
    queue_dir: Any,
    *,
    experiment_id: str,
    local_root: Path,
) -> None:
    if not isinstance(queue_dir, str):
        return
    normalized = normalize_relative_path(queue_dir)
    if normalized != _stable_submitted_queue_dir(experiment_id):
        return
    _append_queue_manifest_candidate(candidates, seen, normalized, local_root=local_root)


def _queue_manifest_candidates(
    manifest: dict[str, Any],
    *,
    local_root: Path,
    local_plan: dict[str, Any] | None,
    local_queue: dict[str, Any] | None,
    recovered_plan: dict[str, Any] | None = None,
    recovered_queue: dict[str, Any] | None = None,
) -> list[Path]:
    experiment_id = manifest["experiment_id"]
    candidates: list[Path] = []
    seen: set[str] = set()

    if isinstance(recovered_queue, dict):
        recovered_queue_artifacts = artifact_contract_from_generated_queue(recovered_queue)
        _append_queue_manifest_candidate(candidates, seen, recovered_queue.get("output_dir"), local_root=local_root)
        _append_recovered_queue_candidates(candidates, seen, recovered_queue_artifacts, local_root=local_root)

    if isinstance(recovered_plan, dict):
        recovered_plan_artifacts = recovered_plan.get("artifacts")
        if isinstance(recovered_plan_artifacts, dict):
            _append_recovered_queue_candidates(candidates, seen, recovered_plan_artifacts, local_root=local_root)

    if isinstance(local_queue, dict):
        local_queue_artifacts = artifact_contract_from_generated_queue(local_queue)
        _append_local_submitted_queue_candidate(
            candidates,
            seen,
            local_queue.get("output_dir"),
            experiment_id=experiment_id,
            local_root=local_root,
        )
        _append_local_submitted_queue_candidate(
            candidates,
            seen,
            local_queue_artifacts.get("queue_dir"),
            experiment_id=experiment_id,
            local_root=local_root,
        )

    if isinstance(local_plan, dict):
        plan_artifacts = local_plan.get("artifacts")
        if isinstance(plan_artifacts, dict):
            _append_local_submitted_queue_candidate(
                candidates,
                seen,
                plan_artifacts.get("queue_dir"),
                experiment_id=experiment_id,
                local_root=local_root,
            )

    _append_queue_manifest_candidate(candidates, seen, _stable_submitted_queue_dir(experiment_id), local_root=local_root)
    return candidates


def _select_submitted_queue_manifest(candidates: Sequence[Path], *, local_root: Path) -> tuple[Path | None, dict[str, Any] | None, list[str]]:
    diagnostics: list[str] = []
    for path in candidates:
        relative = _relative_path(path, local_root)
        if not path.exists():
            diagnostics.append(f"candidate queue manifest is missing: {relative}")
            continue
        queue_manifest = _read_json_object(path)
        remote_queue = _extract_remote_queue(queue_manifest)
        if remote_queue is None:
            diagnostics.append(f"candidate queue manifest is missing queue_spec: {relative}")
            continue
        if not _control_manifest_sha(remote_queue):
            diagnostics.append(f"candidate queue_spec is missing control_manifest_sha256: {relative}")
            continue
        if remote_queue.get("generated_from_manifest") is not True:
            diagnostics.append(f"candidate queue_spec is missing generated_from_manifest=true: {relative}")
            continue
        return path, remote_queue, diagnostics
    return None, None, diagnostics


def _extract_remote_queue(queue_manifest: dict[str, Any] | None) -> dict[str, Any] | None:
    if queue_manifest is None:
        return None
    queue_spec = queue_manifest.get("queue_spec")
    return queue_spec if isinstance(queue_spec, dict) else None


def _control_manifest_sha(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    value = payload.get("control_manifest_sha256")
    return value if isinstance(value, str) else None


def _strict_recovered_queue_contract_errors(
    *,
    manifest: dict[str, Any],
    remote_queue: dict[str, Any],
    artifacts: dict[str, Any],
    source: str = "fetched queue_spec",
) -> list[str]:
    recovered_manifest = {
        "protocol": remote_queue.get("protocol") if isinstance(remote_queue.get("protocol"), dict) else manifest.get("protocol"),
        "submission_policy": (
            remote_queue.get("submission_policy")
            if isinstance(remote_queue.get("submission_policy"), dict)
            else manifest.get("submission_policy")
        ),
        "artifacts": artifacts,
        "requires_complete_manifests": (
            remote_queue.get("requires_complete_manifests")
            if isinstance(remote_queue.get("requires_complete_manifests"), list)
            else manifest.get("requires_complete_manifests", [])
        ),
        "report_contract": remote_queue.get("report_contract") if isinstance(remote_queue.get("report_contract"), dict) else {},
        "jobs": remote_queue.get("jobs") if isinstance(remote_queue.get("jobs"), list) else [],
    }
    errors = validate_strict_contract(recovered_manifest)
    errors.extend(validate_artifact_ownership_contract(recovered_manifest))
    return [f"{source} {error}" for error in errors]


def _build_recovered_plan(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    local_root: Path,
    remote_queue: dict[str, Any],
    submitted_sha: str,
) -> dict[str, Any]:
    experiment_id = manifest["experiment_id"]
    artifacts = artifact_contract_from_generated_queue(remote_queue)
    jobs = remote_queue.get("jobs")
    job_ids = [job["id"] for job in jobs if isinstance(job, dict) and isinstance(job.get("id"), str)] if isinstance(jobs, list) else []
    control_manifest_path = remote_queue.get("control_manifest_path")
    if not isinstance(control_manifest_path, str) or not control_manifest_path:
        control_manifest_path = manifest_path.as_posix()
    queue_name = remote_queue.get("queue_name")
    if not isinstance(queue_name, str) or not queue_name:
        queue_name = experiment_id
    protocol = remote_queue.get("protocol")
    if not isinstance(protocol, dict):
        protocol = manifest.get("protocol", {})
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "control_manifest_path": control_manifest_path,
        "control_manifest_sha256": submitted_sha,
        "generated_queue_path": recovered_queue_path(local_root, experiment_id).as_posix(),
        "generated_queue_source": "submitted_generated_queue_recovered",
        "queue_name": queue_name,
        "job_ids": job_ids,
        "artifacts": artifacts,
        "protocol": protocol,
        "recovered_from": "fetched_queue_manifest.queue_spec",
    }


def build_recovery_diagnostic(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    local_root: Path,
) -> dict[str, Any]:
    experiment_id = manifest["experiment_id"]
    current_manifest_sha = sha256_json(manifest)
    control_path = control_dir(local_root, experiment_id)
    local_plan_path = control_path / "plan.json"
    local_queue_path = control_path / "generated_queue.json"
    local_plan = _read_json_object(local_plan_path)
    local_queue = _read_json_object(local_queue_path)
    recovered_plan = _read_json_object(recovered_plan_path(local_root, experiment_id))
    recovered_queue = _read_json_object(recovered_queue_path(local_root, experiment_id))
    queue_manifest_candidates = _queue_manifest_candidates(
        manifest,
        local_root=local_root,
        local_plan=local_plan,
        local_queue=local_queue,
        recovered_plan=recovered_plan,
        recovered_queue=recovered_queue,
    )
    queue_manifest_path, remote_queue, candidate_diagnostics = _select_submitted_queue_manifest(
        queue_manifest_candidates,
        local_root=local_root,
    )
    submitted_sha = _control_manifest_sha(remote_queue)

    reasons: list[str] = []
    recovered_queue_payload: dict[str, Any] | None = None
    recovered_plan_payload: dict[str, Any] | None = None
    if remote_queue is None:
        reasons.extend(candidate_diagnostics or ["no valid fetched queue manifest candidate found"])
    elif not isinstance(submitted_sha, str) or not submitted_sha:
        reasons.append("fetched queue_spec is missing control_manifest_sha256")
    elif remote_queue.get("generated_from_manifest") is not True:
        reasons.append("fetched queue_spec is missing generated_from_manifest=true")
    else:
        contract = artifact_contract_from_generated_queue(remote_queue)
        contract_errors = artifact_contract_errors(contract, source="fetched queue_spec")
        contract_errors.extend(
            _strict_recovered_queue_contract_errors(
                manifest=manifest,
                remote_queue=remote_queue,
                artifacts=contract,
            )
        )
        if contract_errors:
            reasons.extend(contract_errors)
        else:
            recovered_queue_payload = dict(remote_queue)
            recovered_plan_payload = _build_recovered_plan(
                manifest=manifest,
                manifest_path=manifest_path,
                local_root=local_root,
                remote_queue=remote_queue,
                submitted_sha=submitted_sha,
            )

    local_plan_sha = _control_manifest_sha(local_plan)
    local_queue_sha = _control_manifest_sha(local_queue)
    can_recover = recovered_queue_payload is not None and recovered_plan_payload is not None and isinstance(submitted_sha, str)
    will_write = [f"artifacts/control/{experiment_id}/recovery_audit.json"]
    if can_recover:
        will_write.extend(
            [
                f"artifacts/control/{experiment_id}/submitted_plan_recovered.json",
                f"artifacts/control/{experiment_id}/submitted_generated_queue_recovered.json",
            ]
        )
    else:
        will_write.append(f"artifacts/control/{experiment_id}/freeze_decision.json")

    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "dry_run": True,
        "status": "recoverable" if can_recover else "freeze_required",
        "current_manifest_sha256": current_manifest_sha,
        "submitted_control_manifest_sha256": submitted_sha,
        "local_plan_control_manifest_sha256": local_plan_sha,
        "local_generated_queue_control_manifest_sha256": local_queue_sha,
        "local_plan_drifted_from_submitted": bool(submitted_sha and local_plan_sha != submitted_sha),
        "local_generated_queue_drifted_from_submitted": bool(submitted_sha and local_queue_sha != submitted_sha),
        "current_manifest_drifted_from_submitted": bool(submitted_sha and current_manifest_sha != submitted_sha),
        "can_recover": can_recover,
        "cannot_recover_reasons": reasons,
        "queue_manifest_candidates": [_relative_path(path, local_root) for path in queue_manifest_candidates],
        "queue_manifest_candidate_diagnostics": candidate_diagnostics,
        "queue_manifest_path": _relative_path(queue_manifest_path, local_root) if queue_manifest_path else None,
        "local_plan_path": _relative_path(local_plan_path, local_root),
        "local_generated_queue_path": _relative_path(local_queue_path, local_root),
        "will_write": will_write,
        "_recovered_plan": recovered_plan_payload,
        "_recovered_queue": recovered_queue_payload,
    }


def write_recovery_files(diagnostic: dict[str, Any], *, local_root: Path) -> dict[str, Any]:
    experiment_id = str(diagnostic["experiment_id"])
    audit = {key: value for key, value in diagnostic.items() if not key.startswith("_")}
    audit["dry_run"] = False
    written_files: list[str] = []
    control_path = control_dir(local_root, experiment_id)

    recovered_plan = diagnostic.get("_recovered_plan")
    recovered_queue = diagnostic.get("_recovered_queue")
    if diagnostic.get("can_recover") is True and isinstance(recovered_plan, dict) and isinstance(recovered_queue, dict):
        plan_path = recovered_plan_path(local_root, experiment_id)
        queue_path = recovered_queue_path(local_root, experiment_id)
        write_json(plan_path, recovered_plan)
        write_json(queue_path, recovered_queue)
        written_files.extend([_relative_path(plan_path, local_root), _relative_path(queue_path, local_root)])
    else:
        reasons = diagnostic.get("cannot_recover_reasons")
        reason_text = "; ".join(str(item) for item in reasons) if isinstance(reasons, list) and reasons else "recovery was not possible"
        freeze = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "decision": "freeze",
            "reason": reason_text,
            "submitted_control_manifest_sha256": diagnostic.get("submitted_control_manifest_sha256"),
            "local_plan_control_manifest_sha256": diagnostic.get("local_plan_control_manifest_sha256"),
            "local_generated_queue_control_manifest_sha256": diagnostic.get("local_generated_queue_control_manifest_sha256"),
        }
        path = freeze_decision_path(local_root, experiment_id)
        write_json(path, freeze)
        written_files.append(_relative_path(path, local_root))

    audit["written_files"] = written_files
    audit_path = control_path / "recovery_audit.json"
    write_json(audit_path, audit)
    audit["written_files"] = [_relative_path(audit_path, local_root), *written_files]
    write_json(audit_path, audit)
    return audit


def run_remote_api(command: str, manifest_path: Path, *, local_root: Path, extra_args: Sequence[str]) -> int:
    manifest = load_manifest(manifest_path)
    if command in {"status", "fetch"}:
        try:
            queue_path = existing_queue_path(manifest, local_root=local_root)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        plan = {"experiment_id": manifest["experiment_id"]}
    else:
        plan, _plan_path, queue_path = plan_manifest(
            manifest_path,
            local_root=local_root,
            enforce_clean_git=command == "submit",
        )
    try:
        queue_arg = queue_path.resolve().relative_to(local_root.resolve()).as_posix()
    except ValueError:
        queue_arg = queue_path.as_posix()
    if command == "submit":
        policy = manifest.get("submission_policy", {})
        if isinstance(policy, dict) and policy.get("allow_submit") is not True:
            print(f"Refusing submit for {plan['experiment_id']}: allow_submit is not true", file=sys.stderr)
            return 2
        gate_ok, gate_path, gate_errors = run_local_precloud_gate(queue_path, local_root=local_root)
        if not gate_ok:
            print(
                f"Refusing submit for {plan['experiment_id']}: local pre-cloud gate failed; "
                f"see {_relative_path(gate_path, local_root)}",
                file=sys.stderr,
            )
            for error in gate_errors:
                print(f"- {error}", file=sys.stderr)
            return 2
    remote_api_script = REPO_ROOT / "scripts" / "remote_experiment_api.py"
    argv = [sys.executable, remote_api_script.as_posix(), command, queue_arg, *extra_args]
    return subprocess.call(argv, cwd=local_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["plan", "validate", "submit", "status", "fetch", "report", "inventory", "recover"],
        help="Control-plane action",
    )
    parser.add_argument("manifest", type=Path, help="Control manifest path")
    parser.add_argument("--local-root", type=Path, default=REPO_ROOT, help="Local repository root")
    parser.add_argument(
        "--write-recovery",
        action="store_true",
        help="Write local recovery audit/recovered submitted artifacts or a freeze decision",
    )
    return parser


def print_lines(lines: Iterable[str]) -> None:
    for line in lines:
        print(line)


def main(argv: list[str] | None = None) -> int:
    args, remote_args = build_parser().parse_known_args(argv)
    if remote_args[:1] == ["--"]:
        remote_args = remote_args[1:]
    local_root = args.local_root
    try:
        manifest = load_manifest(args.manifest)
        enforce_clean_git = args.command in {"validate", "plan", "submit"}
        enforce_prerequisites = args.command in {"validate", "plan", "submit"}
        enforce_submit_contract = args.command in {"validate", "plan", "submit"}
        errors, warnings = validate_manifest(
            manifest,
            local_root=local_root,
            enforce_clean_git=enforce_clean_git,
            enforce_prerequisites=enforce_prerequisites,
            enforce_submit_contract=enforce_submit_contract,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.command == "validate":
        if errors:
            print_lines(errors)
            print("; ".join(errors), file=sys.stderr)
            return 2
        print_lines(warnings)
        print(f"manifest valid: {manifest['experiment_id']}")
        return 0

    if errors:
        print("; ".join(errors), file=sys.stderr)
        return 2

    if args.command == "plan":
        plan, plan_path, queue_path = plan_manifest(args.manifest, local_root=local_root, enforce_clean_git=True)
        print(json.dumps({"plan": plan_path.as_posix(), "queue": queue_path.as_posix(), "experiment_id": plan["experiment_id"]}, sort_keys=True))
        return 0

    if args.command == "inventory":
        inventory = write_inventory(manifest, local_root=local_root)
        print(json.dumps({"status": inventory["status"], "missing_count": len(inventory["missing_files"])}, sort_keys=True))
        return 0

    if args.command == "report":
        inventory = write_inventory(manifest, local_root=local_root)
        report_path = write_formal_evidence_stub(manifest, inventory, local_root=local_root)
        print(json.dumps({"report": report_path.as_posix(), "strict_gate": inventory["status"]}, sort_keys=True))
        return 0

    if args.command == "recover":
        diagnostic = build_recovery_diagnostic(manifest, manifest_path=args.manifest, local_root=local_root)
        if args.write_recovery:
            output = write_recovery_files(diagnostic, local_root=local_root)
        else:
            output = {key: value for key, value in diagnostic.items() if not key.startswith("_")}
        print(json.dumps(output, sort_keys=True))
        return 0

    if args.command in {"submit", "status", "fetch"}:
        return run_remote_api(args.command, args.manifest, local_root=local_root, extra_args=remote_args)

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
