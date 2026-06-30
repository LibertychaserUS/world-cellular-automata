#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _fmt(value: Any) -> str:
    number = _float(value)
    if number is not None:
        return f"{number:.7g}"
    return "" if value is None else str(value)


def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _run_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("model", ""), row.get("source_run_dir", ""), row.get("checkpoint_kind", ""))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def summarize_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[_run_key(row)].append(row)
    summaries: list[dict[str, Any]] = []
    for (model, run_dir, checkpoint_kind), group in sorted(grouped.items()):
        rel = [_float(row.get("eval_field_relative_l2")) for row in group]
        mse = [_float(row.get("eval_mse")) for row in group]
        improvement = [_float(row.get("eval_field_mse_improvement_vs_persistence")) for row in group]
        rel_values = [item for item in rel if item is not None]
        mse_values = [item for item in mse if item is not None]
        improvement_values = [item for item in improvement if item is not None]
        h8 = next((row for row in group if row.get("horizon") == "8"), {})
        summaries.append(
            {
                "model": model,
                "run_dir": run_dir,
                "checkpoint_kind": checkpoint_kind,
                "mean_relative_l2": _mean(rel_values),
                "mean_mse": _mean(mse_values),
                "mean_improvement": _mean(improvement_values),
                "h8_relative_l2": _float(h8.get("eval_field_relative_l2")),
                "h8_mse": _float(h8.get("eval_mse")),
                "h8_improvement": _float(h8.get("eval_field_mse_improvement_vs_persistence")),
                "horizon_count": len(group),
            }
        )
    return summaries


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"Strict V20 report validation failed: {message}")


def _start_indices_hash_from_values(starts: list[int]) -> str:
    if not starts:
        return ""
    return hashlib.sha256(struct.pack(f"{len(starts)}q", *starts)).hexdigest()[:16]


def validate_strict_inputs(
    *,
    rows: list[dict[str, str]],
    per_sample_rows: list[dict[str, str]],
    eval_summary: dict[str, Any],
    eval_plan: dict[str, Any],
    audit: dict[str, Any],
    cache_manifest: dict[str, Any],
) -> None:
    _require(bool(rows), "horizon results are empty")
    _require(bool(per_sample_rows), "per-sample rows are required")
    _require(bool(eval_summary.get("strict_claim")), "eval summary must set strict_claim=true")
    _require(int(eval_summary.get("eval_samples", 0) or 0) > 0, "strict eval requires eval_samples > 0")
    _require(str(eval_summary.get("field_split", "")) == "test", "formal V20 report must use locked field_split=test")
    _require(str(eval_plan.get("field_split", "")) == str(eval_summary.get("field_split", "")), "eval_plan field_split must match eval summary")
    _require(bool(eval_plan.get("start_indices_hash")), "eval_plan.json must contain start index hashes")
    _require(str(cache_manifest.get("split_unit", "")) == "trajectory", "cache manifest must use trajectory split")
    _require(bool(cache_manifest.get("trajectory_lengths")), "cache manifest must include trajectory_lengths")
    _require(bool(cache_manifest.get("split_manifest")), "cache manifest must reference split_manifest")
    _require(bool(cache_manifest.get("source_manifest")), "cache manifest must reference source_manifest")
    _require(Path(str(cache_manifest.get("split_manifest"))).exists(), "split_manifest path does not exist")
    _require(Path(str(cache_manifest.get("source_manifest"))).exists(), "source_manifest path does not exist")
    _require(bool(audit.get("datasets") or cache_manifest.get("data_shape")), "dataset audit/cache shape metadata is missing")
    expected_horizons = {str(int(horizon)) for horizon in eval_summary.get("horizons", [])}
    _require(bool(expected_horizons), "eval summary must list horizons")
    expected_checkpoint_kinds = {str(kind) for kind in eval_summary.get("checkpoint_kinds", [])}
    _require(bool(expected_checkpoint_kinds), "eval summary must list checkpoint_kinds")
    expected_eval_samples = int(eval_summary.get("eval_samples", 0) or 0)
    plan_start_hashes = {str(key): str(value) for key, value in dict(eval_plan.get("start_indices_hash", {})).items()}
    _require(set(plan_start_hashes) == expected_horizons, "eval_plan start hash horizons must match eval summary horizons")
    summary_plan_hash = str(eval_summary.get("eval_plan_hash", ""))
    plan_hashes = {row.get("eval_plan_hash", "") for row in rows}
    _require(
        len(plan_hashes) == 1 and summary_plan_hash in plan_hashes and summary_plan_hash,
        f"results eval_plan_hash must exactly match summary, got summary={summary_plan_hash!r} rows={plan_hashes}",
    )

    observed_runs: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    observed_physical_runs: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    row_hash_by_run_horizon: dict[tuple[tuple[str, str, str], str], str] = {}
    for row in rows:
        horizon = str(int(row.get("horizon", 0)))
        _require(horizon in expected_horizons, f"unexpected horizon in results: {horizon}")
        checkpoint_kind = str(row.get("checkpoint_kind", ""))
        _require(checkpoint_kind in expected_checkpoint_kinds, f"unexpected checkpoint kind in results: {checkpoint_kind}")
        observed_runs[_run_key(row)].add(horizon)
        observed_physical_runs[(row.get("model", ""), row.get("source_run_dir", ""))][checkpoint_kind].add(horizon)
        start_hash = str(row.get("eval_start_indices_hash", ""))
        _require(start_hash == plan_start_hashes[horizon], f"result start hash mismatch for horizon {horizon}")
        sample_count = int(row.get("eval_sample_count", 0) or 0)
        _require(sample_count == expected_eval_samples, f"result sample count mismatch for horizon {horizon}: {sample_count}")
        row_hash_by_run_horizon[(_run_key(row), horizon)] = start_hash

    _require(bool(observed_runs), "no evaluated runs found")
    observed_checkpoint_kinds = {key[2] for key in observed_runs}
    _require(
        observed_checkpoint_kinds == expected_checkpoint_kinds,
        f"observed checkpoint kinds {observed_checkpoint_kinds} do not match expected {expected_checkpoint_kinds}",
    )
    for run_key, horizons in observed_runs.items():
        _require(horizons == expected_horizons, f"run {run_key} is missing horizons {expected_horizons - horizons}")
    for physical_run, checkpoint_map in observed_physical_runs.items():
        observed_for_run = set(checkpoint_map)
        _require(
            observed_for_run == expected_checkpoint_kinds,
            f"physical run {physical_run} has checkpoint kinds {observed_for_run}, expected {expected_checkpoint_kinds}",
        )
        for checkpoint_kind, horizons in checkpoint_map.items():
            _require(
                horizons == expected_horizons,
                f"physical run {physical_run} checkpoint {checkpoint_kind} is missing horizons {expected_horizons - horizons}",
            )

    per_sample_counts: dict[tuple[tuple[str, str, str], str], int] = defaultdict(int)
    per_sample_keys: dict[tuple[tuple[str, str, str], str], set[tuple[str, str, str, str]]] = defaultdict(set)
    per_sample_start_order: dict[tuple[tuple[str, str, str], str], list[tuple[int, int]]] = defaultdict(list)
    for row in per_sample_rows:
        run_key = _run_key(row)
        horizon = str(int(row.get("horizon", 0)))
        _require(run_key in observed_runs, f"per-sample row references run not present in horizon results: {run_key}")
        _require(horizon in expected_horizons, f"unexpected horizon in per-sample rows: {horizon}")
        start_hash = str(row.get("eval_start_indices_hash", ""))
        _require(start_hash == plan_start_hashes[horizon], f"per-sample start hash mismatch for horizon {horizon}")
        _require(
            start_hash == row_hash_by_run_horizon.get((run_key, horizon)),
            f"per-sample start hash does not match result row for run={run_key} horizon={horizon}",
        )
        _require(str(row.get("eval_plan_hash", "")) == summary_plan_hash, "per-sample eval_plan_hash must match summary")
        sample_key = (
            str(row.get("horizon", "")),
            str(row.get("trajectory_id", "")),
            str(row.get("start_index", "")),
            str(row.get("target_index", "")),
        )
        _require("" not in sample_key, f"per-sample row must include horizon/trajectory/start/target: {row}")
        try:
            sample_ordinal = int(row.get("sample_ordinal", ""))
            start_index = int(row.get("start_index", ""))
        except ValueError as exc:
            raise SystemExit(f"Strict V20 report validation failed: invalid per-sample ordinal/start_index: {row}") from exc
        group_key = (run_key, horizon)
        _require(sample_key not in per_sample_keys[group_key], f"duplicate per-sample key for run={run_key} sample={sample_key}")
        per_sample_keys[group_key].add(sample_key)
        per_sample_start_order[group_key].append((sample_ordinal, start_index))
        per_sample_counts[group_key] += 1
    for run_key in observed_runs:
        for horizon in expected_horizons:
            count = per_sample_counts.get((run_key, horizon), 0)
            _require(
                count == expected_eval_samples,
                f"per-sample count for run={run_key} horizon={horizon} is {count}, expected {expected_eval_samples}",
            )
            ordered_starts = [start for _ordinal, start in sorted(per_sample_start_order[(run_key, horizon)])]
            _require(
                _start_indices_hash_from_values(ordered_starts) == plan_start_hashes[horizon],
                f"per-sample start_index values do not hash to eval_plan for run={run_key} horizon={horizon}",
            )


def _bootstrap_ci(values: list[float], *, iterations: int = 1000) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(20260622)
    means: list[float] = []
    count = len(values)
    for _ in range(iterations):
        sample = [values[rng.randrange(count)] for _ in range(count)]
        means.append(sum(sample) / count)
    means.sort()
    low = means[int(0.025 * (iterations - 1))]
    high = means[int(0.975 * (iterations - 1))]
    return low, high


def paired_delta_summaries(per_sample_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_run: dict[tuple[str, str, str], dict[tuple[str, str, str, str], dict[str, float]]] = defaultdict(dict)
    persistence_by_sample: dict[tuple[str, str, str, str], float] = {}
    for row in per_sample_rows:
        run_key = _run_key(row)
        sample_key = (
            row.get("horizon", ""),
            row.get("trajectory_id", ""),
            row.get("start_index", ""),
            row.get("target_index", ""),
        )
        mse = _float(row.get("mse"))
        persistence_mse = _float(row.get("persistence_mse"))
        if mse is None or persistence_mse is None:
            continue
        previous_persistence = persistence_by_sample.get(sample_key)
        if previous_persistence is None:
            persistence_by_sample[sample_key] = persistence_mse
        elif abs(previous_persistence - persistence_mse) > 1e-12:
            raise SystemExit(
                "Strict V20 report validation failed: per-sample persistence_mse mismatch "
                f"for sample={sample_key}: {previous_persistence} vs {persistence_mse}"
            )
        by_run[run_key][sample_key] = {"mse": mse, "persistence_mse": persistence_mse}

    wca_runs = [key for key in by_run if "FullRecursive" in key[0] or "WCA" in key[0]]
    baseline_runs = [key for key in by_run if key not in wca_runs]
    summaries: list[dict[str, Any]] = []
    for wca_key in wca_runs:
        for baseline_key in baseline_runs:
            common_all = sorted(set(by_run[wca_key]).intersection(by_run[baseline_key]))
            horizons = sorted({key[0] for key in common_all}, key=lambda item: int(item))
            for horizon in horizons:
                common = [key for key in common_all if key[0] == horizon]
                deltas = [by_run[wca_key][key]["mse"] - by_run[baseline_key][key]["mse"] for key in common]
                if not deltas:
                    continue
                ci_low, ci_high = _bootstrap_ci(deltas)
                summaries.append(
                    {
                        "horizon": horizon,
                        "wca_model": wca_key[0],
                        "wca_run": wca_key[1],
                        "checkpoint_kind": wca_key[2],
                        "baseline_model": baseline_key[0],
                        "baseline_run": baseline_key[1],
                        "baseline_checkpoint_kind": baseline_key[2],
                        "paired_sample_count": len(deltas),
                        "mean_mse_delta": sum(deltas) / len(deltas),
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                    }
                )
    return summaries


def validate_paired_summaries(
    paired_summaries: list[dict[str, Any]],
    *,
    eval_summary: dict[str, Any],
    rows: list[dict[str, str]],
) -> None:
    expected_sample_count = int(eval_summary.get("eval_samples", 0) or 0)
    _require(expected_sample_count > 0, "paired summary validation requires eval_samples and horizons")
    expected_horizons = {str(int(horizon)) for horizon in eval_summary.get("horizons", [])}
    wca_runs = {
        _run_key(row)
        for row in rows
        if "FullRecursive" in row.get("model", "") or "WCA" in row.get("model", "")
    }
    baseline_runs = {_run_key(row) for row in rows if _run_key(row) not in wca_runs}
    _require(bool(wca_runs), "strict report requires at least one WCA run")
    _require(bool(baseline_runs), "strict report requires at least one baseline run")
    _require(bool(paired_summaries), "strict report requires non-empty paired WCA-vs-baseline summaries")
    observed_pairs = {
        (
            str(item.get("horizon", "")),
            str(item.get("wca_run", "")),
            str(item.get("checkpoint_kind", "")),
            str(item.get("baseline_run", "")),
            str(item.get("baseline_checkpoint_kind", "")),
        )
        for item in paired_summaries
    }
    expected_pairs = {
        (horizon, wca_key[1], wca_key[2], baseline_key[1], baseline_key[2])
        for horizon in expected_horizons
        for wca_key in wca_runs
        for baseline_key in baseline_runs
    }
    _require(observed_pairs == expected_pairs, f"paired summaries do not cover all WCA/baseline pairs: missing {expected_pairs - observed_pairs}")
    for item in paired_summaries:
        count = int(item.get("paired_sample_count", 0) or 0)
        _require(
            count == expected_sample_count,
            f"paired sample count for {item.get('wca_run')} vs {item.get('baseline_run')} is {count}, expected {expected_sample_count}",
        )


def validate_paired_summaries_or_internal_only(
    paired_summaries: list[dict[str, Any]],
    *,
    eval_summary: dict[str, Any],
    rows: list[dict[str, str]],
    allow_internal_only: bool,
) -> None:
    if not allow_internal_only:
        validate_paired_summaries(paired_summaries, eval_summary=eval_summary, rows=rows)
        return

    wca_runs = {
        _run_key(row)
        for row in rows
        if "FullRecursive" in row.get("model", "") or "WCA" in row.get("model", "")
    }
    baseline_runs = {_run_key(row) for row in rows if _run_key(row) not in wca_runs}
    _require(bool(wca_runs), "internal-only report requires at least one WCA run")
    if baseline_runs:
        validate_paired_summaries(paired_summaries, eval_summary=eval_summary, rows=rows)


def validate_sentinel_guardrails(
    *,
    sentinel_summary: dict[str, Any],
    sentinel_rows: list[dict[str, str]],
    eval_summary: dict[str, Any],
) -> dict[str, Any]:
    expected_horizons = {str(int(horizon)) for horizon in eval_summary.get("horizons", [])}
    expected_checkpoint_kinds = {str(kind) for kind in eval_summary.get("checkpoint_kinds", [])}
    summary_horizons = {str(int(horizon)) for horizon in sentinel_summary.get("horizons", [])}
    summary_checkpoint_kinds = {str(kind) for kind in sentinel_summary.get("checkpoint_kinds", [])}
    _require(bool(expected_horizons), "sentinel validation requires eval summary horizons")
    _require(summary_horizons == expected_horizons, "sentinel summary horizons must match eval summary horizons")
    _require(
        summary_checkpoint_kinds == expected_checkpoint_kinds,
        "sentinel summary checkpoint_kinds must match eval summary checkpoint_kinds",
    )
    if "eval_seed" in eval_summary:
        _require(
            int(sentinel_summary.get("eval_seed", -1)) == int(eval_summary.get("eval_seed")),
            "sentinel summary eval_seed must match eval summary eval_seed",
        )
    _require(int(sentinel_summary.get("hard_failure_count", -1)) == 0, "sentinel summary hard_failure_count must be 0")
    _require(int(sentinel_summary.get("rows", -1)) == len(sentinel_rows), "sentinel summary rows must match sentinel_results.csv")

    input_shuffle_degraded_count = 0
    input_shuffle_ratios: list[float] = []
    for row in sentinel_rows:
        horizon = str(int(row.get("horizon", 0)))
        _require(horizon in expected_horizons, f"unexpected sentinel horizon: {horizon}")
        checkpoint_kind = str(row.get("checkpoint_kind", ""))
        _require(checkpoint_kind in expected_checkpoint_kinds, f"unexpected sentinel checkpoint kind: {checkpoint_kind}")
        _require(str(row.get("label_leakage_pass", "")).lower() == "true", "sentinel label_leakage_pass must be true")
        _require(str(row.get("horizon_shuffle_pass", "")).lower() == "true", "sentinel horizon_shuffle_pass must be true")
        if str(row.get("input_shuffle_degraded", "")).lower() == "true":
            input_shuffle_degraded_count += 1
        ratio = _float(row.get("input_shuffle_mse_ratio"))
        if ratio is not None:
            input_shuffle_ratios.append(ratio)
    return {
        "rows": len(sentinel_rows),
        "input_shuffle_degraded_count": input_shuffle_degraded_count,
        "input_shuffle_ratio_mean": _mean(input_shuffle_ratios),
        "input_shuffle_degraded_fraction": (
            input_shuffle_degraded_count / len(sentinel_rows) if sentinel_rows else None
        ),
    }


def validate_control_manifest_contract(
    *,
    control_manifest: dict[str, Any],
    eval_summary: dict[str, Any],
) -> None:
    protocol = control_manifest.get("protocol")
    _require(isinstance(protocol, dict) and protocol.get("strict") is True, "control manifest must declare protocol.strict=true")
    eval_contract = control_manifest.get("eval")
    _require(isinstance(eval_contract, dict), "control manifest must define eval contract")
    expected_horizons = {str(int(horizon)) for horizon in eval_summary.get("horizons", [])}
    manifest_horizons = {str(int(horizon)) for horizon in eval_contract.get("horizons", [])}
    _require(manifest_horizons == expected_horizons, "control manifest eval.horizons must match eval summary horizons")
    _require(
        int(eval_contract.get("eval_samples", -1)) == int(eval_summary.get("eval_samples", -2)),
        "control manifest eval.eval_samples must match eval summary eval_samples",
    )
    _require(
        str(eval_contract.get("field_split", "")) == str(eval_summary.get("field_split", "")),
        "control manifest eval.field_split must match eval summary field_split",
    )
    checkpoint_contract = control_manifest.get("checkpoint")
    if isinstance(checkpoint_contract, dict) and checkpoint_contract.get("allowed_kinds") is not None:
        manifest_kinds = {str(kind) for kind in checkpoint_contract.get("allowed_kinds", [])}
        summary_kinds = {str(kind) for kind in eval_summary.get("checkpoint_kinds", [])}
        _require(manifest_kinds == summary_kinds, "control manifest checkpoint.allowed_kinds must match eval summary checkpoint_kinds")


def write_report(
    *,
    output: Path,
    rows: list[dict[str, str]],
    per_sample_rows: list[dict[str, str]],
    summaries: list[dict[str, Any]],
    paired_summaries: list[dict[str, Any]],
    audit: dict[str, Any],
    cache_manifest: dict[str, Any],
    eval_summary: dict[str, Any],
    eval_plan: dict[str, Any],
    sentinel_stats: dict[str, Any] | None = None,
    control_manifest_path: Path | None = None,
    internal_only: bool = False,
) -> None:
    lines = ["# V20 PDEBench Matched Small Results", ""]
    lines.extend(
        [
            "## Data Audit",
            "",
            f"- source file: `{cache_manifest.get('source_path', '')}`",
            f"- source manifest: `{cache_manifest.get('source_manifest', '')}`",
            f"- cache file: `{cache_manifest.get('output_path', '')}`",
            f"- dataset key: `{cache_manifest.get('dataset_key', '')}`",
            f"- layout: `{cache_manifest.get('layout', '')}`",
            f"- resolved layout: `{cache_manifest.get('resolved_layout', '')}`",
            f"- split unit: `{cache_manifest.get('split_unit', '')}`",
            f"- split manifest: `{cache_manifest.get('split_manifest', '')}`",
            f"- data shape: `{cache_manifest.get('data_shape', '')}`",
            f"- audited datasets: `{sorted((audit.get('datasets') or {}).keys())}`",
            f"- eval plan hash: `{eval_summary.get('eval_plan_hash', '')}`",
            f"- eval field split: `{eval_summary.get('field_split', '')}`",
            f"- eval samples: `{eval_summary.get('eval_samples', '')}`",
            f"- eval start hashes: `{eval_plan.get('start_indices_hash', {})}`",
            f"- control manifest: `{control_manifest_path.as_posix() if control_manifest_path else ''}`",
            "",
        ]
    )
    if sentinel_stats is not None:
        lines.extend(
            [
                "## Sentinel Guardrails",
                "",
                f"- sentinel rows: `{sentinel_stats.get('rows', '')}`",
                f"- input shuffle degraded rows: `{sentinel_stats.get('input_shuffle_degraded_count', '')}`",
                f"- input shuffle degraded fraction: `{_fmt(sentinel_stats.get('input_shuffle_degraded_fraction'))}`",
                f"- mean input shuffle MSE ratio: `{_fmt(sentinel_stats.get('input_shuffle_ratio_mean'))}`",
                "- hard gate: label leakage and horizon shuffle sentinels must pass for every row.",
                "- informational gate: input shuffle degradation is reported, but not a formal hard threshold unless a manifest declares one.",
                "",
            ]
        )
    if internal_only:
        lines.extend(
            [
                "## Evidence Scope",
                "",
                "This is an internal WCA-only repair/evaluation report. It is not a matched-baseline comparison.",
                "",
            ]
        )
    lines.extend(
        [
            "## Summary By Run",
            "",
            "| model | ckpt | run | mean rel L2 | mean MSE | mean improvement | h8 rel L2 | h8 improvement | horizons |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summaries:
        run_name = "/".join(str(item["run_dir"]).split("/")[-2:])
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item['model']}`",
                    f"`{item['checkpoint_kind']}`",
                    f"`{run_name}`",
                    _fmt(item.get("mean_relative_l2")),
                    _fmt(item.get("mean_mse")),
                    _fmt(item.get("mean_improvement")),
                    _fmt(item.get("h8_relative_l2")),
                    _fmt(item.get("h8_improvement")),
                    _fmt(item.get("horizon_count")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Paired Delta Summary",
            "",
            "Negative mean delta means WCA has lower per-sample MSE than the baseline on matched samples.",
            "Rows are horizon-stratified; pooled averages are not the primary endpoint.",
            "Final checkpoints are the primary formal endpoint. Best checkpoints are diagnostic unless selected before test evaluation.",
            "",
            "| h | WCA ckpt | WCA run | baseline | baseline ckpt | paired n | mean MSE delta | 95% CI low | 95% CI high |",
            "|---:|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for item in paired_summaries:
        wca_run = "/".join(str(item["wca_run"]).split("/")[-2:])
        baseline_run = "/".join(str(item["baseline_run"]).split("/")[-2:])
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(item.get("horizon")),
                    f"`{item['checkpoint_kind']}`",
                    f"`{wca_run}`",
                    f"`{item['baseline_model']}:{baseline_run}`",
                    f"`{item['baseline_checkpoint_kind']}`",
                    _fmt(item.get("paired_sample_count")),
                    _fmt(item.get("mean_mse_delta")),
                    _fmt(item.get("bootstrap_ci_low")),
                    _fmt(item.get("bootstrap_ci_high")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This report is a deterministic aggregation of horizon-stratified rows. It is not a SOTA claim.",
            "WCA should be described as stronger than a baseline only when it wins on the same source cache, split, horizons, and checkpoint rule.",
            "",
            "## Raw Rows",
            "",
            f"Rows: {len(rows)}",
            f"Per-sample rows: {len(per_sample_rows)}",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the V20 PDEBench matched-small report.")
    parser.add_argument("--horizon-results", type=Path, required=True)
    parser.add_argument("--eval-summary", type=Path, required=True)
    parser.add_argument("--eval-plan", type=Path, required=True)
    parser.add_argument("--per-sample-rows", type=Path, required=True)
    parser.add_argument("--sentinel-summary", type=Path)
    parser.add_argument("--sentinel-results", type=Path)
    parser.add_argument("--control-manifest", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-internal-only",
        action="store_true",
        help="Allow reports with WCA runs but no matched baseline; for internal repair/eval only.",
    )
    args = parser.parse_args()

    rows = _load_rows(args.horizon_results)
    per_sample_rows = _load_rows(args.per_sample_rows)
    eval_summary = _read_json(args.eval_summary)
    eval_plan = _read_json(args.eval_plan)
    if args.control_manifest:
        validate_control_manifest_contract(
            control_manifest=_read_json(args.control_manifest),
            eval_summary=eval_summary,
        )
    sentinel_stats = None
    if args.sentinel_summary or args.sentinel_results:
        _require(bool(args.sentinel_summary and args.sentinel_results), "sentinel summary and results must be provided together")
        sentinel_summary = _read_json(args.sentinel_summary)
        sentinel_rows = _load_rows(args.sentinel_results)
        sentinel_stats = validate_sentinel_guardrails(
            sentinel_summary=sentinel_summary,
            sentinel_rows=sentinel_rows,
            eval_summary=eval_summary,
        )
    audit = _read_json(args.audit)
    cache_manifest = _read_json(args.cache_manifest)
    validate_strict_inputs(
        rows=rows,
        per_sample_rows=per_sample_rows,
        eval_summary=eval_summary,
        eval_plan=eval_plan,
        audit=audit,
        cache_manifest=cache_manifest,
    )
    summaries = summarize_rows(rows)
    paired_summaries = paired_delta_summaries(per_sample_rows)
    validate_paired_summaries_or_internal_only(
        paired_summaries,
        eval_summary=eval_summary,
        rows=rows,
        allow_internal_only=args.allow_internal_only,
    )
    write_report(
        output=args.output,
        rows=rows,
        per_sample_rows=per_sample_rows,
        summaries=summaries,
        paired_summaries=paired_summaries,
        audit=audit,
        cache_manifest=cache_manifest,
        eval_summary=eval_summary,
        eval_plan=eval_plan,
        sentinel_stats=sentinel_stats,
        control_manifest_path=args.control_manifest,
        internal_only=args.allow_internal_only,
    )
    print(json.dumps({"rows": len(rows), "per_sample_rows": len(per_sample_rows), "output": args.output.as_posix()}, sort_keys=True))


if __name__ == "__main__":
    main()
