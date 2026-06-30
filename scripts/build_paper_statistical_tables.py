#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


REPORT_SPECS = [
    {
        "report_id": "v25_recursion_depth_ladder",
        "evidence_status": "formal_establish",
        "report_dir": "artifacts/reports/westb_pdebench_v25_recursion_depth_ladder",
        "per_sample": "per_sample_rows.csv",
        "results": "results_by_horizon.csv",
        "family": "horizon",
    },
    {
        "report_id": "v25e_r3_attribution",
        "evidence_status": "formal_bounded_attribution",
        "report_dir": "artifacts/reports/westb_pdebench_v25e_attribution_closure_r3",
        "per_sample": "per_sample_rows.csv",
        "results": "results_by_horizon.csv",
        "family": "horizon",
    },
    {
        "report_id": "v25d_h8x2_rollout",
        "evidence_status": "formal_token_rollout_diagnostic",
        "report_dir": "artifacts/reports/westb_pdebench_v25d_h8_rollout_twice_r3",
        "per_sample": "per_sample_rollout_rows.csv",
        "results": "results_rollout.csv",
        "family": "rollout",
    },
    {
        "report_id": "v27_n256_patchmean_diagnostic",
        "evidence_status": "diagnostic_adverse_patchmean_scaling",
        "report_dir": "artifacts/reports/westb_pdebench_v27_n256_matched_baselines_formal_token_matched",
        "per_sample": "per_sample_rows.csv",
        "results": "results_by_horizon.csv",
        "family": "horizon",
    },
]

MODEL_ROLES_BY_REPORT = {
    "v25_recursion_depth_ladder": {
        "FullRecursiveWorldStateNCA-heavy-dense": "wca",
        "fno-field-baseline": "external_baseline",
        "unet-field-baseline": "external_baseline",
    },
    "v25e_r3_attribution": {
        "mlp_stem-WCA": "wca",
        "mlp_stem-tokenizer-only": "negative_control",
        "mlp_stem-tokenizer-bypass-o0": "negative_control",
        "token_mlp-field-token-baseline": "token_baseline",
        "token_conv-field-token-baseline": "token_baseline",
        "fno-field-baseline": "external_baseline",
        "unet-field-baseline": "external_baseline",
    },
    "v25d_h8x2_rollout": {
        "mlp_stem-WCA": "wca",
        "mlp_stem-tokenizer-only": "negative_control",
        "mlp_stem-tokenizer-bypass-o0": "negative_control",
        "fno": "external_baseline",
        "unet": "external_baseline",
    },
    "v27_n256_patchmean_diagnostic": {
        "FullRecursiveWorldStateNCA-heavy-dense": "wca",
        "token_mlp-field-token-baseline": "token_baseline",
        "token_conv-field-token-baseline": "token_baseline",
    },
}

CLAIM_SCOPES = [
    {
        "claim_id": "C1_v25_establish_wca_vs_external_baselines",
        "report_id": "v25_recursion_depth_ladder",
        "comparison_scope": "Full dense WCA vs raw-field FNO/UNet baselines across h1/h2/h4/h8",
        "wca_model_contains": ["FullRecursiveWorldStateNCA"],
        "baseline_model_contains": ["fno", "unet"],
        "condition_contains": [],
        "checkpoint_policy": "final_only",
        "paper_use": "establish_signal_only_not_dominance",
        "interpretation_guardrail": "Use to show early WCA signal and recursion-depth motivation; do not cite as a global win over FNO/UNet.",
    },
    {
        "claim_id": "C1b_v25_h8_final_directional_check",
        "report_id": "v25_recursion_depth_ladder",
        "comparison_scope": "Final-checkpoint h8 only: Full dense WCA vs raw-field FNO/UNet baselines",
        "wca_model_contains": ["FullRecursiveWorldStateNCA"],
        "baseline_model_contains": ["fno", "unet"],
        "condition_contains": ["horizon=8"],
        "wca_checkpoint_kind": "final",
        "checkpoint_policy": "final_only",
        "paper_use": "horizon_specific_diagnostic",
        "interpretation_guardrail": "Only discuss as horizon-specific behavior; not enough for a broad scaling claim.",
    },
    {
        "claim_id": "C2_v25e_attribute_wca_core_vs_token_controls",
        "report_id": "v25e_r3_attribution",
        "comparison_scope": "MLP-stem WCA core vs tokenizer-only, outer0, token-MLP, and token-Conv controls",
        "wca_model_contains": ["mlp_stem-WCA"],
        "baseline_model_contains": ["tokenizer", "token_mlp", "token_conv"],
        "condition_contains": [],
        "checkpoint_policy": "final_only",
        "paper_use": "formal_bounded_attribution",
        "interpretation_guardrail": "Supports WCA-core contribution only inside this tokenized PDEBench protocol.",
    },
    {
        "claim_id": "C3_v25d_h8x2_piecewise_rollout_token_controls",
        "report_id": "v25d_h8x2_rollout",
        "comparison_scope": "h8 twice piecewise rollout: MLP-stem WCA vs token controls only",
        "wca_model_contains": ["mlp_stem-WCA"],
        "baseline_model_contains": ["tokenizer"],
        "condition_contains": ["mode=rollout_h8x2_piecewise"],
        "condition_excludes": ["mode=direct_h16_ood"],
        "checkpoint_policy": "final_only",
        "paper_use": "formal_token_rollout_piecewise",
        "metric_space_contract": "token_space_same_decoder_same_eval_indices",
        "claim_mode": "h8x2_piecewise_rollout",
        "comparison_family": "token_equivalent_controls",
        "interpretation_guardrail": "Use as token-space h8x2 piecewise rollout evidence against token controls only; direct h8, teacher-prev, and raw-field anchors are diagnostic context.",
    },
    {
        "claim_id": "C3a_v25d_direct_h8_token_control_diagnostic",
        "report_id": "v25d_h8x2_rollout",
        "comparison_scope": "direct h8 diagnostic: MLP-stem WCA vs token controls only",
        "wca_model_contains": ["mlp_stem-WCA"],
        "baseline_model_contains": ["tokenizer"],
        "condition_contains": ["mode=direct_h8"],
        "condition_excludes": ["mode=direct_h16_ood"],
        "checkpoint_policy": "final_only",
        "paper_use": "diagnostic_direct_h8_reference",
        "metric_space_contract": "token_space_same_decoder_same_eval_indices",
        "claim_mode": "direct_h8_diagnostic",
        "comparison_family": "token_equivalent_controls",
        "manuscript_status_override": "diagnostic_context_only",
        "interpretation_guardrail": "Report only as same-model direct h8 context; do not merge with the formal h8x2 rollout claim.",
    },
    {
        "claim_id": "C3b_v25d_teacher_prev_h8x2_token_control_diagnostic",
        "report_id": "v25d_h8x2_rollout",
        "comparison_scope": "h8 twice teacher-prev diagnostic: MLP-stem WCA vs token controls only",
        "wca_model_contains": ["mlp_stem-WCA"],
        "baseline_model_contains": ["tokenizer"],
        "condition_contains": ["mode=rollout_h8x2_teacher_prev_piecewise"],
        "condition_excludes": ["mode=direct_h16_ood"],
        "checkpoint_policy": "final_only",
        "paper_use": "diagnostic_teacher_prev_reference",
        "metric_space_contract": "token_space_same_decoder_same_eval_indices",
        "claim_mode": "h8x2_teacher_prev_diagnostic",
        "comparison_family": "token_equivalent_controls",
        "manuscript_status_override": "diagnostic_context_only",
        "interpretation_guardrail": "Report only as teacher-forced diagnostic context; do not merge with the formal autoregressive-style piecewise rollout claim.",
    },
    {
        "claim_id": "C3c_v25d_h8x2_raw_anchor_context",
        "report_id": "v25d_h8x2_rollout",
        "comparison_scope": "h8 twice piecewise rollout raw/full-field anchor context: MLP-stem WCA vs FNO/U-Net",
        "wca_model_contains": ["mlp_stem-WCA"],
        "baseline_model_contains": ["fno", "unet"],
        "condition_contains": ["mode=rollout_h8x2_piecewise"],
        "condition_excludes": ["mode=direct_h16_ood"],
        "checkpoint_policy": "final_only",
        "paper_use": "raw_full_field_anchor_context_only",
        "metric_space_contract": "mixed_token_vs_raw_full_field_context_not_formal",
        "claim_mode": "h8x2_piecewise_rollout_raw_anchor_context",
        "comparison_family": "non_equivalent_raw_anchor",
        "manuscript_status_override": "raw_full_field_anchor_context_only",
        "rank_decision_allowed": False,
        "decision_override": "metric_space_mismatch_no_rank_decision",
        "interpretation_guardrail": "May be disclosed as context only; never use as formal evidence that WCA beats native/full-field FNO or U-Net.",
    },
    {
        "claim_id": "C4_v27_n256_patchmean_adverse_scaling",
        "report_id": "v27_n256_patchmean_diagnostic",
        "comparison_scope": "N=256 PatchMean WCA vs token-equivalent MLP/Conv baselines",
        "wca_model_contains": ["FullRecursiveWorldStateNCA"],
        "baseline_model_contains": ["token_mlp", "token_conv"],
        "condition_contains": [],
        "checkpoint_policy": "final_only",
        "paper_use": "adverse_diagnostic",
        "interpretation_guardrail": "Report as PatchMean-interface bottleneck evidence, not as an architecture-wide WCA scaling failure.",
    },
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_md_table(path: Path, rows: list[dict[str, Any]], fields: list[str], *, title: str, limit: int | None = None) -> None:
    shown = rows if limit is None else rows[:limit]
    lines = [f"# {title}", "", f"Rows shown: {len(shown)} / {len(rows)}", ""]
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("|" + "|".join("---" for _ in fields) + "|")
    for row in shown:
        lines.append("| " + " | ".join(_fmt(row.get(field, "")) for field in fields) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    number = _float(value)
    if number is not None:
        if abs(number) < 1e-3 or abs(number) >= 1e4:
            return f"{number:.6e}"
        return f"{number:.6g}"
    return str(value)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _ci(values: list[float], *, iterations: int, seed: int) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means: list[float] = []
    count = len(values)
    for _ in range(iterations):
        sample = [values[rng.randrange(count)] for _ in range(count)]
        means.append(sum(sample) / count)
    means.sort()
    low = means[int(0.025 * (iterations - 1))]
    high = means[int(0.975 * (iterations - 1))]
    return low, high


def _model_role(report_id: str, model: str) -> str:
    roles = MODEL_ROLES_BY_REPORT.get(report_id, {})
    if model not in roles:
        raise ValueError(f"Unknown model role for report={report_id!r} model={model!r}")
    return roles[model]


def _is_wca(report_id: str, model: str) -> bool:
    return _model_role(report_id, model) == "wca"


def _run_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("model", ""),
        row.get("source_run_dir", ""),
        row.get("checkpoint_kind", ""),
        row.get("seed", ""),
    )


def _replicate_id(seed: str) -> str:
    return seed[-2:] if seed else ""


def _condition_key(row: dict[str, str], family: str) -> str:
    if family == "rollout":
        mode = row.get("mode", "")
        step = row.get("step_horizon", "")
        total = row.get("total_horizon", "")
        return f"mode={mode};step={step};total={total}"
    return f"horizon={row.get('horizon', '')}"


def _sample_key(row: dict[str, str], family: str) -> tuple[str, ...]:
    base = (
        row.get("trajectory_id", ""),
        row.get("start_index", ""),
        row.get("target_index", ""),
    )
    if family == "rollout":
        return (
            row.get("mode", ""),
            row.get("step_horizon", ""),
            row.get("total_horizon", ""),
            *base,
        )
    return (row.get("horizon", ""), *base)


def _result_condition_key(row: dict[str, str], family: str) -> str:
    if family == "rollout":
        mode = row.get("mode", "")
        step = row.get("step_horizon", "")
        total = row.get("total_horizon", "")
        return f"mode={mode};step={step};total={total}"
    return f"horizon={row.get('horizon', '')}"


def _result_key(row: dict[str, str], family: str) -> tuple[str, str, str, str, str]:
    return (
        row.get("model", ""),
        row.get("source_run_dir", ""),
        row.get("checkpoint_kind", ""),
        row.get("seed", ""),
        _result_condition_key(row, family),
    )


def build_metric_consistency(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in REPORT_SPECS:
        report_id = str(spec["report_id"])
        family = str(spec["family"])
        report_dir = root / spec["report_dir"]
        per_sample = _read_csv(report_dir / spec["per_sample"])
        results = _read_csv(report_dir / spec["results"])
        result_by_key = {_result_key(row, family): row for row in results}
        grouped: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
        for row in per_sample:
            mse = _float(row.get("mse"))
            if mse is None:
                continue
            grouped[
                (
                    row.get("model", ""),
                    row.get("source_run_dir", ""),
                    row.get("checkpoint_kind", ""),
                    row.get("seed", ""),
                    _condition_key(row, family),
                )
            ].append(mse)
        for key, values in sorted(grouped.items()):
            model, run_dir, checkpoint, seed, condition = key
            result = result_by_key.get(key)
            per_sample_mean = _mean(values)
            result_mse = _float(result.get("eval_mse")) if result else None
            abs_delta = abs(per_sample_mean - result_mse) if per_sample_mean is not None and result_mse is not None else None
            rel_delta = abs_delta / max(abs(result_mse), 1e-30) if abs_delta is not None and result_mse is not None else None
            status = "missing_result_row"
            if result_mse is not None and abs_delta is not None and rel_delta is not None:
                status = "match" if abs_delta <= 1e-12 or rel_delta <= 1e-5 else "mismatch"
            rows.append(
                {
                    "report_id": report_id,
                    "condition": condition,
                    "model": model,
                    "checkpoint_kind": checkpoint,
                    "seed": seed,
                    "source_run_dir": run_dir,
                    "sample_count": len(values),
                    "per_sample_mean_mse": per_sample_mean,
                    "result_eval_mse": result_mse,
                    "abs_delta": abs_delta,
                    "rel_delta": rel_delta,
                    "consistency_status": status,
                }
            )
    return rows


def _consistency_by_report(consistency_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in consistency_rows:
        grouped[str(row.get("report_id", ""))].append(row)
    summary: dict[str, dict[str, Any]] = {}
    for report_id, rows in grouped.items():
        mismatches = [row for row in rows if row.get("consistency_status") != "match"]
        rel_values = [_float(row.get("rel_delta")) for row in rows]
        abs_values = [_float(row.get("abs_delta")) for row in rows]
        summary[report_id] = {
            "metric_consistency_status": "pass" if not mismatches else "fail",
            "metric_consistency_rows": len(rows),
            "metric_consistency_mismatch_count": len(mismatches),
            "metric_consistency_max_rel_delta": max([value for value in rel_values if value is not None], default=None),
            "metric_consistency_max_abs_delta": max([value for value in abs_values if value is not None], default=None),
        }
    return summary


def build_inventory(root: Path, consistency_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    consistency_summary = _consistency_by_report(consistency_rows)
    rows: list[dict[str, Any]] = []
    for spec in REPORT_SPECS:
        report_id = str(spec["report_id"])
        report_dir = root / spec["report_dir"]
        per_sample = _read_csv(report_dir / spec["per_sample"])
        results = _read_csv(report_dir / spec["results"])
        models = sorted({row.get("model", "") for row in per_sample if row.get("model")})
        seeds = sorted({row.get("seed", "") for row in per_sample if row.get("seed")})
        conditions = sorted({_condition_key(row, str(spec["family"])) for row in per_sample})
        sample_counts = defaultdict(int)
        for row in per_sample:
            sample_counts[(_run_key(row), _condition_key(row, str(spec["family"])))] += 1
        rows.append(
            {
                "report_id": spec["report_id"],
                "evidence_status": spec["evidence_status"],
                "family": spec["family"],
                "report_dir": spec["report_dir"],
                "results_file": spec["results"],
                "per_sample_file": spec["per_sample"],
                "results_rows": len(results),
                "per_sample_rows": len(per_sample),
                "model_count": len(models),
                "seed_count": len(seeds),
                "condition_count": len(conditions),
                "min_samples_per_run_condition": min(sample_counts.values()) if sample_counts else "",
                "max_samples_per_run_condition": max(sample_counts.values()) if sample_counts else "",
                "bootstrap_eligible": bool(per_sample),
                "models": "; ".join(models),
                **consistency_summary.get(report_id, {}),
            }
        )
    return rows


def build_condition_summary(root: Path, *, iterations: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in REPORT_SPECS:
        report_dir = root / spec["report_dir"]
        per_sample = _read_csv(report_dir / spec["per_sample"])
        grouped: dict[tuple[str, str, str, str, str, str], list[float]] = defaultdict(list)
        mae_grouped: dict[tuple[str, str, str, str, str, str], list[float]] = defaultdict(list)
        for row in per_sample:
            key = (
                str(spec["report_id"]),
                row.get("model", ""),
                row.get("source_run_dir", ""),
                row.get("checkpoint_kind", ""),
                row.get("seed", ""),
                _condition_key(row, str(spec["family"])),
            )
            mse = _float(row.get("mse"))
            mae = _float(row.get("mae"))
            if mse is not None:
                grouped[key].append(mse)
            if mae is not None:
                mae_grouped[key].append(mae)
        for key, values in sorted(grouped.items()):
            report_id, model, run_dir, checkpoint, seed, condition = key
            low, high = _ci(values, iterations=iterations, seed=20260630)
            rows.append(
                {
                    "report_id": report_id,
                    "condition": condition,
                    "model": model,
                    "checkpoint_kind": checkpoint,
                    "source_run_dir": run_dir,
                    "sample_count": len(values),
                    "mean_mse": _mean(values),
                    "std_mse": _std(values),
                    "bootstrap_mean_mse_ci_low": low,
                    "bootstrap_mean_mse_ci_high": high,
                    "mean_mae": _mean(mae_grouped.get(key, [])),
                    "seed": seed,
                    "model_role": _model_role(report_id, model),
                    "wca_family": _is_wca(report_id, model),
                }
            )
    return rows


def build_paired_bootstrap(root: Path, *, iterations: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in REPORT_SPECS:
        report_dir = root / spec["report_dir"]
        per_sample = _read_csv(report_dir / spec["per_sample"])
        report_id = str(spec["report_id"])
        by_run: dict[tuple[str, str, str, str], dict[tuple[str, ...], float]] = defaultdict(dict)
        condition_by_sample: dict[tuple[str, ...], str] = {}
        for row in per_sample:
            mse = _float(row.get("mse"))
            if mse is None:
                continue
            key = _sample_key(row, str(spec["family"]))
            by_run[_run_key(row)][key] = mse
            condition_by_sample[key] = _condition_key(row, str(spec["family"]))
        wca_runs = [key for key in by_run if _is_wca(report_id, key[0])]
        baseline_runs = [key for key in by_run if not _is_wca(report_id, key[0])]
        for wca in wca_runs:
            for baseline in baseline_runs:
                common = sorted(set(by_run[wca]).intersection(by_run[baseline]))
                by_condition: dict[str, list[float]] = defaultdict(list)
                for sample in common:
                    condition = condition_by_sample[sample]
                    by_condition[condition].append(by_run[wca][sample] - by_run[baseline][sample])
                for condition, deltas in sorted(by_condition.items()):
                    if not deltas:
                        continue
                    low, high = _ci(deltas, iterations=iterations, seed=20260630)
                    mean_delta = _mean(deltas)
                    rows.append(
                        {
                            "report_id": spec["report_id"],
                            "condition": condition,
                            "wca_model": wca[0],
                            "wca_checkpoint_kind": wca[2],
                            "wca_source_run_dir": wca[1],
                            "wca_seed": wca[3],
                            "wca_replicate_id": _replicate_id(wca[3]),
                            "baseline_model": baseline[0],
                            "baseline_checkpoint_kind": baseline[2],
                            "baseline_source_run_dir": baseline[1],
                            "baseline_seed": baseline[3],
                            "baseline_replicate_id": _replicate_id(baseline[3]),
                            "seed_replicate_match": _replicate_id(wca[3]) == _replicate_id(baseline[3]),
                            "paired_sample_count": len(deltas),
                            "mean_mse_delta_wca_minus_baseline": mean_delta,
                            "bootstrap_delta_ci_low": low,
                            "bootstrap_delta_ci_high": high,
                            "wca_better_by_mean": mean_delta is not None and mean_delta < 0,
                            "ci_entirely_wca_better": high is not None and high < 0,
                            "ci_entirely_baseline_better": low is not None and low > 0,
                        }
                    )
    return rows


def _matches_any(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def _checkpoint_policy_match(scope: dict[str, Any], row: dict[str, Any]) -> bool:
    policy = scope.get("checkpoint_policy", "")
    if policy == "final_only":
        return row.get("wca_checkpoint_kind") == "final" and row.get("baseline_checkpoint_kind") == "final"
    if policy == "same_checkpoint_kind":
        return row.get("wca_checkpoint_kind") == row.get("baseline_checkpoint_kind")
    return True


def build_claim_level_summary(paired: list[dict[str, Any]], consistency_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    consistency_summary = _consistency_by_report(consistency_rows)
    rows: list[dict[str, Any]] = []
    for scope in CLAIM_SCOPES:
        scoped = []
        matched = []
        for row in paired:
            if row.get("report_id") != scope["report_id"]:
                continue
            if not _matches_any(str(row.get("wca_model", "")), list(scope["wca_model_contains"])):
                continue
            if not _matches_any(str(row.get("baseline_model", "")), list(scope["baseline_model_contains"])):
                continue
            condition_needles = list(scope.get("condition_contains", []))
            if condition_needles and not _matches_any(str(row.get("condition", "")), condition_needles):
                continue
            condition_excludes = list(scope.get("condition_excludes", []))
            if condition_excludes and _matches_any(str(row.get("condition", "")), condition_excludes):
                continue
            expected_checkpoint = scope.get("wca_checkpoint_kind")
            if expected_checkpoint and row.get("wca_checkpoint_kind") != expected_checkpoint:
                continue
            if not _checkpoint_policy_match(scope, row):
                continue
            scoped.append(row)
            if not row.get("seed_replicate_match"):
                continue
            matched.append(row)

        total = len(matched)
        wca_mean = sum(1 for row in matched if row.get("wca_better_by_mean"))
        wca_ci = sum(1 for row in matched if row.get("ci_entirely_wca_better"))
        baseline_ci = sum(1 for row in matched if row.get("ci_entirely_baseline_better"))
        ambiguous = total - wca_ci - baseline_ci
        sample_counts = [
            int(row["paired_sample_count"])
            for row in matched
            if str(row.get("paired_sample_count", "")).isdigit()
        ]
        conditions = sorted({str(row.get("condition", "")) for row in matched})
        baseline_models = sorted({str(row.get("baseline_model", "")) for row in matched})
        wca_models = sorted({str(row.get("wca_model", "")) for row in matched})
        report_consistency = consistency_summary.get(str(scope["report_id"]), {})
        consistency_status = str(report_consistency.get("metric_consistency_status", "unknown"))
        scope_rank_decision_allowed = bool(scope.get("rank_decision_allowed", True))
        rank_decision_allowed = scope_rank_decision_allowed and consistency_status != "fail"
        if not rank_decision_allowed:
            if consistency_status == "fail":
                decision = "blocked_metric_authority_no_rank_decision"
            else:
                decision = str(scope.get("decision_override", "metric_space_mismatch_no_rank_decision"))
        elif not matched:
            decision = "invalid_no_matching_pairs"
        elif wca_ci / total >= 0.75:
            decision = "sample_level_supports_wca_advantage"
        elif baseline_ci / total >= 0.75:
            decision = "sample_level_adverse_for_wca"
        elif wca_mean / total >= 0.75 and baseline_ci == 0:
            decision = "directional_support_inconclusive_ci"
        else:
            decision = "mixed_or_scope_limited"

        default_manuscript_status = (
            "blocked_metric_authority_mismatch" if consistency_status == "fail" else "usable_with_caveats"
        )
        manuscript_status = scope.get("manuscript_status_override", default_manuscript_status)

        paired_comparison_rows: int | str = total
        context_pair_rows: int | str = ""
        wca_better_by_mean_rows: int | str = wca_mean
        ci_entirely_wca_better_rows: int | str = wca_ci
        ci_entirely_baseline_better_rows: int | str = baseline_ci
        ambiguous_ci_rows: int | str = ambiguous
        if not rank_decision_allowed:
            context_pair_rows = total
            paired_comparison_rows = ""
            wca_better_by_mean_rows = ""
            ci_entirely_wca_better_rows = ""
            ci_entirely_baseline_better_rows = ""
            ambiguous_ci_rows = ""

        rows.append(
            {
                "claim_id": scope["claim_id"],
                "report_id": scope["report_id"],
                "comparison_scope": scope["comparison_scope"],
                "paper_use": scope["paper_use"],
                "metric_space_contract": scope.get("metric_space_contract", ""),
                "claim_mode": scope.get("claim_mode", ""),
                "comparison_family": scope.get("comparison_family", ""),
                "decision": decision,
                "manuscript_status": manuscript_status,
                "metric_consistency_status": consistency_status,
                "metric_consistency_mismatch_count": report_consistency.get("metric_consistency_mismatch_count", ""),
                "metric_consistency_max_rel_delta": report_consistency.get("metric_consistency_max_rel_delta", ""),
                "checkpoint_policy": scope.get("checkpoint_policy", ""),
                "scoped_comparison_rows_before_seed_filter": len(scoped),
                "unmatched_seed_replicate_rows_excluded": len(scoped) - total,
                "rank_decision_allowed": str(rank_decision_allowed).lower(),
                "paired_comparison_rows": paired_comparison_rows,
                "context_pair_rows": context_pair_rows,
                "wca_better_by_mean_rows": wca_better_by_mean_rows,
                "ci_entirely_wca_better_rows": ci_entirely_wca_better_rows,
                "ci_entirely_baseline_better_rows": ci_entirely_baseline_better_rows,
                "ambiguous_ci_rows": ambiguous_ci_rows,
                "min_paired_samples_per_comparison": min(sample_counts) if sample_counts else "",
                "max_paired_samples_per_comparison": max(sample_counts) if sample_counts else "",
                "conditions": "; ".join(conditions),
                "seed_policy": "same_replicate_id_suffix",
                "wca_models": "; ".join(wca_models),
                "baseline_models": "; ".join(baseline_models),
                "interpretation_guardrail": scope["interpretation_guardrail"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper statistical tables from audited WCA report artifacts.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root")
    parser.add_argument("--output-dir", type=Path, default=Path("paper/tables"), help="Output directory")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument(
        "--strict-source-consistency",
        action="store_true",
        help="Exit nonzero if per-sample MSE means do not match source report eval_mse rows.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir

    metric_consistency = build_metric_consistency(root)
    consistency_summary = _consistency_by_report(metric_consistency)
    if args.strict_source_consistency:
        failed = {
            report_id: summary
            for report_id, summary in consistency_summary.items()
            if summary.get("metric_consistency_status") != "pass"
        }
        if failed:
            details = "; ".join(
                f"{report_id}: mismatches={summary.get('metric_consistency_mismatch_count')} "
                f"max_rel={_fmt(summary.get('metric_consistency_max_rel_delta'))}"
                for report_id, summary in sorted(failed.items())
            )
            raise SystemExit(
                "Source metric consistency check failed. "
                "Do not use generated horizon statistics as manuscript-ready until metric authority is resolved. "
                + details
            )
    inventory = build_inventory(root, metric_consistency)
    condition_summary = build_condition_summary(root, iterations=args.bootstrap_iterations)
    paired = build_paired_bootstrap(root, iterations=args.bootstrap_iterations)
    claim_summary = build_claim_level_summary(paired, metric_consistency)

    _write_csv(
        output_dir / "source_metric_consistency_audit.csv",
        metric_consistency,
        [
            "report_id",
            "condition",
            "model",
            "checkpoint_kind",
            "seed",
            "source_run_dir",
            "sample_count",
            "per_sample_mean_mse",
            "result_eval_mse",
            "abs_delta",
            "rel_delta",
            "consistency_status",
        ],
    )
    mismatches = [row for row in metric_consistency if row.get("consistency_status") != "match"]
    _write_md_table(
        output_dir / "source_metric_consistency_audit_mismatches.md",
        mismatches,
        [
            "report_id",
            "condition",
            "model",
            "checkpoint_kind",
            "seed",
            "sample_count",
            "per_sample_mean_mse",
            "result_eval_mse",
            "rel_delta",
            "consistency_status",
        ],
        title="Source Metric Consistency Audit Mismatches",
        limit=120,
    )

    _write_csv(
        output_dir / "statistical_artifact_inventory.csv",
        inventory,
        [
            "report_id",
            "evidence_status",
            "family",
            "report_dir",
            "results_file",
            "per_sample_file",
            "results_rows",
            "per_sample_rows",
            "model_count",
            "seed_count",
            "condition_count",
            "min_samples_per_run_condition",
            "max_samples_per_run_condition",
            "bootstrap_eligible",
            "metric_consistency_status",
            "metric_consistency_rows",
            "metric_consistency_mismatch_count",
            "metric_consistency_max_rel_delta",
            "metric_consistency_max_abs_delta",
            "models",
        ],
    )
    _write_md_table(
        output_dir / "statistical_artifact_inventory.md",
        inventory,
        [
            "report_id",
            "evidence_status",
            "family",
            "per_sample_rows",
            "model_count",
            "seed_count",
            "condition_count",
            "min_samples_per_run_condition",
            "max_samples_per_run_condition",
            "metric_consistency_status",
            "metric_consistency_mismatch_count",
            "metric_consistency_max_rel_delta",
        ],
        title="Statistical Artifact Inventory",
    )

    _write_csv(
        output_dir / "condition_mean_bootstrap_summary.csv",
        condition_summary,
        [
            "report_id",
            "condition",
            "model",
            "model_role",
            "checkpoint_kind",
            "seed",
            "source_run_dir",
            "sample_count",
            "mean_mse",
            "std_mse",
            "bootstrap_mean_mse_ci_low",
            "bootstrap_mean_mse_ci_high",
            "mean_mae",
            "wca_family",
        ],
    )
    _write_md_table(
        output_dir / "condition_mean_bootstrap_summary_preview.md",
        condition_summary,
        [
            "report_id",
            "condition",
            "model",
            "model_role",
            "checkpoint_kind",
            "seed",
            "sample_count",
            "mean_mse",
            "bootstrap_mean_mse_ci_low",
            "bootstrap_mean_mse_ci_high",
        ],
        title="Condition Mean Bootstrap Summary Preview",
        limit=80,
    )

    _write_csv(
        output_dir / "paired_bootstrap_delta_summary.csv",
        paired,
        [
            "report_id",
            "condition",
            "wca_model",
            "wca_checkpoint_kind",
            "wca_seed",
            "wca_source_run_dir",
            "baseline_model",
            "baseline_checkpoint_kind",
            "baseline_seed",
            "baseline_source_run_dir",
            "seed_replicate_match",
            "paired_sample_count",
            "mean_mse_delta_wca_minus_baseline",
            "bootstrap_delta_ci_low",
            "bootstrap_delta_ci_high",
            "wca_better_by_mean",
            "ci_entirely_wca_better",
            "ci_entirely_baseline_better",
        ],
    )
    key_pairs = [
        row
        for row in paired
        if row.get("ci_entirely_wca_better") or row.get("ci_entirely_baseline_better")
    ]
    _write_md_table(
        output_dir / "paired_bootstrap_delta_key_findings.md",
        key_pairs,
        [
            "report_id",
            "condition",
            "wca_model",
            "wca_checkpoint_kind",
            "wca_seed",
            "wca_source_run_dir",
            "baseline_model",
            "baseline_checkpoint_kind",
            "baseline_seed",
            "baseline_source_run_dir",
            "seed_replicate_match",
            "paired_sample_count",
            "mean_mse_delta_wca_minus_baseline",
            "bootstrap_delta_ci_low",
            "bootstrap_delta_ci_high",
            "ci_entirely_wca_better",
            "ci_entirely_baseline_better",
        ],
        title="Paired Bootstrap Delta Key Findings",
        limit=120,
    )

    _write_csv(
        output_dir / "claim_level_statistical_summary.csv",
        claim_summary,
        [
            "claim_id",
            "report_id",
            "comparison_scope",
            "paper_use",
            "metric_space_contract",
            "claim_mode",
            "comparison_family",
            "decision",
            "manuscript_status",
            "metric_consistency_status",
            "metric_consistency_mismatch_count",
            "metric_consistency_max_rel_delta",
            "checkpoint_policy",
            "scoped_comparison_rows_before_seed_filter",
            "unmatched_seed_replicate_rows_excluded",
            "rank_decision_allowed",
            "paired_comparison_rows",
            "context_pair_rows",
            "wca_better_by_mean_rows",
            "ci_entirely_wca_better_rows",
            "ci_entirely_baseline_better_rows",
            "ambiguous_ci_rows",
            "min_paired_samples_per_comparison",
            "max_paired_samples_per_comparison",
            "conditions",
            "seed_policy",
            "wca_models",
            "baseline_models",
            "interpretation_guardrail",
        ],
    )
    _write_md_table(
        output_dir / "claim_level_statistical_summary.md",
        claim_summary,
        [
            "claim_id",
            "paper_use",
            "metric_space_contract",
            "claim_mode",
            "decision",
            "manuscript_status",
            "metric_consistency_status",
            "checkpoint_policy",
            "rank_decision_allowed",
            "paired_comparison_rows",
            "context_pair_rows",
            "wca_better_by_mean_rows",
            "ci_entirely_wca_better_rows",
            "ci_entirely_baseline_better_rows",
            "conditions",
            "seed_policy",
            "interpretation_guardrail",
        ],
        title="Claim-Level Statistical Summary",
    )

    print(f"wrote {output_dir / 'statistical_artifact_inventory.csv'}")
    print(f"wrote {output_dir / 'source_metric_consistency_audit.csv'}")
    print(f"wrote {output_dir / 'condition_mean_bootstrap_summary.csv'}")
    print(f"wrote {output_dir / 'paired_bootstrap_delta_summary.csv'}")
    print(f"wrote {output_dir / 'claim_level_statistical_summary.csv'}")


if __name__ == "__main__":
    main()
