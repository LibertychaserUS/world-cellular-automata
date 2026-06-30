#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


REQUIRED_TABLES = (
    "claim_level_statistical_summary.csv",
    "condition_mean_bootstrap_summary.csv",
    "paired_bootstrap_delta_summary.csv",
    "statistical_artifact_inventory.csv",
    "source_metric_consistency_audit.csv",
)

TOL = {
    "blue": "#0077BB",
    "cyan": "#33BBEE",
    "teal": "#009988",
    "orange": "#EE7733",
    "red": "#CC3311",
    "magenta": "#EE3377",
    "grey": "#BBBBBB",
    "black": "#000000",
}

DELTA_MSE_SCALE = 1e5

V25D_FORMAL_ROLLOUT_CONDITIONS = (
    "mode=rollout_h8x2_piecewise;step=8;total=16",
)
V25D_CONTEXT_CONDITIONS = (
    "mode=direct_h8;step=8;total=8",
    "mode=rollout_h8x2_piecewise;step=8;total=16",
    "mode=rollout_h8x2_teacher_prev_piecewise;step=8;total=16",
)

RAW_FULL_FIELD_ANCHOR_MODELS = {"fno", "unet", "fno-field-baseline", "unet-field-baseline"}
V25D_TOKEN_MAIN_MODELS = (
    "mlp_stem-WCA",
    "mlp_stem-tokenizer-only",
    "mlp_stem-tokenizer-bypass-o0",
)
V25D_TOKEN_MAIN_BASELINES = (
    "mlp_stem-tokenizer-only",
    "mlp_stem-tokenizer-bypass-o0",
)

CONDITION_LABELS = {
    "mode=direct_h8;step=8;total=8": "direct h8",
    "mode=rollout_h8x2_piecewise;step=8;total=16": "h8x2 rollout",
    "mode=rollout_h8x2_teacher_prev_piecewise;step=8;total=16": "h8x2 teacher-prev",
    "mode=direct_h16_ood;step=8;total=16": "direct h16 OOD",
}

MODEL_LABELS = {
    "mlp_stem-WCA": "WCA",
    "fno": "FNO",
    "unet": "U-Net",
    "mlp_stem-tokenizer-only": "tokenizer-only",
    "mlp_stem-tokenizer-bypass-o0": "outer0",
    "FullRecursiveWorldStateNCA-heavy-dense": "PatchMean WCA",
    "fno-field-baseline": "FNO",
    "unet-field-baseline": "U-Net",
    "token_mlp-field-token-baseline": "token MLP",
    "token_conv-field-token-baseline": "token Conv",
}

MODEL_COLORS = {
    "mlp_stem-WCA": TOL["teal"],
    "fno": TOL["blue"],
    "unet": TOL["orange"],
    "mlp_stem-tokenizer-only": TOL["grey"],
    "mlp_stem-tokenizer-bypass-o0": TOL["magenta"],
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    if value in {"", "None"}:
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in {"", "None"}:
        return 0.0
    return float(value)


def _bool(row: dict[str, str], key: str) -> bool:
    return row.get(key, "").strip().lower() == "true"


def _short_claim(claim_id: str) -> str:
    mapping = {
        "C1_v25_establish_wca_vs_external_baselines": "V25\nWCA vs\nFNO/UNet",
        "C1b_v25_h8_final_directional_check": "V25 h8\nfinal only",
        "C2_v25e_attribute_wca_core_vs_token_controls": "V25e\ncore vs\ncontrols",
        "C3_v25d_h8x2_piecewise_rollout_token_controls": "V25d\nh8x2\nrollout",
        "C3a_v25d_direct_h8_token_control_diagnostic": "V25d\nh8\ndiagnostic",
        "C3b_v25d_teacher_prev_h8x2_token_control_diagnostic": "V25d\nteacher-prev\ndiagnostic",
        "C3c_v25d_h8x2_raw_anchor_context": "V25d\nraw anchor\ncontext",
        "C4_v27_n256_patchmean_adverse_scaling": "V27 N256\nPatchMean",
    }
    return mapping.get(claim_id, claim_id)


def _short_decision(row: dict[str, str]) -> str:
    if row.get("manuscript_status") == "blocked_metric_authority_mismatch":
        return "blocked: metric authority"
    decision = row.get("decision", "")
    mapping = {
        "sample_level_supports_wca_advantage": "supports WCA",
        "sample_level_adverse_for_wca": "adverse for WCA",
        "directional_support_inconclusive_ci": "directional",
        "mixed_or_scope_limited": "mixed",
        "metric_space_mismatch_no_rank_decision": "context only: no rank decision",
        "invalid_no_matching_pairs": "invalid",
    }
    return mapping.get(decision, decision.replace("_", " "))


def _short_report(report_id: str) -> str:
    mapping = {
        "v25_recursion_depth_ladder": "V25\nrecursion",
        "v25e_r3_attribution": "V25e\nattribution",
        "v25d_h8x2_rollout": "V25d\nrollout",
        "v27_n256_patchmean_diagnostic": "V27\nN256",
    }
    return mapping.get(report_id, report_id)


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def _condition_label(condition: str) -> str:
    return CONDITION_LABELS.get(condition, condition)


def _raw_full_field_anchors_present(rows: list[dict[str, str]], key: str) -> list[str]:
    return sorted({row.get(key, "") for row in rows if row.get(key, "") in RAW_FULL_FIELD_ANCHOR_MODELS})


def _validate_source_tables(root: Path) -> list[dict[str, object]]:
    tables_dir = root / "paper/tables"
    manifest_rows: list[dict[str, object]] = []
    for name in REQUIRED_TABLES:
        path = tables_dir / name
        if not path.exists():
            raise FileNotFoundError(f"required generated paper table is missing: {path}")
        rows = _read_csv(path)
        if not rows:
            raise ValueError(f"required generated paper table is empty: {path}")
        manifest_rows.append(
            {
                "path": str(path),
                "rows": len(rows),
                "sha256": _sha256(path),
            }
        )
    return manifest_rows


def _claim_rows_for_mode(root: Path, figure_mode: str) -> list[dict[str, str]]:
    table = root / "paper/tables/claim_level_statistical_summary.csv"
    rows = _read_csv(table)
    if not rows:
        raise ValueError(f"claim-level table is empty: {table}")

    for row in rows:
        claim_id = row.get("claim_id", "<missing claim_id>")
        metric_status = row.get("metric_consistency_status")
        manuscript_status = row.get("manuscript_status")
        if metric_status != "pass" and manuscript_status != "blocked_metric_authority_mismatch":
            raise ValueError(
                f"claim {claim_id} has metric_consistency_status={metric_status!r} "
                f"but manuscript_status={manuscript_status!r}; blocked data must be explicitly blocked"
            )

    if figure_mode == "audit":
        return rows

    usable = [
        row
        for row in rows
        if row.get("manuscript_status") == "usable_with_caveats"
        and row.get("metric_consistency_status") == "pass"
    ]
    if not usable:
        raise ValueError("no claim rows are eligible for a main-result figure")
    blocked = [row["claim_id"] for row in rows if row not in usable]
    if blocked and figure_mode == "main-result":
        # Main-result figures are allowed to exclude blocked rows, but they may
        # never silently include them.
        pass
    return usable


def _check_figure_file(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"figure was not written: {path}")
    size_bytes = path.stat().st_size
    if size_bytes <= 0:
        raise ValueError(f"figure is empty: {path}")

    if path.suffix.lower() == ".png":
        image = mpimg.imread(path)
        if image.size == 0:
            raise ValueError(f"PNG figure has no pixels: {path}")
        height, width = image.shape[:2]
        if width < 100 or height < 100:
            raise ValueError(f"PNG figure is unexpectedly small: {path} ({width}x{height})")
        pixel_range = float(image.max() - image.min())
        if pixel_range <= 1e-6:
            raise ValueError(f"PNG figure appears blank: {path}")
        return {
            "path": str(path),
            "format": "png",
            "size_bytes": size_bytes,
            "width": int(width),
            "height": int(height),
            "pixel_range": pixel_range,
            "sha256": _sha256(path),
            "check_status": "pass",
        }

    if path.suffix.lower() == ".svg":
        text = path.read_text(encoding="utf-8", errors="replace")
        if "<svg" not in text:
            raise ValueError(f"SVG figure does not contain an <svg> element: {path}")
        return {
            "path": str(path),
            "format": "svg",
            "size_bytes": size_bytes,
            "sha256": _sha256(path),
            "check_status": "pass",
        }

    raise ValueError(f"unsupported figure format: {path}")


def build_claim_level_strength(root: Path, output_dir: Path, figure_mode: str) -> dict[str, object]:
    rows = _claim_rows_for_mode(root, figure_mode)

    labels = [_short_claim(row["claim_id"]) for row in rows]
    wca = [_int(row, "ci_entirely_wca_better_rows") for row in rows]
    baseline = [_int(row, "ci_entirely_baseline_better_rows") for row in rows]
    total = [_int(row, "paired_comparison_rows") for row in rows]
    ambiguous = [max(t - a - b, 0) for t, a, b in zip(total, wca, baseline)]

    y = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.barh(y, wca, color="#2f855a", label="CI supports WCA")
    ax.barh(y, ambiguous, left=wca, color="#a0aec0", label="CI ambiguous")
    ax.barh(
        y,
        baseline,
        left=[a + b for a, b in zip(wca, ambiguous)],
        color="#c53030",
        label="CI supports baseline/control",
    )

    for i, row in enumerate(rows):
        ax.text(
            total[i] + max(total) * 0.01,
            i,
            _short_decision(row),
            va="center",
            fontsize=8.5,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Matched-replicate paired comparison rows")
    title = "Claim-Level Statistical Evidence Strength"
    if figure_mode == "main-result":
        title = "Main-Result Statistical Evidence Strength"
    elif figure_mode == "audit":
        title = "Audit: Claim-Level Statistical Evidence Strength"
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "claim_level_statistical_strength"
    if figure_mode == "main-result":
        stem = "main_result_statistical_strength"
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return {
        "figure_id": stem,
        "figure_mode": figure_mode,
        "source": "paper/tables/claim_level_statistical_summary.csv",
        "included_claim_ids": [row["claim_id"] for row in rows],
        "included_rows": len(rows),
        "blocked_rows_included": sum(
            1 for row in rows if row.get("manuscript_status") == "blocked_metric_authority_mismatch"
        ),
        "main_result_ready": figure_mode == "main-result",
        "files": [_check_figure_file(png), _check_figure_file(svg)],
    }


def build_metric_consistency_audit(root: Path, output_dir: Path) -> dict[str, object]:
    rows = _read_csv(root / "paper/tables/statistical_artifact_inventory.csv")
    report_rows = [
        row
        for row in rows
        if row.get("family") in {"horizon", "rollout"}
        and row.get("metric_consistency_rows") not in {"", "0", None}
    ]
    if not report_rows:
        raise ValueError("no metric-consistency inventory rows available")

    labels = [_short_report(row["report_id"]) for row in report_rows]
    mismatch_counts = [_int(row, "metric_consistency_mismatch_count") for row in report_rows]
    total_counts = [_int(row, "metric_consistency_rows") for row in report_rows]
    mismatch_rates = [
        (mismatch / total) if total else 0.0 for mismatch, total in zip(mismatch_counts, total_counts)
    ]
    max_rel = [_float(row, "metric_consistency_max_rel_delta") for row in report_rows]
    colors = [TOL["red"] if row.get("metric_consistency_status") == "fail" else TOL["teal"] for row in report_rows]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(labels, mismatch_rates, color=colors, edgecolor=TOL["black"], linewidth=0.4)
    ax.set_ylabel("Mismatch rate")
    ax.set_xlabel("Report")
    ax.set_ylim(0, min(max(mismatch_rates + [0.05]) * 1.08, 1.08))
    ax.set_title("Source Metric Consistency Audit")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    for bar, mismatches, total, rel in zip(bars, mismatch_counts, total_counts, max_rel):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(mismatch_rates + [0.05]) * 0.025,
            f"{mismatches}/{total}\nmax rel {rel:.3g}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "source_metric_consistency_audit.png"
    svg = output_dir / "source_metric_consistency_audit.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return {
        "figure_id": "source_metric_consistency_audit",
        "figure_mode": "audit",
        "source": "paper/tables/statistical_artifact_inventory.csv",
        "included_report_ids": [row["report_id"] for row in report_rows],
        "included_rows": len(report_rows),
        "blocked_rows_included": sum(1 for row in report_rows if row.get("metric_consistency_status") == "fail"),
        "main_result_ready": False,
        "files": [_check_figure_file(png), _check_figure_file(svg)],
    }


def build_v25d_rollout_mean_mse(root: Path, output_dir: Path) -> dict[str, object]:
    candidate_rows = [
        row
        for row in _read_csv(root / "paper/tables/condition_mean_bootstrap_summary.csv")
        if row.get("report_id") == "v25d_h8x2_rollout"
        and row.get("checkpoint_kind") == "final"
        and row.get("condition") in V25D_CONTEXT_CONDITIONS
    ]
    excluded_raw_anchors = _raw_full_field_anchors_present(candidate_rows, "model")
    rows = [
        row
        for row in candidate_rows
        if row.get("condition") in V25D_FORMAL_ROLLOUT_CONDITIONS
        and row.get("model") in V25D_TOKEN_MAIN_MODELS
    ]
    if not rows:
        raise ValueError("no V25d final rollout rows available")

    models = list(V25D_TOKEN_MAIN_MODELS)
    conditions = list(V25D_FORMAL_ROLLOUT_CONDITIONS)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["condition"], row["model"]), []).append(row)

    fig, axes = plt.subplots(1, len(conditions), figsize=(13.8, 4.8), sharey=True)
    if len(conditions) == 1:
        axes = [axes]
    y_positions = list(range(len(models)))
    for ax, condition in zip(axes, conditions):
        means: list[float] = []
        low: list[float] = []
        high: list[float] = []
        for model in models:
            group = grouped.get((condition, model), [])
            if not group:
                means.append(float("nan"))
                low.append(float("nan"))
                high.append(float("nan"))
                continue
            values = [_float(row, "mean_mse") for row in group]
            lows = [_float(row, "bootstrap_mean_mse_ci_low") for row in group]
            highs = [_float(row, "bootstrap_mean_mse_ci_high") for row in group]
            means.append(sum(values) / len(values))
            low.append(min(lows))
            high.append(max(highs))

        colors = [MODEL_COLORS.get(model, TOL["grey"]) for model in models]
        ax.barh(y_positions, means, color=colors, edgecolor=TOL["black"], linewidth=0.4)
        for ypos, mean, lo, hi in zip(y_positions, means, low, high):
            ax.plot([lo, hi], [ypos, ypos], color=TOL["black"], linewidth=1.0)
            ax.plot(mean, ypos, marker="o", markersize=3.0, color=TOL["black"])
        ax.set_xscale("log")
        ax.set_title(_condition_label(condition))
        ax.set_xlabel("Mean MSE (log)")
        ax.grid(axis="x", alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels([_model_label(model) for model in models])
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    fig.suptitle("V25d Main Result: Token-Level h8x2 Piecewise Rollout Mean MSE", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "v25d_rollout_mean_mse.png"
    svg = output_dir / "v25d_rollout_mean_mse.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return {
        "figure_id": "v25d_rollout_mean_mse",
        "figure_mode": "main-result",
        "source": "paper/tables/condition_mean_bootstrap_summary.csv",
        "included_report_ids": ["v25d_h8x2_rollout"],
        "included_conditions": conditions,
        "included_model_families": models,
        "included_rows": len(rows),
        "excluded_raw_full_field_anchor_models": excluded_raw_anchors,
        "metric_space_contract": "token_level_controls_only",
        "blocked_rows_included": 0,
        "main_result_ready": True,
        "files": [_check_figure_file(png), _check_figure_file(svg)],
    }


def build_v25d_rollout_paired_delta(root: Path, output_dir: Path) -> dict[str, object]:
    candidate_rows = [
        row
        for row in _read_csv(root / "paper/tables/paired_bootstrap_delta_summary.csv")
        if row.get("report_id") == "v25d_h8x2_rollout"
        and row.get("wca_checkpoint_kind") == "final"
        and row.get("baseline_checkpoint_kind") == "final"
        and _bool(row, "seed_replicate_match")
        and row.get("condition") in V25D_CONTEXT_CONDITIONS
    ]
    excluded_raw_anchors = _raw_full_field_anchors_present(candidate_rows, "baseline_model")
    rows = [
        row
        for row in candidate_rows
        if row.get("condition") in V25D_FORMAL_ROLLOUT_CONDITIONS
        and row.get("baseline_model") in V25D_TOKEN_MAIN_BASELINES
    ]
    if not rows:
        raise ValueError("no V25d paired-delta rows available")

    baselines = list(V25D_TOKEN_MAIN_BASELINES)
    conditions = list(V25D_FORMAL_ROLLOUT_CONDITIONS)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["condition"], row["baseline_model"]), []).append(row)

    fig, axes = plt.subplots(1, len(conditions), figsize=(13.8, 4.8), sharey=True)
    if len(conditions) == 1:
        axes = [axes]
    y_positions = list(range(len(baselines)))
    for ax, condition in zip(axes, conditions):
        for ypos, baseline in zip(y_positions, baselines):
            group = grouped.get((condition, baseline), [])
            if not group:
                continue
            deltas = [_float(row, "mean_mse_delta_wca_minus_baseline") for row in group]
            lows = [_float(row, "bootstrap_delta_ci_low") for row in group]
            highs = [_float(row, "bootstrap_delta_ci_high") for row in group]
            mean_delta = (sum(deltas) / len(deltas)) * DELTA_MSE_SCALE
            lo = min(lows) * DELTA_MSE_SCALE
            hi = max(highs) * DELTA_MSE_SCALE
            color = TOL["teal"] if hi < 0 else TOL["red"] if lo > 0 else TOL["grey"]
            ax.plot([lo, hi], [ypos, ypos], color=color, linewidth=2.0)
            ax.plot(mean_delta, ypos, marker="o", markersize=5.0, color=color)
        ax.axvline(0, color=TOL["black"], linewidth=0.8, linestyle="--")
        ax.set_title(_condition_label(condition))
        ax.set_xlabel("Delta MSE x 1e5: WCA - baseline")
        ax.grid(axis="x", alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels([_model_label(model) for model in baselines])
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    fig.suptitle("V25d Main Result: Token-Level h8x2 Piecewise Rollout Paired Deltas", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "v25d_rollout_paired_delta.png"
    svg = output_dir / "v25d_rollout_paired_delta.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return {
        "figure_id": "v25d_rollout_paired_delta",
        "figure_mode": "main-result",
        "source": "paper/tables/paired_bootstrap_delta_summary.csv",
        "included_report_ids": ["v25d_h8x2_rollout"],
        "included_conditions": conditions,
        "included_model_families": ["mlp_stem-WCA", *baselines],
        "included_baseline_model_families": baselines,
        "included_rows": len(rows),
        "excluded_raw_full_field_anchor_models": excluded_raw_anchors,
        "metric_space_contract": "token_level_controls_only",
        "blocked_rows_included": 0,
        "main_result_ready": True,
        "files": [_check_figure_file(png), _check_figure_file(svg)],
    }


def build_figures(root: Path, output_dir: Path, figure_mode: str) -> dict[str, object]:
    table_manifest = _validate_source_tables(root)
    figures: list[dict[str, object]] = []
    if figure_mode in {"audit", "all"}:
        figures.append(build_claim_level_strength(root, output_dir, "audit"))
        figures.append(build_metric_consistency_audit(root, output_dir))
    if figure_mode in {"main-result", "all"}:
        figures.append(build_claim_level_strength(root, output_dir, "main-result"))
        figures.append(build_v25d_rollout_mean_mse(root, output_dir))
        figures.append(build_v25d_rollout_paired_delta(root, output_dir))
    manifest = {
        "schema": "wca.paper.figure_manifest.v1",
        "source_policy": "quantitative figures read generated paper/tables only; raw artifacts feed table scripts, not figures",
        "source_tables": table_manifest,
        "figures": figures,
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build WCA paper figures from generated statistical tables.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument(
        "--figure-mode",
        choices=["audit", "main-result", "all"],
        default="all",
        help="audit may show blocked rows; main-result includes only source-consistent manuscript-usable rows",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = build_figures(root, output_dir, args.figure_mode)
    for figure in manifest["figures"]:
        for file_info in figure["files"]:
            print(f"wrote {file_info['path']} ({file_info['check_status']})")
    print(f"wrote {manifest['manifest_path']}")


if __name__ == "__main__":
    main()
