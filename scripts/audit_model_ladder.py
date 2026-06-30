#!/usr/bin/env python3
"""Audit model size, run cost, and horizon metrics for a control manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from wca.config import Config, load_config  # noqa: E402
from wca.models.field_wca import FieldTokenizerWCA  # noqa: E402
from wca.models.rws_nca import FullRecursiveWorldStateNCA  # noqa: E402
from scripts.train_field_baseline import ConvFieldNet, TinyFNO2d, TinyUNet2d  # noqa: E402


HORIZONS = (1, 2, 4, 8)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return output


def _safe_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _format(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.8g}"
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format(row.get(key, "")) for key in fieldnames})


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _find_run_file(run_dir: Path, name: str) -> tuple[Path | None, list[Path]]:
    direct = run_dir / name
    if direct.exists():
        return direct, [direct]
    if not run_dir.exists():
        return None, []
    matches = sorted(run_dir.rglob(name))
    if not matches:
        return None, []
    return matches[-1], matches


def _config_from_model_entry(entry: dict[str, Any]) -> tuple[Config, str, list[str]]:
    warnings: list[str] = []
    run_dir_raw = str(entry.get("run_dir") or "")
    config_path: Path | None = None
    if run_dir_raw:
        found, matches = _find_run_file(ROOT / run_dir_raw, "config.json")
        if found is not None:
            config_path = found
            if len(matches) > 1:
                warnings.append(f"multiple config.json files under {run_dir_raw}; using {found.as_posix()}")
    if config_path is not None:
        payload = _read_json(config_path)
        allowed = set(Config.__dataclass_fields__)  # type: ignore[attr-defined]
        cfg = Config(**{key: value for key, value in payload.items() if key in allowed})
        return cfg, config_path.relative_to(ROOT).as_posix(), warnings

    config_path_raw = str(entry.get("config_path") or "")
    if not config_path_raw:
        raise ValueError(f"model entry {entry.get('id')} has neither run config nor config_path")
    cfg = load_config(ROOT / config_path_raw)
    return cfg, config_path_raw, warnings


def _count_parameters(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def _wca_model(cfg: Config) -> Any:
    if str(getattr(cfg, "field_tokenizer", "patch_mean")) != "patch_mean":
        return FieldTokenizerWCA(cfg)
    return FullRecursiveWorldStateNCA(
        n_nodes=int(cfg.n_nodes),
        hidden_dim=int(cfg.hidden_dim),
        edge_dim=int(cfg.edge_dim),
        inner_steps=int(cfg.inner_steps),
        pair_chunk_size=int(cfg.pair_chunk_size),
        output_dim=int(cfg.field_output_dim) if cfg.task == "field" else 1,
        activation_checkpoint_inner=bool(cfg.activation_checkpoint_inner),
    )


def _baseline_model(cfg: Config, model_kind: str) -> Any:
    condition_channels = 4 if bool(cfg.field_horizon_conditioning) else 0
    width = int(cfg.baseline_width or 64)
    depth = int(cfg.baseline_depth or 4)
    out_channels = int(cfg.field_output_dim)
    if model_kind == "convnet":
        return ConvFieldNet(out_channels, out_channels, width=width, depth=depth, condition_channels=condition_channels)
    if model_kind == "fno":
        modes = int(cfg.fno_modes or 12)
        return TinyFNO2d(
            out_channels,
            out_channels,
            width=width,
            modes=modes,
            depth=depth,
            condition_channels=condition_channels,
        )
    if model_kind == "unet":
        return TinyUNet2d(out_channels, out_channels, width=width, depth=depth, condition_channels=condition_channels)
    raise ValueError(f"unsupported baseline model: {model_kind}")


def _model_kind(cfg: Config, entry: dict[str, Any]) -> str:
    if cfg.baseline_model:
        return str(cfg.baseline_model)
    family = str(entry.get("family") or "").lower()
    if family in {"convnet", "fno", "unet"}:
        return family
    return "wca"


def _parameter_count(cfg: Config, entry: dict[str, Any]) -> tuple[str, int]:
    kind = _model_kind(cfg, entry)
    model = _wca_model(cfg) if kind == "wca" else _baseline_model(cfg, kind)
    return kind, _count_parameters(model)


def _parameter_breakdown(cfg: Config, kind: str) -> dict[str, int]:
    if kind != "wca":
        return {
            "field_tokenizer_params": 0,
            "wca_core_params": 0,
            "field_decoder_params": 0,
        }
    model = _wca_model(cfg)
    if isinstance(model, FieldTokenizerWCA):
        breakdown = model.parameter_breakdown()
        return {
            "field_tokenizer_params": int(breakdown.get("field_tokenizer_params", 0)),
            "wca_core_params": int(breakdown.get("wca_core_params", 0)),
            "field_decoder_params": int(breakdown.get("field_decoder_params", 0)),
        }
    return {
        "field_tokenizer_params": 0,
        "wca_core_params": _count_parameters(model),
        "field_decoder_params": 0,
    }


def _train_log_summary(run_dir: Path) -> tuple[dict[str, Any], list[str]]:
    path, matches = _find_run_file(run_dir, "train_log.csv")
    warnings: list[str] = []
    if path is None:
        return {}, warnings
    if len(matches) > 1:
        warnings.append(f"multiple train_log.csv files under {run_dir.relative_to(ROOT).as_posix()}; using {path.as_posix()}")
    rows = _load_csv(path)
    if not rows:
        return {"train_log_path": path.relative_to(ROOT).as_posix()}, warnings
    final = rows[-1]
    step_values = [_safe_float(row.get("step_seconds")) for row in rows]
    peak_alloc = max((_safe_float(row.get("cuda_peak_memory_allocated_mb")) for row in rows), default=float("nan"))
    peak_reserved = max((_safe_float(row.get("cuda_peak_memory_reserved_mb")) for row in rows), default=float("nan"))
    finite_steps = [value for value in step_values if not math.isnan(value)]
    mean_step = sum(finite_steps) / len(finite_steps) if finite_steps else float("nan")
    batch = _safe_float(final.get("field_patch_count"))
    return (
        {
            "train_log_path": path.relative_to(ROOT).as_posix(),
            "final_epoch": _safe_float(final.get("epoch")),
            "final_train_loss": _safe_float(final.get("loss")),
            "mean_step_seconds": mean_step,
            "final_step_seconds": _safe_float(final.get("step_seconds")),
            "peak_memory_allocated_mb": peak_alloc,
            "peak_memory_reserved_mb": peak_reserved,
            "final_eval_mse": _safe_float(final.get("eval_mse")),
            "final_checkpoint_score": _safe_float(final.get("eval_field_horizon_stratified_score")),
            "field_patch_count": batch,
        },
        warnings,
    )


def _run_matches_result(run_dir: str, source_run_dir: str) -> bool:
    run_dir = run_dir.strip("/")
    source_run_dir = source_run_dir.strip("/")
    return (
        source_run_dir == run_dir
        or source_run_dir.startswith(f"{run_dir}/")
        or run_dir.endswith(source_run_dir)
        or f"/{source_run_dir}/" in f"/{run_dir}/"
    )


def _horizon_metrics(run_dir: str, rows: list[dict[str, str]], checkpoint_kind: str = "final") -> dict[str, float]:
    metrics: dict[str, float] = {}
    matched = [
        row for row in rows
        if row.get("checkpoint_kind") == checkpoint_kind and _run_matches_result(run_dir, row.get("source_run_dir", ""))
    ]
    for horizon in HORIZONS:
        horizon_rows = [row for row in matched if _safe_int(row.get("horizon")) == horizon]
        if horizon_rows:
            row = horizon_rows[-1]
            metrics[f"h{horizon}_mse"] = _safe_float(row.get("eval_mse"))
            metrics[f"h{horizon}_relative_l2"] = _safe_float(row.get("eval_field_relative_l2"))
            metrics[f"h{horizon}_improvement_vs_persistence"] = _safe_float(
                row.get("eval_field_mse_improvement_vs_persistence")
            )
    if metrics:
        values = [metrics[f"h{h}_mse"] for h in HORIZONS if f"h{h}_mse" in metrics]
        metrics["mean_horizon_mse"] = sum(values) / len(values)
    return metrics


def _size_class(param_count: int, reference_params: int, kind: str) -> str:
    if kind == "wca":
        return "wca"
    if reference_params <= 0:
        return "unknown"
    ratio = param_count / reference_params
    if ratio < 0.8:
        return "smaller_than_wca"
    if ratio <= 1.25:
        return "near_wca"
    if ratio < 5.0:
        return "larger_than_wca"
    return "much_larger_than_wca"


def _write_markdown(path: Path, rows: list[dict[str, Any]], *, reference_params: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Model Ladder Audit",
        "",
        f"- reference_wca_params: `{reference_params}`",
        "- parameter counts are computed by live PyTorch model instantiation.",
        "- quality rows are copied from deterministic horizon-stratified evaluation when provided.",
        "",
        "| id | kind | role | params | ratio_to_wca | size_class | final_epoch | peak_mem_mb | h1_mse | h2_mse | h4_mse | h8_mse |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['id']}`",
                    f"`{row['model_kind']}`",
                    f"`{row.get('role', '')}`",
                    _format(row.get("parameter_count")),
                    _format(row.get("param_ratio_to_wca")),
                    f"`{row.get('size_class', '')}`",
                    _format(row.get("final_epoch")),
                    _format(row.get("peak_memory_allocated_mb")),
                    _format(row.get("h1_mse")),
                    _format(row.get("h2_mse")),
                    _format(row.get("h4_mse")),
                    _format(row.get("h8_mse")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "A WCA claim is stronger when the matched non-WCA baselines are standard-sized or larger than the WCA model. "
            "If a baseline is smaller than WCA, it should be treated as a weak/sanity baseline, not the main authority.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_audit(manifest_path: Path, horizon_results: Path | None) -> list[dict[str, Any]]:
    manifest = _read_json(manifest_path)
    results = _load_csv(horizon_results) if horizon_results else []
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for entry in manifest.get("model_matrix", []):
        if not isinstance(entry, dict):
            continue
        cfg, config_source, config_warnings = _config_from_model_entry(entry)
        warnings.extend(config_warnings)
        kind, parameter_count = _parameter_count(cfg, entry)
        parameter_breakdown = _parameter_breakdown(cfg, kind)
        run_dir = str(entry.get("run_dir") or "")
        train_summary, train_warnings = _train_log_summary(ROOT / run_dir) if run_dir else ({}, [])
        warnings.extend(train_warnings)
        row: dict[str, Any] = {
            "id": str(entry.get("id") or run_dir or config_source),
            "family": str(entry.get("family") or kind),
            "role": str(entry.get("role") or ""),
            "run_dir": run_dir,
            "config_source": config_source,
            "model_kind": kind,
            "parameter_count": parameter_count,
            "raw_grid_height": int(cfg.field_grid_height or cfg.field_grid_size),
            "raw_grid_width": int(cfg.field_grid_width or cfg.field_grid_size),
            "patch_height": int(cfg.field_patch_height or cfg.field_patch_size),
            "patch_width": int(cfg.field_patch_width or cfg.field_patch_size),
            "n_nodes": int(cfg.n_nodes),
            "hidden_dim": int(cfg.hidden_dim),
            "edge_dim": int(cfg.edge_dim),
            "outer_steps": int(cfg.outer_steps),
            "inner_steps": int(cfg.inner_steps),
            "baseline_width": int(cfg.baseline_width),
            "baseline_depth": int(cfg.baseline_depth),
            "fno_modes": int(cfg.fno_modes),
            "batch_size": int(cfg.batch_size),
            "grad_accum_steps": int(cfg.grad_accum_steps),
            "epochs": int(cfg.epochs),
            "lr": float(cfg.lr),
            "precision": str(cfg.precision),
            "field_target_steps_choices": str(cfg.field_target_steps_choices),
            "field_horizon_conditioning": bool(cfg.field_horizon_conditioning),
            "field_residual_readout": bool(cfg.field_residual_readout),
            "field_tokenizer": str(cfg.field_tokenizer),
            "field_token_dim": int(cfg.field_token_dim),
            "field_tokenizer_width": int(cfg.field_tokenizer_width),
            "field_decoder_width": int(cfg.field_decoder_width),
            "field_tokenizer_only": bool(cfg.field_tokenizer_only),
        }
        row.update(parameter_breakdown)
        row.update(train_summary)
        if run_dir:
            row.update(_horizon_metrics(run_dir, results, checkpoint_kind="final"))
        rows.append(row)

    wca_params = [int(row["parameter_count"]) for row in rows if row["model_kind"] == "wca"]
    reference_params = wca_params[0] if wca_params else 0
    for row in rows:
        row["reference_wca_params"] = reference_params
        row["param_ratio_to_wca"] = (
            float(row["parameter_count"]) / float(reference_params) if reference_params > 0 else float("nan")
        )
        row["size_class"] = _size_class(int(row["parameter_count"]), reference_params, str(row["model_kind"]))
        row["audit_warnings"] = "; ".join(warnings)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit model parameter/cost ladder for a WCA control manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--horizon-results", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = build_audit(args.manifest, args.horizon_results)
    fieldnames = [
        "id",
        "family",
        "role",
        "run_dir",
        "config_source",
        "model_kind",
        "parameter_count",
        "field_tokenizer_params",
        "wca_core_params",
        "field_decoder_params",
        "reference_wca_params",
        "param_ratio_to_wca",
        "size_class",
        "raw_grid_height",
        "raw_grid_width",
        "patch_height",
        "patch_width",
        "n_nodes",
        "hidden_dim",
        "edge_dim",
        "outer_steps",
        "inner_steps",
        "baseline_width",
        "baseline_depth",
        "fno_modes",
        "batch_size",
        "grad_accum_steps",
        "epochs",
        "lr",
        "precision",
        "field_target_steps_choices",
        "field_horizon_conditioning",
        "field_residual_readout",
        "field_tokenizer",
        "field_token_dim",
        "field_tokenizer_width",
        "field_decoder_width",
        "field_tokenizer_only",
        "final_epoch",
        "final_train_loss",
        "mean_step_seconds",
        "final_step_seconds",
        "peak_memory_allocated_mb",
        "peak_memory_reserved_mb",
        "final_eval_mse",
        "final_checkpoint_score",
        "h1_mse",
        "h2_mse",
        "h4_mse",
        "h8_mse",
        "mean_horizon_mse",
        "h1_improvement_vs_persistence",
        "h2_improvement_vs_persistence",
        "h4_improvement_vs_persistence",
        "h8_improvement_vs_persistence",
        "audit_warnings",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "model_ladder_audit.csv", rows, fieldnames)
    (args.output_dir / "model_ladder_audit.json").write_text(
        json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reference_params = int(rows[0].get("reference_wca_params", 0)) if rows else 0
    _write_markdown(args.output_dir / "model_ladder_audit.md", rows, reference_params=reference_params)
    print(json.dumps({"rows": len(rows), "output_dir": args.output_dir.as_posix()}, sort_keys=True))


if __name__ == "__main__":
    main()
