#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from wca.config import Config
from wca.data.field.real_cache import make_real_field_batch
from wca.data.field.synthetic import field_token_shape, parse_field_target_steps_choices
from wca.utils.device import resolve_device
from wca.utils.precision import autocast_context
from wca.utils.seed import set_seed

from scripts import eval_weatherbench_baseline_horizons as baseline_eval
from scripts import eval_weatherbench_horizons as wca_eval
from scripts.field_eval_plan import eval_plan_hash, fixed_eval_plan_for_horizons, write_fixed_eval_plan
from scripts.field_eval_plan import attach_fixed_eval_plan, horizon_eval_seed, start_indices_hash
from scripts.train_field_token_baseline import build_model as build_token_model
from scripts.train_field_token_baseline import evaluate as evaluate_token_model
from scripts.train_field_token_baseline import input_dim_for_config, predict_tokens as predict_token_baseline


RAW_FIELD_EXTERNAL_BASELINES = {"convnet", "fno", "unet"}
TOKEN_EQUIVALENT_BASELINES = {"token_mlp", "token_conv"}

BASE_FIELDS = [
    "source_run_dir",
    "checkpoint_kind",
    "checkpoint_path",
    "checkpoint_legacy_model_pt",
    "horizon",
    "eval_plan_seed",
    "eval_plan_hash",
    "eval_horizon_seed",
    "eval_start_indices_hash",
    "eval_sample_count",
    "model",
    "seed",
    "hidden_dim",
    "edge_dim",
    "inner_steps",
    "outer_steps",
    "baseline_model",
    "baseline_width",
    "baseline_depth",
    "fno_modes",
    "field_horizon_conditioning",
    "field_tendency_baseline",
    "field_tendency_scale",
    "field_residual_scale",
    "field_tokenizer",
    "field_token_dim",
    "field_tokenizer_width",
    "field_decoder_width",
    "field_tokenizer_only",
    "field_baseline_scope",
]
CSV_FIELDS = BASE_FIELDS + wca_eval.METRIC_FIELDS
PER_SAMPLE_FIELDS = wca_eval.PER_SAMPLE_FIELDS


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.7g}"
    return str(value)


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _validate_matched_persistence(rows: list[dict[str, Any]], *, rel_tol: float = 1e-6, abs_tol: float = 1e-12) -> None:
    by_horizon: dict[int, list[tuple[str, str, float]]] = {}
    for row in rows:
        horizon = int(row["horizon"])
        value = _float_value(row.get("eval_field_persistence_mse"))
        if math.isnan(value) or math.isinf(value):
            continue
        by_horizon.setdefault(horizon, []).append((str(row.get("source_run_dir", "")), str(row.get("checkpoint_kind", "")), value))

    mismatches: list[str] = []
    for horizon, items in sorted(by_horizon.items()):
        if len(items) <= 1:
            continue
        reference_run, reference_ckpt, reference = items[0]
        for run, ckpt, value in items[1:]:
            tolerance = max(abs_tol, abs(reference) * rel_tol)
            if abs(value - reference) > tolerance:
                mismatches.append(
                    "h="
                    f"{horizon}: {run} ({ckpt}) persistence_mse={value:.12g} "
                    f"differs from {reference_run} ({reference_ckpt}) persistence_mse={reference:.12g}"
                )
                break
    if mismatches:
        raise SystemExit(
            "Horizon-stratified evaluation is not a fair matched comparison: "
            "persistence baselines differ within the same horizon. "
            "Use a shared eval seed/index plan or pass --allow-mismatched-persistence for exploratory reports only.\n"
            + "\n".join(mismatches)
        )


def _validate_matched_eval_plan(rows: list[dict[str, Any]]) -> None:
    by_horizon: dict[int, set[tuple[str, str, str]]] = {}
    for row in rows:
        horizon = int(row["horizon"])
        plan_hash = str(row.get("eval_plan_hash", ""))
        start_hash = str(row.get("eval_start_indices_hash", ""))
        sample_count = str(row.get("eval_sample_count", ""))
        by_horizon.setdefault(horizon, set()).add((plan_hash, start_hash, sample_count))
    mismatches = [
        f"h={horizon}: {sorted(values)}"
        for horizon, values in sorted(by_horizon.items())
        if len(values) > 1
    ]
    if mismatches:
        raise SystemExit(
            "Horizon-stratified evaluation is not a fair matched comparison: "
            "eval plan hashes differ within the same horizon.\n"
            + "\n".join(mismatches)
        )


def _run_config_payload(run_dir: Path) -> dict[str, Any]:
    return _read_json(run_dir / "config.json")


def _strict_label_contract_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    grid_height = int(payload.get("field_grid_height") or payload.get("field_grid_size") or 0)
    grid_width = int(payload.get("field_grid_width") or payload.get("field_grid_size") or 0)
    patch_height = int(payload.get("field_patch_height") or payload.get("field_patch_size") or 0)
    patch_width = int(payload.get("field_patch_width") or payload.get("field_patch_size") or 0)
    token_height = grid_height // patch_height if patch_height > 0 else 0
    token_width = grid_width // patch_width if patch_width > 0 else 0
    return {
        "field_dataset": str(payload.get("field_dataset", "")),
        "field_data_path": str(payload.get("field_data_path", "")),
        "field_grid_height": grid_height,
        "field_grid_width": grid_width,
        "field_patch_height": patch_height,
        "field_patch_width": patch_width,
        "field_output_dim": int(payload.get("field_output_dim", 1) or 1),
        "field_input_steps": int(payload.get("field_input_steps", 1) or 1),
        "field_stride": int(payload.get("field_stride", 1) or 1),
        "field_train_start": int(payload.get("field_train_start", 0) or 0),
        "field_train_size": int(payload.get("field_train_size", 0) or 0),
        "field_eval_start": int(payload.get("field_eval_start", 0) or 0),
        "field_eval_size": int(payload.get("field_eval_size", 0) or 0),
        "field_val_start": int(payload.get("field_val_start", 0) or 0),
        "field_val_size": int(payload.get("field_val_size", 0) or 0),
        "field_test_start": int(payload.get("field_test_start", 0) or 0),
        "field_test_size": int(payload.get("field_test_size", 0) or 0),
        "token_height": token_height,
        "token_width": token_width,
        "n_nodes": token_height * token_width,
    }


def _validate_strict_run_contracts(run_dirs: list[Path]) -> None:
    contracts: list[tuple[Path, dict[str, Any]]] = [
        (run_dir, _strict_label_contract_from_payload(_run_config_payload(run_dir)))
        for run_dir in run_dirs
    ]
    if len(contracts) <= 1:
        return
    reference_run, reference = contracts[0]
    mismatches: list[str] = []
    for run_dir, contract in contracts[1:]:
        if contract != reference:
            changed = {
                key: {"reference": reference.get(key), "current": contract.get(key)}
                for key in sorted(set(reference) | set(contract))
                if reference.get(key) != contract.get(key)
            }
            mismatches.append(f"{run_dir.as_posix()} differs from {reference_run.as_posix()}: {json.dumps(changed, sort_keys=True)}")
    if mismatches:
        raise SystemExit(
            "Strict horizon-stratified evaluation requires one shared label/eval-token contract. "
            "Split runs with different grid/patch/token geometry, dataset path, split windows, output channels, "
            "input steps, or stride into separate formal tables; raw anchors with different contracts may only be "
            "reported as exploratory diagnostics.\n"
            + "\n".join(mismatches)
        )


def _strict_scope_kind_from_payload(payload: dict[str, Any]) -> str:
    baseline_model = str(payload.get("baseline_model", ""))
    baseline_scope = str(payload.get("field_baseline_scope", ""))
    if baseline_model in RAW_FIELD_EXTERNAL_BASELINES and baseline_scope != "token_equivalent":
        return "raw_field_external_anchor_not_token_equivalent"
    if baseline_model in TOKEN_EQUIVALENT_BASELINES or baseline_scope == "token_equivalent":
        return "token_equivalent"
    return "token_level_wca"


def _validate_strict_matched_run_scope(run_dirs: list[Path]) -> None:
    by_kind: dict[str, list[str]] = {}
    for run_dir in run_dirs:
        by_kind.setdefault(_strict_scope_kind_from_payload(_run_config_payload(run_dir)), []).append(run_dir.as_posix())
    raw = by_kind.get("raw_field_external_anchor_not_token_equivalent", [])
    formal_token = [
        path
        for kind, paths in by_kind.items()
        if kind != "raw_field_external_anchor_not_token_equivalent"
        for path in paths
    ]
    if raw and formal_token:
        raise SystemExit(
            "Strict horizon-stratified evaluation cannot mix token-level WCA/token baselines "
            "with raw-field external anchors in one matched table. "
            "Run raw-field anchors as exploratory (--allow-mismatched-persistence) or split them into a separate eval group.\n"
            f"raw_field_external_anchor_not_token_equivalent={raw[:5]}\n"
            f"formal_token_level={formal_token[:5]}"
        )


def _is_baseline_run(run_dir: Path) -> bool:
    if _is_token_baseline_run(run_dir):
        return False
    config_path = run_dir / "config.json"
    if config_path.exists():
        model = str(_read_json(config_path).get("baseline_model", ""))
        if model in {"convnet", "fno", "unet"}:
            return True
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        model = str(_read_json(summary_path).get("model", ""))
        if model.endswith("-field-baseline"):
            return True
    try:
        import torch

        checkpoint_path = run_dir / "best_model.pt"
        if not checkpoint_path.exists():
            checkpoint_path = run_dir / "model.pt"
        if checkpoint_path.exists():
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state_dict, dict):
                try:
                    baseline_eval.state_dict_model_kind(state_dict)
                    return True
                except ValueError:
                    return False
    except Exception:
        return False
    return False


def _is_token_baseline_run(run_dir: Path) -> bool:
    config_path = run_dir / "config.json"
    if config_path.exists():
        payload = _read_json(config_path)
        model = str(payload.get("baseline_model", ""))
        if model in {"token_mlp", "token_conv"}:
            return True
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        payload = _read_json(summary_path)
        model = str(payload.get("model", ""))
        baseline_model = str(payload.get("baseline_model", ""))
        if model.endswith("-field-token-baseline") or baseline_model in {"token_mlp", "token_conv"}:
            return True
    return False


def _discover_field_run_dirs(paths: list[Path]) -> list[Path]:
    run_dirs: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        for candidate in wca_eval.discover_run_dirs([path]):
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                run_dirs.append(candidate)
    return sorted(run_dirs, key=lambda item: str(item))


def _checkpoint_kinds(raw: str) -> list[str]:
    kinds = [item.strip() for item in raw.split(",") if item.strip()]
    if any(kind not in {"final", "best"} for kind in kinds):
        raise SystemExit("--checkpoint-kinds may only contain final,best")
    return kinds


def _evaluate_wca_run(
    run_dir: Path,
    horizons: list[int],
    kinds: list[str],
    *,
    eval_batches: int,
    device_name: str,
    eval_seed: int,
    eval_samples: int,
    eval_batch_size: int,
    per_sample_rows: list[dict[str, Any]] | None = None,
    field_split: str = "eval",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint_kind, checkpoint_path in wca_eval.checkpoint_paths(run_dir, kinds):
        for row in wca_eval.evaluate_run_checkpoint(
            run_dir,
            checkpoint_kind,
            checkpoint_path,
            horizons,
            eval_batches=eval_batches,
            device_name=device_name,
            eval_seed=eval_seed,
            eval_samples=eval_samples,
            eval_batch_size=eval_batch_size,
            per_sample_rows=per_sample_rows,
            field_split=field_split,
        ):
            row.setdefault("checkpoint_legacy_model_pt", False)
            row.setdefault("baseline_model", "")
            row.setdefault("baseline_width", "")
            row.setdefault("baseline_depth", "")
            row.setdefault("fno_modes", "")
            rows.append(row)
    return rows


def _evaluate_baseline_run(
    run_dir: Path,
    horizons: list[int],
    kinds: list[str],
    *,
    eval_batches: int,
    device_name: str,
    eval_seed: int,
    eval_samples: int,
    eval_batch_size: int,
    per_sample_rows: list[dict[str, Any]] | None = None,
    field_split: str = "eval",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint_kind, checkpoint_path, checkpoint_legacy_model_pt in baseline_eval.checkpoint_paths(run_dir, kinds):
        for row in baseline_eval.evaluate_run_checkpoint(
            run_dir,
            checkpoint_kind,
            checkpoint_path,
            checkpoint_legacy_model_pt,
            horizons,
            eval_batches=eval_batches,
            device_name=device_name,
            eval_seed=eval_seed,
            eval_samples=eval_samples,
            eval_batch_size=eval_batch_size,
            per_sample_rows=per_sample_rows,
            field_split=field_split,
        ):
            row.setdefault("hidden_dim", "")
            row.setdefault("edge_dim", "")
            row.setdefault("inner_steps", "")
            row.setdefault("outer_steps", "")
            rows.append(row)
    return rows


def _token_config_from_run_dir(run_dir: Path) -> Config:
    cfg = baseline_eval.config_from_run_dir(run_dir)
    cfg.field_baseline_scope = "token_equivalent"
    return cfg


def _token_checkpoint_paths(run_dir: Path, kinds: list[str]) -> list[tuple[str, Path, bool]]:
    return baseline_eval.checkpoint_paths(run_dir, kinds)


def _token_spec_from_config(cfg: Config) -> dict[str, int | str]:
    model = str(getattr(cfg, "baseline_model", ""))
    if model not in {"token_mlp", "token_conv"}:
        raise ValueError(f"Token baseline config must use token_mlp or token_conv, got {model!r}")
    return {
        "baseline_model": model,
        "baseline_width": int(getattr(cfg, "baseline_width", 0) or (128 if model == "token_mlp" else 64)),
        "baseline_depth": int(getattr(cfg, "baseline_depth", 0) or (3 if model == "token_mlp" else 4)),
        "fno_modes": int(getattr(cfg, "fno_modes", 0) or 0),
    }


def _build_token_baseline_model(cfg: Config, state_dict: dict[str, Tensor], device: torch.device) -> tuple[nn.Module, dict[str, int | str]]:
    spec = _token_spec_from_config(cfg)
    model = build_token_model(
        str(spec["baseline_model"]),
        input_dim_for_config(cfg),
        int(getattr(cfg, "field_output_dim", 1)),
        field_token_shape(cfg),
        int(spec["baseline_width"]),
        int(spec["baseline_depth"]),
    ).to(device)
    model.load_state_dict(state_dict)
    return model, spec


def _per_sample_errors(prediction: Tensor, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    label = batch["label"].to(device=prediction.device, dtype=prediction.dtype)
    errors = prediction - label
    mse = errors.pow(2).flatten(start_dim=1).mean(dim=1)
    mae = errors.abs().flatten(start_dim=1).mean(dim=1)
    persistence = batch.get("field_prediction_baseline")
    if isinstance(persistence, Tensor) and persistence.shape == label.shape:
        persistence = persistence.to(device=prediction.device, dtype=prediction.dtype)
        persistence_mse = (persistence - label).pow(2).flatten(start_dim=1).mean(dim=1)
        improvement = (persistence_mse - mse) / persistence_mse.clamp_min(1e-8)
    else:
        persistence_mse = torch.full_like(mse, float("nan"))
        improvement = torch.full_like(mse, float("nan"))
    return mse, mae, persistence_mse, improvement


def _fork_rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [device.index if device.index is not None else torch.cuda.current_device()]


@torch.no_grad()
def _append_token_per_sample_rows(
    rows: list[dict[str, Any]],
    *,
    model: nn.Module,
    cfg: Config,
    device: torch.device,
    run_dir: Path,
    checkpoint_kind: str,
    checkpoint_path: Path,
    checkpoint_legacy_model_pt: bool,
    horizon: int,
    plan_hash: str,
    start_hash: str,
    spec: dict[str, int | str],
    field_split: str = "eval",
) -> None:
    model.eval()
    sample_ordinal = 0
    for _ in range(cfg.eval_batches):
        batch = make_real_field_batch(cfg, device, split=field_split)
        with autocast_context(device, cfg.precision):
            prediction, _target = predict_token_baseline(model, batch, cfg)
        mse, mae, persistence_mse, improvement = _per_sample_errors(prediction, batch)
        starts = batch.get("field_start_index")
        targets = batch.get("field_target_index")
        trajectories = batch.get("field_trajectory_id")
        for index in range(int(mse.numel())):
            rows.append(
                {
                    "source_run_dir": run_dir.as_posix(),
                    "checkpoint_kind": checkpoint_kind,
                    "checkpoint_path": checkpoint_path.as_posix(),
                    "checkpoint_legacy_model_pt": checkpoint_legacy_model_pt,
                    "horizon": int(horizon),
                    "eval_plan_hash": plan_hash,
                    "eval_start_indices_hash": start_hash,
                    "model": f"{spec['baseline_model']}-field-token-baseline",
                    "seed": cfg.seed,
                    "sample_ordinal": sample_ordinal,
                    "start_index": int(starts[index].detach().cpu().item()) if isinstance(starts, Tensor) else "",
                    "target_index": int(targets[index].detach().cpu().item()) if isinstance(targets, Tensor) else "",
                    "trajectory_id": int(trajectories[index].detach().cpu().item()) if isinstance(trajectories, Tensor) else "",
                    "mse": float(mse[index].detach().cpu().item()),
                    "mae": float(mae[index].detach().cpu().item()),
                    "persistence_mse": float(persistence_mse[index].detach().cpu().item()),
                    "improvement_vs_persistence": float(improvement[index].detach().cpu().item()),
                }
            )
            sample_ordinal += 1


def _evaluate_token_baseline_run(
    run_dir: Path,
    horizons: list[int],
    kinds: list[str],
    *,
    eval_batches: int,
    device_name: str,
    eval_seed: int,
    eval_samples: int,
    eval_batch_size: int,
    per_sample_rows: list[dict[str, Any]] | None = None,
    field_split: str = "eval",
) -> list[dict[str, Any]]:
    base_cfg = _token_config_from_run_dir(run_dir)
    device = resolve_device(device_name)
    set_seed(base_cfg.seed)
    rows: list[dict[str, Any]] = []
    starts_by_horizon = (
        fixed_eval_plan_for_horizons(
            base_cfg,
            horizons,
            eval_seed=eval_seed,
            eval_samples=eval_samples,
            field_split=field_split,
        )
        if eval_samples > 0
        else {}
    )
    plan_hash = eval_plan_hash(
        horizons,
        eval_batches,
        eval_seed,
        eval_samples=eval_samples,
        eval_batch_size=eval_batch_size,
        start_indices_by_horizon=starts_by_horizon,
        field_split=field_split,
    )
    for checkpoint_kind, checkpoint_path, checkpoint_legacy_model_pt in _token_checkpoint_paths(run_dir, kinds):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model, spec = _build_token_baseline_model(base_cfg, state_dict, device)
        for horizon in horizons:
            cfg = baseline_eval.prepare_horizon_config(base_cfg, horizon, eval_batches=eval_batches, device=str(device))
            cfg.field_baseline_scope = "token_equivalent"
            cfg._field_eval_split_override = field_split
            horizon_seed = horizon_eval_seed(eval_seed, horizon)
            if eval_samples > 0:
                starts = starts_by_horizon[int(horizon)]
                attach_fixed_eval_plan(cfg, starts, eval_batch_size=eval_batch_size or cfg.batch_size)
                start_hash = start_indices_hash(starts)
                sample_count = int(starts.numel())
            else:
                start_hash = ""
                sample_count = 0
            with torch.random.fork_rng(devices=_fork_rng_devices(device)):
                torch.manual_seed(horizon_seed)
                metrics = evaluate_token_model(model, cfg, device)
            if per_sample_rows is not None and eval_samples > 0:
                cfg._field_fixed_start_cursor = 0
                _append_token_per_sample_rows(
                    per_sample_rows,
                    model=model,
                    cfg=cfg,
                    device=device,
                    run_dir=run_dir,
                    checkpoint_kind=checkpoint_kind,
                    checkpoint_path=checkpoint_path,
                    checkpoint_legacy_model_pt=checkpoint_legacy_model_pt,
                    horizon=int(horizon),
                    plan_hash=plan_hash,
                    start_hash=start_hash,
                    spec=spec,
                    field_split=field_split,
                )
            rows.append(
                {
                    "source_run_dir": run_dir.as_posix(),
                    "checkpoint_kind": checkpoint_kind,
                    "checkpoint_path": checkpoint_path.as_posix(),
                    "checkpoint_legacy_model_pt": checkpoint_legacy_model_pt,
                    "horizon": horizon,
                    "eval_plan_seed": eval_seed,
                    "eval_plan_hash": plan_hash,
                    "eval_horizon_seed": horizon_seed,
                    "eval_start_indices_hash": start_hash,
                    "eval_sample_count": sample_count,
                    "model": f"{spec['baseline_model']}-field-token-baseline",
                    "seed": cfg.seed,
                    **spec,
                    "field_horizon_conditioning": cfg.field_horizon_conditioning,
                    "field_tendency_baseline": cfg.field_tendency_baseline,
                    "field_tendency_scale": cfg.field_tendency_scale,
                    "field_residual_scale": cfg.field_residual_scale,
                    "field_baseline_scope": "token_equivalent",
                    **metrics,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    extra_fields = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in CSV_FIELDS and (key.startswith("eval_field_") or key.startswith("eval_h"))
        }
    )
    fieldnames = CSV_FIELDS + extra_fields
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format(row.get(key)) for key in fieldnames})


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# Field Horizon-Stratified Evaluation", "", f"Rows: {len(rows)}", ""]
    lines.extend(
        [
            "| run | model | ckpt | h | mse | persistence_mse | rel_l2 | improvement |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        name = "/".join(str(row["source_run_dir"]).split("/")[-2:])
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{name}`",
                    f"`{row.get('model', '')}`",
                    f"`{row['checkpoint_kind']}`",
                    _format(row.get("horizon")),
                    _format(row.get("eval_mse")),
                    _format(row.get("eval_field_persistence_mse")),
                    _format(row.get("eval_field_relative_l2")),
                    _format(row.get("eval_field_mse_improvement_vs_persistence")),
                ]
            )
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_per_sample_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PER_SAMPLE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format(row.get(key)) for key in PER_SAMPLE_FIELDS})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate field checkpoints at deterministic fixed horizons.")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--horizons", default="1,2,4,8")
    parser.add_argument("--checkpoint-kinds", default="final")
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--eval-seed", type=int, default=wca_eval.DEFAULT_EVAL_SEED)
    parser.add_argument("--eval-samples", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--field-split", default="eval", choices=["eval", "val", "test"])
    parser.add_argument(
        "--allow-mismatched-persistence",
        action="store_true",
        help="Allow exploratory reports where runs are evaluated on different samples. Strict matched reports should not use this.",
    )
    parser.add_argument(
        "--contract-check-only",
        action="store_true",
        help="Validate strict run scope and label/token contracts, then exit before loading checkpoints.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports/field_horizon_stratified_eval"))
    args = parser.parse_args(argv)
    if args.eval_samples <= 0 and not args.allow_mismatched_persistence:
        raise SystemExit(
            "Strict horizon-stratified evaluation requires --eval-samples > 0 so every model is evaluated "
            "on the same explicit start-index plan. Pass --allow-mismatched-persistence only for exploratory reports."
        )
    if args.eval_samples > 0:
        if args.eval_batch_size <= 0:
            raise SystemExit("--eval-batch-size must be positive when --eval-samples is used.")
        if args.eval_samples % args.eval_batch_size != 0:
            raise SystemExit("--eval-samples must be divisible by --eval-batch-size.")
        actual_eval_batches = args.eval_samples // args.eval_batch_size
        if args.eval_batches != actual_eval_batches:
            raise SystemExit(
                "--eval-batches must equal --eval-samples / --eval-batch-size in strict fixed-plan evaluation. "
                f"got eval_batches={args.eval_batches}, actual={actual_eval_batches}."
            )

    horizons = parse_field_target_steps_choices(args.horizons)
    kinds = _checkpoint_kinds(args.checkpoint_kinds)
    run_dirs = _discover_field_run_dirs(args.run_dirs)
    if not run_dirs:
        raise SystemExit("No field run directories with config.json and model.pt found.")
    if not args.allow_mismatched_persistence:
        _validate_strict_matched_run_scope(run_dirs)
        _validate_strict_run_contracts(run_dirs)
    if args.contract_check_only:
        print(json.dumps({"run_count": len(run_dirs), "strict_contract_ok": True}, sort_keys=True))
        return

    rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        if _is_token_baseline_run(run_dir):
            rows.extend(
                _evaluate_token_baseline_run(
                    run_dir,
                    horizons,
                    kinds,
                    eval_batches=args.eval_batches,
                    device_name=args.device,
                    eval_seed=args.eval_seed,
                    eval_samples=args.eval_samples,
                    eval_batch_size=args.eval_batch_size,
                    per_sample_rows=per_sample_rows,
                    field_split=args.field_split,
                )
            )
        elif _is_baseline_run(run_dir):
            rows.extend(
                _evaluate_baseline_run(
                    run_dir,
                    horizons,
                    kinds,
                    eval_batches=args.eval_batches,
                    device_name=args.device,
                    eval_seed=args.eval_seed,
                    eval_samples=args.eval_samples,
                    eval_batch_size=args.eval_batch_size,
                    per_sample_rows=per_sample_rows,
                    field_split=args.field_split,
                )
            )
        else:
            rows.extend(
                _evaluate_wca_run(
                    run_dir,
                    horizons,
                    kinds,
                    eval_batches=args.eval_batches,
                    device_name=args.device,
                    eval_seed=args.eval_seed,
                    eval_samples=args.eval_samples,
                    eval_batch_size=args.eval_batch_size,
                    per_sample_rows=per_sample_rows,
                    field_split=args.field_split,
                )
            )
    if not rows:
        raise SystemExit("No checkpoints were evaluated.")
    if not args.allow_mismatched_persistence:
        _validate_matched_eval_plan(rows)
        _validate_matched_persistence(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    starts_by_horizon = {}
    if args.eval_samples > 0:
        reference_cfg = wca_eval.config_from_run_dir(run_dirs[0])
        starts_by_horizon = fixed_eval_plan_for_horizons(
            reference_cfg,
            horizons,
            eval_seed=args.eval_seed,
            eval_samples=args.eval_samples,
            field_split=args.field_split,
        )
        write_fixed_eval_plan(
            args.output_dir / "eval_plan.json",
            starts_by_horizon,
            eval_seed=args.eval_seed,
            field_split=args.field_split,
        )
    write_csv(args.output_dir / "results_by_horizon.csv", rows)
    write_per_sample_csv(args.output_dir / "per_sample_rows.csv", per_sample_rows)
    write_markdown(args.output_dir / "results_by_horizon.md", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "rows": len(rows),
                "per_sample_rows": len(per_sample_rows),
                "horizons": horizons,
                "checkpoint_kinds": kinds,
                "eval_batches": args.eval_batches,
                "eval_batch_size": args.eval_batch_size,
                "eval_samples": args.eval_samples,
                "eval_seed": args.eval_seed,
                "field_split": args.field_split,
                "eval_plan_hash": eval_plan_hash(
                    horizons,
                    args.eval_batches,
                    args.eval_seed,
                    eval_samples=args.eval_samples,
                    eval_batch_size=args.eval_batch_size,
                    start_indices_by_horizon=starts_by_horizon,
                    field_split=args.field_split,
                ),
                "sample_rule": "explicit_start_indices_without_replacement_v1" if args.eval_samples > 0 else "torch_randint_seed_per_horizon_v1",
                "matched_persistence_required": not args.allow_mismatched_persistence,
                "strict_claim": not args.allow_mismatched_persistence and args.eval_samples > 0,
                "primary_checkpoint_kind": "final",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "output_dir": args.output_dir.as_posix()}, sort_keys=True))


if __name__ == "__main__":
    main()
