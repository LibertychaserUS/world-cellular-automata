#!/usr/bin/env python3
"""Evaluate two-step h8 autoregressive field rollout on fixed PDEBench windows.

This is an attribution/generalization evaluator, not a training script. It keeps
the trained checkpoint fixed and measures whether an h8 predictor can be applied
twice to reach the h16 target.

The current field interface predicts token-level patch means. For learnable
tokenizers, the second autoregressive input is reconstructed as a piecewise
constant field with ``unpatchify_field``. This is intentionally reported as
``rollout_h8x2_piecewise`` so it cannot be confused with native full-resolution
rollout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import eval_weatherbench_baseline_horizons as baseline_eval
from scripts import eval_weatherbench_horizons as wca_eval
from scripts.field_eval_plan import (
    DEFAULT_EVAL_SEED,
    attach_fixed_eval_plan,
    eval_plan_hash,
    fixed_eval_plan_for_horizons,
    horizon_eval_seed,
    start_indices_hash,
    write_fixed_eval_plan,
)
from scripts.train_field_baseline import field_prediction_baseline, model_input_from_batch
from wca.config import Config
from wca.data.field.real_cache import make_real_field_batch
from wca.data.field.synthetic import configure_field_nodes, field_patch_shape, patchify_field, unpatchify_field
from wca.models.field_wca import FieldTokenizerWCA
from wca.training.checkpointing import load_checkpoint
from wca.utils.device import resolve_device
from wca.utils.precision import autocast_context
from wca.utils.seed import set_seed


ROW_FIELDS = [
    "source_run_dir",
    "checkpoint_kind",
    "checkpoint_path",
    "model",
    "seed",
    "mode",
    "step_horizon",
    "total_horizon",
    "eval_plan_seed",
    "eval_plan_hash",
    "eval_start_indices_hash",
    "eval_sample_count",
    "field_split",
    "field_tokenizer",
    "field_token_dim",
    "field_tokenizer_width",
    "field_decoder_width",
    "field_baseline_scope",
    "outer_steps",
    "inner_steps",
    "eval_mse",
    "eval_mae",
    "eval_field_relative_l2",
    "eval_field_persistence_mse",
    "eval_field_persistence_mae",
    "eval_field_mse_improvement_vs_persistence",
    "eval_field_rollout_degradation_vs_direct_h8",
    "eval_field_rollout_degradation_vs_direct_h16",
]

PER_SAMPLE_FIELDS = [
    "source_run_dir",
    "checkpoint_kind",
    "checkpoint_path",
    "model",
    "seed",
    "mode",
    "step_horizon",
    "total_horizon",
    "eval_plan_hash",
    "eval_start_indices_hash",
    "sample_ordinal",
    "start_index",
    "target_index",
    "trajectory_id",
    "mse",
    "mae",
    "persistence_mse",
    "improvement_vs_persistence",
]


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


def _clone_config(cfg: Config) -> Config:
    allowed = {field.name for field in fields(Config)}
    return Config(**{key: value for key, value in cfg.to_dict().items() if key in allowed})


def _prepare_cfg(base_cfg: Config, *, horizon: int, eval_batch_size: int, device_name: str) -> Config:
    cfg = _clone_config(base_cfg)
    cfg.field_horizon_max_steps = int(getattr(base_cfg, "field_horizon_max_steps", 0) or horizon)
    cfg.field_target_steps_choices = ""
    cfg.field_target_steps = int(horizon)
    cfg.batch_size = int(eval_batch_size)
    cfg.eval_batches = 1
    cfg.device = device_name
    configure_field_nodes(cfg)
    return cfg


def _discover_run_dirs(paths: Iterable[Path]) -> list[Path]:
    discovered = wca_eval.discover_run_dirs(paths)
    return sorted(discovered, key=lambda item: str(item))


def _checkpoint_kinds(raw: str) -> list[str]:
    kinds = [item.strip() for item in raw.split(",") if item.strip()]
    if any(kind not in {"final", "best"} for kind in kinds):
        raise SystemExit("--checkpoint-kinds may only contain final,best")
    return kinds


def _is_baseline_run(run_dir: Path) -> bool:
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
    return False


def _build_model(run_dir: Path, cfg: Config, checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, str, dict[str, Any]]:
    if _is_baseline_run(run_dir):
        state_dict = torch.load(checkpoint_path, map_location=device)
        if not isinstance(state_dict, dict):
            raise ValueError(f"Baseline checkpoint must contain a state dict: {checkpoint_path}")
        model, spec = baseline_eval._build_baseline_model(cfg, state_dict, run_dir, device)
        model.eval()
        return model, str(spec["baseline_model"]), dict(spec)

    model = wca_eval._build_model(cfg, device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    return model, wca_eval.model_name_for_config(cfg), {}


def _checkpoint_paths(run_dir: Path, kinds: list[str]) -> list[tuple[str, Path]]:
    if _is_baseline_run(run_dir):
        return [(kind, path) for kind, path, _legacy in baseline_eval.checkpoint_paths(run_dir, kinds)]
    return wca_eval.checkpoint_paths(run_dir, kinds)


def _patchify(cfg: Config, field: Tensor) -> Tensor:
    tokens = patchify_field(field, field_patch_shape(cfg))
    return tokens.squeeze(-1) if tokens.shape[-1] == 1 else tokens


def _unpatchify(cfg: Config, tokens: Tensor) -> Tensor:
    token_tensor = tokens.unsqueeze(-1) if tokens.ndim == 2 else tokens
    return unpatchify_field(
        token_tensor,
        field_patch_shape(cfg),
        (int(cfg.field_grid_height), int(cfg.field_grid_width)),
    )


def _wca_prediction_tokens(model: nn.Module, cfg: Config, batch: dict[str, Any]) -> Tensor:
    with autocast_context(torch.device(cfg.device), cfg.precision):
        prediction = wca_eval._forward_prediction(model, cfg, batch)
    return prediction


def _baseline_prediction_tokens_and_field(model: nn.Module, cfg: Config, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
    raw_field = model(model_input_from_batch(batch, cfg))
    if bool(getattr(cfg, "field_residual_readout", False)):
        prediction_field = field_prediction_baseline(batch, cfg, raw_field)
        prediction_field = prediction_field + float(getattr(cfg, "field_residual_scale", 1.0)) * raw_field
    else:
        prediction_field = raw_field
    return _patchify(cfg, prediction_field), prediction_field


def _predict_tokens_and_piecewise_field(
    model: nn.Module,
    cfg: Config,
    batch: dict[str, Any],
    *,
    is_baseline: bool,
) -> tuple[Tensor, Tensor]:
    if is_baseline:
        return _baseline_prediction_tokens_and_field(model, cfg, batch)
    tokens = _wca_prediction_tokens(model, cfg, batch)
    return tokens, _unpatchify(cfg, tokens)


def _refresh_rollout_batch(
    batch: dict[str, Any],
    cfg: Config,
    *,
    previous_field: Tensor,
    current_field: Tensor,
    target_batch: dict[str, Any],
    step_horizon: int,
) -> dict[str, Any]:
    refreshed = dict(batch)
    field_input = torch.stack([previous_field, current_field], dim=1)
    current_tokens = _patchify(cfg, current_field)
    previous_tokens = _patchify(cfg, previous_field)
    H = batch["H"].clone()
    token_channels = int(current_tokens.shape[-1]) if current_tokens.ndim == 3 else 1
    if current_tokens.ndim == 2:
        H[:, :, 0] = current_tokens
    else:
        H[:, :, :token_channels] = current_tokens
    if H.shape[-1] >= token_channels * 2:
        if previous_tokens.ndim == 2:
            H[:, :, token_channels] = previous_tokens
        else:
            H[:, :, token_channels : token_channels * 2] = previous_tokens
    refreshed.update(
        {
            "H": H,
            "field_input": field_input,
            "label": target_batch["label"],
            "field_target": target_batch["field_target"],
            "field_target_index": target_batch["field_target_index"],
            "field_target_steps_actual": torch.full_like(batch["field_target_steps_actual"], int(step_horizon)),
            "field_prediction_baseline": current_tokens,
            "field_previous_tokens": previous_tokens,
        }
    )
    return refreshed


def _field_metrics(prediction: Tensor, target: Tensor, persistence: Tensor) -> dict[str, float]:
    errors = prediction - target
    mse = errors.pow(2).mean()
    mae = errors.abs().mean()
    relative_l2 = errors.pow(2).sum().sqrt() / target.pow(2).sum().sqrt().clamp_min(1e-8)
    persistence_errors = persistence - target
    persistence_mse = persistence_errors.pow(2).mean()
    persistence_mae = persistence_errors.abs().mean()
    improvement = (persistence_mse - mse) / persistence_mse.clamp_min(1e-8)
    return {
        "eval_mse": float(mse.detach().cpu().item()),
        "eval_mae": float(mae.detach().cpu().item()),
        "eval_field_relative_l2": float(relative_l2.detach().cpu().item()),
        "eval_field_persistence_mse": float(persistence_mse.detach().cpu().item()),
        "eval_field_persistence_mae": float(persistence_mae.detach().cpu().item()),
        "eval_field_mse_improvement_vs_persistence": float(improvement.detach().cpu().item()),
    }


def _per_sample_metrics(prediction: Tensor, target: Tensor, persistence: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    errors = prediction - target
    mse = errors.flatten(start_dim=1).pow(2).mean(dim=1)
    mae = errors.flatten(start_dim=1).abs().mean(dim=1)
    persistence_mse = (persistence - target).flatten(start_dim=1).pow(2).mean(dim=1)
    improvement = (persistence_mse - mse) / persistence_mse.clamp_min(1e-8)
    return mse, mae, persistence_mse, improvement


def _mean(rows: list[dict[str, float]], key: str) -> float:
    values = [float(row[key]) for row in rows if not math.isnan(float(row[key]))]
    return sum(values) / len(values) if values else float("nan")


@torch.no_grad()
def _evaluate_checkpoint(
    run_dir: Path,
    checkpoint_kind: str,
    checkpoint_path: Path,
    *,
    starts: Tensor,
    eval_plan_hash_value: str,
    eval_seed: int,
    step_horizon: int,
    total_horizon: int,
    eval_batch_size: int,
    device_name: str,
    field_split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base_cfg = wca_eval.config_from_run_dir(run_dir) if not _is_baseline_run(run_dir) else baseline_eval.config_from_run_dir(run_dir)
    device = resolve_device(device_name)
    base_cfg.device = str(device)
    set_seed(int(base_cfg.seed))
    model, model_name, _spec = _build_model(run_dir, base_cfg, checkpoint_path, device)
    is_baseline = _is_baseline_run(run_dir)

    cfg_h8 = _prepare_cfg(base_cfg, horizon=step_horizon, eval_batch_size=eval_batch_size, device_name=str(device))
    cfg_h8._field_eval_split_override = field_split
    cfg_h16 = _prepare_cfg(base_cfg, horizon=total_horizon, eval_batch_size=eval_batch_size, device_name=str(device))
    cfg_h16._field_eval_split_override = field_split
    cfg_h7 = _prepare_cfg(base_cfg, horizon=max(1, step_horizon - 1), eval_batch_size=eval_batch_size, device_name=str(device))
    cfg_h7._field_eval_split_override = field_split

    rows_by_mode: dict[str, list[dict[str, float]]] = {
        "direct_h8": [],
        "direct_h16_ood": [],
        "rollout_h8x2_piecewise": [],
        "rollout_h8x2_teacher_prev_piecewise": [],
    }
    per_sample_rows: list[dict[str, Any]] = []
    start_hash = start_indices_hash(starts)
    sample_ordinal_by_mode = {mode: 0 for mode in rows_by_mode}

    for cursor in range(0, int(starts.numel()), eval_batch_size):
        chunk = starts[cursor : cursor + eval_batch_size]
        for cfg in (cfg_h8, cfg_h16, cfg_h7):
            attach_fixed_eval_plan(cfg, chunk, eval_batch_size=eval_batch_size)
        batch_h8 = make_real_field_batch(cfg_h8, device, split=field_split)
        batch_h16 = make_real_field_batch(cfg_h16, device, split=field_split)
        batch_h7 = make_real_field_batch(cfg_h7, device, split=field_split)

        with torch.random.fork_rng(devices=wca_eval._fork_rng_devices(device)):
            torch.manual_seed(horizon_eval_seed(eval_seed, step_horizon))
            pred_h8_tokens, pred_h8_field = _predict_tokens_and_piecewise_field(
                model,
                cfg_h8,
                batch_h8,
                is_baseline=is_baseline,
            )
        with torch.random.fork_rng(devices=wca_eval._fork_rng_devices(device)):
            torch.manual_seed(horizon_eval_seed(eval_seed, max(1, step_horizon - 1)))
            _pred_h7_tokens, pred_h7_field = _predict_tokens_and_piecewise_field(
                model,
                cfg_h7,
                batch_h7,
                is_baseline=is_baseline,
            )
        with torch.random.fork_rng(devices=wca_eval._fork_rng_devices(device)):
            torch.manual_seed(horizon_eval_seed(eval_seed, total_horizon))
            pred_h16_direct_tokens, _pred_h16_direct_field = _predict_tokens_and_piecewise_field(
                model,
                cfg_h16,
                batch_h16,
                is_baseline=is_baseline,
            )

        pure_rollout_batch = _refresh_rollout_batch(
            batch_h8,
            cfg_h8,
            previous_field=pred_h7_field,
            current_field=pred_h8_field,
            target_batch=batch_h16,
            step_horizon=step_horizon,
        )
        teacher_prev_batch = _refresh_rollout_batch(
            batch_h8,
            cfg_h8,
            previous_field=batch_h7["field_target"],
            current_field=pred_h8_field,
            target_batch=batch_h16,
            step_horizon=step_horizon,
        )
        pred_rollout_tokens, _ = _predict_tokens_and_piecewise_field(model, cfg_h8, pure_rollout_batch, is_baseline=is_baseline)
        pred_teacher_tokens, _ = _predict_tokens_and_piecewise_field(model, cfg_h8, teacher_prev_batch, is_baseline=is_baseline)

        targets = {
            "direct_h8": batch_h8["label"],
            "direct_h16_ood": batch_h16["label"],
            "rollout_h8x2_piecewise": batch_h16["label"],
            "rollout_h8x2_teacher_prev_piecewise": batch_h16["label"],
        }
        predictions = {
            "direct_h8": pred_h8_tokens,
            "direct_h16_ood": pred_h16_direct_tokens,
            "rollout_h8x2_piecewise": pred_rollout_tokens,
            "rollout_h8x2_teacher_prev_piecewise": pred_teacher_tokens,
        }
        persistences = {
            "direct_h8": batch_h8["field_prediction_baseline"],
            "direct_h16_ood": batch_h16["field_prediction_baseline"],
            "rollout_h8x2_piecewise": batch_h16["field_prediction_baseline"],
            "rollout_h8x2_teacher_prev_piecewise": batch_h16["field_prediction_baseline"],
        }
        for mode, prediction in predictions.items():
            target = targets[mode].to(device=prediction.device, dtype=prediction.dtype)
            persistence = persistences[mode].to(device=prediction.device, dtype=prediction.dtype)
            rows_by_mode[mode].append(_field_metrics(prediction, target, persistence))
            mse, mae, persistence_mse, improvement = _per_sample_metrics(prediction, target, persistence)
            starts_tensor = batch_h16["field_start_index"] if mode != "direct_h8" else batch_h8["field_start_index"]
            targets_tensor = batch_h16["field_target_index"] if mode != "direct_h8" else batch_h8["field_target_index"]
            trajectories = batch_h16["field_trajectory_id"] if mode != "direct_h8" else batch_h8["field_trajectory_id"]
            for index in range(int(mse.numel())):
                per_sample_rows.append(
                    {
                        "source_run_dir": run_dir.as_posix(),
                        "checkpoint_kind": checkpoint_kind,
                        "checkpoint_path": checkpoint_path.as_posix(),
                        "model": model_name,
                        "seed": base_cfg.seed,
                        "mode": mode,
                        "step_horizon": step_horizon,
                        "total_horizon": step_horizon if mode == "direct_h8" else total_horizon,
                        "eval_plan_hash": eval_plan_hash_value,
                        "eval_start_indices_hash": start_hash,
                        "sample_ordinal": sample_ordinal_by_mode[mode],
                        "start_index": int(starts_tensor[index].detach().cpu().item()),
                        "target_index": int(targets_tensor[index].detach().cpu().item()),
                        "trajectory_id": int(trajectories[index].detach().cpu().item()),
                        "mse": float(mse[index].detach().cpu().item()),
                        "mae": float(mae[index].detach().cpu().item()),
                        "persistence_mse": float(persistence_mse[index].detach().cpu().item()),
                        "improvement_vs_persistence": float(improvement[index].detach().cpu().item()),
                    }
                )
                sample_ordinal_by_mode[mode] += 1

    direct_h8_mse = _mean(rows_by_mode["direct_h8"], "eval_mse")
    direct_h16_mse = _mean(rows_by_mode["direct_h16_ood"], "eval_mse")
    rows: list[dict[str, Any]] = []
    for mode, mode_rows in rows_by_mode.items():
        metrics = {key: _mean(mode_rows, key) for key in mode_rows[0]}
        metrics["eval_field_rollout_degradation_vs_direct_h8"] = (
            metrics["eval_mse"] / direct_h8_mse if direct_h8_mse > 0 and mode.startswith("rollout_") else float("nan")
        )
        metrics["eval_field_rollout_degradation_vs_direct_h16"] = (
            metrics["eval_mse"] / direct_h16_mse if direct_h16_mse > 0 and mode.startswith("rollout_") else float("nan")
        )
        rows.append(
            {
                "source_run_dir": run_dir.as_posix(),
                "checkpoint_kind": checkpoint_kind,
                "checkpoint_path": checkpoint_path.as_posix(),
                "model": model_name,
                "seed": base_cfg.seed,
                "mode": mode,
                "step_horizon": step_horizon,
                "total_horizon": step_horizon if mode == "direct_h8" else total_horizon,
                "eval_plan_seed": eval_seed,
                "eval_plan_hash": eval_plan_hash_value,
                "eval_start_indices_hash": start_hash,
                "eval_sample_count": int(starts.numel()),
                "field_split": field_split,
                "field_tokenizer": getattr(base_cfg, "field_tokenizer", "patch_mean"),
                "field_token_dim": getattr(base_cfg, "field_token_dim", ""),
                "field_tokenizer_width": getattr(base_cfg, "field_tokenizer_width", ""),
                "field_decoder_width": getattr(base_cfg, "field_decoder_width", ""),
                "field_baseline_scope": getattr(base_cfg, "field_baseline_scope", ""),
                "outer_steps": getattr(base_cfg, "outer_steps", ""),
                "inner_steps": getattr(base_cfg, "inner_steps", ""),
                **metrics,
            }
        )
    return rows, per_sample_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields_: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields_, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format(row.get(key)) for key in fields_})


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# H8x2 Field Rollout Evaluation",
        "",
        "This evaluates token-level two-step h8 autoregressive rollout. For learnable-tokenizer WCA, "
        "the second input is a piecewise-constant unpatchified field; this is a diagnostic, not a native "
        "full-resolution decoder claim.",
        "",
        f"Rows: {len(rows)}",
        "",
        "| run | model | ckpt | mode | rel_l2 | mse | persistence_mse | improvement | deg_vs_h8 | deg_vs_h16 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        name = "/".join(str(row["source_run_dir"]).split("/")[-2:])
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{name}`",
                    f"`{row.get('model', '')}`",
                    f"`{row.get('checkpoint_kind', '')}`",
                    f"`{row.get('mode', '')}`",
                    _format(row.get("eval_field_relative_l2")),
                    _format(row.get("eval_mse")),
                    _format(row.get("eval_field_persistence_mse")),
                    _format(row.get("eval_field_mse_improvement_vs_persistence")),
                    _format(row.get("eval_field_rollout_degradation_vs_direct_h8")),
                    _format(row.get("eval_field_rollout_degradation_vs_direct_h16")),
                ]
            )
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate h8x2 token-level field rollout on fixed starts.")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--step-horizon", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=2)
    parser.add_argument("--checkpoint-kinds", default="final,best")
    parser.add_argument("--eval-seed", type=int, default=DEFAULT_EVAL_SEED)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--field-split", default="test", choices=["eval", "val", "test"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports/field_h8_rollout_twice"))
    args = parser.parse_args()

    if args.rollout_steps != 2:
        raise SystemExit("This strict evaluator currently supports only --rollout-steps 2.")
    if args.eval_samples <= 0:
        raise SystemExit("--eval-samples must be positive for fixed-plan rollout evaluation.")
    if args.eval_batch_size <= 0:
        raise SystemExit("--eval-batch-size must be positive.")
    if args.eval_samples % args.eval_batch_size != 0:
        raise SystemExit("--eval-samples must be divisible by --eval-batch-size.")

    total_horizon = int(args.step_horizon) * int(args.rollout_steps)
    run_dirs = _discover_run_dirs(args.run_dirs)
    if not run_dirs:
        raise SystemExit("No field run directories with config.json and model.pt found.")

    reference_cfg = wca_eval.config_from_run_dir(run_dirs[0]) if not _is_baseline_run(run_dirs[0]) else baseline_eval.config_from_run_dir(run_dirs[0])
    starts_by_horizon = fixed_eval_plan_for_horizons(
        reference_cfg,
        [total_horizon],
        eval_seed=args.eval_seed,
        eval_samples=args.eval_samples,
        field_split=args.field_split,
    )
    starts = starts_by_horizon[total_horizon]
    plan_hash = eval_plan_hash(
        [total_horizon],
        args.eval_samples // args.eval_batch_size,
        args.eval_seed,
        eval_samples=args.eval_samples,
        eval_batch_size=args.eval_batch_size,
        start_indices_by_horizon=starts_by_horizon,
        field_split=args.field_split,
    )

    rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    kinds = _checkpoint_kinds(args.checkpoint_kinds)
    for run_dir in run_dirs:
        for checkpoint_kind, checkpoint_path in _checkpoint_paths(run_dir, kinds):
            checkpoint_rows, checkpoint_per_sample_rows = _evaluate_checkpoint(
                run_dir,
                checkpoint_kind,
                checkpoint_path,
                starts=starts,
                eval_plan_hash_value=plan_hash,
                eval_seed=args.eval_seed,
                step_horizon=args.step_horizon,
                total_horizon=total_horizon,
                eval_batch_size=args.eval_batch_size,
                device_name=args.device,
                field_split=args.field_split,
            )
            rows.extend(checkpoint_rows)
            per_sample_rows.extend(checkpoint_per_sample_rows)

    if not rows:
        raise SystemExit("No checkpoints were evaluated.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_fixed_eval_plan(args.output_dir / "eval_plan.json", starts_by_horizon, eval_seed=args.eval_seed, field_split=args.field_split)
    _write_csv(args.output_dir / "results_rollout.csv", rows, ROW_FIELDS)
    _write_csv(args.output_dir / "per_sample_rollout_rows.csv", per_sample_rows, PER_SAMPLE_FIELDS)
    _write_markdown(args.output_dir / "results_rollout.md", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "rows": len(rows),
                "per_sample_rows": len(per_sample_rows),
                "step_horizon": int(args.step_horizon),
                "rollout_steps": int(args.rollout_steps),
                "total_horizon": total_horizon,
                "checkpoint_kinds": kinds,
                "eval_samples": int(args.eval_samples),
                "eval_batch_size": int(args.eval_batch_size),
                "eval_seed": int(args.eval_seed),
                "field_split": args.field_split,
                "eval_plan_hash": plan_hash,
                "sample_rule": "explicit_start_indices_without_replacement_total_horizon_v1",
                "strict_claim": True,
                "rollout_input_rule": "piecewise_constant_unpatchify_predicted_h7_and_h8_then_second_h8",
                "primary_mode": "rollout_h8x2_piecewise",
                "diagnostic_modes": ["direct_h8", "direct_h16_ood", "rollout_h8x2_teacher_prev_piecewise"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "output_dir": args.output_dir.as_posix()}, sort_keys=True))


if __name__ == "__main__":
    main()
