#!/usr/bin/env python3
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wca.config import Config
from wca.data.field.synthetic import configure_field_nodes, parse_field_target_steps_choices
from wca.models.field_wca import FieldTokenizerWCA
from wca.models.rws_nca import FullRecursiveWorldStateNCA
from wca.training.checkpointing import load_checkpoint
from wca.training.evaluator import evaluate, make_batch
from wca.utils.device import resolve_device
from wca.utils.precision import autocast_context
from wca.utils.seed import set_seed
from wca.training.prediction import predict_for_task

from scripts.field_eval_plan import (
    DEFAULT_EVAL_SEED,
    attach_fixed_eval_plan,
    eval_plan_hash,
    fixed_eval_plan_for_horizons,
    horizon_eval_seed,
    start_indices_hash,
    write_fixed_eval_plan,
)


BASE_FIELDS = [
    "source_run_dir",
    "checkpoint_kind",
    "checkpoint_path",
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
METRIC_FIELDS = [
    "eval_mse",
    "eval_mae",
    "eval_field_relative_l2",
    "eval_field_persistence_mse",
    "eval_field_persistence_mae",
    "eval_field_mse_improvement_vs_persistence",
    "eval_field_mse_2m_temperature",
    "eval_field_mse_improvement_vs_persistence_2m_temperature",
    "eval_field_mse_10m_u_component_of_wind",
    "eval_field_mse_improvement_vs_persistence_10m_u_component_of_wind",
    "eval_field_mse_10m_v_component_of_wind",
    "eval_field_mse_improvement_vs_persistence_10m_v_component_of_wind",
    "eval_field_mse_mean_sea_level_pressure",
    "eval_field_mse_improvement_vs_persistence_mean_sea_level_pressure",
]
CSV_FIELDS = BASE_FIELDS + METRIC_FIELDS
PER_SAMPLE_FIELDS = [
    "source_run_dir",
    "checkpoint_kind",
    "checkpoint_path",
    "horizon",
    "eval_plan_hash",
    "eval_start_indices_hash",
    "model",
    "seed",
    "sample_ordinal",
    "start_index",
    "target_index",
    "trajectory_id",
    "mse",
    "mae",
    "persistence_mse",
    "improvement_vs_persistence",
]


def model_name_for_config(cfg: Config) -> str:
    tokenizer_name = str(getattr(cfg, "field_tokenizer", "patch_mean"))
    tokenizer_only = bool(getattr(cfg, "field_tokenizer_only", False))
    outer_steps = int(getattr(cfg, "outer_steps", 0))
    if tokenizer_name == "patch_mean":
        return "FullRecursiveWorldStateNCA-heavy-dense"
    if tokenizer_only:
        return f"{tokenizer_name}-tokenizer-only"
    if outer_steps <= 0:
        return f"{tokenizer_name}-tokenizer-bypass-o0"
    return f"{tokenizer_name}-WCA"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def config_from_run_dir(run_dir: Path) -> Config:
    payload = _read_json(run_dir / "config.json")
    allowed = {field.name for field in fields(Config)}
    filtered = {key: value for key, value in payload.items() if key in allowed}
    cfg = Config(**filtered)
    if cfg.task != "field":
        raise ValueError(f"Expected field run config in {run_dir}, got task={cfg.task!r}")
    configure_field_nodes(cfg)
    return cfg


def horizon_max_from_config(cfg: Config, horizons: Iterable[int]) -> int:
    explicit = int(getattr(cfg, "field_horizon_max_steps", 0) or 0)
    if explicit > 0:
        return explicit
    choices = parse_field_target_steps_choices(cfg.field_target_steps_choices) if cfg.field_target_steps_choices else []
    return max([*choices, *[int(item) for item in horizons], int(cfg.field_target_steps)])


def prepare_horizon_config(base_cfg: Config, horizon: int, *, eval_batches: int, device: str) -> Config:
    cfg = Config(**base_cfg.to_dict())
    cfg.field_horizon_max_steps = horizon_max_from_config(base_cfg, [horizon])
    cfg.field_target_steps_choices = ""
    cfg.field_target_steps = int(horizon)
    cfg.eval_batches = int(eval_batches)
    cfg.device = device
    configure_field_nodes(cfg)
    return cfg


def discover_run_dirs(paths: Iterable[Path]) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        candidates: list[Path]
        if (path / "config.json").exists() and (path / "model.pt").exists():
            candidates = [path]
        else:
            candidates = sorted(parent for parent in path.rglob("config.json") if (parent.parent / "model.pt").exists())
            candidates = [candidate.parent for candidate in candidates]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                discovered.append(candidate)
    return sorted(discovered, key=lambda item: str(item))


def checkpoint_paths(run_dir: Path, kinds: list[str]) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    mapping = {"final": run_dir / "model.pt", "best": run_dir / "best_model.pt"}
    for kind in kinds:
        path = mapping[kind]
        if path.exists():
            paths.append((kind, path))
    return paths


def _build_model(cfg: Config, device: torch.device) -> torch.nn.Module:
    if str(getattr(cfg, "field_tokenizer", "patch_mean")) != "patch_mean":
        return FieldTokenizerWCA(cfg).to(device)
    return FullRecursiveWorldStateNCA(
        n_nodes=cfg.n_nodes,
        hidden_dim=cfg.hidden_dim,
        edge_dim=cfg.edge_dim,
        inner_steps=cfg.inner_steps,
        pair_chunk_size=cfg.pair_chunk_size,
        output_dim=int(getattr(cfg, "field_output_dim", 1)),
        activation_checkpoint_inner=cfg.activation_checkpoint_inner,
    ).to(device)


def _forward_prediction(model: torch.nn.Module, cfg: Config, batch: dict[str, Any]) -> torch.Tensor:
    module = model.module if hasattr(model, "module") else model
    if isinstance(module, FieldTokenizerWCA):
        prediction, _diagnostics = model(batch, cfg.outer_steps)
        return prediction
    input_visibility = batch.get("input_visibility")
    if input_visibility is None:
        H_final, _diagnostics = model(batch["H"], batch["adjacency"], cfg.outer_steps)
    else:
        H_final, _diagnostics = model(
            batch["H"],
            batch["adjacency"],
            cfg.outer_steps,
            input_visibility=input_visibility,
            input_visibility_channels=batch.get("input_visibility_channels"),
        )
    return predict_for_task(module, cfg, H_final, batch)


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.7g}"
    return str(value)


def _fork_rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [device.index if device.index is not None else torch.cuda.current_device()]


def _per_sample_errors(prediction: torch.Tensor, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    label = batch["label"].to(device=prediction.device, dtype=prediction.dtype)
    errors = prediction - label
    mse = errors.pow(2).flatten(start_dim=1).mean(dim=1)
    mae = errors.abs().flatten(start_dim=1).mean(dim=1)
    persistence = batch.get("field_prediction_baseline")
    if isinstance(persistence, torch.Tensor) and persistence.shape == label.shape:
        persistence = persistence.to(device=prediction.device, dtype=prediction.dtype)
        persistence_mse = (persistence - label).pow(2).flatten(start_dim=1).mean(dim=1)
        improvement = (persistence_mse - mse) / persistence_mse.clamp_min(1e-8)
    else:
        persistence_mse = torch.full_like(mse, float("nan"))
        improvement = torch.full_like(mse, float("nan"))
    return mse, mae, persistence_mse, improvement


@torch.no_grad()
def append_per_sample_rows(
    rows: list[dict[str, Any]],
    *,
    model: torch.nn.Module,
    cfg: Config,
    device: torch.device,
    run_dir: Path,
    checkpoint_kind: str,
    checkpoint_path: Path,
    horizon: int,
    plan_hash: str,
    start_hash: str,
    field_split: str = "eval",
) -> None:
    model.eval()
    sample_ordinal = 0
    for _ in range(cfg.eval_batches):
        batch = make_batch(cfg, device, field_split=field_split)
        with autocast_context(device, cfg.precision):
            prediction = _forward_prediction(model, cfg, batch)
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
                    "horizon": int(horizon),
                    "eval_plan_hash": plan_hash,
                    "eval_start_indices_hash": start_hash,
                    "model": model_name_for_config(cfg),
                    "seed": cfg.seed,
                    "sample_ordinal": sample_ordinal,
                    "start_index": int(starts[index].detach().cpu().item()) if isinstance(starts, torch.Tensor) else "",
                    "target_index": int(targets[index].detach().cpu().item()) if isinstance(targets, torch.Tensor) else "",
                    "trajectory_id": int(trajectories[index].detach().cpu().item()) if isinstance(trajectories, torch.Tensor) else "",
                    "mse": float(mse[index].detach().cpu().item()),
                    "mae": float(mae[index].detach().cpu().item()),
                    "persistence_mse": float(persistence_mse[index].detach().cpu().item()),
                    "improvement_vs_persistence": float(improvement[index].detach().cpu().item()),
                }
            )
            sample_ordinal += 1


def evaluate_run_checkpoint(
    run_dir: Path,
    checkpoint_kind: str,
    checkpoint_path: Path,
    horizons: list[int],
    *,
    eval_batches: int,
    device_name: str,
    eval_seed: int = DEFAULT_EVAL_SEED,
    eval_samples: int = 0,
    eval_batch_size: int = 0,
    per_sample_rows: list[dict[str, Any]] | None = None,
    field_split: str = "eval",
) -> list[dict[str, Any]]:
    base_cfg = config_from_run_dir(run_dir)
    device = resolve_device(device_name)
    set_seed(base_cfg.seed)
    model = _build_model(base_cfg, device)
    load_checkpoint(checkpoint_path, model, map_location=device)
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
    for horizon in horizons:
        cfg = prepare_horizon_config(base_cfg, horizon, eval_batches=eval_batches, device=str(device))
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
            metrics = evaluate(model, cfg, device)
        if per_sample_rows is not None and eval_samples > 0:
            cfg._field_fixed_start_cursor = 0
            append_per_sample_rows(
                per_sample_rows,
                model=model,
                cfg=cfg,
                device=device,
                run_dir=run_dir,
                checkpoint_kind=checkpoint_kind,
                checkpoint_path=checkpoint_path,
                horizon=int(horizon),
                plan_hash=plan_hash,
                start_hash=start_hash,
                field_split=field_split,
            )
        tokenizer_name = str(getattr(cfg, "field_tokenizer", "patch_mean"))
        tokenizer_only = bool(getattr(cfg, "field_tokenizer_only", False))
        model_name = model_name_for_config(cfg)
        rows.append(
            {
                "source_run_dir": run_dir.as_posix(),
                "checkpoint_kind": checkpoint_kind,
                "checkpoint_path": checkpoint_path.as_posix(),
                "horizon": horizon,
                "eval_plan_seed": eval_seed,
                "eval_plan_hash": plan_hash,
                "eval_horizon_seed": horizon_seed,
                "eval_start_indices_hash": start_hash,
                "eval_sample_count": sample_count,
                "model": model_name,
                "seed": cfg.seed,
                "hidden_dim": cfg.hidden_dim,
                "edge_dim": cfg.edge_dim,
                "inner_steps": cfg.inner_steps,
                "outer_steps": cfg.outer_steps,
                "field_horizon_conditioning": cfg.field_horizon_conditioning,
                "field_tendency_baseline": cfg.field_tendency_baseline,
                "field_tendency_scale": cfg.field_tendency_scale,
                "field_residual_scale": cfg.field_residual_scale,
                "field_tokenizer": tokenizer_name,
                "field_token_dim": getattr(cfg, "field_token_dim", 0),
                "field_tokenizer_width": getattr(cfg, "field_tokenizer_width", 0),
                "field_decoder_width": getattr(cfg, "field_decoder_width", 0),
                "field_tokenizer_only": tokenizer_only,
                "field_baseline_scope": getattr(cfg, "field_baseline_scope", ""),
                **metrics,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format(row.get(key)) for key in CSV_FIELDS})


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# WeatherBench2 Horizon-Stratified Evaluation", "", f"Rows: {len(rows)}", ""]
    lines.extend(
        [
            "| run | ckpt | h | mse | persistence_mse | improvement | t2m_impr | u10_impr | v10_impr | mslp_impr |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        name = "/".join(str(row["source_run_dir"]).split("/")[-2:])
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{name}`",
                    f"`{row['checkpoint_kind']}`",
                    _format(row.get("horizon")),
                    _format(row.get("eval_mse")),
                    _format(row.get("eval_field_persistence_mse")),
                    _format(row.get("eval_field_mse_improvement_vs_persistence")),
                    _format(row.get("eval_field_mse_improvement_vs_persistence_2m_temperature")),
                    _format(row.get("eval_field_mse_improvement_vs_persistence_10m_u_component_of_wind")),
                    _format(row.get("eval_field_mse_improvement_vs_persistence_10m_v_component_of_wind")),
                    _format(row.get("eval_field_mse_improvement_vs_persistence_mean_sea_level_pressure")),
                ]
            )
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate WCA WeatherBench2 checkpoints at fixed forecast horizons.")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--horizons", default="1,2,4,8")
    parser.add_argument("--checkpoint-kinds", default="final,best")
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--eval-seed", type=int, default=DEFAULT_EVAL_SEED)
    parser.add_argument("--eval-samples", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--field-split", default="eval", choices=["eval", "val", "test"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports/weatherbench2_horizon_eval"))
    args = parser.parse_args()

    horizons = parse_field_target_steps_choices(args.horizons)
    kinds = [item.strip() for item in args.checkpoint_kinds.split(",") if item.strip()]
    if any(kind not in {"final", "best"} for kind in kinds):
        raise SystemExit("--checkpoint-kinds may only contain final,best")
    run_dirs = discover_run_dirs(args.run_dirs)
    if not run_dirs:
        raise SystemExit("No run directories with config.json and model.pt found.")

    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        checkpoints = checkpoint_paths(run_dir, kinds)
        if not checkpoints:
            continue
        for checkpoint_kind, checkpoint_path in checkpoints:
            rows.extend(
                evaluate_run_checkpoint(
                    run_dir,
                    checkpoint_kind,
                    checkpoint_path,
                    horizons,
                    eval_batches=args.eval_batches,
                    device_name=args.device,
                    eval_seed=args.eval_seed,
                    eval_samples=args.eval_samples,
                    eval_batch_size=args.eval_batch_size,
                    field_split=args.field_split,
                )
            )
    if not rows:
        raise SystemExit("No checkpoints were evaluated.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    starts_by_horizon = {}
    if args.eval_samples > 0:
        starts_by_horizon = fixed_eval_plan_for_horizons(
            config_from_run_dir(run_dirs[0]),
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
    write_markdown(args.output_dir / "results_by_horizon.md", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "rows": len(rows),
                "horizons": horizons,
                "checkpoint_kinds": kinds,
                "eval_batches": args.eval_batches,
                "eval_seed": args.eval_seed,
                "eval_batch_size": args.eval_batch_size,
                "eval_samples": args.eval_samples,
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
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "output_dir": args.output_dir.as_posix()}, sort_keys=True))


if __name__ == "__main__":
    main()
