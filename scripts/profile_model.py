#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wca.config import add_common_cli_args, config_from_args
from wca.training.evaluator import make_batch
from wca.models.rws_nca import FullRecursiveWorldStateNCA
from wca.training.prediction import predict_for_task
from wca.utils.device import resolve_device, sync_device
from wca.utils.precision import autocast_context, cuda_memory_metrics, make_grad_scaler
from wca.utils.seed import set_seed


PROFILE_COLUMNS = [
    "profile_id",
    "repeat",
    "step",
    "task",
    "field_dataset",
    "device",
    "precision",
    "batch_size",
    "n_nodes",
    "hidden_dim",
    "edge_dim",
    "field_adjacency_mode",
    "field_input_scope",
    "field_input_steps",
    "field_output_dim",
    "field_target_steps",
    "field_target_steps_choices",
    "outer_steps",
    "inner_steps",
    "pair_chunk_size",
    "activation_checkpoint_inner",
    "loss",
    "step_seconds",
    "samples_per_second",
    "cuda_memory_allocated_mb",
    "cuda_memory_reserved_mb",
    "cuda_peak_memory_allocated_mb",
    "cuda_peak_memory_reserved_mb",
]


def _finite_values(values: list[float]) -> list[float]:
    return [value for value in values if not (math.isnan(value) or math.isinf(value))]


def _percentile(values: list[float], fraction: float) -> float:
    finite = sorted(_finite_values(values))
    if not finite:
        return float("nan")
    if len(finite) == 1:
        return finite[0]
    index = fraction * (len(finite) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return finite[lower]
    weight = index - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def _summary(values: list[float]) -> dict[str, float]:
    finite = _finite_values(values)
    if not finite:
        return {
            "count": 0.0,
            "mean": float("nan"),
            "median": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": float(len(finite)),
        "mean": sum(finite) / len(finite),
        "median": float(median(finite)),
        "p25": _percentile(finite, 0.25),
        "p75": _percentile(finite, 0.75),
        "min": min(finite),
        "max": max(finite),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _profile_id(cfg: Any) -> str:
    return (
        f"{cfg.task}_{cfg.field_dataset}_n{cfg.n_nodes}_d{cfg.hidden_dim}_"
        f"o{cfg.outer_steps}_i{cfg.inner_steps}_b{cfg.batch_size}_{cfg.precision}"
    )


def _make_model(cfg: Any, device: torch.device) -> FullRecursiveWorldStateNCA:
    return FullRecursiveWorldStateNCA(
        n_nodes=cfg.n_nodes,
        hidden_dim=cfg.hidden_dim,
        edge_dim=cfg.edge_dim,
        inner_steps=cfg.inner_steps,
        pair_chunk_size=cfg.pair_chunk_size,
        output_dim=int(getattr(cfg, "field_output_dim", 1)) if cfg.task == "field" else 1,
        activation_checkpoint_inner=cfg.activation_checkpoint_inner,
    ).to(device)


def _forward_loss(model: FullRecursiveWorldStateNCA, cfg: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    input_visibility = batch.get("input_visibility")
    if input_visibility is None:
        H_final, _ = model(batch["H"], batch["adjacency"], cfg.outer_steps)
    else:
        H_final, _ = model(
            batch["H"],
            batch["adjacency"],
            cfg.outer_steps,
            input_visibility=input_visibility,
            input_visibility_channels=batch.get("input_visibility_channels"),
        )
    prediction = predict_for_task(model, cfg, H_final, batch)

    if cfg.task == "maze":
        loss = ((prediction - batch["distance_field"]) ** 2 * batch["distance_mask"]).sum()
        return loss / batch["distance_mask"].sum().clamp_min(1.0)
    return torch.nn.functional.mse_loss(prediction, batch["label"])


def _run_profile_step(
    *,
    model: FullRecursiveWorldStateNCA,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg: Any,
    device: torch.device,
) -> tuple[float, float, dict[str, float]]:
    batch = make_batch(cfg, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with autocast_context(device, cfg.precision):
        loss = _forward_loss(model, cfg, batch)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    if scaler.is_enabled():
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.max_grad_norm))
    scaler.step(optimizer)
    scaler.update()
    sync_device(device)
    elapsed = time.perf_counter() - start
    return float(loss.detach().cpu()), elapsed, cuda_memory_metrics(device)


def _artifact_rows(rows: list[dict[str, Any]], *, metric: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        key = f"repeat_{row['repeat']}"
        grouped.setdefault(key, []).append(float(row[metric]))
        grouped.setdefault("all", []).append(float(row[metric]))
    output: list[dict[str, Any]] = []
    for group, values in sorted(grouped.items()):
        summary = _summary(values)
        output.append({"group": group, "metric": metric, **summary})
    return output


def write_profile_artifacts(
    output_dir: Path,
    *,
    cfg: Any,
    device: torch.device,
    rows: list[dict[str, Any]],
    warmup_steps: int,
    measured_steps: int,
    repeats: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "profile_long.csv", rows, PROFILE_COLUMNS)
    performance_rows = _artifact_rows(rows, metric="samples_per_second") + _artifact_rows(rows, metric="step_seconds")
    _write_csv(
        output_dir / "performance_long.csv",
        performance_rows,
        ["group", "metric", "count", "mean", "median", "p25", "p75", "min", "max"],
    )
    memory_rows: list[dict[str, Any]] = []
    for metric in [
        "cuda_memory_allocated_mb",
        "cuda_memory_reserved_mb",
        "cuda_peak_memory_allocated_mb",
        "cuda_peak_memory_reserved_mb",
    ]:
        memory_rows.extend(_artifact_rows(rows, metric=metric))
    _write_csv(
        output_dir / "memory_long.csv",
        memory_rows,
        ["group", "metric", "count", "mean", "median", "p25", "p75", "min", "max"],
    )
    _write_json(
        output_dir / "optimization_gate.json",
        {
            "optimization_role": "capacity_probe",
            "baseline_reference": "FullRecursiveWorldStateNCA",
            "equivalence_required": False,
            "equivalence_status": "not_applicable",
            "formal_claim_eligible": False,
            "reason": "profile_model.py produces capacity/profiling evidence only, not formal model-capability evidence.",
        },
    )
    _write_json(
        output_dir / "profile_metadata.json",
        {
            "profile_id": _profile_id(cfg),
            "python": sys.executable,
            "platform": platform.platform(),
            "device": str(device),
            "precision": cfg.precision,
            "warmup_steps": int(warmup_steps),
            "measured_steps": int(measured_steps),
            "repeats": int(repeats),
            "task": cfg.task,
            "field_dataset": cfg.field_dataset,
            "n_nodes": int(cfg.n_nodes),
            "hidden_dim": int(cfg.hidden_dim),
            "edge_dim": int(cfg.edge_dim),
            "field_adjacency_mode": getattr(cfg, "field_adjacency_mode", ""),
            "field_input_scope": getattr(cfg, "field_input_scope", ""),
            "field_input_steps": int(getattr(cfg, "field_input_steps", 0)),
            "field_output_dim": int(getattr(cfg, "field_output_dim", 1)),
            "field_target_steps": int(getattr(cfg, "field_target_steps", 0)),
            "field_target_steps_choices": str(getattr(cfg, "field_target_steps_choices", "")),
            "outer_steps": int(cfg.outer_steps),
            "inner_steps": int(cfg.inner_steps),
            "pair_chunk_size": int(cfg.pair_chunk_size),
            "activation_checkpoint_inner": bool(cfg.activation_checkpoint_inner),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile one WCA forward/backward step.")
    add_common_cli_args(parser)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()
    cfg = config_from_args(args)
    if cfg.task == "maze":
        cfg.n_nodes = cfg.grid_size * cfg.grid_size
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)
    model = _make_model(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = make_grad_scaler(device, cfg.precision)
    rows: list[dict[str, Any]] = []
    profile_id = _profile_id(cfg)
    if args.warmup_steps < 0 or args.steps <= 0 or args.repeats <= 0:
        raise ValueError("--warmup-steps must be >= 0 and --steps/--repeats must be positive")
    for repeat in range(args.repeats):
        for _ in range(args.warmup_steps):
            _run_profile_step(model=model, optimizer=optimizer, scaler=scaler, cfg=cfg, device=device)
        for step in range(args.steps):
            loss, elapsed, memory = _run_profile_step(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                cfg=cfg,
                device=device,
            )
            rows.append(
                {
                    "profile_id": profile_id,
                    "repeat": repeat,
                    "step": step,
                    "task": cfg.task,
                    "field_dataset": cfg.field_dataset,
                    "device": str(device),
                    "precision": cfg.precision,
                    "batch_size": int(cfg.batch_size),
                    "n_nodes": int(cfg.n_nodes),
                    "hidden_dim": int(cfg.hidden_dim),
                    "edge_dim": int(cfg.edge_dim),
                    "field_adjacency_mode": getattr(cfg, "field_adjacency_mode", ""),
                    "field_input_scope": getattr(cfg, "field_input_scope", ""),
                    "field_input_steps": int(getattr(cfg, "field_input_steps", 0)),
                    "field_output_dim": int(getattr(cfg, "field_output_dim", 1)),
                    "field_target_steps": int(getattr(cfg, "field_target_steps", 0)),
                    "field_target_steps_choices": str(getattr(cfg, "field_target_steps_choices", "")),
                    "outer_steps": int(cfg.outer_steps),
                    "inner_steps": int(cfg.inner_steps),
                    "pair_chunk_size": int(cfg.pair_chunk_size),
                    "activation_checkpoint_inner": bool(cfg.activation_checkpoint_inner),
                    "loss": loss,
                    "step_seconds": elapsed,
                    "samples_per_second": float(cfg.batch_size) / elapsed if elapsed > 0 else float("nan"),
                    **memory,
                }
            )
    elapsed_values = [float(row["step_seconds"]) for row in rows]
    print(f"device={device}")
    print(f"precision={cfg.precision}")
    print(f"steps={args.steps}")
    print(f"warmup_steps={args.warmup_steps}")
    print(f"repeats={args.repeats}")
    print(f"avg_step_seconds={sum(elapsed_values) / max(1, len(elapsed_values)):.6f}")
    for key, value in (rows[-1] if rows else cuda_memory_metrics(device)).items():
        if not str(key).startswith("cuda_"):
            continue
        print(f"{key}={value:.6f}")
    if args.output_dir:
        write_profile_artifacts(
            Path(args.output_dir),
            cfg=cfg,
            device=device,
            rows=rows,
            warmup_steps=args.warmup_steps,
            measured_steps=args.steps,
            repeats=args.repeats,
        )
        print(f"profile_artifacts={args.output_dir}")


if __name__ == "__main__":
    main()
