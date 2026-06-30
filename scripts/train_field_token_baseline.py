#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wca.config import add_common_cli_args, config_from_args
from wca.data.field.real_cache import make_real_field_batch
from wca.data.field.synthetic import (
    configure_field_nodes,
    field_horizon_features,
    field_token_shape,
    parse_field_target_steps_choices,
)
from wca.utils.device import resolve_device, sync_device
from wca.utils.precision import autocast_context, cuda_memory_metrics, make_grad_scaler
from wca.utils.seed import set_seed


def _as_token_channels(tokens: Tensor) -> Tensor:
    if tokens.ndim == 2:
        return tokens.unsqueeze(-1)
    if tokens.ndim == 3:
        return tokens
    raise ValueError(f"Expected token tensor [B,N] or [B,N,C], got {tuple(tokens.shape)}")


def token_features_from_batch(batch: Dict[str, Tensor], cfg) -> Tensor:
    current = _as_token_channels(batch["field_prediction_baseline"])
    previous_raw = batch.get("field_previous_tokens")
    previous = current if not isinstance(previous_raw, Tensor) else _as_token_channels(previous_raw)
    if previous.shape != current.shape:
        raise ValueError(f"field_previous_tokens shape {tuple(previous.shape)} must match {tuple(current.shape)}")

    batch_size, n_nodes, _channels = current.shape
    token_height, token_width = field_token_shape(cfg)
    if n_nodes != token_height * token_width:
        raise ValueError(f"Token count {n_nodes} does not match configured token grid {token_height}x{token_width}")

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, token_height, device=current.device, dtype=current.dtype),
        torch.linspace(-1.0, 1.0, token_width, device=current.device, dtype=current.dtype),
        indexing="ij",
    )
    coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1).unsqueeze(0).expand(batch_size, n_nodes, 2)
    parts = [current, previous.to(device=current.device, dtype=current.dtype), coords]
    if bool(getattr(cfg, "field_horizon_conditioning", False)):
        target_steps = int(batch["field_target_steps_actual"][0].item())
        features = field_horizon_features(cfg, target_steps, current.device, current.dtype)
        parts.append(features.view(1, 1, 4).expand(batch_size, n_nodes, 4))
    return torch.cat(parts, dim=-1)


class TokenMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int = 128, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(max(1, depth - 1)):
            layers.extend([nn.Linear(current_dim, width), nn.SiLU()])
            current_dim = width
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, token_features: Tensor) -> Tensor:
        return self.net(token_features)


class TokenConvNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, token_shape: tuple[int, int], width: int = 64, depth: int = 4) -> None:
        super().__init__()
        self.token_shape = token_shape
        layers: list[nn.Module] = [nn.Conv2d(input_dim, width, 3, padding=1), nn.SiLU()]
        for _ in range(max(1, depth - 2)):
            layers.extend([nn.Conv2d(width, width, 3, padding=1), nn.SiLU()])
        layers.append(nn.Conv2d(width, output_dim, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, token_features: Tensor) -> Tensor:
        batch_size, n_nodes, channels = token_features.shape
        token_height, token_width = self.token_shape
        if n_nodes != token_height * token_width:
            raise ValueError(f"Token count {n_nodes} does not match token grid {token_height}x{token_width}")
        grid = token_features.transpose(1, 2).reshape(batch_size, channels, token_height, token_width)
        output = self.net(grid)
        return output.reshape(batch_size, output.shape[1], n_nodes).transpose(1, 2)


def token_prediction_baseline(batch: Dict[str, Tensor], cfg, reference: Tensor) -> Tensor:
    current = _as_token_channels(batch["field_prediction_baseline"]).to(device=reference.device, dtype=reference.dtype)
    if not bool(getattr(cfg, "field_tendency_baseline", False)):
        return current
    previous = _as_token_channels(batch["field_previous_tokens"]).to(device=reference.device, dtype=reference.dtype)
    horizon = batch.get("field_target_steps_actual")
    if isinstance(horizon, Tensor):
        horizon_value = horizon.to(device=reference.device, dtype=reference.dtype).view(-1, 1, 1)
    else:
        horizon_value = reference.new_tensor(float(getattr(cfg, "field_target_steps", 1))).view(1, 1, 1)
    return current + float(getattr(cfg, "field_tendency_scale", 1.0)) * horizon_value * (current - previous)


def _restore_target_rank(prediction: Tensor, target: Tensor) -> Tensor:
    if target.ndim == 2 and prediction.ndim == 3 and prediction.shape[-1] == 1:
        return prediction.squeeze(-1)
    return prediction


def predict_tokens(model: nn.Module, batch: Dict[str, Tensor], cfg) -> Tuple[Tensor, Tensor]:
    raw_prediction = model(token_features_from_batch(batch, cfg))
    target = batch["label"]
    if bool(getattr(cfg, "field_residual_readout", False)):
        baseline = token_prediction_baseline(batch, cfg, raw_prediction)
        prediction = baseline + float(getattr(cfg, "field_residual_scale", 1.0)) * raw_prediction
    else:
        prediction = raw_prediction
    return _restore_target_rank(prediction, target), target


def _metric_suffix(name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in name).strip("_")


def field_metrics(prediction: Tensor, batch: Dict[str, Tensor]) -> Dict[str, float]:
    target = batch["label"]
    persistence = batch["field_prediction_baseline"]
    mse = F.mse_loss(prediction, target)
    mae = (prediction - target).abs().mean()
    persistence_mse = F.mse_loss(persistence, target)
    persistence_mae = (persistence - target).abs().mean()
    relative_l2 = (prediction - target).pow(2).sum().sqrt() / target.pow(2).sum().sqrt().clamp_min(1e-8)
    improvement = (persistence_mse - mse) / persistence_mse.clamp_min(1e-8)
    metrics = {
        "eval_mse": float(mse.item()),
        "eval_mae": float(mae.item()),
        "eval_field_relative_l2": float(relative_l2.item()),
        "eval_field_persistence_mse": float(persistence_mse.item()),
        "eval_field_persistence_mae": float(persistence_mae.item()),
        "eval_field_mse_improvement_vs_persistence": float(improvement.item()),
        "eval_field_patch_count": float(target.shape[1]),
    }
    variable_names = [item.strip() for item in str(batch.get("field_variable", "")).split(",") if item.strip()]
    if prediction.ndim == 3 and target.ndim == 3 and variable_names and prediction.shape[-1] == len(variable_names):
        for index, variable_name in enumerate(variable_names):
            suffix = _metric_suffix(variable_name)
            variable_prediction = prediction[..., index]
            variable_target = target[..., index]
            variable_persistence = persistence[..., index]
            variable_mse = F.mse_loss(variable_prediction, variable_target)
            variable_mae = (variable_prediction - variable_target).abs().mean()
            variable_persistence_mse = F.mse_loss(variable_persistence, variable_target)
            variable_persistence_mae = (variable_persistence - variable_target).abs().mean()
            variable_improvement = (variable_persistence_mse - variable_mse) / variable_persistence_mse.clamp_min(1e-8)
            metrics.update(
                {
                    f"eval_field_mse_{suffix}": float(variable_mse.item()),
                    f"eval_field_mae_{suffix}": float(variable_mae.item()),
                    f"eval_field_persistence_mse_{suffix}": float(variable_persistence_mse.item()),
                    f"eval_field_persistence_mae_{suffix}": float(variable_persistence_mae.item()),
                    f"eval_field_mse_improvement_vs_persistence_{suffix}": float(variable_improvement.item()),
                }
            )
    return metrics


@torch.no_grad()
def evaluate(model: nn.Module, cfg, device: torch.device) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, float] = {}
    for _ in range(cfg.eval_batches):
        batch = make_real_field_batch(cfg, device, split="eval")
        prediction, _target = predict_tokens(model, batch, cfg)
        metrics = field_metrics(prediction, batch)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value)
    return {key: value / float(cfg.eval_batches) for key, value in sums.items()}


def _clone_config(cfg) -> object:
    return type(cfg)(**asdict(cfg))


def _checkpoint_horizons(cfg) -> list[int]:
    raw = getattr(cfg, "field_checkpoint_horizons", "")
    if raw:
        return parse_field_target_steps_choices(raw)
    raw_choices = getattr(cfg, "field_target_steps_choices", "")
    if raw_choices:
        return parse_field_target_steps_choices(raw_choices)
    return [int(cfg.field_target_steps)]


def _checkpoint_weights(cfg, count: int) -> list[float]:
    raw = getattr(cfg, "field_checkpoint_score_weights", "")
    if not raw:
        return [1.0 / float(count)] * count
    weights = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(weights) != count:
        raise ValueError(
            "field_checkpoint_score_weights must match field_checkpoint_horizons "
            f"({len(weights)} weights for {count} horizons)."
        )
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("field_checkpoint_score_weights must sum to a positive value.")
    return [weight / total for weight in weights]


def _horizon_max(cfg, horizons: list[int]) -> int:
    explicit = int(getattr(cfg, "field_horizon_max_steps", 0) or 0)
    if explicit > 0:
        return explicit
    choices = parse_field_target_steps_choices(cfg.field_target_steps_choices) if cfg.field_target_steps_choices else []
    return max([*choices, *horizons, int(cfg.field_target_steps)])


def evaluate_horizon_stratified(model: nn.Module, cfg, device: torch.device) -> Dict[str, float]:
    horizons = _checkpoint_horizons(cfg)
    weights = _checkpoint_weights(cfg, len(horizons))
    score_metric = getattr(cfg, "field_checkpoint_score_metric", "relative_mse")
    if score_metric not in {"relative_mse", "mse"}:
        raise ValueError(f"Unsupported field_checkpoint_score_metric: {score_metric!r}")
    horizon_max = _horizon_max(cfg, horizons)
    output: Dict[str, float] = {}
    score = 0.0
    valid = 0
    for horizon, weight in zip(horizons, weights, strict=True):
        horizon_cfg = _clone_config(cfg)
        horizon_cfg.field_horizon_max_steps = int(horizon_max)
        horizon_cfg.field_target_steps_choices = ""
        horizon_cfg.field_target_steps = int(horizon)
        horizon_cfg.eval_batches = int(getattr(cfg, "field_checkpoint_eval_batches", 0) or cfg.eval_batches)
        torch.manual_seed(int(cfg.seed) + 910000 + int(horizon) * 1009)
        metrics = evaluate(model, horizon_cfg, device)
        mse = float(metrics.get("eval_mse", float("nan")))
        persistence_mse = float(metrics.get("eval_field_persistence_mse", float("nan")))
        relative_mse = mse / persistence_mse if math.isfinite(mse) and persistence_mse > 0.0 else float("nan")
        output[f"eval_h{horizon}_mse"] = mse
        output[f"eval_h{horizon}_field_persistence_mse"] = persistence_mse
        output[f"eval_h{horizon}_field_relative_mse"] = relative_mse
        score_term = relative_mse if score_metric == "relative_mse" else mse
        if math.isfinite(score_term):
            score += weight * score_term
            valid += 1
    output["eval_field_horizon_stratified_score"] = score if valid else float("nan")
    output["eval_field_horizon_stratified_horizon_count"] = float(len(horizons))
    output["eval_field_horizon_stratified_valid_count"] = float(valid)
    return output


def build_model(kind: str, input_dim: int, output_dim: int, token_shape: tuple[int, int], width: int, depth: int) -> nn.Module:
    if kind == "token_mlp":
        return TokenMLP(input_dim=input_dim, output_dim=output_dim, width=width, depth=depth)
    if kind == "token_conv":
        return TokenConvNet(input_dim=input_dim, output_dim=output_dim, token_shape=token_shape, width=width, depth=depth)
    raise ValueError(f"Unsupported token baseline model: {kind}")


def write_summary(run_dir: Path, model_name: str, cfg, final_metrics: Dict[str, float], best_metrics: Dict[str, float]) -> None:
    summary = {
        "model": model_name,
        "baseline_model": getattr(cfg, "baseline_model", ""),
        "field_baseline_scope": "token_equivalent",
        "baseline_width": int(getattr(cfg, "baseline_width", 0) or 0),
        "baseline_depth": int(getattr(cfg, "baseline_depth", 0) or 0),
        "run_dir": str(run_dir),
        "config": asdict(cfg),
        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
        "checkpoint_score": getattr(cfg, "checkpoint_score", ""),
        "checkpoint_selection_metric": best_metrics.get("best_metric_name", ""),
        "best_checkpoint": str(run_dir / "best_model.pt"),
        "final_checkpoint": str(run_dir / "final_model.pt"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def input_dim_for_config(cfg) -> int:
    output_dim = int(getattr(cfg, "field_output_dim", 1))
    horizon_dim = 4 if bool(getattr(cfg, "field_horizon_conditioning", False)) else 0
    return output_dim * 2 + 2 + horizon_dim


def main() -> None:
    parser = argparse.ArgumentParser(description="Train token-equivalent field baselines on real field cache data.")
    add_common_cli_args(parser)
    parser.add_argument(
        "--token-baseline-model",
        choices=["token_mlp", "token_conv"],
        default=None,
        help="Token-equivalent baseline kind. Overrides baseline_model from the config file.",
    )
    args = parser.parse_args()
    cfg = config_from_args(args)
    if args.token_baseline_model is not None:
        cfg.baseline_model = args.token_baseline_model
    if cfg.task != "field":
        raise SystemExit("scripts/train_field_token_baseline.py requires task: field")
    if cfg.baseline_model not in {"token_mlp", "token_conv"}:
        raise SystemExit("--token-baseline-model or baseline_model is required and must be token_mlp or token_conv")
    if cfg.baseline_width <= 0:
        cfg.baseline_width = 128 if cfg.baseline_model == "token_mlp" else 64
    if cfg.baseline_depth <= 0:
        cfg.baseline_depth = 3 if cfg.baseline_model == "token_mlp" else 4
    cfg.field_baseline_scope = "token_equivalent"
    configure_field_nodes(cfg)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    cfg.device = str(device)

    model = build_model(
        cfg.baseline_model,
        input_dim_for_config(cfg),
        int(getattr(cfg, "field_output_dim", 1)),
        field_token_shape(cfg),
        cfg.baseline_width,
        cfg.baseline_depth,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = make_grad_scaler(device, cfg.precision)
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fieldnames = [
        "epoch",
        "loss",
        "eval_mse",
        "eval_mae",
        "eval_field_relative_l2",
        "eval_field_persistence_mse",
        "eval_field_persistence_mae",
        "eval_field_mse_improvement_vs_persistence",
        "eval_field_horizon_stratified_score",
        "eval_field_horizon_stratified_horizon_count",
        "eval_field_horizon_stratified_valid_count",
        "cuda_peak_memory_allocated_mb",
        "cuda_peak_memory_reserved_mb",
        "step_seconds",
    ]
    best_metrics: Dict[str, float] = {}
    best_score = math.inf
    final_metrics: Dict[str, float] = {}
    with (run_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            batch = make_real_field_batch(cfg, device, split="train")
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, cfg.precision):
                prediction, target = predict_tokens(model, batch, cfg)
                loss = F.mse_loss(prediction, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.max_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            sync_device(device)
            step_seconds = time.perf_counter() - start

            if epoch == 1 or epoch % cfg.log_every == 0 or epoch == cfg.epochs:
                final_metrics = evaluate(model, cfg, device)
                checkpoint_metrics = dict(final_metrics)
                if str(getattr(cfg, "checkpoint_score", "")) == "field_horizon_stratified":
                    checkpoint_metrics.update(evaluate_horizon_stratified(model, cfg, device))
                metric_name = (
                    "eval_field_horizon_stratified_score"
                    if str(getattr(cfg, "checkpoint_score", "")) == "field_horizon_stratified"
                    else "eval_mse"
                )
                candidate_score = float(checkpoint_metrics.get(metric_name, float("inf")))
                if candidate_score < best_score:
                    best_score = candidate_score
                    best_metrics = {
                        "best_metric_name": metric_name,
                        "best_metric": best_score,
                        "best_metric_epoch": float(epoch),
                        "best_eval_mse": checkpoint_metrics["eval_mse"],
                        "best_eval_mse_epoch": float(epoch),
                        "best_eval_mae": checkpoint_metrics["eval_mae"],
                        "best_eval_mae_epoch": float(epoch),
                        "best_loss": float(loss.item()),
                        "best_loss_epoch": float(epoch),
                    }
                    torch.save(model.state_dict(), run_dir / "best_model.pt")
                    torch.save(model.state_dict(), run_dir / "model.pt")
                row = {
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    **checkpoint_metrics,
                    **cuda_memory_metrics(device),
                    "step_seconds": step_seconds,
                }
                writer.writerow({key: row.get(key, "") for key in fieldnames})
                file.flush()

    torch.save(model.state_dict(), run_dir / "final_model.pt")
    if not (run_dir / "best_model.pt").exists():
        torch.save(model.state_dict(), run_dir / "best_model.pt")
        torch.save(model.state_dict(), run_dir / "model.pt")
    write_summary(run_dir, f"{cfg.baseline_model}-field-token-baseline", cfg, final_metrics, best_metrics)
    print(json.dumps({"run_dir": str(run_dir), "final_metrics": final_metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
