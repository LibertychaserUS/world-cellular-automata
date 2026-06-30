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
    field_patch_shape,
    parse_field_target_steps_choices,
    patchify_field,
)
from wca.utils.device import resolve_device, sync_device
from wca.utils.precision import autocast_context, cuda_memory_metrics, make_grad_scaler
from wca.utils.seed import set_seed


class ConvFieldNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 64,
        depth: int = 6,
        condition_channels: int = 0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels * 2 + 2 + condition_channels, width, 3, padding=1), nn.SiLU()]
        for _ in range(max(1, depth - 2)):
            layers.extend([nn.Conv2d(width, width, 3, padding=1), nn.SiLU()])
        layers.append(nn.Conv2d(width, out_channels, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, model_input: Tensor) -> Tensor:
        return self.net(model_input)


class UNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.SiLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TinyUNet2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, width: int = 64, depth: int = 3, condition_channels: int = 0) -> None:
        super().__init__()
        input_channels = in_channels * 2 + 2 + condition_channels
        levels = max(2, depth)
        widths = [width * (2**level) for level in range(levels)]
        self.down_blocks = nn.ModuleList()
        current_channels = input_channels
        for block_width in widths:
            self.down_blocks.append(UNetBlock(current_channels, block_width))
            current_channels = block_width
        self.bottleneck = UNetBlock(widths[-1], widths[-1] * 2)
        self.up_blocks = nn.ModuleList()
        self.up_projections = nn.ModuleList()
        current_channels = widths[-1] * 2
        for skip_width in reversed(widths):
            self.up_projections.append(nn.Conv2d(current_channels, skip_width, 1))
            self.up_blocks.append(UNetBlock(skip_width * 2, skip_width))
            current_channels = skip_width
        self.head = nn.Conv2d(widths[0], out_channels, 1)

    def forward(self, model_input: Tensor) -> Tensor:
        skips: list[Tensor] = []
        x = model_input
        for block in self.down_blocks:
            x = block(x)
            skips.append(x)
            x = F.avg_pool2d(x, kernel_size=2)
        x = self.bottleneck(x)
        for projection, block, skip in zip(self.up_projections, self.up_blocks, reversed(skips), strict=True):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = projection(x)
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x)


class SpectralConv2d(nn.Module):
    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.width = width
        self.modes = modes
        scale = 1.0 / max(1, width * width)
        self.weights = nn.Parameter(scale * torch.randn(width, width, modes, modes, dtype=torch.cfloat))

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        output_dtype = x.dtype
        # torch.fft does not support bfloat16 on CPU/CUDA. Keep the spectral
        # path in fp32 while allowing the surrounding baseline to run under AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_ft = torch.fft.rfft2(x.float())
            out_ft = torch.zeros(batch, channels, height, width // 2 + 1, device=x.device, dtype=torch.cfloat)
            weights = self.weights.to(device=x.device)
            modes_y = min(self.modes, height)
            modes_x = min(self.modes, width // 2 + 1)
            out_ft[:, :, :modes_y, :modes_x] = torch.einsum(
                "bixy,ioxy->boxy",
                x_ft[:, :, :modes_y, :modes_x],
                weights[:, :, :modes_y, :modes_x],
            )
            output = torch.fft.irfft2(out_ft, s=(height, width))
        return output.to(dtype=output_dtype)


class TinyFNO2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 48,
        modes: int = 12,
        depth: int = 4,
        condition_channels: int = 0,
    ) -> None:
        super().__init__()
        self.lift = nn.Conv2d(in_channels * 2 + 2 + condition_channels, width, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, modes) for _ in range(depth)])
        self.pointwise = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.head = nn.Sequential(nn.Conv2d(width, width, 1), nn.SiLU(), nn.Conv2d(width, out_channels, 1))

    def forward(self, model_input: Tensor) -> Tensor:
        x = self.lift(model_input)
        for spectral, pointwise in zip(self.spectral, self.pointwise, strict=True):
            x = F.silu(spectral(x) + pointwise(x))
        return self.head(x)


def model_input_from_batch(batch: Dict[str, Tensor], cfg) -> Tensor:
    current = batch["field_input"][:, -1]
    previous = batch["field_input"][:, -2] if batch["field_input"].shape[1] > 1 else current
    batch_size, _, height, width = current.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=current.device, dtype=current.dtype),
        torch.linspace(-1.0, 1.0, width, device=current.device, dtype=current.dtype),
        indexing="ij",
    )
    coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(batch_size, 2, height, width)
    parts = [current, previous, coords]
    if bool(getattr(cfg, "field_horizon_conditioning", False)):
        target_steps = int(batch["field_target_steps_actual"][0].item())
        features = field_horizon_features(cfg, target_steps, current.device, current.dtype)
        horizon_planes = features.view(1, 4, 1, 1).expand(batch_size, 4, height, width)
        parts.append(horizon_planes)
    return torch.cat(parts, dim=1)


def field_prediction_baseline(batch: Dict[str, Tensor], cfg, raw_field: Tensor) -> Tensor:
    current = batch["field_input"][:, -1].to(device=raw_field.device, dtype=raw_field.dtype)
    if not bool(getattr(cfg, "field_tendency_baseline", False)):
        return current
    previous = batch["field_input"][:, -2].to(device=raw_field.device, dtype=raw_field.dtype)
    horizon = batch.get("field_target_steps_actual")
    if isinstance(horizon, Tensor):
        horizon_value = horizon.to(device=raw_field.device, dtype=raw_field.dtype).view(-1, 1, 1, 1)
    else:
        horizon_value = torch.tensor(float(getattr(cfg, "field_target_steps", 1)), device=raw_field.device, dtype=raw_field.dtype)
    return current + float(getattr(cfg, "field_tendency_scale", 1.0)) * horizon_value * (current - previous)


def predict_tokens(model: nn.Module, batch: Dict[str, Tensor], cfg) -> Tuple[Tensor, Tensor]:
    raw_field = model(model_input_from_batch(batch, cfg))
    if bool(getattr(cfg, "field_residual_readout", False)):
        baseline = field_prediction_baseline(batch, cfg, raw_field)
        prediction_field = baseline + float(getattr(cfg, "field_residual_scale", 1.0)) * raw_field
    else:
        prediction_field = raw_field
    prediction = patchify_field(prediction_field, field_patch_shape(cfg))
    if prediction.shape[-1] == 1:
        prediction = prediction.squeeze(-1)
    target = batch["label"]
    return prediction, target


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


def build_model(kind: str, output_dim: int, width: int, depth: int, modes: int, condition_channels: int = 0) -> nn.Module:
    if kind == "convnet":
        return ConvFieldNet(
            in_channels=output_dim,
            out_channels=output_dim,
            width=width,
            depth=depth,
            condition_channels=condition_channels,
        )
    if kind == "unet":
        return TinyUNet2d(
            in_channels=output_dim,
            out_channels=output_dim,
            width=width,
            depth=depth,
            condition_channels=condition_channels,
        )
    if kind == "fno":
        return TinyFNO2d(
            in_channels=output_dim,
            out_channels=output_dim,
            width=width,
            modes=modes,
            depth=depth,
            condition_channels=condition_channels,
        )
    raise ValueError(f"Unsupported baseline model: {kind}")


def write_summary(run_dir: Path, model_name: str, cfg, final_metrics: Dict[str, float], best_metrics: Dict[str, float]) -> None:
    summary = {
        "model": model_name,
        "baseline_model": getattr(cfg, "baseline_model", ""),
        "baseline_width": int(getattr(cfg, "baseline_width", 0) or 0),
        "baseline_depth": int(getattr(cfg, "baseline_depth", 0) or 0),
        "fno_modes": int(getattr(cfg, "fno_modes", 0) or 0),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train simple ConvNet/FNO field baselines on real field cache data.")
    add_common_cli_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)
    if cfg.task != "field":
        raise SystemExit("scripts/train_field_baseline.py requires task: field")
    if cfg.baseline_model not in {"convnet", "fno", "unet"}:
        raise SystemExit("--baseline-model is required and must be convnet, fno, or unet")
    if cfg.baseline_width <= 0:
        cfg.baseline_width = 64
    if cfg.baseline_depth <= 0:
        cfg.baseline_depth = 6
    if cfg.fno_modes <= 0:
        cfg.fno_modes = 12
    configure_field_nodes(cfg)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    cfg.device = str(device)

    model = build_model(
        cfg.baseline_model,
        int(getattr(cfg, "field_output_dim", 1)),
        cfg.baseline_width,
        cfg.baseline_depth,
        cfg.fno_modes,
        condition_channels=4 if bool(getattr(cfg, "field_horizon_conditioning", False)) else 0,
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
    write_summary(run_dir, f"{cfg.baseline_model}-field-baseline", cfg, final_metrics, best_metrics)
    print(json.dumps({"run_dir": str(run_dir), "final_metrics": final_metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
