from __future__ import annotations

from dataclasses import fields
import math
from typing import Dict, List, Optional

import torch
from torch import nn

from wca.config import Config
from wca.data.batch import generate_source_distractor_batch
from wca.data.field.real_cache import SUPPORTED_REAL_FIELD_DATASETS, make_real_field_batch
from wca.data.field.synthetic import make_field_batch, parse_field_target_steps_choices
from wca.data.maze.batch import make_maze_batch
from wca.data.maze.generator import MazeSpec
from wca.data.maze.metrics import compute_metrics
from wca.models.diagnostics import diagnostics_metrics
from wca.models.field_wca import FieldTokenizerWCA
from wca.schemas import TensorBatch
from wca.utils.distributed import DistributedContext, reduce_metric_dict
from wca.training.prediction import predict_for_task
from wca.utils.precision import autocast_context


def make_batch(
    cfg: Config,
    device: torch.device,
    maze_pool: Optional[List[MazeSpec]] = None,
    field_split: str = "train",
) -> TensorBatch:
    if cfg.task == "source":
        return generate_source_distractor_batch(cfg, device)
    if cfg.task == "maze":
        return make_maze_batch(cfg, device, maze_pool=maze_pool)
    if cfg.task == "field":
        if cfg.field_dataset in SUPPORTED_REAL_FIELD_DATASETS:
            return make_real_field_batch(cfg, device, split=field_split)
        return make_field_batch(cfg, device)
    raise ValueError(f"Unknown task: {cfg.task}")


def _model_module(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _forward_task_prediction(model: nn.Module, cfg: Config, batch: TensorBatch) -> tuple[Tensor, Dict[str, Tensor]]:
    module = _model_module(model)
    if isinstance(module, FieldTokenizerWCA):
        return model(batch, cfg.outer_steps)
    input_visibility = batch.get("input_visibility")
    if input_visibility is None:
        H_final, diagnostics = model(batch["H"], batch["adjacency"], cfg.outer_steps)
    else:
        H_final, diagnostics = model(
            batch["H"],
            batch["adjacency"],
            cfg.outer_steps,
            input_visibility=input_visibility,
            input_visibility_channels=batch.get("input_visibility_channels"),
        )
    prediction = predict_for_task(module, cfg, H_final, batch)
    return prediction, diagnostics


def _clone_config(cfg: Config) -> Config:
    allowed = {field.name for field in fields(Config)}
    data = {key: value for key, value in cfg.to_dict().items() if key in allowed}
    return Config(**data)


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def field_checkpoint_horizons(cfg: Config) -> list[int]:
    raw = getattr(cfg, "field_checkpoint_horizons", "")
    if raw:
        return parse_field_target_steps_choices(raw)
    raw_choices = getattr(cfg, "field_target_steps_choices", "")
    if raw_choices:
        return parse_field_target_steps_choices(raw_choices)
    return [int(cfg.field_target_steps)]


def _field_checkpoint_horizon_max(cfg: Config, horizons: list[int]) -> int:
    explicit = int(getattr(cfg, "field_horizon_max_steps", 0) or 0)
    if explicit > 0:
        return explicit
    raw_choices = getattr(cfg, "field_target_steps_choices", "")
    choices = parse_field_target_steps_choices(raw_choices) if raw_choices else []
    return max([*choices, *horizons, int(cfg.field_target_steps)])


def field_checkpoint_weights(cfg: Config, horizon_count: int) -> list[float]:
    raw = getattr(cfg, "field_checkpoint_score_weights", "")
    if not raw:
        return [1.0 / float(horizon_count)] * horizon_count
    weights = _parse_float_list(raw)
    if len(weights) != horizon_count:
        raise ValueError(
            "field_checkpoint_score_weights must match the number of checkpoint horizons. "
            f"got {len(weights)} weights for {horizon_count} horizons."
        )
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("field_checkpoint_score_weights must sum to a positive value.")
    return [weight / total for weight in weights]


def _finite(value: float) -> bool:
    return not (math.isnan(value) or math.isinf(value))


def _horizon_eval_seed(cfg: Config, horizon: int, rank: int) -> int:
    offset = int(getattr(cfg, "field_checkpoint_seed_offset", 910000))
    return int(cfg.seed) + offset + int(horizon) * 1009 + int(rank) * 1000003


def _fork_rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [device.index if device.index is not None else torch.cuda.current_device()]


def _prepare_fixed_horizon_eval_config(cfg: Config, horizon: int, horizon_max: int) -> Config:
    horizon_cfg = _clone_config(cfg)
    horizon_cfg.field_horizon_max_steps = int(horizon_max)
    horizon_cfg.field_target_steps_choices = ""
    horizon_cfg.field_target_steps = int(horizon)
    horizon_cfg.eval_batches = int(getattr(cfg, "field_checkpoint_eval_batches", 0) or cfg.eval_batches)
    return horizon_cfg


@torch.no_grad()
def evaluate_field_horizon_stratified(
    model: nn.Module,
    cfg: Config,
    device: torch.device,
    ctx: Optional[DistributedContext] = None,
    maze_pool: Optional[List[MazeSpec]] = None,
) -> Dict[str, float]:
    if cfg.task != "field":
        return {}
    horizons = field_checkpoint_horizons(cfg)
    if not horizons:
        return {}
    weights = field_checkpoint_weights(cfg, len(horizons))
    score_metric = getattr(cfg, "field_checkpoint_score_metric", "relative_mse")
    if score_metric not in {"relative_mse", "mse"}:
        raise ValueError(f"Unsupported field_checkpoint_score_metric: {score_metric!r}")

    output: Dict[str, float] = {}
    score = 0.0
    valid_score_count = 0
    rank = ctx.rank if ctx is not None else 0
    horizon_max = _field_checkpoint_horizon_max(cfg, horizons)
    for horizon, weight in zip(horizons, weights, strict=True):
        horizon_cfg = _prepare_fixed_horizon_eval_config(cfg, horizon, horizon_max)
        with torch.random.fork_rng(devices=_fork_rng_devices(device)):
            torch.manual_seed(_horizon_eval_seed(cfg, horizon, rank))
            metrics = evaluate(model, horizon_cfg, device, ctx=ctx, maze_pool=maze_pool)
        mse = float(metrics.get("eval_mse", float("nan")))
        persistence_mse = float(metrics.get("eval_field_persistence_mse", float("nan")))
        relative_mse = mse / persistence_mse if _finite(mse) and persistence_mse > 0.0 else float("nan")
        output[f"eval_h{horizon}_mse"] = mse
        output[f"eval_h{horizon}_field_persistence_mse"] = persistence_mse
        output[f"eval_h{horizon}_field_relative_mse"] = relative_mse
        output[f"eval_h{horizon}_field_mse_improvement_vs_persistence"] = float(
            metrics.get("eval_field_mse_improvement_vs_persistence", float("nan"))
        )
        output[f"eval_h{horizon}_field_relative_l2"] = float(
            metrics.get("eval_field_relative_l2", float("nan"))
        )
        score_term = relative_mse if score_metric == "relative_mse" else mse
        if _finite(score_term):
            score += weight * score_term
            valid_score_count += 1
    output["eval_field_horizon_stratified_score"] = score if valid_score_count > 0 else float("nan")
    output["eval_field_horizon_stratified_horizon_count"] = float(len(horizons))
    output["eval_field_horizon_stratified_valid_count"] = float(valid_score_count)
    return output


@torch.no_grad()
def evaluate(
    model: nn.Module,
    cfg: Config,
    device: torch.device,
    ctx: Optional[DistributedContext] = None,
    maze_pool: Optional[List[MazeSpec]] = None,
) -> Dict[str, float]:
    return _evaluate_with_prefix(model, cfg, device, prefix="eval", ctx=ctx, maze_pool=maze_pool)


@torch.no_grad()
def evaluate_pool(
    model: nn.Module,
    cfg: Config,
    device: torch.device,
    prefix: str,
    ctx: Optional[DistributedContext] = None,
    maze_pool: Optional[List[MazeSpec]] = None,
) -> Dict[str, float]:
    return _evaluate_with_prefix(model, cfg, device, prefix=prefix, ctx=ctx, maze_pool=maze_pool)


def _evaluate_with_prefix(
    model: nn.Module,
    cfg: Config,
    device: torch.device,
    prefix: str,
    ctx: Optional[DistributedContext] = None,
    maze_pool: Optional[List[MazeSpec]] = None,
) -> Dict[str, float]:
    if not prefix:
        raise ValueError("Metric prefix must be non-empty.")

    model.eval()
    metric_sums: Dict[str, float] = {}
    valid_counts: Dict[str, int] = {}
    module = _model_module(model)
    for _ in range(cfg.eval_batches):
        if cfg.task == "field":
            batch = make_batch(cfg, device, maze_pool=maze_pool, field_split="eval")
        else:
            batch = make_batch(cfg, device, maze_pool=maze_pool)
        with autocast_context(device, cfg.precision):
            prediction, diagnostics = _forward_task_prediction(model, cfg, batch)
        metrics = compute_metrics(cfg.task, prediction, batch, cfg.grid_size)
        metrics.update(diagnostics_metrics(diagnostics))
        for key, value in metrics.items():
            value = float(value)
            out_key = f"{prefix}_{key}"
            metric_sums.setdefault(out_key, 0.0)
            valid_counts.setdefault(out_key, 0)
            if math.isnan(value):
                continue
            metric_sums[out_key] = metric_sums.get(out_key, 0.0) + value
            valid_counts[out_key] = valid_counts.get(out_key, 0) + 1
    averaged = {
        key: (metric_sums[key] / valid_counts[key] if valid_counts[key] > 0 else float("nan"))
        for key in metric_sums
    }
    if ctx is not None:
        averaged = reduce_metric_dict(averaged, device, ctx)
    return averaged
