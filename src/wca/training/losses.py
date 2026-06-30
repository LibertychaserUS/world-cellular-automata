from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from wca.schemas import TensorBatch


def masked_maze_mse(prediction: Tensor, batch: TensorBatch) -> Tensor:
    field = batch["distance_field"]
    mask = batch["distance_mask"].to(dtype=prediction.dtype)
    masked_count = mask.sum().clamp_min(1.0)
    return (((prediction - field) ** 2) * mask).sum() / masked_count


def _nonself_open_edges(batch: TensorBatch, prediction: Tensor) -> Tensor:
    adjacency = batch["adjacency"].to(device=prediction.device)
    if adjacency.ndim == 2:
        adjacency = adjacency.unsqueeze(0).expand(prediction.shape[0], -1, -1)
    open_mask = batch["distance_mask"].to(device=prediction.device) > 0.0
    n_nodes = prediction.shape[1]
    nonself = ~torch.eye(n_nodes, dtype=torch.bool, device=prediction.device).unsqueeze(0)
    return (adjacency > 0.0) & open_mask.unsqueeze(1) & open_mask.unsqueeze(2) & nonself


def _goal_excluded_open_mask(batch: TensorBatch, prediction: Tensor) -> Tensor:
    open_mask = batch["distance_mask"].to(device=prediction.device) > 0.0
    batch_idx = torch.arange(prediction.shape[0], device=prediction.device)
    node_mask = open_mask.clone()
    node_mask[batch_idx, batch["goal_idx"].to(device=prediction.device).long()] = False
    return node_mask


def _zero_like_prediction(prediction: Tensor) -> Tensor:
    return prediction.sum() * 0.0


def _large_finite_sentinel(prediction: Tensor, shape: torch.Size) -> Tensor:
    # Avoid +/-inf sentinels here: masked min with inf can keep the forward
    # value finite while producing non-finite gradients through autograd.
    sentinel = prediction.detach().abs().max().clamp_min(1.0) + 1_000_000.0
    return sentinel.expand(shape)


def neighbor_rank_loss(prediction: Tensor, batch: TensorBatch, margin: float = 0.01) -> Tensor:
    true_field = batch["distance_field"].to(device=prediction.device)
    edges = _nonself_open_edges(batch, prediction)
    descending_edges = edges & (true_field.unsqueeze(1) < true_field.unsqueeze(2))
    if not bool(descending_edges.any().item()):
        return _zero_like_prediction(prediction)
    violations = prediction.unsqueeze(1) - prediction.unsqueeze(2) + margin
    return F.relu(violations[descending_edges]).mean()


def descent_margin_loss(prediction: Tensor, batch: TensorBatch, margin: float = 0.01) -> Tensor:
    true_field = batch["distance_field"].to(device=prediction.device)
    edges = _nonself_open_edges(batch, prediction)
    descending_edges = edges & (true_field.unsqueeze(1) < true_field.unsqueeze(2))
    node_mask = _goal_excluded_open_mask(batch, prediction) & descending_edges.any(dim=2)
    if not bool(node_mask.any().item()):
        return _zero_like_prediction(prediction)

    sentinel = _large_finite_sentinel(prediction, descending_edges.shape)
    lower_neighbor_predictions = torch.where(descending_edges, prediction.unsqueeze(1), sentinel)
    best_lower_neighbor = lower_neighbor_predictions.min(dim=2).values
    violations = best_lower_neighbor - prediction + margin
    return F.relu(violations[node_mask]).mean()


def bellman_residual_loss(prediction: Tensor, batch: TensorBatch, step_scale: float = 1.0) -> Tensor:
    edges = _nonself_open_edges(batch, prediction)
    node_mask = _goal_excluded_open_mask(batch, prediction) & edges.any(dim=2)
    if not bool(node_mask.any().item()):
        return _zero_like_prediction(prediction)

    sentinel = _large_finite_sentinel(prediction, edges.shape)
    neighbor_predictions = torch.where(edges, prediction.unsqueeze(1), sentinel)
    best_neighbor = neighbor_predictions.min(dim=2).values
    target = float(step_scale) + best_neighbor
    return (((prediction - target) ** 2)[node_mask]).mean()


def _weight_from_cfg(cfg: Any, name: str, default: float) -> float:
    return float(getattr(cfg, name, default)) if cfg is not None else default


def compute_task_loss(task: str, prediction: Tensor, batch: TensorBatch, cfg: Any = None) -> Tensor:
    if task == "maze":
        loss = masked_maze_mse(prediction, batch) * _weight_from_cfg(cfg, "field_loss_weight", 1.0)
        neighbor_weight = _weight_from_cfg(cfg, "neighbor_rank_loss_weight", 0.0)
        descent_weight = _weight_from_cfg(cfg, "descent_margin_loss_weight", 0.0)
        bellman_weight = _weight_from_cfg(cfg, "bellman_loss_weight", 0.0)
        margin = _weight_from_cfg(cfg, "loss_margin", 0.01)
        if neighbor_weight:
            loss = loss + neighbor_weight * neighbor_rank_loss(prediction, batch, margin=margin)
        if descent_weight:
            loss = loss + descent_weight * descent_margin_loss(prediction, batch, margin=margin)
        if bellman_weight:
            step_scale = _weight_from_cfg(cfg, "bellman_step_scale", 1.0)
            loss = loss + bellman_weight * bellman_residual_loss(prediction, batch, step_scale=step_scale)
        return loss
    if task == "field":
        return F.mse_loss(prediction, batch["label"])
    return F.mse_loss(prediction, batch["label"])
