from __future__ import annotations

import math
import re
from typing import Dict, List, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from wca.data.maze.oracle import get_grid_neighbors
from wca.schemas import TensorBatch


def source_metrics(prediction: Tensor, batch: TensorBatch) -> Dict[str, float]:
    label = batch["label"]
    distractor = batch["distractor_sign"]
    mse = F.mse_loss(prediction, label).item()
    sign_acc = ((prediction.sign() == label.sign()).float().mean()).item()
    return {
        "mse": mse,
        "mae": (prediction - label).abs().mean().item(),
        "sign_acc": sign_acc,
        "source_alignment": (prediction * label).mean().item(),
        "distractor_alignment": (prediction * distractor).mean().item(),
    }


def _field_variable_names(batch: TensorBatch, channels: int) -> List[str]:
    raw = batch.get("field_variable", "")
    if isinstance(raw, str) and raw:
        names = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        names = []
    if len(names) != channels:
        names = [f"ch{idx}" for idx in range(channels)]
    return [re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower() or f"ch{idx}" for idx, name in enumerate(names)]


def field_metrics(prediction: Tensor, batch: TensorBatch) -> Dict[str, float]:
    label = batch["label"]
    mse = F.mse_loss(prediction, label)
    mae = (prediction - label).abs().mean()
    relative_l2 = (prediction - label).pow(2).sum().sqrt() / label.pow(2).sum().sqrt().clamp_min(1e-8)
    persistence = batch.get("field_prediction_baseline")
    if isinstance(persistence, Tensor) and persistence.shape == label.shape:
        persistence_mse = F.mse_loss(persistence, label)
        persistence_mae = (persistence - label).abs().mean()
        persistence_relative_l2 = (persistence - label).pow(2).sum().sqrt()
        persistence_relative_l2 = persistence_relative_l2 / label.pow(2).sum().sqrt().clamp_min(1e-8)
        target_delta = label - persistence
        pred_delta = prediction - persistence
        delta_mse = F.mse_loss(pred_delta, target_delta)
        delta_mae = (pred_delta - target_delta).abs().mean()
        improvement = (persistence_mse - mse) / persistence_mse.clamp_min(1e-8)
    else:
        persistence_mse = torch.tensor(float("nan"), device=label.device)
        persistence_mae = torch.tensor(float("nan"), device=label.device)
        persistence_relative_l2 = torch.tensor(float("nan"), device=label.device)
        delta_mse = torch.tensor(float("nan"), device=label.device)
        delta_mae = torch.tensor(float("nan"), device=label.device)
        improvement = torch.tensor(float("nan"), device=label.device)
    target_energy = label.pow(2).mean().clamp_min(1e-8)
    pred_energy = prediction.pow(2).mean()
    adjacency = batch.get("adjacency")
    if isinstance(adjacency, Tensor):
        adjacency_density = adjacency.float().mean()
        adjacency_degree = adjacency.float().sum(dim=-1).mean()
    else:
        adjacency_density = torch.tensor(float("nan"), device=label.device)
        adjacency_degree = torch.tensor(float("nan"), device=label.device)
    input_visibility = batch.get("input_visibility")
    if isinstance(input_visibility, Tensor):
        input_visibility_density = input_visibility.float().mean()
        input_visibility_degree = input_visibility.float().sum(dim=-1).mean()
    else:
        input_visibility_density = torch.tensor(float("nan"), device=label.device)
        input_visibility_degree = torch.tensor(float("nan"), device=label.device)
    metrics = {
        "mse": mse.item(),
        "mae": mae.item(),
        "sign_acc": float("nan"),
        "source_alignment": float("nan"),
        "distractor_alignment": float("nan"),
        "mean_label": label.mean().item(),
        "mean_pred": prediction.mean().item(),
        "field_relative_l2": relative_l2.item(),
        "field_energy_error": ((pred_energy - target_energy).abs() / target_energy).item(),
        "field_patch_count": float(label.shape[1]) if label.ndim > 1 else float(label.numel()),
        "field_adjacency_density": adjacency_density.item(),
        "field_adjacency_degree": adjacency_degree.item(),
        "field_input_visibility_density": input_visibility_density.item(),
        "field_input_visibility_degree": input_visibility_degree.item(),
        "field_persistence_mse": persistence_mse.item(),
        "field_persistence_mae": persistence_mae.item(),
        "field_persistence_relative_l2": persistence_relative_l2.item(),
        "field_delta_mse": delta_mse.item(),
        "field_delta_mae": delta_mae.item(),
        "field_mse_improvement_vs_persistence": improvement.item(),
    }
    if label.ndim == 3 and prediction.ndim == 3:
        channels = int(label.shape[-1])
        names = _field_variable_names(batch, channels)
        per_variable: Dict[str, float] = {}
        for idx, name in enumerate(names):
            pred_ch = prediction[..., idx]
            label_ch = label[..., idx]
            ch_mse = F.mse_loss(pred_ch, label_ch)
            ch_mae = (pred_ch - label_ch).abs().mean()
            per_variable[f"field_mse_{name}"] = ch_mse.item()
            per_variable[f"field_mae_{name}"] = ch_mae.item()
            if isinstance(persistence, Tensor) and persistence.shape == label.shape:
                persistence_ch = persistence[..., idx]
                p_mse = F.mse_loss(persistence_ch, label_ch)
                p_mae = (persistence_ch - label_ch).abs().mean()
                ch_improvement = (p_mse - ch_mse) / p_mse.clamp_min(1e-8)
                per_variable[f"field_persistence_mse_{name}"] = p_mse.item()
                per_variable[f"field_persistence_mae_{name}"] = p_mae.item()
                per_variable[f"field_mse_improvement_vs_persistence_{name}"] = ch_improvement.item()
        metrics.update(per_variable)
    return metrics


def maze_metrics(prediction: Tensor, batch: TensorBatch) -> Dict[str, float]:
    raw_distance = batch["raw_distance"]

    if prediction.ndim == 1:
        label = batch["label"]
        mse = F.mse_loss(prediction, label).item()
        mae = (prediction - label).abs().mean().item()
        return {
            "mse": mse,
            "mae": mae,
            "sign_acc": float("nan"),
            "source_alignment": (prediction * label).mean().item(),
            "distractor_alignment": float("nan"),
            "mean_label": label.mean().item(),
            "mean_pred": prediction.mean().item(),
            "mean_raw_distance": raw_distance.mean().item(),
            "distance_mae_steps": float("nan"),
            "start_exact_acc": float("nan"),
        }

    field = batch["distance_field"]
    mask = batch["distance_mask"]
    masked_count = mask.sum().clamp_min(1.0)
    field_mse = (((prediction - field) ** 2) * mask).sum() / masked_count
    field_mae = ((prediction - field).abs() * mask).sum() / masked_count

    batch_arange = torch.arange(prediction.shape[0], device=prediction.device)
    start_pred = prediction[batch_arange, batch["start_idx"]]
    start_label = batch["label"]
    start_mse = F.mse_loss(start_pred, start_label)
    start_mae = (start_pred - start_label).abs().mean()

    max_distance_est = raw_distance / start_label.clamp_min(1e-6)
    pred_steps = torch.round(start_pred * max_distance_est).clamp_min(0.0)
    exact_acc = (pred_steps == raw_distance).float().mean()
    distance_mae_steps = (pred_steps - raw_distance).abs().mean()

    return {
        "mse": field_mse.item(),
        "mae": field_mae.item(),
        "sign_acc": float("nan"),
        "source_alignment": (start_pred * start_label).mean().item(),
        "distractor_alignment": float("nan"),
        "mean_label": start_label.mean().item(),
        "mean_pred": start_pred.mean().item(),
        "mean_raw_distance": raw_distance.mean().item(),
        "start_mse": start_mse.item(),
        "start_mae": start_mae.item(),
        "distance_mae_steps": distance_mae_steps.item(),
        "start_exact_acc": exact_acc.item(),
    }


def _open_mask_from_batch(batch: TensorBatch) -> Tensor:
    if "open_mask" in batch:
        return batch["open_mask"].bool()
    return batch["distance_mask"] > 0.0


def greedy_path_nodes(prediction_field: Tensor, batch: TensorBatch, grid_size: int, sample_idx: int = 0) -> List[int]:
    if prediction_field.ndim != 2:
        return []

    starts = batch["start_idx"].detach()
    goals = batch["goal_idx"].detach()
    open_mask = _open_mask_from_batch(batch).detach()
    n_nodes = prediction_field.shape[1]

    current = int(starts[sample_idx].item())
    goal = int(goals[sample_idx].item())
    path = [current]
    visited = {current}
    max_steps = n_nodes * 2

    for _ in range(max_steps):
        if current == goal:
            break
        candidates = [nxt for nxt in get_grid_neighbors(grid_size)[current] if bool(open_mask[sample_idx, nxt].item())]
        if not candidates:
            break

        next_node = min(candidates, key=lambda idx: float(prediction_field[sample_idx, idx].item()))
        path.append(next_node)
        current = next_node
        if current in visited and current != goal:
            break
        visited.add(current)

    return path


def classify_path(path: List[int], goal: int, true_distance: float) -> Tuple[bool, bool, bool]:
    if not path:
        return False, False, False
    reached = path[-1] == goal
    looped = len(set(path)) < len(path) and not reached
    path_len = max(0, len(path) - 1)
    optimal = reached and abs(float(path_len) - float(true_distance)) < 1e-6
    return reached, optimal, looped


def greedy_path_metrics(prediction_field: Tensor, batch: TensorBatch, grid_size: int) -> Dict[str, float]:
    if prediction_field.ndim != 2:
        return {
            "path_success_rate": float("nan"),
            "path_optimal_rate": float("nan"),
            "path_length_ratio": float("nan"),
            "path_loop_rate": float("nan"),
        }

    batch_size, n_nodes = prediction_field.shape
    raw_distance = batch["raw_distance"].detach()
    starts = batch["start_idx"].detach()
    goals = batch["goal_idx"].detach()
    open_mask = _open_mask_from_batch(batch).detach()

    successes = 0
    optimal = 0
    loops = 0
    ratios: List[float] = []
    max_steps = n_nodes * 2

    for b in range(batch_size):
        current = int(starts[b].item())
        goal = int(goals[b].item())
        visited = {current}
        path_len = 0
        reached = False
        looped = False

        for _ in range(max_steps):
            if current == goal:
                reached = True
                break
            candidates = [nxt for nxt in get_grid_neighbors(grid_size)[current] if bool(open_mask[b, nxt].item())]
            if not candidates:
                break

            next_node = min(candidates, key=lambda idx: float(prediction_field[b, idx].item()))
            path_len += 1
            current = next_node
            if current in visited and current != goal:
                looped = True
                break
            visited.add(current)

        if reached:
            successes += 1
            true_dist = max(1.0, float(raw_distance[b].item()))
            ratios.append(float(path_len) / true_dist)
            if abs(float(path_len) - true_dist) < 1e-6:
                optimal += 1
        if looped:
            loops += 1

    return {
        "path_success_rate": successes / max(1, batch_size),
        "path_optimal_rate": optimal / max(1, batch_size),
        "path_length_ratio": sum(ratios) / len(ratios) if ratios else float("nan"),
        "path_loop_rate": loops / max(1, batch_size),
    }


def goal_rank(prediction_field: Tensor, batch: TensorBatch) -> float:
    ranks: List[float] = []
    open_mask = _open_mask_from_batch(batch)
    for b in range(prediction_field.shape[0]):
        goal = int(batch["goal_idx"][b].item())
        open_values = prediction_field[b][open_mask[b]]
        goal_value = prediction_field[b, goal]
        rank = (open_values < goal_value).float().sum().item() + 1.0
        ranks.append(rank)
    return sum(ranks) / max(1, len(ranks))


def spurious_local_minima_count(prediction_field: Tensor, batch: TensorBatch, grid_size: int) -> float:
    open_mask = _open_mask_from_batch(batch)
    counts: List[float] = []
    for b in range(prediction_field.shape[0]):
        goal = int(batch["goal_idx"][b].item())
        count = 0
        for node in range(prediction_field.shape[1]):
            if node == goal or not bool(open_mask[b, node].item()):
                continue
            value = float(prediction_field[b, node].item())
            neighbors = [nxt for nxt in get_grid_neighbors(grid_size)[node] if bool(open_mask[b, nxt].item())]
            if neighbors and all(value <= float(prediction_field[b, nxt].item()) for nxt in neighbors):
                count += 1
        counts.append(float(count))
    return sum(counts) / max(1, len(counts))


def monotonic_descent_accuracy(prediction_field: Tensor, batch: TensorBatch, grid_size: int) -> float:
    true_field = batch["distance_field"]
    open_mask = _open_mask_from_batch(batch)
    correct = 0
    total = 0
    for b in range(prediction_field.shape[0]):
        goal = int(batch["goal_idx"][b].item())
        for node in range(prediction_field.shape[1]):
            if node == goal or not bool(open_mask[b, node].item()):
                continue
            true_value = float(true_field[b, node].item())
            lower_neighbors = [
                nxt
                for nxt in get_grid_neighbors(grid_size)[node]
                if bool(open_mask[b, nxt].item()) and float(true_field[b, nxt].item()) < true_value
            ]
            if not lower_neighbors:
                continue
            total += 1
            pred_value = float(prediction_field[b, node].item())
            if any(float(prediction_field[b, nxt].item()) < pred_value for nxt in lower_neighbors):
                correct += 1
    return correct / max(1, total)


def neighbor_order_accuracy(prediction_field: Tensor, batch: TensorBatch, grid_size: int) -> float:
    true_field = batch["distance_field"]
    open_mask = _open_mask_from_batch(batch)
    correct = 0
    total = 0
    for b in range(prediction_field.shape[0]):
        for node in range(prediction_field.shape[1]):
            if not bool(open_mask[b, node].item()):
                continue
            neighbors = [nxt for nxt in get_grid_neighbors(grid_size)[node] if bool(open_mask[b, nxt].item())]
            for i, left in enumerate(neighbors):
                for right in neighbors[i + 1 :]:
                    true_delta = float(true_field[b, left].item() - true_field[b, right].item())
                    if abs(true_delta) < 1e-8:
                        continue
                    pred_delta = float(prediction_field[b, left].item() - prediction_field[b, right].item())
                    total += 1
                    if math.copysign(1.0, true_delta) == math.copysign(1.0, pred_delta):
                        correct += 1
    return correct / max(1, total)


def functional_field_metrics(prediction_field: Tensor, batch: TensorBatch, grid_size: int) -> Dict[str, float]:
    if prediction_field.ndim != 2:
        return {}
    return {
        "goal_rank": goal_rank(prediction_field, batch),
        "spurious_local_minima_count": spurious_local_minima_count(prediction_field, batch, grid_size),
        "monotonic_descent_accuracy": monotonic_descent_accuracy(prediction_field, batch, grid_size),
        "neighbor_order_accuracy": neighbor_order_accuracy(prediction_field, batch, grid_size),
    }


def compute_metrics(task: str, prediction: Tensor, batch: TensorBatch, grid_size: int) -> Dict[str, float]:
    if task == "source":
        return source_metrics(prediction, batch)
    if task == "maze":
        metrics = maze_metrics(prediction, batch)
        if prediction.ndim == 2:
            metrics.update(greedy_path_metrics(prediction, batch, grid_size))
            metrics.update(functional_field_metrics(prediction, batch, grid_size))
        return metrics
    if task == "field":
        return field_metrics(prediction, batch)
    raise ValueError(f"Unknown task: {task}")
