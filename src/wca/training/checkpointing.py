from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch


@dataclass(frozen=True)
class CheckpointState:
    epoch: int = 0
    best_metrics: Dict[str, float] | None = None
    first_threshold_epochs: Dict[str, int] | None = None
    best_checkpoint_row: Dict[str, float] | None = None
    has_optimizer_state: bool = False

    def __post_init__(self) -> None:
        if self.best_metrics is None:
            object.__setattr__(self, "best_metrics", {})
        if self.first_threshold_epochs is None:
            object.__setattr__(self, "first_threshold_epochs", {})


def save_raw_model_state(path: str | Path, model: torch.nn.Module) -> None:
    torch.save(model.state_dict(), Path(path))


def save_training_state(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    best_metrics: Mapping[str, float],
    first_threshold_epochs: Mapping[str, int],
    best_checkpoint_row: Mapping[str, float] | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_metrics": dict(best_metrics),
        "first_threshold_epochs": dict(first_threshold_epochs),
    }
    if best_checkpoint_row is not None:
        payload["best_checkpoint_row"] = dict(best_checkpoint_row)
    torch.save(payload, Path(path))


def _is_training_payload(checkpoint: Any) -> bool:
    return isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    reset_optimizer: bool = False,
    map_location: str | torch.device = "cpu",
) -> CheckpointState:
    checkpoint = torch.load(Path(path), map_location=map_location)
    if _is_training_payload(checkpoint):
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer is not None and optimizer_state is not None and not reset_optimizer:
            optimizer.load_state_dict(optimizer_state)
        return CheckpointState(
            epoch=int(checkpoint.get("epoch", 0)),
            best_metrics=dict(checkpoint.get("best_metrics", {})),
            first_threshold_epochs=dict(checkpoint.get("first_threshold_epochs", {})),
            best_checkpoint_row=_coerce_best_checkpoint_row(checkpoint.get("best_checkpoint_row")),
            has_optimizer_state=optimizer_state is not None,
        )

    model.load_state_dict(checkpoint)
    return CheckpointState(epoch=0, best_metrics={}, first_threshold_epochs={}, has_optimizer_state=False)


def _coerce_best_checkpoint_row(value: Any) -> Optional[Dict[str, float]]:
    if not isinstance(value, Mapping):
        return None
    row: Dict[str, float] = {}
    for key, raw_value in value.items():
        if not isinstance(key, str):
            continue
        try:
            row[key] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return row or None


def _finite_metric(row: Mapping[str, float], key: str) -> Optional[float]:
    try:
        value = float(row.get(key, float("nan")))
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _best_score(row: Mapping[str, float]) -> Optional[tuple[float, float]]:
    path_opt = _finite_metric(row, "heldout_path_optimal_rate")
    loop_rate = _finite_metric(row, "heldout_path_loop_rate")
    if path_opt is None:
        path_opt = _finite_metric(row, "eval_path_optimal_rate")
        loop_rate = _finite_metric(row, "eval_path_loop_rate")
    if path_opt is None:
        field_horizon_score = _finite_metric(row, "eval_field_horizon_stratified_score")
        if field_horizon_score is not None:
            return (-field_horizon_score, float("-inf"))
        eval_mse = _finite_metric(row, "eval_mse")
        if eval_mse is None:
            return None
        return (-eval_mse, float("-inf"))
    return (path_opt, -loop_rate if loop_rate is not None else float("-inf"))


def best_checkpoint_improved(
    previous_row: Optional[Mapping[str, float]],
    candidate_row: Mapping[str, float],
) -> bool:
    candidate_score = _best_score(candidate_row)
    if candidate_score is None:
        return False
    if previous_row is None:
        return True
    previous_score = _best_score(previous_row)
    if previous_score is None:
        return True
    return candidate_score > previous_score


def best_checkpoint_row_from_metrics(best_metrics: Mapping[str, float]) -> Optional[Dict[str, float]]:
    """Recover a resume comparison row for older payloads without best_checkpoint_row."""
    path_opt = _finite_metric(best_metrics, "best_path_opt")
    if path_opt is not None:
        row: Dict[str, float] = {"eval_path_optimal_rate": path_opt}
        epoch = _finite_metric(best_metrics, "best_path_opt_epoch")
        if epoch is not None:
            row["epoch"] = epoch
        return row
    field_horizon_score = _finite_metric(best_metrics, "best_eval_field_horizon_stratified_score")
    if field_horizon_score is not None:
        row = {"eval_field_horizon_stratified_score": field_horizon_score}
        epoch = _finite_metric(best_metrics, "best_eval_field_horizon_stratified_score_epoch")
        if epoch is not None:
            row["epoch"] = epoch
        return row
    eval_mse = _finite_metric(best_metrics, "best_eval_mse")
    if eval_mse is not None:
        row = {"eval_mse": eval_mse}
        epoch = _finite_metric(best_metrics, "best_eval_mse_epoch")
        if epoch is not None:
            row["epoch"] = epoch
        return row
    return None
