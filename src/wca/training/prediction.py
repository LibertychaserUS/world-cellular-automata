from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from wca.schemas import TensorBatch


def _field_residual_baseline(cfg: Any, raw_prediction: Tensor, batch: TensorBatch) -> Tensor:
    baseline = batch["field_prediction_baseline"].to(device=raw_prediction.device, dtype=raw_prediction.dtype)
    previous = batch.get("field_previous_tokens")
    if bool(getattr(cfg, "field_tendency_baseline", False)) and isinstance(previous, Tensor):
        previous = previous.to(device=raw_prediction.device, dtype=raw_prediction.dtype)
        horizon = batch.get("field_target_steps_actual")
        if isinstance(horizon, Tensor):
            horizon = horizon.to(device=raw_prediction.device, dtype=raw_prediction.dtype).view(-1, 1)
            while horizon.ndim < baseline.ndim:
                horizon = horizon.unsqueeze(-1)
        else:
            horizon = raw_prediction.new_tensor(float(getattr(cfg, "field_target_steps", 1)))
        tendency_scale = float(getattr(cfg, "field_tendency_scale", 1.0))
        baseline = baseline + tendency_scale * horizon * (baseline - previous)
    if baseline.ndim == 2 and raw_prediction.ndim == 3 and raw_prediction.shape[-1] == 1:
        baseline = baseline.unsqueeze(-1)
    return baseline


def predict_for_task(module: nn.Module, cfg: Any, H_final: Tensor, batch: TensorBatch) -> Tensor:
    task = getattr(cfg, "task", "")
    if task in {"maze", "field"}:
        raw_prediction = module.predict_all_nodes(H_final)
    else:
        raw_prediction = module.predict_target(H_final, batch["target_idx"])

    if task == "field" and bool(getattr(cfg, "field_residual_readout", False)):
        baseline = _field_residual_baseline(cfg, raw_prediction, batch)
        return baseline + float(getattr(cfg, "field_residual_scale", 1.0)) * raw_prediction

    return raw_prediction
