from __future__ import annotations

import math
from typing import Dict

from torch import Tensor

LOCAL_WORLD_DIAGNOSTIC_KEYS = ("center_diversity", "collapse_score", "state_energy", "diag_energy")


def local_world_diagnostics(local_worlds: Tensor) -> Dict[str, float]:
    center_mean = local_worlds.mean(dim=2)
    global_center_mean = center_mean.mean(dim=1, keepdim=True)
    center_diversity = (center_mean - global_center_mean).pow(2).mean().sqrt().item()
    collapse_score = 1.0 / (center_diversity + 1e-6)
    state_energy = local_worlds.pow(2).mean().item()
    center_diag = local_worlds.diagonal(dim1=1, dim2=2).transpose(1, 2)
    diag_energy = center_diag.pow(2).mean().item()
    return {
        "center_diversity": center_diversity,
        "collapse_score": collapse_score,
        "state_energy": state_energy,
        "diag_energy": diag_energy,
    }


def diagnostics_metrics(diagnostics: Dict[str, Tensor]) -> Dict[str, float]:
    """Return logging metrics without inventing local-world values for bypass runs."""

    metrics: Dict[str, float] = {}
    local_worlds = diagnostics.get("last_local_worlds")
    if local_worlds is None:
        metrics.update({key: math.nan for key in LOCAL_WORLD_DIAGNOSTIC_KEYS})
    else:
        metrics.update(local_world_diagnostics(local_worlds))
    core_executed = diagnostics.get("field_core_executed")
    if core_executed is not None:
        metrics["field_core_executed"] = float(core_executed.detach().float().mean().cpu().item())
    return metrics
