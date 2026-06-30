from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ExperimentPreset:
    name: str
    config_path: str
    description: str


EXPERIMENTS: Dict[str, ExperimentPreset] = {
    "baseline_6x6_pool": ExperimentPreset(
        name="baseline_6x6_pool",
        config_path="configs/baseline_6x6_pool.yaml",
        description="Protected WNCA v0.2-heavy baseline on a structured 6x6 maze pool.",
    ),
    "heldout_6x6_pool": ExperimentPreset(
        name="heldout_6x6_pool",
        config_path="configs/heldout_6x6_pool.yaml",
        description="Held-out 6x6 structured pool for generalization checks.",
    ),
}
