from __future__ import annotations

from wca.config import Config
from wca.training.trainer import train


def run_ablation(cfg: Config) -> None:
    """Run an ablation config through the default trainer.

    Baseline protection is enforced by the trainer's default model selection.
    Low-reference variants require an explicit future variant-aware trainer.
    """
    train(cfg)
