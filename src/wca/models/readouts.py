from __future__ import annotations

from torch import nn


def make_scalar_readout(hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, 1),
    )
