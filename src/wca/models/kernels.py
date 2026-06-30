from __future__ import annotations

import torch
from torch import Tensor, nn


class PairInteractionKernel(nn.Module):
    """
    Shared dense pairwise local interaction kernel.

    This receives states from inside each local world and produces a delta for
    every receiver/sender pair. It is deliberately expensive and explicit.
    """

    def __init__(self, hidden_dim: int, edge_dim: int) -> None:
        super().__init__()
        input_dim = hidden_dim * 4 + edge_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 3),
            nn.SiLU(),
            nn.Linear(hidden_dim * 3, hidden_dim * 3),
            nn.SiLU(),
            nn.Linear(hidden_dim * 3, hidden_dim),
        )

    def forward(self, receiver: Tensor, sender: Tensor, edge_features: Tensor) -> Tensor:
        features = torch.cat(
            [receiver, sender, sender - receiver, receiver * sender, edge_features],
            dim=-1,
        )
        return self.net(features)
