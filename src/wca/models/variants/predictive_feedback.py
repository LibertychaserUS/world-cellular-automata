from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from wca.models.rws_nca import FullRecursiveWorldStateNCA


class PredictiveFeedbackWorldStateNCA(FullRecursiveWorldStateNCA):
    """Self-predictive variant from the low-reference v0.3 path.

    This class is deliberately not imported by default training.
    """

    def __init__(self, n_nodes: int, hidden_dim: int, edge_dim: int, inner_steps: int, pair_chunk_size: int = 0) -> None:
        super().__init__(n_nodes, hidden_dim, edge_dim, inner_steps, pair_chunk_size)
        self.surprise_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

    def project_full_world(self, H: Tensor, surprise: Optional[Tensor] = None) -> Tensor:  # type: ignore[override]
        batch_size, n_nodes, hidden_dim = H.shape
        if surprise is None:
            return super().project_full_world(H)

        L = H.unsqueeze(1).expand(batch_size, n_nodes, n_nodes, hidden_dim).clone()
        centers = torch.arange(n_nodes, device=H.device)
        world_nodes = torch.arange(n_nodes, device=H.device)
        center_code = self.center_embedding(centers).view(1, n_nodes, 1, hidden_dim)

        shift = self.surprise_encoder(surprise)
        surprise_norm = surprise.detach().norm(dim=-1, keepdim=True)
        gain = 1.0 + torch.exp(-surprise_norm * 5.0)
        center_code = center_code + (shift * gain).view(batch_size, n_nodes, 1, hidden_dim)

        rel_index = world_nodes.view(1, -1) - centers.view(-1, 1) + (n_nodes - 1)
        relative_code = self.relative_embedding(rel_index).view(1, n_nodes, n_nodes, hidden_dim)
        return L + 0.1 * center_code + 0.1 * relative_code

    def forward(self, H: Tensor, adjacency: Tensor, outer_steps: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        diagnostics: Dict[str, Tensor] = {}
        last_L = None
        energies: List[Tensor] = []
        surprises: List[Tensor] = []

        H_prev2 = H.clone()
        H_prev = H.clone()
        surprise = None

        for step in range(outer_steps):
            L = self.project_full_world(H, surprise)
            L = self.evolve_local_worlds(L, adjacency)
            H_new = self.outer_norm(self.compose_world(L))
            H_pred = H if step == 0 else H_prev + (H_prev - H_prev2)
            surprise = H_new - H_pred
            surprises.append(surprise.pow(2).mean().detach())
            H_prev2 = H_prev
            H_prev = H_new
            H = H_new
            last_L = L
            energies.append(H.pow(2).mean())

        if last_L is not None:
            diagnostics["last_local_worlds"] = last_L
        diagnostics["outer_energy"] = torch.stack(energies) if energies else torch.zeros(1, device=H.device)
        diagnostics["surprise_history"] = torch.stack(surprises) if surprises else torch.zeros(1, device=H.device)
        if surprise is not None:
            diagnostics["final_surprise"] = surprise
        return H, diagnostics
