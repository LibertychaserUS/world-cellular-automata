from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from wca.models.kernels import PairInteractionKernel


class FullRecursiveWorldStateNCA(nn.Module):
    """
    Heavy full Recursive World-State NCA.

    Explicit tensors:
        H: [B, N, D]
        L: [B, N, N, D]
        receiver/sender pair states during dense evolution: [B, N, N, N, D]
    """

    def __init__(
        self,
        n_nodes: int,
        hidden_dim: int,
        edge_dim: int,
        inner_steps: int,
        pair_chunk_size: int = 0,
        output_dim: int = 1,
        activation_checkpoint_inner: bool = False,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.inner_steps = inner_steps
        self.pair_chunk_size = pair_chunk_size
        self.output_dim = output_dim
        self.activation_checkpoint_inner = activation_checkpoint_inner

        self.center_embedding = nn.Embedding(n_nodes, hidden_dim)
        self.relative_embedding = nn.Embedding(2 * n_nodes - 1, hidden_dim)
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim),
        )
        self.kernel = PairInteractionKernel(hidden_dim, edge_dim)
        self.update_norm = nn.LayerNorm(hidden_dim)
        self.outer_norm = nn.LayerNorm(hidden_dim)
        self.readout = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def project_full_world(
        self,
        H: Tensor,
        input_visibility: Optional[Tensor] = None,
        input_visibility_channels: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, n_nodes, hidden_dim = H.shape
        if n_nodes != self.n_nodes or hidden_dim != self.hidden_dim:
            raise ValueError(f"Expected H [B,{self.n_nodes},{self.hidden_dim}], got {tuple(H.shape)}")

        L = H.unsqueeze(1).expand(batch_size, n_nodes, n_nodes, hidden_dim).clone()
        if input_visibility is not None:
            if input_visibility.shape != (batch_size, self.n_nodes, self.n_nodes):
                raise ValueError(f"Expected input_visibility [B,N,N], got {tuple(input_visibility.shape)}")
            visibility = input_visibility.to(device=H.device, dtype=H.dtype).unsqueeze(-1)
            if input_visibility_channels is None:
                channel_gate = torch.ones(self.hidden_dim, device=H.device, dtype=H.dtype)
            else:
                if input_visibility_channels.shape != (self.hidden_dim,):
                    raise ValueError(
                        f"Expected input_visibility_channels [D], got {tuple(input_visibility_channels.shape)}"
                    )
                channel_gate = input_visibility_channels.to(device=H.device, dtype=H.dtype)
            channel_gate = channel_gate.view(1, 1, 1, self.hidden_dim)
            L = L * (1.0 - channel_gate + visibility * channel_gate)

        centers = torch.arange(n_nodes, device=H.device)
        world_nodes = torch.arange(n_nodes, device=H.device)
        center_code = self.center_embedding(centers).view(1, n_nodes, 1, hidden_dim)
        rel_index = world_nodes.view(1, -1) - centers.view(-1, 1) + (n_nodes - 1)
        relative_code = self.relative_embedding(rel_index).view(1, n_nodes, n_nodes, hidden_dim)

        return L + 0.1 * center_code + 0.1 * relative_code

    def make_edge_features(self, adjacency: Tensor) -> Tensor:
        batch_size, n_nodes, _ = adjacency.shape
        device = adjacency.device
        receiver = torch.arange(n_nodes, device=device).view(n_nodes, 1).expand(n_nodes, n_nodes)
        sender = torch.arange(n_nodes, device=device).view(1, n_nodes).expand(n_nodes, n_nodes)
        distance = (receiver - sender).abs().float() / max(1, n_nodes - 1)
        identity = (receiver == sender).float()
        direction = torch.sign((sender - receiver).float())

        edge_raw = torch.stack([distance, identity, direction, torch.zeros_like(distance)], dim=-1)
        edge_raw = edge_raw.view(1, n_nodes, n_nodes, 4).expand(batch_size, n_nodes, n_nodes, 4).clone()
        edge_raw[..., 3] = adjacency
        return self.edge_encoder(edge_raw)

    def _evolve_local_world_step(
        self,
        L: Tensor,
        edge_features: Tensor,
        adjacency_mask: Tensor,
        denom: Tensor,
    ) -> Tensor:
        batch_size, _, _, hidden_dim = L.shape

        if self.pair_chunk_size <= 0 or self.pair_chunk_size >= self.n_nodes:
            receiver = L.unsqueeze(3).expand(
                batch_size,
                self.n_nodes,
                self.n_nodes,
                self.n_nodes,
                hidden_dim,
            )
            sender = L.unsqueeze(2).expand(
                batch_size,
                self.n_nodes,
                self.n_nodes,
                self.n_nodes,
                hidden_dim,
            )
            edge = edge_features.expand(batch_size, self.n_nodes, -1, -1, -1)

            pair_delta = self.kernel(receiver, sender, edge)
            pair_delta = pair_delta * adjacency_mask
            delta = pair_delta.sum(dim=3) / denom
        else:
            delta_sum = torch.zeros_like(L)
            chunk = self.pair_chunk_size
            for start in range(0, self.n_nodes, chunk):
                end = min(start + chunk, self.n_nodes)
                sender_chunk = L[:, :, start:end, :]
                sender = sender_chunk.unsqueeze(2).expand(
                    batch_size,
                    self.n_nodes,
                    self.n_nodes,
                    end - start,
                    hidden_dim,
                )
                receiver = L.unsqueeze(3).expand(
                    batch_size,
                    self.n_nodes,
                    self.n_nodes,
                    end - start,
                    hidden_dim,
                )
                edge = edge_features[:, :, :, start:end, :].expand(batch_size, self.n_nodes, -1, -1, -1)
                mask = adjacency_mask[:, :, :, start:end, :]

                pair_delta = self.kernel(receiver, sender, edge)
                pair_delta = pair_delta * mask
                delta_sum = delta_sum + pair_delta.sum(dim=3)
            delta = delta_sum / denom

        return self.update_norm(L + 0.20 * delta)

    def evolve_local_worlds(self, L: Tensor, adjacency: Tensor) -> Tensor:
        batch_size, n_centers, n_world_nodes, hidden_dim = L.shape
        if (n_centers, n_world_nodes, hidden_dim) != (self.n_nodes, self.n_nodes, self.hidden_dim):
            raise ValueError(f"Unexpected local-world shape: {tuple(L.shape)}")
        if adjacency.shape != (batch_size, self.n_nodes, self.n_nodes):
            raise ValueError(f"Expected adjacency [B,N,N], got {tuple(adjacency.shape)}")

        edge_features = self.make_edge_features(adjacency)
        edge_features = edge_features.view(batch_size, 1, self.n_nodes, self.n_nodes, self.edge_dim)
        adjacency_mask = adjacency.view(batch_size, 1, self.n_nodes, self.n_nodes, 1)

        for _ in range(self.inner_steps):
            denom = adjacency_mask.sum(dim=3).clamp_min(1.0)
            if self.activation_checkpoint_inner and self.training:
                L = checkpoint(
                    self._evolve_local_world_step,
                    L,
                    edge_features,
                    adjacency_mask,
                    denom,
                    use_reentrant=False,
                )
            else:
                L = self._evolve_local_world_step(L, edge_features, adjacency_mask, denom)

        return L

    def compose_world(self, L: Tensor) -> Tensor:
        idx = torch.arange(self.n_nodes, device=L.device)
        return L[:, idx, idx, :]

    def forward(
        self,
        H: Tensor,
        adjacency: Tensor,
        outer_steps: int,
        input_visibility: Optional[Tensor] = None,
        input_visibility_channels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        diagnostics: Dict[str, Tensor] = {}
        last_L: Optional[Tensor] = None
        energies: List[Tensor] = []

        for _ in range(outer_steps):
            L = self.project_full_world(
                H,
                input_visibility=input_visibility,
                input_visibility_channels=input_visibility_channels,
            )
            L = self.evolve_local_worlds(L, adjacency)
            H = self.outer_norm(self.compose_world(L))
            last_L = L
            energies.append(H.pow(2).mean())

        if last_L is not None:
            diagnostics["last_local_worlds"] = last_L
        if energies:
            diagnostics["outer_energy"] = torch.stack(energies)
        return H, diagnostics

    def predict_target(self, H: Tensor, target_idx: Tensor) -> Tensor:
        batch_arange = torch.arange(H.shape[0], device=H.device)
        target_state = H[batch_arange, target_idx, :]
        prediction = self.readout(target_state)
        return prediction.squeeze(-1) if self.output_dim == 1 else prediction

    def predict_all_nodes(self, H: Tensor) -> Tensor:
        prediction = self.readout(H)
        return prediction.squeeze(-1) if self.output_dim == 1 else prediction
