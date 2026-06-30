from __future__ import annotations

import math
import torch
from torch import Tensor

from wca.config import Config
from wca.schemas import TensorBatch


def make_full_adjacency(n_nodes: int, device: torch.device) -> Tensor:
    return torch.ones(n_nodes, n_nodes, device=device)


def make_ring_adjacency(n_nodes: int, device: torch.device) -> Tensor:
    adjacency = torch.zeros(n_nodes, n_nodes, device=device)
    for i in range(n_nodes):
        adjacency[i, i] = 1.0
        adjacency[i, (i - 1) % n_nodes] = 1.0
        adjacency[i, (i + 1) % n_nodes] = 1.0
    return adjacency


def balanced_signs(batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
    patterns = torch.tensor(
        [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]],
        device=device,
    )
    repeats = math.ceil(batch_size / 4)
    signs = patterns.repeat(repeats, 1)[:batch_size]
    signs = signs[torch.randperm(batch_size, device=device)]
    return signs[:, 0], signs[:, 1]


def generate_source_distractor_batch(cfg: Config, device: torch.device) -> TensorBatch:
    batch_size = cfg.batch_size
    n_nodes = cfg.n_nodes
    hidden_dim = cfg.hidden_dim

    H = torch.zeros(batch_size, n_nodes, hidden_dim, device=device)
    source_idx = torch.randint(0, n_nodes, (batch_size,), device=device)
    distractor_idx = torch.randint(0, n_nodes, (batch_size,), device=device)
    target_idx = torch.randint(0, n_nodes, (batch_size,), device=device)

    for b in range(batch_size):
        while distractor_idx[b].item() == source_idx[b].item():
            distractor_idx[b] = torch.randint(0, n_nodes, (1,), device=device)
        while target_idx[b].item() in {source_idx[b].item(), distractor_idx[b].item()}:
            target_idx[b] = torch.randint(0, n_nodes, (1,), device=device)

    source_sign, distractor_sign = balanced_signs(batch_size, device)
    batch_arange = torch.arange(batch_size, device=device)

    H[batch_arange, source_idx, 0] = source_sign
    H[batch_arange, distractor_idx, 1] = distractor_sign
    H[batch_arange, target_idx, 2] = 1.0
    H[:, :, 3] = torch.linspace(-1.0, 1.0, n_nodes, device=device).unsqueeze(0)

    adjacency = make_full_adjacency(n_nodes, device).unsqueeze(0).expand(batch_size, n_nodes, n_nodes).clone()

    return {
        "H": H,
        "adjacency": adjacency,
        "target_idx": target_idx,
        "label": source_sign,
        "baseline_label": source_sign,
        "source_sign": source_sign,
        "distractor_sign": distractor_sign,
        "raw_distance": torch.zeros(batch_size, device=device),
    }
