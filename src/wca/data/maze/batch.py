from __future__ import annotations

import random
from typing import List, Optional

import torch
from torch import Tensor

from wca.config import Config
from wca.constants import GOAL_CH, OPEN_CH, START_CH, WALL_CH, X_CH, Y_CH
from wca.data.maze.generator import (
    MazeSpec,
    generate_fixed_hard_maze,
    generate_fixed_maze_set,
    generate_single_maze,
    generate_structured_maze,
)
from wca.data.maze.oracle import bfs_distance_field
from wca.data.maze.pools import ensure_maze_pool
from wca.schemas import TensorBatch


def make_grid_adjacency(grid_size: int, device: torch.device) -> Tensor:
    n_nodes = grid_size * grid_size
    adjacency = torch.zeros(n_nodes, n_nodes, device=device)
    for r in range(grid_size):
        for c in range(grid_size):
            i = r * grid_size + c
            adjacency[i, i] = 1.0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    j = nr * grid_size + nc
                    adjacency[i, j] = 1.0
    return adjacency


def _select_maze(cfg: Config, maze_pool: Optional[List[MazeSpec]]) -> MazeSpec:
    if cfg.maze_mode == "fixed":
        return generate_fixed_hard_maze(cfg)
    if cfg.maze_mode == "fixed-set":
        return random.choice(generate_fixed_maze_set(cfg))
    if cfg.maze_mode == "structured-random":
        pool = maze_pool if maze_pool is not None else ensure_maze_pool(cfg)
        if pool:
            return random.choice(pool)
        return generate_structured_maze(cfg)
    return generate_single_maze(cfg.grid_size, cfg.wall_prob)


def make_maze_batch(cfg: Config, device: torch.device, maze_pool: Optional[List[MazeSpec]] = None) -> TensorBatch:
    batch_size = cfg.batch_size
    grid_size = cfg.grid_size
    n_nodes = grid_size * grid_size
    hidden_dim = cfg.hidden_dim

    H = torch.zeros(batch_size, n_nodes, hidden_dim, device=device)
    start_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
    goal_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
    labels = torch.zeros(batch_size, device=device)
    raw_distances = torch.zeros(batch_size, device=device)
    distance_field = torch.zeros(batch_size, n_nodes, device=device)
    distance_mask = torch.zeros(batch_size, n_nodes, device=device)
    open_masks = torch.zeros(batch_size, n_nodes, dtype=torch.bool, device=device)
    maze_ids: List[str] = []

    base_adjacency = make_grid_adjacency(grid_size, device)
    adjacency = base_adjacency.unsqueeze(0).expand(batch_size, n_nodes, n_nodes).clone()
    max_distance = max(1, 2 * (grid_size - 1))

    for b in range(batch_size):
        walls, start, goal, dist = _select_maze(cfg, maze_pool)
        start_idx[b] = start
        goal_idx[b] = goal
        raw_distances[b] = float(dist)
        labels[b] = float(dist) / float(max_distance)
        maze_ids.append(f"{grid_size}x{grid_size}:s{start}:g{goal}:w{hash(tuple(sorted(walls))) & 0xfffffff:x}")

        wall_mask = torch.zeros(n_nodes, device=device)
        if walls:
            wall_tensor = torch.tensor(walls, dtype=torch.long, device=device)
            wall_mask[wall_tensor] = 1.0
            H[b, wall_tensor, WALL_CH] = 1.0

        H[b, start, START_CH] = 1.0
        H[b, goal, GOAL_CH] = 1.0
        H[b, :, OPEN_CH] = 1.0 - wall_mask

        open_mask = 1.0 - wall_mask
        open_masks[b] = open_mask > 0.5
        adjacency[b] = adjacency[b] * open_mask.view(1, n_nodes) * open_mask.view(n_nodes, 1)
        adjacency[b].fill_diagonal_(1.0)

        bfs_field = bfs_distance_field(grid_size, walls, goal)
        for node_idx, node_dist in enumerate(bfs_field):
            if node_dist >= 0.0:
                distance_field[b, node_idx] = float(node_dist) / float(max_distance)
                distance_mask[b, node_idx] = 1.0

    for r in range(grid_size):
        for c in range(grid_size):
            idx = r * grid_size + c
            H[:, idx, X_CH] = -1.0 + 2.0 * c / max(1, grid_size - 1)
            H[:, idx, Y_CH] = -1.0 + 2.0 * r / max(1, grid_size - 1)

    return {
        "H": H,
        "adjacency": adjacency,
        "target_idx": start_idx,
        "start_idx": start_idx,
        "goal_idx": goal_idx,
        "label": labels,
        "baseline_label": labels,
        "distance_field": distance_field,
        "distance_mask": distance_mask,
        "open_mask": open_masks,
        "raw_distance": raw_distances,
        "source_sign": labels,
        "distractor_sign": torch.zeros_like(labels),
        "maze_id": maze_ids,  # type: ignore[dict-item]
    }
