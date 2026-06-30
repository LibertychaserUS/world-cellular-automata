from __future__ import annotations

from wca.config import Config
from wca.training.trainer import train


def run_maze(cfg: Config) -> None:
    cfg.task = "maze"
    cfg.n_nodes = cfg.grid_size * cfg.grid_size
    train(cfg)
