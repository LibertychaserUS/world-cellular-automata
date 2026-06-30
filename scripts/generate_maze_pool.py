#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wca.config import add_common_cli_args, config_from_args
from wca.data.maze.pools import ensure_maze_pool, save_maze_pool


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a structured maze pool.")
    add_common_cli_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)
    if cfg.task != "maze":
        cfg.task = "maze"
    cfg.n_nodes = cfg.grid_size * cfg.grid_size
    if not cfg.pool_path:
        cfg.pool_path = f"artifacts/maze_pools/{cfg.maze_mode}_{cfg.grid_size}x{cfg.grid_size}_seed{cfg.seed}_n{cfg.maze_pool_size}.json"
    pool = ensure_maze_pool(cfg)
    if cfg.pool_path:
        save_maze_pool(pool, cfg.pool_path)
    print(f"maze_pool_path={cfg.pool_path}")
    print(f"maze_pool_count={len(pool)}")


if __name__ == "__main__":
    main()
