#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wca.config import add_common_cli_args, config_from_args
from wca.data.maze.pools import ensure_maze_pool
from wca.models.rws_nca import FullRecursiveWorldStateNCA
from wca.training.evaluator import evaluate
from wca.utils.device import resolve_device
from wca.utils.seed import set_seed


def write_eval_outputs(output_dir: str | Path, metrics: Mapping[str, float]) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "eval_metrics.json").write_text(json.dumps(dict(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# WCA Eval Metrics", "", "| metric | value |", "|---|---:|"]
    for key, value in sorted(metrics.items()):
        lines.append(f"| `{key}` | {float(value):.6g} |")
    lines.append("")
    (target / "eval_metrics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a WCA maze checkpoint.")
    add_common_cli_args(parser)
    parser.add_argument("--output-dir", type=str, default="", help="Optional directory for eval_metrics.json/md")
    args = parser.parse_args()
    cfg = config_from_args(args)
    if not cfg.checkpoint:
        raise SystemExit("--checkpoint is required")
    device = resolve_device(cfg.device)
    if cfg.task == "maze":
        cfg.n_nodes = cfg.grid_size * cfg.grid_size
    set_seed(cfg.seed)
    maze_pool = ensure_maze_pool(cfg) if cfg.task == "maze" and cfg.maze_pool_size > 0 else []
    model = FullRecursiveWorldStateNCA(
        n_nodes=cfg.n_nodes,
        hidden_dim=cfg.hidden_dim,
        edge_dim=cfg.edge_dim,
        inner_steps=cfg.inner_steps,
        pair_chunk_size=cfg.pair_chunk_size,
        activation_checkpoint_inner=cfg.activation_checkpoint_inner,
    ).to(device)
    model.load_state_dict(torch.load(cfg.checkpoint, map_location=device))
    metrics = evaluate(model, cfg, device, maze_pool=maze_pool)
    print(json.dumps(metrics, indent=2))
    if args.output_dir:
        write_eval_outputs(args.output_dir, metrics)


if __name__ == "__main__":
    main()
