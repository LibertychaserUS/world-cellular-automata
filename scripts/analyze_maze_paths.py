#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wca.config import add_common_cli_args, config_from_args
from wca.data.maze.forensics import render_trace_markdown, summarize_traces, trace_greedy_path
from wca.data.maze.pools import ensure_maze_pool
from wca.models.rws_nca import FullRecursiveWorldStateNCA
from wca.training.evaluator import make_batch
from wca.utils.device import resolve_device
from wca.utils.seed import set_seed


def _load_model(cfg: Any, device: torch.device) -> FullRecursiveWorldStateNCA:
    model = FullRecursiveWorldStateNCA(
        n_nodes=cfg.n_nodes,
        hidden_dim=cfg.hidden_dim,
        edge_dim=cfg.edge_dim,
        inner_steps=cfg.inner_steps,
        pair_chunk_size=cfg.pair_chunk_size,
        activation_checkpoint_inner=cfg.activation_checkpoint_inner,
    ).to(device)
    payload = torch.load(cfg.checkpoint, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    model.load_state_dict(payload)
    model.eval()
    return model


def _trace_model_batch(model: FullRecursiveWorldStateNCA, cfg: Any, batch: Dict[str, Any], traces: List[Dict[str, Any]], pool_index: int | None = None) -> None:
    H_final, _diagnostics = model(batch["H"], batch["adjacency"], cfg.outer_steps)
    prediction = model.predict_all_nodes(H_final)
    for sample_idx in range(prediction.shape[0]):
        trace = trace_greedy_path(prediction, batch, cfg.grid_size, sample_idx)
        trace["trace_idx"] = len(traces)
        if pool_index is not None:
            trace["pool_index"] = pool_index
        traces.append(trace)


@torch.no_grad()
def collect_traces(cfg: Any, device: torch.device, batches: int, max_samples: int, random_samples: bool = False) -> Dict[str, Any]:
    set_seed(cfg.seed)
    maze_pool = ensure_maze_pool(cfg) if cfg.task == "maze" and cfg.maze_pool_size > 0 else []
    model = _load_model(cfg, device)
    traces: List[Dict[str, Any]] = []

    if maze_pool and not random_samples:
        old_batch_size = cfg.batch_size
        cfg.batch_size = 1
        try:
            for pool_index, maze in enumerate(maze_pool):
                batch = make_batch(cfg, device, maze_pool=[maze])
                _trace_model_batch(model, cfg, batch, traces, pool_index=pool_index)
                if len(traces) >= max_samples:
                    return {"summary": summarize_traces(traces), "traces": traces}
        finally:
            cfg.batch_size = old_batch_size
        return {"summary": summarize_traces(traces), "traces": traces}

    for _ in range(batches):
        batch = make_batch(cfg, device, maze_pool=maze_pool)
        _trace_model_batch(model, cfg, batch, traces)
        if len(traces) >= max_samples:
            summary = summarize_traces(traces)
            return {"summary": summary, "traces": traces}

    summary = summarize_traces(traces)
    return {"summary": summary, "traces": traces}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze greedy maze paths from a WCA checkpoint.")
    add_common_cli_args(parser)
    parser.add_argument("--batches", type=int, default=None, help="Number of generated eval batches to inspect.")
    parser.add_argument("--max-samples", type=int, default=64, help="Maximum number of samples to trace.")
    parser.add_argument("--output-dir", type=str, default="", help="Directory for path_forensics.json/md.")
    parser.add_argument("--max-examples", type=int, default=12, help="Maximum examples in markdown report.")
    parser.add_argument("--random-samples", action="store_true", help="Sample randomly from the pool instead of iterating it in order.")
    args = parser.parse_args()
    cfg = config_from_args(args)
    if cfg.task != "maze":
        raise SystemExit("Path forensics only supports --task maze")
    if not cfg.checkpoint:
        raise SystemExit("--checkpoint is required")
    cfg.n_nodes = cfg.grid_size * cfg.grid_size
    batches = args.batches if args.batches is not None else cfg.eval_batches
    device = resolve_device(cfg.device)
    result = collect_traces(cfg, device, batches=batches, max_samples=args.max_samples, random_samples=args.random_samples)

    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg.checkpoint).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "path_forensics.json"
    md_path = output_dir / "path_forensics.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(
        render_trace_markdown(result["summary"], result["traces"], max_examples=args.max_examples),
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
