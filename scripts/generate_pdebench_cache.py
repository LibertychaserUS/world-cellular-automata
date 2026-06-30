#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wca.data.field.pdebench_cache import convert_hdf5_time_series_to_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a PDEBench HDF5 time series to WCA field cache.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset-key", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.add_argument(
        "--layout",
        type=str,
        default="auto",
        choices=["auto", "thw", "tchw", "thwc", "sthw", "sthwc", "stchw"],
    )
    parser.add_argument("--max-trajectories", type=int, default=0)
    parser.add_argument("--variables", type=str, required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--eval-size", type=int, required=True)
    args = parser.parse_args()
    convert_hdf5_time_series_to_cache(
        source_path=args.source,
        dataset_key=args.dataset_key,
        output_path=args.output,
        manifest_path=args.manifest,
        source_manifest_path=args.source_manifest,
        layout=args.layout,
        max_trajectories=args.max_trajectories,
        variables=[item.strip() for item in args.variables.split(",") if item.strip()],
        train_size=args.train_size,
        eval_size=args.eval_size,
    )


if __name__ == "__main__":
    main()
