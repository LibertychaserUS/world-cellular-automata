#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wca.config import add_common_cli_args, config_from_args
from wca.training.trainer import smoke_test_shapes, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the protected WCA v0.2-heavy baseline.")
    add_common_cli_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if cfg.smoke_test_only or world_size <= 1:
        smoke_test_shapes(cfg)
    if cfg.smoke_test_only:
        return
    train(cfg)


if __name__ == "__main__":
    main()
