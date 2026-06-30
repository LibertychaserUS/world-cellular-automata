#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wca.data.field.pdebench_audit import audit_hdf5_file, write_audit_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a PDEBench HDF5 file without loading full tensors.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = audit_hdf5_file(args.path)
    write_audit_json(audit, args.output)


if __name__ == "__main__":
    main()
