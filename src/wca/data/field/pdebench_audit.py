from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py


def _visit_hdf5_dataset(datasets: dict[str, dict[str, Any]], name: str, obj: Any) -> None:
    if isinstance(obj, h5py.Dataset):
        datasets[name] = {
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "ndim": int(obj.ndim),
            "size": int(obj.size),
        }


def audit_hdf5_file(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    datasets: dict[str, dict[str, Any]] = {}
    with h5py.File(file_path, "r") as handle:
        handle.visititems(lambda name, obj: _visit_hdf5_dataset(datasets, name, obj))
    return {
        "path": str(file_path),
        "format": "hdf5",
        "datasets": datasets,
    }


def write_audit_json(audit: dict[str, Any], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
