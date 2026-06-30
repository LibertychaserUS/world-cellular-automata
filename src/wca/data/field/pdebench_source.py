from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_source_manifest(
    *,
    source_path: str | Path,
    manifest_path: str | Path,
    source_url: str,
    dataset_family: str,
    license_name: str,
    variables: Sequence[str],
) -> None:
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(source)
    payload = {
        "source_name": "pdebench",
        "dataset_family": dataset_family,
        "source_url": source_url,
        "source_file": str(source),
        "sha256": sha256_file(source),
        "license": license_name,
        "variables": list(variables),
    }
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
