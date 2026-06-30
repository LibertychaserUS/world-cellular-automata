from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor

from wca.config import Config
from wca.data.field.real_cache import SUPPORTED_REAL_FIELD_DATASETS, _load_cache, valid_start_indices

DEFAULT_EVAL_SEED = 2026062201


def horizon_eval_seed(eval_seed: int, horizon: int) -> int:
    return int(eval_seed) + int(horizon) * 1009


def eval_plan_hash(
    horizons: list[int],
    eval_batches: int,
    eval_seed: int,
    *,
    eval_samples: int = 0,
    eval_batch_size: int = 0,
    start_indices_by_horizon: dict[int, Tensor] | None = None,
    field_split: str = "eval",
) -> str:
    start_index_hashes: dict[str, str] = {}
    start_index_counts: dict[str, int] = {}
    if start_indices_by_horizon:
        for horizon, starts in sorted(start_indices_by_horizon.items()):
            start_index_hashes[str(int(horizon))] = start_indices_hash(starts)
            start_index_counts[str(int(horizon))] = int(starts.numel())
    payload = {
        "eval_batches": int(eval_batches),
        "eval_batch_size": int(eval_batch_size),
        "eval_samples": int(eval_samples),
        "eval_seed": int(eval_seed),
        "field_split": field_split,
        "horizon_seeds": {str(horizon): horizon_eval_seed(eval_seed, horizon) for horizon in horizons},
        "horizons": [int(horizon) for horizon in horizons],
        "sample_rule": "explicit_start_indices_v1" if eval_samples > 0 else "torch_randint_seed_per_horizon_v1",
        "start_index_counts": start_index_counts,
        "start_index_hashes": start_index_hashes,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def start_indices_hash(starts: Tensor) -> str:
    tensor = starts.detach().cpu().to(dtype=torch.long).contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()[:16]


def make_fixed_eval_start_indices(
    cfg: Config,
    horizon: int,
    eval_seed: int,
    eval_samples: int,
    *,
    field_split: str = "eval",
) -> Tensor:
    if cfg.field_dataset not in SUPPORTED_REAL_FIELD_DATASETS:
        raise ValueError("Explicit fixed eval start indices are only supported for real field cache datasets.")
    if eval_samples <= 0:
        raise ValueError(f"eval_samples must be positive, got {eval_samples}.")
    payload = _load_cache(cfg.field_data_path)
    data = payload["data"]
    valid_indices = valid_start_indices(cfg, field_split, int(data.shape[0]), target_steps=int(horizon), payload=payload)
    if int(valid_indices.numel()) < int(eval_samples):
        raise ValueError(
            f"Requested eval_samples={eval_samples}, but only {valid_indices.numel()} unique valid "
            f"eval start indices are available for horizon={horizon}."
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(horizon_eval_seed(eval_seed, horizon))
    permutation = torch.randperm(int(valid_indices.numel()), generator=generator)
    return valid_indices[permutation[: int(eval_samples)]].to(dtype=torch.long)


def fixed_eval_plan_for_horizons(
    cfg: Config,
    horizons: list[int],
    *,
    eval_seed: int,
    eval_samples: int,
    field_split: str = "eval",
) -> dict[int, Tensor]:
    return {
        int(horizon): make_fixed_eval_start_indices(
            cfg,
            int(horizon),
            eval_seed,
            eval_samples,
            field_split=field_split,
        )
        for horizon in horizons
    }


def attach_fixed_eval_plan(cfg: Config, starts: Tensor, *, eval_batch_size: int) -> None:
    if eval_batch_size <= 0:
        raise ValueError(f"eval_batch_size must be positive, got {eval_batch_size}.")
    if int(starts.numel()) % int(eval_batch_size) != 0:
        raise ValueError(
            f"eval_samples={starts.numel()} must be divisible by eval_batch_size={eval_batch_size}."
        )
    cfg.batch_size = int(eval_batch_size)
    cfg.eval_batches = int(starts.numel()) // int(eval_batch_size)
    cfg._field_fixed_start_indices = starts.cpu()
    cfg._field_fixed_start_cursor = 0


def write_fixed_eval_plan(
    path: str | Path,
    starts_by_horizon: dict[int, Tensor],
    *,
    eval_seed: int,
    field_split: str = "eval",
) -> None:
    payload = {
        "schema_version": 1,
        "eval_seed": int(eval_seed),
        "field_split": field_split,
        "sample_rule": "explicit_start_indices_without_replacement_v1",
        "horizons": [int(horizon) for horizon in sorted(starts_by_horizon)],
        "start_indices": {
            str(int(horizon)): starts.detach().cpu().to(dtype=torch.long).tolist()
            for horizon, starts in sorted(starts_by_horizon.items())
        },
        "start_indices_hash": {
            str(int(horizon)): start_indices_hash(starts)
            for horizon, starts in sorted(starts_by_horizon.items())
        },
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
