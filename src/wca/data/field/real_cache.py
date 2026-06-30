from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch
from torch import Tensor

from wca.config import Config
from wca.data.field.synthetic import (
    choose_field_target_steps,
    field_grid_shape,
    field_patch_shape,
    field_token_shape,
    inject_field_horizon_conditioning,
    make_field_adjacency,
    make_field_input_channel_mask,
    make_field_input_visibility,
    patchify_field,
)
from wca.schemas import TensorBatch


SUPPORTED_REAL_FIELD_DATASETS = {"weatherbench_cache", "weatherbench2_era5_cache", "pdebench_cache"}


@lru_cache(maxsize=8)
def _load_cache(path: str) -> Dict[str, Any]:
    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Real field cache not found: {cache_path}")
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected cache dict in {cache_path}, got {type(payload).__name__}")
    data = payload.get("data")
    if not isinstance(data, Tensor):
        raise ValueError(f"Real field cache {cache_path} must contain a Tensor under key 'data'.")
    if data.ndim != 4:
        raise ValueError(f"Expected real field data [T,C,H,W], got {tuple(data.shape)}")
    payload["data"] = data.to(dtype=torch.float32).contiguous()
    return payload


def _split_bounds(cfg: Config, split: str, total_steps: int, target_steps: int | None = None) -> Tuple[int, int]:
    if split == "test":
        start = int(getattr(cfg, "field_test_start", 0))
        size = int(getattr(cfg, "field_test_size", 0))
        if size <= 0:
            raise ValueError("field_test_size must be positive when evaluating the locked test split.")
    elif split == "val":
        start = int(getattr(cfg, "field_val_start", 0) or getattr(cfg, "field_eval_start", 0))
        size = int(getattr(cfg, "field_val_size", 0) or getattr(cfg, "field_eval_size", 0))
    elif split == "eval":
        start = int(getattr(cfg, "field_eval_start", 0))
        size = int(getattr(cfg, "field_eval_size", 0))
    else:
        start = int(getattr(cfg, "field_train_start", 0))
        size = int(getattr(cfg, "field_train_size", 0))
    if size <= 0:
        size = total_steps - start
    end = min(total_steps, start + size)
    if start < 0 or start >= total_steps:
        raise ValueError(f"Invalid {split} start={start} for real field cache with {total_steps} steps")
    target_steps = int(cfg.field_target_steps) if target_steps is None else int(target_steps)
    min_required = int(cfg.field_input_steps) + target_steps * max(1, int(cfg.field_stride))
    if end - start < min_required:
        raise ValueError(
            f"Real field {split} range [{start}, {end}) is too short for "
            f"input_steps={cfg.field_input_steps}, target_steps={target_steps}, "
            f"stride={cfg.field_stride}."
        )
    return start, end


def _target_offset(cfg: Config, target_steps: int) -> int:
    stride = max(1, int(cfg.field_stride))
    return (int(cfg.field_input_steps) + int(target_steps) - 1) * stride


def _trajectory_lengths(payload: Dict[str, Any]) -> list[int]:
    if str(payload.get("split_unit", "")) != "trajectory":
        return []
    raw = payload.get("trajectory_lengths", [])
    if not isinstance(raw, Sequence):
        return []
    lengths = [int(item) for item in raw]
    return [item for item in lengths if item > 0]


def _trajectory_offsets(payload: Dict[str, Any], lengths: Sequence[int]) -> list[int]:
    raw = payload.get("trajectory_offsets", [])
    if isinstance(raw, Sequence) and len(raw) == len(lengths):
        return [int(item) for item in raw]
    offsets: list[int] = []
    cursor = 0
    for length in lengths:
        offsets.append(cursor)
        cursor += int(length)
    return offsets


def valid_start_indices(cfg: Config, split: str, total_steps: int, target_steps: int, payload: Dict[str, Any] | None = None) -> Tensor:
    """Return all valid sequence start indices for a split.

    For grouped PDEBench caches, ``field_train_start/size`` and
    ``field_eval_start/size`` are trajectory ranges, not flat frame ranges.
    This prevents trajectory leakage and windows crossing between trajectories.
    """

    payload = payload or {}
    lengths = _trajectory_lengths(payload)
    target_offset = _target_offset(cfg, int(target_steps))
    if lengths:
        offsets = _trajectory_offsets(payload, lengths)
        if sum(lengths) != int(total_steps):
            raise ValueError(
                "Trajectory metadata does not match cache data length: "
                f"sum(trajectory_lengths)={sum(lengths)}, total_steps={total_steps}."
            )
        if split == "test":
            start = int(getattr(cfg, "field_test_start", 0))
            size = int(getattr(cfg, "field_test_size", 0))
            if size <= 0:
                raise ValueError("field_test_size must be positive when evaluating the locked test split.")
        elif split == "val":
            start = int(getattr(cfg, "field_val_start", 0) or getattr(cfg, "field_eval_start", 0))
            size = int(getattr(cfg, "field_val_size", 0) or getattr(cfg, "field_eval_size", 0))
        elif split == "eval":
            start = int(getattr(cfg, "field_eval_start", 0))
            size = int(getattr(cfg, "field_eval_size", 0))
        else:
            start = int(getattr(cfg, "field_train_start", 0))
            size = int(getattr(cfg, "field_train_size", 0))
        if size <= 0:
            size = len(lengths) - start
        end = min(len(lengths), start + size)
        if start < 0 or start >= len(lengths):
            raise ValueError(
                f"Invalid {split} trajectory start={start} for cache with {len(lengths)} trajectories"
            )
        ranges: list[Tensor] = []
        too_short: list[int] = []
        for trajectory_id in range(start, end):
            length = int(lengths[trajectory_id])
            count = length - target_offset
            if count <= 0:
                too_short.append(trajectory_id)
                continue
            offset = int(offsets[trajectory_id])
            ranges.append(torch.arange(offset, offset + count, dtype=torch.long))
        if not ranges:
            raise ValueError(
                f"No valid start indices for real field {split} trajectory range [{start}, {end}) "
                f"with target_steps={target_steps}."
            )
        if too_short and len(too_short) == end - start:
            raise ValueError(
                f"All selected {split} trajectories are too short for target_steps={target_steps}: {too_short}"
            )
        return torch.cat(ranges, dim=0)

    start, end = _split_bounds(cfg, split, total_steps, target_steps=target_steps)
    max_start = end - target_offset
    if max_start <= start:
        raise ValueError(f"No valid start indices for real field {split} range [{start}, {end})")
    return torch.arange(start, max_start, dtype=torch.long)


def _validate_fixed_start_indices(fixed_indices: Tensor, valid_indices: Tensor, split: str) -> None:
    if fixed_indices.numel() == 0:
        raise ValueError("Fixed field eval start-index plan is empty.")
    if valid_indices.numel() == 0:
        raise ValueError(f"No valid field {split} start indices are available.")
    fixed_cpu = fixed_indices.cpu().to(dtype=torch.long)
    valid_cpu = valid_indices.cpu().to(dtype=torch.long)
    if not torch.isin(fixed_cpu, valid_cpu).all():
        invalid = fixed_cpu[~torch.isin(fixed_cpu, valid_cpu)][:8].tolist()
        raise ValueError(
            "Fixed field eval start-index plan contains indices outside the configured split/window bounds: "
            f"{invalid}"
        )


def _sample_start_indices(
    cfg: Config,
    split: str,
    total_steps: int,
    device: torch.device,
    target_steps: int,
    payload: Dict[str, Any] | None = None,
) -> Tensor:
    valid_indices = valid_start_indices(cfg, split, total_steps, target_steps, payload=payload)
    fixed_indices = getattr(cfg, "_field_fixed_start_indices", None)
    if split in {"eval", "val", "test"} and isinstance(fixed_indices, Tensor):
        _validate_fixed_start_indices(fixed_indices, valid_indices, split)
        cursor = int(getattr(cfg, "_field_fixed_start_cursor", 0) or 0)
        next_cursor = cursor + int(cfg.batch_size)
        if next_cursor > int(fixed_indices.numel()):
            raise ValueError(
                "Fixed field eval start-index plan is exhausted. "
                f"cursor={cursor}, batch_size={cfg.batch_size}, plan_size={fixed_indices.numel()}."
            )
        cfg._field_fixed_start_cursor = next_cursor
        return fixed_indices[cursor:next_cursor].to(device=device, dtype=torch.long)

    selected = torch.randint(0, int(valid_indices.numel()), (cfg.batch_size,), device=device)
    return valid_indices.to(device=device)[selected]


def _frame_indices(cfg: Config, starts: Tensor, target_steps: int) -> Tensor:
    stride = max(1, int(cfg.field_stride))
    total_steps = int(cfg.field_input_steps) + int(target_steps)
    offsets = torch.arange(total_steps, device=starts.device, dtype=starts.dtype) * stride
    return starts.unsqueeze(1) + offsets.unsqueeze(0)


def _trajectory_ids_for_starts(payload: Dict[str, Any], starts: Tensor, device: torch.device) -> Tensor:
    lengths = _trajectory_lengths(payload)
    if not lengths:
        return torch.full_like(starts, -1, device=device, dtype=torch.long)
    offsets = _trajectory_offsets(payload, lengths)
    ids = torch.full_like(starts, -1, device=device, dtype=torch.long)
    for trajectory_id, (offset, length) in enumerate(zip(offsets, lengths, strict=True)):
        mask = (starts >= int(offset)) & (starts < int(offset) + int(length))
        ids = torch.where(mask, torch.full_like(ids, int(trajectory_id)), ids)
    return ids


def _validate_cache_grid(cfg: Config, data: Tensor) -> None:
    expected_shape = field_grid_shape(cfg)
    if data.shape[-2:] != expected_shape:
        raise ValueError(
            f"Real field cache grid {tuple(data.shape[-2:])} does not match configured field grid {expected_shape}. "
            "Create a deterministic cache with matching native or projected dimensions first."
        )
    if int(getattr(cfg, "field_output_dim", 1)) != int(data.shape[1]):
        raise ValueError(
            f"field_output_dim={getattr(cfg, 'field_output_dim', 1)} does not match cache channel count {data.shape[1]}"
        )
    if cfg.hidden_dim < int(data.shape[1]):
        raise ValueError(
            f"hidden_dim={cfg.hidden_dim} is too small for {data.shape[1]} current field channels."
        )
    field_token_shape(cfg)


def _parse_variable_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _validate_cache_variables(cfg: Config, payload: Dict[str, Any]) -> None:
    expected = _parse_variable_list(getattr(cfg, "field_variables", "") or getattr(cfg, "field_variable", ""))
    if not expected:
        return
    actual = [str(item) for item in payload.get("variables", [])]
    if actual != expected:
        raise ValueError(
            "Real field cache variables do not match config. "
            f"expected={expected}, actual={actual}. Rebuild the cache or fix field_variables ordering."
        )


def _validate_pdebench_trajectory_contract(payload: Dict[str, Any], total_steps: int) -> None:
    if str(payload.get("split_unit", "")) != "trajectory":
        raise ValueError(
            "pdebench_cache requires trajectory-level metadata. "
            "Regenerate the cache with the current generate_pdebench_cache.py instead of using a flattened frame cache."
        )
    lengths = _trajectory_lengths(payload)
    if not lengths:
        raise ValueError("pdebench_cache requires non-empty trajectory_lengths metadata.")
    if sum(lengths) != int(total_steps):
        raise ValueError(
            "pdebench_cache trajectory_lengths do not match data length: "
            f"sum={sum(lengths)}, data_steps={total_steps}."
        )


def _tokens_for_readout(tokens: Tensor) -> Tensor:
    if tokens.shape[-1] == 1:
        return tokens.squeeze(-1)
    return tokens


def make_real_field_batch(cfg: Config, device: torch.device, split: str = "train") -> TensorBatch:
    actual_split = str(getattr(cfg, "_field_eval_split_override", "") or split) if split == "eval" else split
    if cfg.field_dataset not in SUPPORTED_REAL_FIELD_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_REAL_FIELD_DATASETS))
        raise ValueError(f"Unsupported real field_dataset={cfg.field_dataset!r}. Supported: {supported}")
    if not cfg.field_data_path:
        raise ValueError("field_data_path is required for real field cache datasets.")

    payload = _load_cache(cfg.field_data_path)
    data_cpu = payload["data"]
    _validate_cache_grid(cfg, data_cpu)
    _validate_cache_variables(cfg, payload)
    if cfg.field_dataset == "pdebench_cache":
        _validate_pdebench_trajectory_contract(payload, int(data_cpu.shape[0]))
    token_height, token_width = field_token_shape(cfg)
    cfg.grid_size = token_height
    cfg.n_nodes = token_height * token_width

    target_steps = choose_field_target_steps(cfg, device)
    starts = _sample_start_indices(cfg, actual_split, data_cpu.shape[0], device, target_steps=target_steps, payload=payload)
    indices = _frame_indices(cfg, starts, target_steps=target_steps).cpu()
    n_channels = int(data_cpu.shape[1])
    sequence = data_cpu[indices.reshape(-1)].view(
        cfg.batch_size,
        -1,
        n_channels,
        data_cpu.shape[-2],
        data_cpu.shape[-1],
    )
    sequence = sequence.to(device=device)

    inputs = sequence[:, : cfg.field_input_steps]
    target_index = int(cfg.field_input_steps) + int(target_steps) - 1
    target = sequence[:, target_index]
    patch_shape = field_patch_shape(cfg)
    current_tokens = patchify_field(inputs[:, -1], patch_shape)
    previous_tokens = patchify_field(inputs[:, -2] if cfg.field_input_steps > 1 else inputs[:, -1], patch_shape)
    target_tokens = patchify_field(target, patch_shape)
    output_current_tokens = _tokens_for_readout(current_tokens)
    output_target_tokens = _tokens_for_readout(target_tokens)

    H = torch.zeros(cfg.batch_size, cfg.n_nodes, cfg.hidden_dim, device=device)
    token_channels = int(current_tokens.shape[-1])
    H[:, :, :token_channels] = current_tokens
    coord_start = token_channels
    if cfg.hidden_dim >= token_channels * 2:
        H[:, :, token_channels : token_channels * 2] = previous_tokens
        coord_start = token_channels * 2
    coord_start = inject_field_horizon_conditioning(H, cfg, target_steps, coord_start)
    if cfg.hidden_dim > coord_start + 1:
        y_coords = torch.linspace(-1.0, 1.0, token_height, device=device)
        x_coords = torch.linspace(-1.0, 1.0, token_width, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        H[:, :, coord_start] = xx.reshape(1, cfg.n_nodes)
        H[:, :, coord_start + 1] = yy.reshape(1, cfg.n_nodes)

    adjacency = make_field_adjacency(cfg, device).unsqueeze(0).expand(cfg.batch_size, cfg.n_nodes, cfg.n_nodes).clone()
    input_visibility = (
        make_field_input_visibility(cfg, device).unsqueeze(0).expand(cfg.batch_size, cfg.n_nodes, cfg.n_nodes).clone()
    )
    variables = payload.get("variables", [getattr(cfg, "field_variable", "")])
    variable = ",".join(str(item) for item in variables)
    return {
        "H": H,
        "adjacency": adjacency,
        "input_visibility": input_visibility,
        "input_visibility_channels": make_field_input_channel_mask(cfg, device),
        "target_idx": torch.zeros(cfg.batch_size, dtype=torch.long, device=device),
        "label": output_target_tokens,
        "field_input": inputs,
        "field_target": target,
        "field_start_index": starts,
        "field_trajectory_id": _trajectory_ids_for_starts(payload, starts, device),
        "field_target_index": starts + target_index * max(1, int(cfg.field_stride)),
        "field_target_steps_actual": torch.full((cfg.batch_size,), target_steps, dtype=torch.long, device=device),
        "field_prediction_baseline": output_current_tokens,
        "field_previous_tokens": _tokens_for_readout(previous_tokens),
        "field_source_dataset": cfg.field_dataset,
        "field_variable": variable,
        "source_sign": torch.zeros(cfg.batch_size, device=device),
        "distractor_sign": torch.zeros(cfg.batch_size, device=device),
        "raw_distance": torch.zeros(cfg.batch_size, device=device),
    }
