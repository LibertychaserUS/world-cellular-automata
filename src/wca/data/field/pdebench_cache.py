from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import h5py
import torch


def _resolve_layout(tensor: torch.Tensor, *, variables: Sequence[str], layout: str) -> str:
    resolved = layout.lower()
    if resolved == "auto":
        if tensor.ndim == 3:
            resolved = "thw"
        elif tensor.ndim == 4 and tensor.shape[-1] == len(variables):
            resolved = "thwc"
        elif tensor.ndim == 4 and tensor.shape[1] == len(variables):
            resolved = "tchw"
        elif tensor.ndim == 4:
            resolved = "sthw"
        elif tensor.ndim == 5 and tensor.shape[-1] == len(variables):
            resolved = "sthwc"
        elif tensor.ndim == 5 and tensor.shape[2] == len(variables):
            resolved = "stchw"
        else:
            raise ValueError(f"Cannot infer PDEBench layout for shape {tuple(tensor.shape)} and variables={variables}")
    return resolved


def _sample_time_lengths(tensor: torch.Tensor, resolved_layout: str) -> list[int]:
    if resolved_layout in {"sthw", "sthwc", "stchw"}:
        return [int(tensor.shape[1])] * int(tensor.shape[0])
    return []


def _to_time_channel_height_width(tensor: torch.Tensor, *, variables: Sequence[str], layout: str) -> torch.Tensor:
    resolved = _resolve_layout(tensor, variables=variables, layout=layout)

    if resolved == "thw":
        if tensor.ndim != 3:
            raise ValueError(f"Layout thw expects 3D tensor, got {tuple(tensor.shape)}")
        return tensor.unsqueeze(1)
    if resolved == "tchw":
        if tensor.ndim != 4:
            raise ValueError(f"Layout tchw expects 4D tensor, got {tuple(tensor.shape)}")
        return tensor
    if resolved == "thwc":
        if tensor.ndim != 4:
            raise ValueError(f"Layout thwc expects 4D tensor, got {tuple(tensor.shape)}")
        return tensor.permute(0, 3, 1, 2)
    if resolved == "sthw":
        if tensor.ndim != 4:
            raise ValueError(f"Layout sthw expects 4D tensor, got {tuple(tensor.shape)}")
        samples, time, height, width = tensor.shape
        return tensor.reshape(samples * time, 1, height, width)
    if resolved == "sthwc":
        if tensor.ndim != 5:
            raise ValueError(f"Layout sthwc expects 5D tensor, got {tuple(tensor.shape)}")
        samples, time, height, width, channels = tensor.shape
        return tensor.permute(0, 1, 4, 2, 3).reshape(samples * time, channels, height, width)
    if resolved == "stchw":
        if tensor.ndim != 5:
            raise ValueError(f"Layout stchw expects 5D tensor, got {tuple(tensor.shape)}")
        samples, time, channels, height, width = tensor.shape
        return tensor.reshape(samples * time, channels, height, width)
    raise ValueError(f"Unsupported PDEBench layout={layout!r} for tensor shape {tuple(tensor.shape)}")


def _first_dataset_key(handle: h5py.File) -> str:
    candidates: list[str] = []

    def visit(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 3:
            candidates.append(name)

    handle.visititems(visit)
    if not candidates:
        raise ValueError("No high-dimensional dataset found in PDEBench HDF5 file.")
    return sorted(candidates)[0]


def _grouped_dataset_keys(handle: h5py.File, dataset_key: str, max_trajectories: int) -> list[str]:
    if "/" in dataset_key or dataset_key == "auto":
        return []
    keys: list[str] = []
    for group_name in sorted(handle.keys()):
        item = handle[group_name]
        if not isinstance(item, h5py.Group):
            continue
        if dataset_key in item and isinstance(item[dataset_key], h5py.Dataset):
            keys.append(f"{group_name}/{dataset_key}")
        if max_trajectories > 0 and len(keys) >= max_trajectories:
            break
    return keys


def _load_hdf5_dataset(
    handle: h5py.File,
    *,
    dataset_key: str,
    max_trajectories: int,
    variables: Sequence[str],
    layout: str,
) -> tuple[torch.Tensor, str, int, list[int], str]:
    if dataset_key == "auto":
        dataset_key = _first_dataset_key(handle)
    grouped_keys = _grouped_dataset_keys(handle, dataset_key, max_trajectories)
    if grouped_keys:
        trajectories: list[torch.Tensor] = []
        resolved_layouts: list[str] = []
        for key in grouped_keys:
            raw = torch.as_tensor(handle[key][()], dtype=torch.float32)
            resolved_layouts.append(_resolve_layout(raw, variables=variables, layout=layout))
            trajectories.append(_to_time_channel_height_width(raw, variables=variables, layout=layout))
        trajectory_lengths = [int(trajectory.shape[0]) for trajectory in trajectories]
        resolved_layout = resolved_layouts[0] if resolved_layouts else layout
        if any(item != resolved_layout for item in resolved_layouts):
            raise ValueError(f"Grouped PDEBench datasets have mixed resolved layouts: {sorted(set(resolved_layouts))}")
        return torch.cat(trajectories, dim=0), f"*/{dataset_key}", len(grouped_keys), trajectory_lengths, resolved_layout
    if dataset_key not in handle:
        raise KeyError(f"Dataset key {dataset_key!r} not found in HDF5 file")
    raw = torch.as_tensor(handle[dataset_key][()], dtype=torch.float32)
    resolved_layout = _resolve_layout(raw, variables=variables, layout=layout)
    data = _to_time_channel_height_width(
        raw,
        variables=variables,
        layout=layout,
    )
    trajectory_lengths = _sample_time_lengths(raw, resolved_layout)
    return data, dataset_key, len(trajectory_lengths), trajectory_lengths, resolved_layout


def _trajectory_offsets(lengths: Sequence[int]) -> list[int]:
    offsets: list[int] = []
    cursor = 0
    for length in lengths:
        offsets.append(cursor)
        cursor += int(length)
    return offsets


def convert_hdf5_time_series_to_cache(
    *,
    source_path: str | Path,
    dataset_key: str,
    output_path: str | Path,
    manifest_path: str | Path,
    variables: Sequence[str],
    train_size: int,
    eval_size: int,
    source_manifest_path: str | Path | None = None,
    layout: str = "auto",
    max_trajectories: int = 0,
) -> None:
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(source)
    with h5py.File(source, "r") as handle:
        data, resolved_dataset_key, trajectory_count, trajectory_lengths, resolved_layout = _load_hdf5_dataset(
            handle,
            dataset_key=dataset_key,
            max_trajectories=max_trajectories,
            variables=variables,
            layout=layout,
        )
    data = data.contiguous()
    trajectory_offsets = _trajectory_offsets(trajectory_lengths)
    uniform_time_steps = sorted(set(trajectory_lengths))
    split_unit = "trajectory" if trajectory_lengths else "frame"
    if split_unit == "trajectory" and int(train_size) + int(eval_size) > len(trajectory_lengths):
        raise ValueError(
            "Grouped PDEBench cache uses trajectory-level split sizes. "
            f"train_size + eval_size = {int(train_size) + int(eval_size)} exceeds "
            f"trajectory_count={len(trajectory_lengths)}."
        )
    train_ids = list(range(int(train_size))) if split_unit == "trajectory" else list(range(int(train_size)))
    eval_ids = (
        list(range(int(train_size), min(int(train_size) + int(eval_size), len(trajectory_lengths))))
        if split_unit == "trajectory"
        else list(range(int(train_size), int(train_size) + int(eval_size)))
    )
    test_ids = (
        list(range(int(train_size) + int(eval_size), len(trajectory_lengths)))
        if split_unit == "trajectory"
        else []
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "dataset_id": "pdebench_cache",
        "data": data,
        "variables": list(variables),
        "source_format": "pdebench_hdf5",
        "source_path": str(source),
        "source_manifest": str(source_manifest_path) if source_manifest_path else "",
        "dataset_key": resolved_dataset_key,
        "layout": layout,
        "resolved_layout": resolved_layout,
        "trajectory_count": trajectory_count,
        "trajectory_lengths": trajectory_lengths,
        "trajectory_offsets": trajectory_offsets,
        "time_steps_per_trajectory": uniform_time_steps[0] if len(uniform_time_steps) == 1 else 0,
        "max_trajectories": int(max_trajectories),
        "split_unit": split_unit,
    }
    torch.save(payload, output)
    split_manifest = {
        "schema_version": 1,
        "split_unit": split_unit,
        "train_ids": train_ids,
        "eval_ids": eval_ids,
        "test_ids": test_ids,
        "trajectory_count": int(trajectory_count),
        "trajectory_lengths": trajectory_lengths,
        "trajectory_offsets": trajectory_offsets,
    }
    split_manifest_path = Path(manifest_path).with_name("split_manifest.json")
    split_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    split_manifest_path.write_text(json.dumps(split_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "source_path": str(source),
        "source_manifest": str(source_manifest_path) if source_manifest_path else "",
        "dataset_key": resolved_dataset_key,
        "layout": layout,
        "resolved_layout": resolved_layout,
        "trajectory_count": trajectory_count,
        "trajectory_lengths": trajectory_lengths,
        "trajectory_offsets": trajectory_offsets,
        "time_steps_per_trajectory": uniform_time_steps[0] if len(uniform_time_steps) == 1 else 0,
        "max_trajectories": int(max_trajectories),
        "output_path": str(output),
        "split_manifest": str(split_manifest_path),
        "split_unit": split_unit,
        "variables": list(variables),
        "data_shape": list(data.shape),
        "train_start": 0,
        "train_size": int(train_size),
        "eval_start": int(train_size),
        "eval_size": int(eval_size),
        "test_start": int(train_size) + int(eval_size) if split_unit == "trajectory" else 0,
        "test_size": len(test_ids),
    }
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
