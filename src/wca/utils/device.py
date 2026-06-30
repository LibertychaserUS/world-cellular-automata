from __future__ import annotations

import torch


def resolve_device(device_name: str, local_rank: int = 0, distributed: bool = False) -> torch.device:
    if distributed and torch.cuda.is_available() and device_name in {"auto", "cuda"}:
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    if device_name == "auto":
        if torch.backends.mps.is_available() and not distributed:
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device_name == "cuda" and distributed and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device(device_name)


def sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
