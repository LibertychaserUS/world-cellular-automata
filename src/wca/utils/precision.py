from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch


def autocast_dtype(precision: str) -> torch.dtype | None:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return None
    raise ValueError(f"Unsupported precision: {precision}")


def autocast_context(device: torch.device, precision: str) -> Any:
    dtype = autocast_dtype(precision)
    if dtype is None or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_grad_scaler(device: torch.device, precision: str) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and precision == "fp16"))


def cuda_memory_metrics(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {
            "cuda_memory_allocated_mb": float("nan"),
            "cuda_memory_reserved_mb": float("nan"),
            "cuda_peak_memory_allocated_mb": float("nan"),
            "cuda_peak_memory_reserved_mb": float("nan"),
        }
    return {
        "cuda_memory_allocated_mb": torch.cuda.memory_allocated(device) / 1024.0 / 1024.0,
        "cuda_memory_reserved_mb": torch.cuda.memory_reserved(device) / 1024.0 / 1024.0,
        "cuda_peak_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0,
        "cuda_peak_memory_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024.0 / 1024.0,
    }
