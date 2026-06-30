from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.distributed as dist


@dataclass
class DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    backend: str

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0


def init_distributed_from_env() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return DistributedContext(False, 0, 1, 0, "")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return DistributedContext(True, rank, world_size, local_rank, backend)


def barrier(ctx: DistributedContext) -> None:
    if ctx.enabled:
        if ctx.backend == "nccl":
            dist.barrier(device_ids=[ctx.local_rank])
        else:
            dist.barrier()


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.destroy_process_group()


def broadcast_object(value: Any, ctx: DistributedContext, src: int = 0) -> Any:
    if not ctx.enabled:
        return value
    payload = [value]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def reduce_scalar(value: float, device: torch.device, ctx: DistributedContext, average: bool = True) -> float:
    if not ctx.enabled:
        return float(value)
    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if average:
        tensor = tensor / ctx.world_size
    return float(tensor.item())


def reduce_metric_dict(metrics: Dict[str, float], device: torch.device, ctx: DistributedContext) -> Dict[str, float]:
    if not ctx.enabled:
        return metrics
    reduced: Dict[str, float] = {}
    for key, value in metrics.items():
        value = float(value)
        valid = 0.0 if math.isnan(value) or math.isinf(value) else 1.0
        payload = torch.tensor([0.0 if valid == 0.0 else value, valid], device=device)
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
        if payload[1].item() > 0:
            reduced[key] = float((payload[0] / payload[1]).item())
        else:
            reduced[key] = float("nan")
    return reduced
