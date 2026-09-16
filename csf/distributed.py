"""
Distributed runtime helpers (torch.distributed, launched with `torchrun` or plain `python`).

Backend selection: NCCL on Linux with CUDA, Gloo on Windows (NCCL is not available there)
or on CPU. When the script is started without torchrun it runs as a single process
(world_size=1) and every helper degrades to a no-op.

Input : environment variables set by torchrun (RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, ...) or by
        csf.launch (same variables + CSF_INIT_METHOD=file://... rendezvous, used on Windows).
Output: `DistInfo` (rank, local_rank, world_size, device) plus collective helpers.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, List

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def init_distributed(timeout_minutes: int = 180) -> DistInfo:
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if (torch.cuda.is_available() and platform.system() != "Windows") else "gloo"
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        kwargs = {"device_id": device} if backend == "nccl" else {}
        if os.environ.get("CSF_INIT_METHOD"):
            kwargs.update(init_method=os.environ["CSF_INIT_METHOD"], rank=rank, world_size=world_size)
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_minutes), **kwargs)
    return DistInfo(rank, local_rank, world_size, device)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_gather_objects(obj: Any) -> List[Any]:
    """Gather an arbitrary picklable object from every rank (returns [obj] when not distributed)."""
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]
    out: List[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def broadcast_object(obj: Any, src: int = 0) -> Any:
    if not (dist.is_available() and dist.is_initialized()):
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=src)
    return box[0]


def all_reduce_mean(value: float, device: torch.device) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device if dist.get_backend() == "nccl" else "cpu")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / dist.get_world_size())


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
