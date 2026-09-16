"""Minimal multi-node NCCL communication smoke for the frozen reference fleet."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


rank = int(os.environ["RANK"])
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
try:
    value = torch.tensor([rank + 1], dtype=torch.float32, device="cuda")
    dist.all_reduce(value)
    expected = sum(range(1, int(os.environ["WORLD_SIZE"]) + 1))
    if float(value.item()) != float(expected):
        raise RuntimeError(f"all-reduce mismatch: {value.item()} != {expected}")
    print(f"rank={rank} world_size={dist.get_world_size()} all_reduce={value.item()}")
    dist.barrier()
finally:
    dist.destroy_process_group()
