"""NCCL/Gloo point-to-point exchange for the independent reference."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import torch
import torch.distributed as dist
from torch import Tensor

TensorArtifacts = Mapping[str, Mapping[str, Tensor]]


class PairExchange(Protocol):
    def exchange(
        self,
        *,
        peer_rank: int,
        artifacts: TensorArtifacts,
    ) -> dict[str, dict[str, Tensor]]: ...


class TorchDistributedPairExchange:
    """Exchange a fixed named tensor schema over the default process group."""

    def exchange(
        self,
        *,
        peer_rank: int,
        artifacts: TensorArtifacts,
    ) -> dict[str, dict[str, Tensor]]:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Torch distributed process group is not initialized")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if not 0 <= peer_rank < world_size or peer_rank == rank:
            raise ValueError("peer rank is outside the process group")
        backend = str(dist.get_backend())
        if backend not in {"gloo", "nccl"}:
            raise ValueError(f"unsupported reference backend: {backend}")
        entries = _ordered_entries(artifacts, backend=backend)
        received = {
            artifact_name: {
                tensor_name: torch.zeros_like(value)
                for tensor_name, value in tensors.items()
            }
            for artifact_name, tensors in artifacts.items()
        }

        def send_all() -> None:
            for tag, (artifact_name, tensor_name, value) in enumerate(entries):
                del artifact_name, tensor_name
                dist.send(value.contiguous(), dst=peer_rank, tag=tag)

        def receive_all() -> None:
            for tag, (artifact_name, tensor_name, _) in enumerate(entries):
                dist.recv(
                    received[artifact_name][tensor_name],
                    src=peer_rank,
                    tag=tag,
                )

        if rank < peer_rank:
            send_all()
            receive_all()
        else:
            receive_all()
            send_all()
        return received


def _ordered_entries(
    artifacts: TensorArtifacts,
    *,
    backend: str,
) -> tuple[tuple[str, str, Tensor], ...]:
    if not artifacts or any(not tensors for tensors in artifacts.values()):
        raise ValueError("reference exchange artifacts must not be empty")
    entries = tuple(
        (artifact_name, tensor_name, artifacts[artifact_name][tensor_name])
        for artifact_name in sorted(artifacts)
        for tensor_name in sorted(artifacts[artifact_name])
    )
    for artifact_name, tensor_name, value in entries:
        label = f"{artifact_name}.{tensor_name}"
        if value.dtype != torch.float32:
            raise ValueError(f"{label} must use float32")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{label} must be finite")
        if backend == "gloo" and value.device.type != "cpu":
            raise ValueError("Gloo development exchange requires CPU tensors")
        if backend == "nccl" and value.device.type != "cuda":
            raise ValueError("NCCL reference exchange requires CUDA tensors")
    return entries


__all__ = ["PairExchange", "TensorArtifacts", "TorchDistributedPairExchange"]
