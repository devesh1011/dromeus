"""Canonical per-rank slow-weight trajectory evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import torch
from safetensors.torch import (
    save_file as _save_file,  # pyright: ignore[reportUnknownVariableType]
)
from torch import Tensor

_SaveSafetensors = Callable[[dict[str, Tensor], str], None]
save_safetensors = cast(_SaveSafetensors, _save_file)


@dataclass(frozen=True, slots=True)
class TrajectoryContext:
    experiment_sha256: str
    run_config_sha256: str
    initial_checkpoint_sha256: str
    tensor_schema_hash: str
    world_size: int
    benchmark_seed: int
    rank: int
    node_id: str
    interval: int
    total_outer_steps: int
    backend: Literal["gloo", "nccl"] = "gloo"
    acceptance_eligible: bool = False
    source_repository: str = "gensyn-ai/noloco"
    source_commit: str = "a1b4a425bdc4050a356cf9f4bae7c383419703ab"
    source_path: str = "src/noloco/sparse_optimizer_c.py"
    outer_gradient_convention: str = "phi-minus-theta-v1"
    pairing_convention: str = "dromeus-peer-scheduler-v1"

    def __post_init__(self) -> None:
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("trajectory rank is outside world size")
        if self.interval <= 0 or self.total_outer_steps <= 0:
            raise ValueError("trajectory cadence and total steps must be positive")
        if self.acceptance_eligible != (self.backend == "nccl"):
            raise ValueError("only NCCL trajectory evidence is acceptance eligible")
        if (
            self.source_repository,
            self.source_commit,
            self.source_path,
            self.outer_gradient_convention,
            self.pairing_convention,
        ) != (
            "gensyn-ai/noloco",
            "a1b4a425bdc4050a356cf9f4bae7c383419703ab",
            "src/noloco/sparse_optimizer_c.py",
            "phi-minus-theta-v1",
            "dromeus-peer-scheduler-v1",
        ):
            raise ValueError(
                "trajectory source provenance does not match frozen NoLoCo"
            )


@dataclass(frozen=True, slots=True)
class TrajectoryRecord:
    completed_outer_steps: int
    peer_rank: int | None
    snapshot_path: Path
    snapshot_sha256: str


@dataclass(frozen=True, slots=True)
class TrajectoryWriter:
    root: Path
    context: TrajectoryContext

    def should_write(self, *, completed_outer_steps: int) -> bool:
        if not 0 <= completed_outer_steps <= self.context.total_outer_steps:
            raise ValueError("completed outer steps are outside trajectory")
        return (
            completed_outer_steps == 0
            or completed_outer_steps == self.context.total_outer_steps
            or completed_outer_steps % self.context.interval == 0
        )

    def write(
        self,
        *,
        completed_outer_steps: int,
        peer_rank: int | None,
        slow_weights: Mapping[str, Tensor],
    ) -> TrajectoryRecord:
        if not self.should_write(completed_outer_steps=completed_outer_steps):
            raise ValueError("trajectory step is outside frozen evidence cadence")
        if completed_outer_steps == 0 and peer_rank is not None:
            raise ValueError("initial trajectory snapshot cannot have a peer")
        if completed_outer_steps > 0 and peer_rank is None:
            raise ValueError("post-outer trajectory snapshot requires a peer")
        tensors = _validated_cpu_tensors(slow_weights)
        rank_root = self.root / f"rank-{self.context.rank}"
        snapshot_root = rank_root / "snapshots"
        snapshot_root.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_root / (
            f"outer-{completed_outer_steps:06d}.safetensors"
        )
        temporary = snapshot_path.with_suffix(".tmp")
        try:
            save_safetensors(tensors, str(temporary))
            temporary.replace(snapshot_path)
        finally:
            temporary.unlink(missing_ok=True)
        record = TrajectoryRecord(
            completed_outer_steps=completed_outer_steps,
            peer_rank=peer_rank,
            snapshot_path=snapshot_path,
            snapshot_sha256=_file_sha256(snapshot_path),
        )
        rank_root.mkdir(parents=True, exist_ok=True)
        metadata = {
            **asdict(self.context),
            "phase": (
                "initialization" if completed_outer_steps == 0 else "post_outer"
            ),
            "completed_outer_steps": completed_outer_steps,
            "peer_rank": peer_rank,
            "snapshot_path": str(snapshot_path.relative_to(rank_root)),
            "snapshot_sha256": record.snapshot_sha256,
        }
        with (rank_root / "trajectory.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    metadata,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
        return record


def _validated_cpu_tensors(values: Mapping[str, Tensor]) -> dict[str, Tensor]:
    if not values:
        raise ValueError("trajectory slow weights must not be empty")
    result: dict[str, Tensor] = {}
    for name in sorted(values):
        value = values[name]
        if value.dtype != torch.float32:
            raise ValueError(f"trajectory tensor {name} must use float32")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"trajectory tensor {name} must be finite")
        result[name] = value.detach().cpu().contiguous().clone()
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["TrajectoryContext", "TrajectoryRecord", "TrajectoryWriter"]
