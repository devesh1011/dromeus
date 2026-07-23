"""Training algorithm interface."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from dromeus.manifests.models import RoundId


class SerializableState(Protocol):
    """State seam shared by M1 and future algorithm/codec implementations."""

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: Mapping[str, object]) -> None: ...


def checksum_tensors(tensors: dict[str, np.ndarray]) -> str:
    """Hash named tensor values independently of the algorithm implementation."""
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = np.ascontiguousarray(tensors[name])
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tensor.shape).encode())
        digest.update(tensor.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class AlgorithmSnapshot:
    round_id: RoundId
    phase: str
    weights: dict[str, np.ndarray]


@dataclass(frozen=True)
class TrainedWeightsBundle:
    round_id: RoundId
    tensors: dict[str, np.ndarray]
    checksum: str


__all__ = [
    "AlgorithmSnapshot",
    "SerializableState",
    "TrainedWeightsBundle",
    "checksum_tensors",
]
