"""Training algorithm interface."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from dromeus.manifests.models import RoundId


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
