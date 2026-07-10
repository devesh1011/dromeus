"""Training algorithm interface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dromeus.manifests.models import RoundId


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
