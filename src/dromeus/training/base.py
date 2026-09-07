"""Trainer interface."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Named finite metrics with application-defined meaning and units."""

    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        if any(
            not name.strip() or not math.isfinite(value)
            for name, value in self.metrics.items()
        ):
            raise ValueError("evaluation metrics need nonblank names and finite values")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


class WeightTrainer(Protocol):
    """Owns model, optimizer, and local training state."""

    def train_local_steps(self, step_count: int) -> None: ...

    def weights(self) -> dict[str, np.ndarray]:
        """Return a copy of current model weights."""
        ...

    def load_weights(self, weights: dict[str, np.ndarray]) -> None:
        """Replace current model weights."""
        ...

    @property
    def local_loss(self) -> float | None:
        """Return the latest local loss, or none before training."""
        ...

    def evaluate(self) -> EvaluationResult | tuple[float, float] | None:
        """Return named metrics, a legacy loss/accuracy pair, or no evaluation."""
        ...


@runtime_checkable
class CheckpointTrainer(WeightTrainer, Protocol):
    """Trainer whose complete durable state is tensor-serializable."""

    def checkpoint_tensors(self) -> dict[str, np.ndarray]: ...

    def load_checkpoint_tensors(self, state: dict[str, np.ndarray]) -> None: ...
