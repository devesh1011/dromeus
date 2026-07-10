"""Trainer interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


class WeightTrainer(Protocol):
    """Owns model, optimizer, and local training state."""

    def train_local_steps(self, step_count: int) -> None: ...

    def weights(self) -> dict[str, np.ndarray]:
        """Return a copy of current model weights."""
        ...

    def load_weights(self, weights: dict[str, np.ndarray]) -> None:
        """Replace current model weights."""
        ...


@runtime_checkable
class StochasticGradientTrainer(WeightTrainer, Protocol):
    """Trainer that can expose one stochastic gradient at current weights."""

    def stochastic_gradients(self) -> dict[str, np.ndarray]: ...
