"""Decentralized parallel SGD algorithm."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from dromeus.algorithms.base import (
    AlgorithmSnapshot,
    TrainedWeightsBundle,
    checksum_tensors,
)
from dromeus.manifests.models import RoundId, TensorSchema
from dromeus.training.base import StochasticGradientTrainer, WeightTrainer

_DTYPES = {
    "float16": np.dtype(np.float16),
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
}


@dataclass
class DPSGDAdapter:
    trainer: WeightTrainer
    tensor_schema: TensorSchema
    local_steps: int
    learning_rate: float | None = None
    _round_id: RoundId = 0
    _phase: str = "created"

    def __post_init__(self) -> None:
        if self.local_steps <= 0:
            raise ValueError("local_steps must be positive")
        if self.learning_rate is not None and (
            self.learning_rate <= 0 or not math.isfinite(self.learning_rate)
        ):
            raise ValueError("learning_rate must be positive and finite")

    def pre_local(self, round_id: RoundId) -> AlgorithmSnapshot:
        self._round_id = round_id
        self._phase = "pre-local"
        return self.snapshot()

    def local_training(self) -> AlgorithmSnapshot:
        if self.learning_rate is not None and isinstance(
            self.trainer, StochasticGradientTrainer
        ):
            for _ in range(self.local_steps):
                self.step()
        else:
            self.trainer.train_local_steps(self.local_steps)
        self._phase = "post-local"
        return self.snapshot()

    def step(
        self,
        peer_bundles: Sequence[TrainedWeightsBundle] = (),
        peer_weights: Sequence[float] | None = None,
    ) -> AlgorithmSnapshot:
        if self.learning_rate is None:
            raise ValueError("learning_rate is required for paper D-PSGD step")
        if not isinstance(self.trainer, StochasticGradientTrainer):
            raise TypeError("trainer must expose stochastic_gradients")
        local = self.trainer.weights()
        self._validate_tensors(local)
        gradients = self.trainer.stochastic_gradients()
        self._validate_tensors(gradients)
        mixed = self._weighted_average(local, peer_bundles, peer_weights)
        updated = {
            name: (
                mixed[name].astype(np.float32)
                - self.learning_rate * gradients[name].astype(np.float32)
            ).astype(np.float32)
            for name in mixed
        }
        self.trainer.load_weights(updated)
        self._phase = "post-local"
        return self.snapshot()

    def post_local_bundle(self) -> TrainedWeightsBundle:
        tensors = self.trainer.weights()
        self._validate_tensors(tensors)
        self._phase = "bundled"
        return TrainedWeightsBundle(
            round_id=self._round_id,
            tensors=tensors,
            checksum=checksum_tensors(tensors),
        )

    def validate_peer(self, peer_bundle: TrainedWeightsBundle) -> None:
        """Validate a peer update without mutating local model state."""
        if peer_bundle.round_id != self._round_id:
            raise ValueError("peer bundle round does not match current round")
        self._validate_tensors(peer_bundle.tensors)
        if checksum_tensors(peer_bundle.tensors) != peer_bundle.checksum:
            raise ValueError("peer bundle checksum mismatch")

    def peer_apply(self, peer_bundle: TrainedWeightsBundle) -> AlgorithmSnapshot:
        self.validate_peer(peer_bundle)
        local = self.trainer.weights()
        self._validate_tensors(local)
        mixed = self._weighted_average(local, (peer_bundle,), (0.5,))
        self.trainer.load_weights(mixed)
        self._phase = "post-mix"
        return self.snapshot()

    def snapshot(self) -> AlgorithmSnapshot:
        return AlgorithmSnapshot(
            round_id=self._round_id,
            phase=self._phase,
            weights=self.trainer.weights(),
        )

    def _validate_tensors(self, tensors: dict[str, np.ndarray]) -> None:
        expected = {tensor.name: tensor for tensor in self.tensor_schema.tensors}
        if set(tensors) != set(expected):
            raise ValueError("tensor names do not match schema")
        for name, tensor in tensors.items():
            spec = expected[name]
            if spec.dtype not in _DTYPES:
                raise ValueError("D-PSGD only supports floating point tensors")
            if tensor.dtype != _DTYPES[spec.dtype]:
                raise ValueError(f"tensor {name} dtype does not match schema")
            if tensor.shape != spec.shape:
                raise ValueError(f"tensor {name} shape does not match schema")
            if not np.isfinite(tensor).all():
                raise ValueError(f"tensor {name} contains non-finite values")

    def _weighted_average(
        self,
        local: dict[str, np.ndarray],
        peer_bundles: Sequence[TrainedWeightsBundle],
        peer_weights: Sequence[float] | None,
    ) -> dict[str, np.ndarray]:
        if peer_weights is None:
            weight = 1.0 / (len(peer_bundles) + 1)
            peer_weights = tuple(weight for _ in peer_bundles)
            self_weight = weight
        else:
            if len(peer_weights) != len(peer_bundles):
                raise ValueError("peer weights must match peer bundles")
            self_weight = 1.0 - sum(peer_weights)
        if not math.isfinite(self_weight) or any(
            not math.isfinite(weight) for weight in peer_weights
        ):
            raise ValueError("mixing weights must be finite")
        if self_weight < 0 or any(weight < 0 for weight in peer_weights):
            raise ValueError("mixing weights must be non-negative")
        if not np.isclose(self_weight + sum(peer_weights), 1.0):
            raise ValueError("mixing weights must sum to one")
        mixed = {name: (local[name].astype(np.float32) * self_weight) for name in local}
        for bundle, weight in zip(peer_bundles, peer_weights, strict=True):
            if bundle.round_id != self._round_id:
                raise ValueError("peer bundle round does not match current round")
            self._validate_tensors(bundle.tensors)
            if checksum_tensors(bundle.tensors) != bundle.checksum:
                raise ValueError("peer bundle checksum mismatch")
            for name in mixed:
                mixed[name] += bundle.tensors[name].astype(np.float32) * weight
        return {name: value.astype(np.float32) for name, value in mixed.items()}
