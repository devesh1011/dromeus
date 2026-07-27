"""Decentralized parallel SGD algorithm."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from dromeus.algorithms.base import (
    AlgorithmSnapshot,
    TrainedWeightsBundle,
    checksum_tensors,
)
from dromeus.algorithms.codec import IdentityCodec, UpdateCodec
from dromeus.manifests.models import RoundId, TensorSchema
from dromeus.training.base import (
    CheckpointTrainer,
    StochasticGradientTrainer,
    WeightTrainer,
)

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
    training_round_count: int | None = None
    codec: UpdateCodec = field(default_factory=IdentityCodec)
    _round_id: RoundId = 0
    _phase: str = "created"

    def __post_init__(self) -> None:
        if self.local_steps <= 0:
            raise ValueError("local_steps must be positive")
        if self.learning_rate is not None and (
            self.learning_rate <= 0 or not math.isfinite(self.learning_rate)
        ):
            raise ValueError("learning_rate must be positive and finite")
        if self.training_round_count is not None and self.training_round_count <= 0:
            raise ValueError("training_round_count must be positive")

    def pre_local(self, round_id: RoundId) -> AlgorithmSnapshot:
        self._round_id = round_id
        self._phase = "pre-local"
        return self.snapshot()

    def local_training(self) -> AlgorithmSnapshot:
        if (
            self.training_round_count is None
            or self._round_id < self.training_round_count
        ):
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
        tensors = self.codec.encode(self.trainer.weights())
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
        decoded = self.codec.decode(peer_bundle.tensors)
        self._validate_tensors(decoded)

    def peer_apply(self, peer_bundle: TrainedWeightsBundle) -> AlgorithmSnapshot:
        self.validate_peer(peer_bundle)
        local = self.trainer.weights()
        self._validate_tensors(local)
        decoded = self.codec.decode(peer_bundle.tensors)
        decoded_bundle = TrainedWeightsBundle(
            round_id=peer_bundle.round_id,
            tensors=decoded,
            checksum=checksum_tensors(decoded),
        )
        mixed = self._weighted_average(local, (decoded_bundle,), (0.5,))
        self.trainer.load_weights(mixed)
        self._phase = "post-mix"
        return self.snapshot()

    def snapshot(self) -> AlgorithmSnapshot:
        return AlgorithmSnapshot(
            round_id=self._round_id,
            phase=self._phase,
            weights=self.trainer.weights(),
        )

    def checkpoint_tensors(self) -> dict[str, np.ndarray]:
        """Return complete durable state when the trainer exposes it."""
        if isinstance(self.trainer, CheckpointTrainer):
            return self.trainer.checkpoint_tensors()
        return self.trainer.weights()

    def evaluate(self) -> tuple[float, float]:
        """Evaluate through the trainer's local test-data seam."""
        evaluator = getattr(self.trainer, "evaluate", None)
        if not callable(evaluator):
            raise TypeError("trainer does not expose evaluation")
        result = evaluator()
        if not isinstance(result, tuple):
            raise ValueError("trainer evaluation must return loss and accuracy")
        values = cast(tuple[object, object], result)
        if len(values) != 2:
            raise ValueError("trainer evaluation must return loss and accuracy")
        loss = float(cast(float, values[0]))
        accuracy = float(cast(float, values[1]))
        if not math.isfinite(loss) or loss < 0:
            raise ValueError("evaluation loss must be finite and non-negative")
        if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError("evaluation accuracy must be finite in [0, 1]")
        return loss, accuracy

    @property
    def local_loss(self) -> float | None:
        """Return the most recent local minibatch loss when the trainer exposes it."""
        value = getattr(self.trainer, "last_local_loss", None)
        if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
            return float(value)
        return None

    def state_dict(self) -> dict[str, object]:
        """Return serializable algorithm, model, and codec state."""
        state: dict[str, object] = {
            "round_id": self._round_id,
            "phase": self._phase,
            "codec_id": self.codec.codec_id,
            "weights": self.trainer.weights(),
            "codec": self.codec.state_dict(),
        }
        if isinstance(self.trainer, CheckpointTrainer):
            state["trainer"] = self.trainer.checkpoint_tensors()
        return state

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore state after validating all tensors through the public schema."""
        round_id = state.get("round_id")
        phase = state.get("phase")
        codec_id = state.get("codec_id")
        weights_value = state.get("weights")
        codec_value = state.get("codec", {})
        trainer_value = state.get("trainer")
        if not isinstance(round_id, int) or round_id < 0:
            raise ValueError("algorithm state round_id is invalid")
        if not isinstance(phase, str) or not phase:
            raise ValueError("algorithm state phase is invalid")
        if codec_id != self.codec.codec_id:
            raise ValueError("algorithm state codec does not match")
        if not isinstance(weights_value, Mapping) or not all(
            isinstance(name, str) and isinstance(value, np.ndarray)
            for name, value in cast(Mapping[object, object], weights_value).items()
        ):
            raise ValueError("algorithm state weights are invalid")
        if not isinstance(codec_value, Mapping):
            raise ValueError("algorithm codec state is invalid")
        weights = dict(cast(Mapping[str, np.ndarray], weights_value))
        self._validate_tensors(weights)
        self.codec.load_state_dict(cast(Mapping[str, object], codec_value))
        if trainer_value is not None:
            if not isinstance(self.trainer, CheckpointTrainer) or not isinstance(
                trainer_value, Mapping
            ):
                raise ValueError("algorithm trainer state is invalid")
            if not all(
                isinstance(name, str) and isinstance(value, np.ndarray)
                for name, value in cast(
                    Mapping[object, object], trainer_value
                ).items()
            ):
                raise ValueError("algorithm trainer state is invalid")
            self.trainer.load_checkpoint_tensors(
                dict(cast(Mapping[str, np.ndarray], trainer_value))
            )
            restored = self.trainer.weights()
            self._validate_tensors(restored)
            if any(
                not np.array_equal(restored[name], weights[name])
                for name in weights
            ):
                raise ValueError("algorithm trainer state weights do not match")
        else:
            self.trainer.load_weights(weights)
        self._round_id = round_id
        self._phase = phase

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
            decoded = self.codec.decode(bundle.tensors)
            self._validate_tensors(decoded)
            for name in mixed:
                mixed[name] += decoded[name].astype(np.float32) * weight
        return {name: value.astype(np.float32) for name, value in mixed.items()}
