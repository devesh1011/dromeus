"""Decentralized local SGD with pairwise averaging."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np

from dromeus.algorithms.base import (
    AlgorithmSnapshot,
    AlgorithmUpdate,
    UpdateBundle,
    ValidatedUpdate,
    checksum_tensors,
)
from dromeus.algorithms.codec import (
    IdentityCodec,
    NamedSafetensorsUpdateBundleCodec,
    NamedUpdateBundleCodec,
    UpdateCodec,
)
from dromeus.manifests.models import RoundId, TensorSchema, UpdateCodecBinding
from dromeus.training.base import (
    CheckpointTrainer,
    WeightTrainer,
)

_DTYPES = {
    "float16": np.dtype(np.float16),
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
}
_TRAINED_WEIGHTS = "trained_weights"


@dataclass
class DPSGDAdapter:
    trainer: WeightTrainer
    tensor_schema: TensorSchema
    local_steps: int
    training_round_count: int | None = None
    codec: UpdateCodec = field(default_factory=IdentityCodec)
    bundle_codec: NamedUpdateBundleCodec | None = None
    manifest_codec_id: str | None = None
    _round_id: RoundId = 0
    _phase: str = "created"
    _local_decoded: dict[str, np.ndarray] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.local_steps <= 0:
            raise ValueError("local_steps must be positive")
        if self.training_round_count is not None and self.training_round_count <= 0:
            raise ValueError("training_round_count must be positive")
        if (
            self.manifest_codec_id is not None
            and self.manifest_codec_id != self.codec.codec_id
        ):
            raise ValueError("algorithm codec does not match manifest")

    def pre_local(self, round_id: RoundId) -> None:
        self._round_id = round_id
        self._phase = "pre-local"
        self._local_decoded = None

    def local_training(self) -> None:
        if (
            self.training_round_count is None
            or self._round_id < self.training_round_count
        ):
            self.trainer.train_local_steps(self.local_steps)
        self._phase = "post-local"

    def post_local_bundle(self) -> UpdateBundle:
        if self.bundle_codec is None:
            raise RuntimeError("update bundle codec is not configured")
        tensors = self.trainer.weights()
        self._validate_tensors(tensors)
        encoded = self.codec.encode(tensors)
        bundle = self.bundle_codec.encode(
            round_id=self._round_id,
            artifacts={_TRAINED_WEIGHTS: encoded},
            codec_bindings={_TRAINED_WEIGHTS: self._codec_binding()},
        )
        try:
            decoded_local = self.codec.decode(encoded)
            self._validate_tensors(decoded_local)
        except BaseException:
            self.bundle_codec.release(bundle)
            raise
        self._local_decoded = {
            name: np.ascontiguousarray(value).copy()
            for name, value in decoded_local.items()
        }
        self._phase = "bundled"
        return bundle

    def validate_peer(self, peer_bundle: UpdateBundle) -> ValidatedUpdate:
        """Validate a peer update without mutating local model state."""
        if self.bundle_codec is None:
            raise RuntimeError("update bundle codec is not configured")
        if peer_bundle.metadata.round_id != self._round_id:
            raise ValueError("peer bundle round does not match current round")
        artifacts = self.bundle_codec.decode(
            peer_bundle,
            codec_bindings={_TRAINED_WEIGHTS: self._codec_binding()},
        )
        encoded = artifacts[_TRAINED_WEIGHTS]
        decoded = self.codec.decode(encoded)
        self._validate_tensors(decoded)
        return ValidatedUpdate(
            round_id=self._round_id,
            tensors=decoded,
            checksum=checksum_tensors(decoded),
        )

    def peer_apply(self, peer_update: AlgorithmUpdate) -> AlgorithmSnapshot:
        if not isinstance(peer_update, ValidatedUpdate):
            raise TypeError("D-PSGD requires a one-artifact validated update")
        if peer_update.round_id != self._round_id:
            raise ValueError("peer update round does not match current round")
        local = self._local_decoded or self.trainer.weights()
        self._validate_tensors(local)
        decoded = dict(peer_update.tensors)
        self._validate_tensors(decoded)
        if checksum_tensors(decoded) != peer_update.checksum:
            raise ValueError("peer update checksum mismatch")
        mixed = {
            name: (
                local[name].astype(np.float32) * np.float32(0.5)
                + decoded[name].astype(np.float32) * np.float32(0.5)
            ).astype(np.float32)
            for name in local
        }
        self.trainer.load_weights(mixed)
        self._phase = "post-mix"
        return self.snapshot()

    def release_bundle(self, bundle: UpdateBundle) -> None:
        if self.bundle_codec is None:
            raise RuntimeError("update bundle codec is not configured")
        self.bundle_codec.release(bundle)

    def configure_bundle_codec(
        self,
        *,
        artifact_root: Path,
        run_id: str,
        manifest_hash: str,
        sender_public_key: str,
        algorithm_id: str,
    ) -> None:
        """Bind formed run context without exposing codec details to runtime."""
        if self.bundle_codec is None:
            self.bundle_codec = NamedSafetensorsUpdateBundleCodec(
                artifact_root=artifact_root,
                run_id=run_id,
                manifest_hash=manifest_hash,
                sender_public_key=sender_public_key,
                algorithm_id=algorithm_id,
                artifact_schemas={_TRAINED_WEIGHTS: self.tensor_schema},
            )

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
                for name, value in cast(Mapping[object, object], trainer_value).items()
            ):
                raise ValueError("algorithm trainer state is invalid")
            self.trainer.load_checkpoint_tensors(
                dict(cast(Mapping[str, np.ndarray], trainer_value))
            )
            restored = self.trainer.weights()
            self._validate_tensors(restored)
            if any(
                not np.array_equal(restored[name], weights[name]) for name in weights
            ):
                raise ValueError("algorithm trainer state weights do not match")
        else:
            self.trainer.load_weights(weights)
        self._round_id = round_id
        self._phase = phase
        self._local_decoded = None

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

    def _codec_binding(self) -> UpdateCodecBinding:
        return UpdateCodecBinding(
            codec_id=self.codec.codec_id,
            codec_version=1,
            logical_schema=self.tensor_schema,
        )
