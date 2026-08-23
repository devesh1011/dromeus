"""NoLoCo pairwise outer optimization."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from dromeus.algorithms.base import (
    AlgorithmEvaluation,
    AlgorithmObservations,
    AlgorithmSnapshot,
    AlgorithmUpdate,
    NamedValidatedUpdate,
    UpdateBundle,
    checksum_artifacts,
)
from dromeus.algorithms.codec import (
    IdentityCodec,
    NamedSafetensorsUpdateBundleCodec,
    NamedUpdateBundleCodec,
    UpdateCodec,
    validate_tensor_map,
)
from dromeus.manifests.models import (
    NoLoCoConfig,
    RoundId,
    TensorSchema,
    UpdateCodecBinding,
)
from dromeus.training.base import CheckpointTrainer, WeightTrainer

_ARTIFACT_NAMES = ("outer_gradient", "slow_weights")
_STATE_PREFIX = "noloco.v1."
_PHASES = {
    "created": 0,
    "pre-local": 1,
    "post-local": 2,
    "bundled": 3,
    "post-outer": 4,
}


def _identity_codecs() -> dict[str, UpdateCodec]:
    return {name: IdentityCodec("identity-v1") for name in _ARTIFACT_NAMES}


@dataclass
class NoLoCoAlgorithm:
    """Own NoLoCo slow weights, outer momentum, and opaque bundle lifecycle.

    Paper ``phi`` is ``_slow_weights``, post-local ``theta`` is the trainer's fast
    weights, and Dromeus names the descent-oriented outer gradient ``g = phi -
    theta``. ``alpha``, ``beta``, and ``gamma`` map directly to ``config``. The
    identity path subtracts the averaged gradients and slow-weight correction. This
    is algebraically equivalent to the pinned executable upstream convention, which
    adds ``beta * mean(theta - phi)`` and effectively uses ``gamma = beta``. It does
    not follow the divergent sign printed in paper Equation 2.
    """

    trainer: WeightTrainer
    tensor_schema: TensorSchema
    config: NoLoCoConfig
    bundle_codec: NamedUpdateBundleCodec | None = None
    artifact_codecs: Mapping[str, UpdateCodec] = field(default_factory=_identity_codecs)
    manifest_codec_ids: Mapping[str, str] | None = None
    _round_id: RoundId = 0
    _phase: str = "created"
    _slow_weights: dict[str, np.ndarray] = field(init=False, repr=False)
    _outer_momentum: dict[str, np.ndarray] = field(init=False, repr=False)
    _error_feedback_residual: dict[str, np.ndarray] = field(init=False, repr=False)
    _completed_outer_steps: int = field(default=0, init=False, repr=False)
    _local_artifacts: dict[str, dict[str, np.ndarray]] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if set(self.artifact_codecs) != set(_ARTIFACT_NAMES):
            raise ValueError("NoLoCo codecs must cover both named artifacts")
        if self.manifest_codec_ids is not None:
            actual = {
                name: codec.codec_id for name, codec in self.artifact_codecs.items()
            }
            if dict(self.manifest_codec_ids) != actual:
                raise ValueError("NoLoCo codecs do not match manifest")
        weights = self.trainer.weights()
        self._validate_tensors(weights)
        self._slow_weights = self._copy_tensors(weights)
        self._outer_momentum = {
            name: np.zeros_like(value, dtype=np.float32)
            for name, value in weights.items()
        }
        self._error_feedback_residual = {
            name: np.zeros_like(value, dtype=np.float32)
            for name, value in weights.items()
        }

    def pre_local(self, round_id: RoundId) -> None:
        if self._phase == "created":
            weights = self.trainer.weights()
            self._validate_tensors(weights)
            self._slow_weights = self._copy_tensors(weights)
            self._outer_momentum = {
                name: np.zeros_like(value, dtype=np.float32)
                for name, value in weights.items()
            }
            self._error_feedback_residual = {
                name: np.zeros_like(value, dtype=np.float32)
                for name, value in weights.items()
            }
        self._round_id = round_id
        self.trainer.load_weights(self._copy_tensors(self._slow_weights))
        self._local_artifacts = None
        self._phase = "pre-local"

    def local_training(self) -> None:
        self.trainer.train_local_steps(self.config.inner_steps)
        self._phase = "post-local"

    def post_local_bundle(self) -> UpdateBundle:
        if self.bundle_codec is None:
            raise RuntimeError("update bundle codec is not configured")
        fast_weights = self.trainer.weights()
        self._validate_tensors(fast_weights)
        logical = {
            "outer_gradient": {
                name: (
                    self._slow_weights[name].astype(np.float32)
                    - fast_weights[name].astype(np.float32)
                    + (
                        self._error_feedback_residual[name]
                        if self._codec_is_lossy("outer_gradient")
                        else np.float32(0.0)
                    )
                ).astype(np.float32)
                for name in self._slow_weights
            },
            "slow_weights": self._copy_tensors(self._slow_weights),
        }
        encoded = {
            name: self.artifact_codecs[name].encode(logical[name])
            for name in _ARTIFACT_NAMES
        }
        bundle = self.bundle_codec.encode(
            round_id=self._round_id,
            artifacts=encoded,
            codec_bindings=self._codec_bindings(),
        )
        try:
            local = {
                name: self.artifact_codecs[name].decode(encoded[name])
                for name in _ARTIFACT_NAMES
            }
            self._validate_artifacts(local)
        except BaseException:
            self.bundle_codec.release(bundle)
            raise
        self._local_artifacts = {
            name: self._copy_tensors(tensors) for name, tensors in local.items()
        }
        self._error_feedback_residual = {
            name: (
                logical["outer_gradient"][name]
                - local["outer_gradient"][name]
                if self._codec_is_lossy("outer_gradient")
                else np.zeros_like(value, dtype=np.float32)
            ).astype(np.float32)
            for name, value in self._slow_weights.items()
        }
        self._phase = "bundled"
        return bundle

    def validate_peer(self, peer_bundle: UpdateBundle) -> NamedValidatedUpdate:
        if self.bundle_codec is None:
            raise RuntimeError("update bundle codec is not configured")
        if peer_bundle.metadata.round_id != self._round_id:
            raise ValueError("peer bundle round does not match current round")
        encoded = self.bundle_codec.decode(
            peer_bundle, codec_bindings=self._codec_bindings()
        )
        artifacts = {
            name: self.artifact_codecs[name].decode(encoded[name])
            for name in _ARTIFACT_NAMES
        }
        self._validate_artifacts(artifacts)
        return NamedValidatedUpdate(
            round_id=self._round_id,
            artifacts=artifacts,
            checksum=checksum_artifacts(artifacts),
        )

    def peer_apply(self, peer_update: AlgorithmUpdate) -> AlgorithmSnapshot:
        if not isinstance(peer_update, NamedValidatedUpdate):
            raise TypeError("NoLoCo requires named validated artifacts")
        if peer_update.round_id != self._round_id:
            raise ValueError("peer update round does not match current round")
        self._validate_artifacts(peer_update.artifacts)
        if checksum_artifacts(peer_update.artifacts) != peer_update.checksum:
            raise ValueError("peer update checksum mismatch")
        if self._local_artifacts is None:
            raise RuntimeError("local update bundle has not been built")
        local_gradient = self._local_artifacts["outer_gradient"]
        local_slow = self._local_artifacts["slow_weights"]
        peer_gradient = peer_update.artifacts["outer_gradient"]
        peer_slow = peer_update.artifacts["slow_weights"]
        next_momentum: dict[str, np.ndarray] = {}
        next_slow: dict[str, np.ndarray] = {}
        for name in self._slow_weights:
            momentum = (
                np.float32(self.config.alpha) * self._outer_momentum[name]
                - np.float32(self.config.beta / 2.0)
                * (local_gradient[name] + peer_gradient[name])
                - np.float32(self.config.gamma / 2.0)
                * (local_slow[name] - peer_slow[name])
            ).astype(np.float32)
            next_momentum[name] = momentum
            next_slow[name] = (
                self._slow_weights[name].astype(np.float32) + momentum
            ).astype(np.float32)
        self._outer_momentum = next_momentum
        self._slow_weights = next_slow
        self._completed_outer_steps += 1
        self.trainer.load_weights(self._copy_tensors(next_slow))
        self._phase = "post-outer"
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
                artifact_schemas={
                    name: (
                        self._codec_encoded_schema(name)
                        if self._codec_is_lossy(name)
                        else self.tensor_schema
                    )
                    for name in _ARTIFACT_NAMES
                },
            )

    def snapshot(self) -> AlgorithmSnapshot:
        return AlgorithmSnapshot(
            round_id=self._round_id,
            phase=self._phase,
            weights=self._slow_weights,
        )

    def evaluate(self) -> AlgorithmEvaluation | None:
        result = self.trainer.evaluate()
        if result is None:
            return None
        loss, accuracy = result
        return AlgorithmEvaluation(loss=float(loss), accuracy=float(accuracy))

    def observations(self) -> AlgorithmObservations:
        observations = AlgorithmObservations(local_loss=self.trainer.local_loss)
        if self._local_artifacts is None:
            return observations
        residual_norm = _tensor_l2_norm(self._error_feedback_residual)
        signal = {
            name: (
                self._local_artifacts["outer_gradient"][name]
                + self._error_feedback_residual[name]
            ).astype(np.float32)
            for name in self._error_feedback_residual
        }
        signal_norm = _tensor_l2_norm(signal)
        ratio = residual_norm / signal_norm if signal_norm > 0.0 else 0.0
        return AlgorithmObservations(
            local_loss=observations.local_loss,
            error_feedback_residual_l2_norm=residual_norm,
            error_feedback_signal_l2_norm=signal_norm,
            error_feedback_residual_to_signal_ratio=ratio,
        )

    def checkpoint_tensors(self) -> dict[str, np.ndarray]:
        """Return complete durable state in the versioned flat namespace."""
        if not isinstance(self.trainer, CheckpointTrainer):
            raise TypeError("NoLoCo trainer must expose checkpoint state")
        state: dict[str, np.ndarray] = {
            f"{_STATE_PREFIX}schema_version": np.array([1], dtype=np.int64),
            f"{_STATE_PREFIX}round_id": np.array([self._round_id], dtype=np.int64),
            f"{_STATE_PREFIX}completed_outer_steps": np.array(
                [self._completed_outer_steps], dtype=np.int64
            ),
            f"{_STATE_PREFIX}phase": np.array([_PHASES[self._phase]], dtype=np.int64),
        }
        for name, value in self._slow_weights.items():
            state[f"{_STATE_PREFIX}slow_weights.{name}"] = value.copy()
        for name, value in self._outer_momentum.items():
            state[f"{_STATE_PREFIX}outer_momentum.{name}"] = value.copy()
        for name, value in self._error_feedback_residual.items():
            state[f"{_STATE_PREFIX}error_feedback_residual.{name}"] = value.copy()
        for name, value in self.trainer.checkpoint_tensors().items():
            state[f"{_STATE_PREFIX}trainer.{name}"] = np.ascontiguousarray(value).copy()
        for artifact_name, codec in self.artifact_codecs.items():
            for name, value in codec.state_dict().items():
                if not isinstance(value, np.ndarray):
                    raise TypeError("NoLoCo codec checkpoint state must be tensors")
                state[f"{_STATE_PREFIX}codec.{artifact_name}.{name}"] = (
                    np.ascontiguousarray(cast(Any, value)).copy()
                )
        return state

    def load_checkpoint_tensors(self, state: Mapping[str, np.ndarray]) -> None:
        """Restore a validated checkpoint produced by `checkpoint_tensors`."""
        if not isinstance(self.trainer, CheckpointTrainer):
            raise TypeError("NoLoCo trainer must expose checkpoint state")
        version = self._counter(state, "schema_version")
        if version != 1:
            raise ValueError("unsupported NoLoCo checkpoint version")
        round_id = self._counter(state, "round_id")
        completed_outer_steps = self._counter(state, "completed_outer_steps")
        phase_value = self._counter(state, "phase")
        phase_by_value = {value: name for name, value in _PHASES.items()}
        if phase_value not in phase_by_value:
            raise ValueError("NoLoCo checkpoint phase is invalid")
        names = {tensor.name for tensor in self.tensor_schema.tensors}
        slow_weights = self._checkpoint_group(state, "slow_weights", names)
        outer_momentum = self._checkpoint_group(state, "outer_momentum", names)
        residual = self._checkpoint_group(state, "error_feedback_residual", names)
        self._validate_tensors(slow_weights)
        self._validate_tensors(outer_momentum)
        self._validate_tensors(residual)
        trainer_prefix = f"{_STATE_PREFIX}trainer."
        trainer_state = {
            name.removeprefix(trainer_prefix): np.ascontiguousarray(value).copy()
            for name, value in state.items()
            if name.startswith(trainer_prefix)
        }
        if not trainer_state:
            raise ValueError("NoLoCo trainer checkpoint state is missing")
        self.trainer.load_checkpoint_tensors(trainer_state)
        restored_weights = self.trainer.weights()
        self._validate_tensors(restored_weights)
        if any(
            not np.array_equal(restored_weights[name], slow_weights[name])
            for name in names
        ):
            raise ValueError("NoLoCo trainer and slow weights do not match")
        for artifact_name, codec in self.artifact_codecs.items():
            codec_prefix = f"{_STATE_PREFIX}codec.{artifact_name}."
            codec.load_state_dict(
                {
                    name.removeprefix(codec_prefix): value.copy()
                    for name, value in state.items()
                    if name.startswith(codec_prefix)
                }
            )
        self._round_id = round_id
        self._completed_outer_steps = completed_outer_steps
        self._phase = phase_by_value[phase_value]
        self._slow_weights = self._copy_tensors(slow_weights)
        self._outer_momentum = self._copy_tensors(outer_momentum)
        self._error_feedback_residual = self._copy_tensors(residual)
        self._local_artifacts = None

    def state_dict(self) -> dict[str, object]:
        return dict(self.checkpoint_tensors())

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if not all(isinstance(value, np.ndarray) for value in state.values()):
            raise ValueError("NoLoCo algorithm state must contain only tensors")
        tensor_state: dict[str, np.ndarray] = {
            name: cast(np.ndarray, value) for name, value in state.items()
        }
        self.load_checkpoint_tensors(tensor_state)

    def _validate_artifacts(
        self, artifacts: Mapping[str, Mapping[str, np.ndarray]]
    ) -> None:
        if set(artifacts) != set(_ARTIFACT_NAMES):
            raise ValueError("NoLoCo update requires both named artifacts")
        for tensors in artifacts.values():
            self._validate_tensors(tensors)

    def _validate_tensors(self, tensors: Mapping[str, np.ndarray]) -> None:
        validate_tensor_map(tensors, self.tensor_schema)
        for value in tensors.values():
            if value.dtype != np.float32:
                raise ValueError("NoLoCo requires FP32 logical tensors")

    def _codec_bindings(self) -> dict[str, UpdateCodecBinding]:
        return {
            name: UpdateCodecBinding(
                codec_id=self.artifact_codecs[name].codec_id,
                codec_version=1,
                logical_schema=self.tensor_schema,
            )
            for name in _ARTIFACT_NAMES
        }

    def _codec_is_lossy(self, name: str) -> bool:
        value = getattr(self.artifact_codecs[name], "lossy", False)
        if not isinstance(value, bool):
            raise TypeError("codec lossy marker must be boolean")
        return value

    def _codec_encoded_schema(self, name: str) -> TensorSchema:
        value = getattr(self.artifact_codecs[name], "encoded_schema", None)
        if not isinstance(value, TensorSchema):
            raise TypeError("lossy codec must declare an encoded schema")
        return value

    @staticmethod
    def _counter(state: Mapping[str, np.ndarray], name: str) -> int:
        value = np.asarray(state.get(f"{_STATE_PREFIX}{name}"))
        if value.dtype != np.int64 or value.shape != (1,) or int(value[0]) < 0:
            raise ValueError(f"NoLoCo checkpoint {name} is invalid")
        return int(value[0])

    @staticmethod
    def _checkpoint_group(
        state: Mapping[str, np.ndarray],
        group: str,
        names: set[str],
    ) -> dict[str, np.ndarray]:
        prefix = f"{_STATE_PREFIX}{group}."
        values = {
            name.removeprefix(prefix): np.ascontiguousarray(value).copy()
            for name, value in state.items()
            if name.startswith(prefix)
        }
        if set(values) != names:
            raise ValueError(f"NoLoCo checkpoint {group} is incomplete")
        return values

    @staticmethod
    def _copy_tensors(
        tensors: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        return {
            name: np.ascontiguousarray(value).copy() for name, value in tensors.items()
        }


def _tensor_l2_norm(tensors: Mapping[str, np.ndarray]) -> float:
    total = 0.0
    for value in tensors.values():
        flattened = np.asarray(value, dtype=np.float32).reshape(-1)
        for start in range(0, flattened.size, 1_048_576):
            chunk = flattened[start : start + 1_048_576].astype(
                np.float64, copy=False
            )
            total += float(np.sum(np.square(chunk), dtype=np.float64))
    return math.sqrt(total)


__all__ = ["NoLoCoAlgorithm"]
