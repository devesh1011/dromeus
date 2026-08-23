"""Training algorithm interface."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

import numpy as np

from dromeus.manifests.canonical import file_sha256, update_bundle_digest
from dromeus.manifests.models import (
    Identifier,
    OpaqueUpdateBundleMetadata,
    RoundId,
    TensorSchema,
)


class SerializableState(Protocol):
    """State seam shared by M1 and future algorithm/codec implementations."""

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: Mapping[str, object]) -> None: ...


@dataclass(frozen=True, slots=True)
class AlgorithmObservations:
    """Immutable algorithm observations captured for one round."""

    local_loss: float | None = None
    error_feedback_residual_l2_norm: float | None = None
    error_feedback_signal_l2_norm: float | None = None
    error_feedback_residual_to_signal_ratio: float | None = None

    def __post_init__(self) -> None:
        values = (
            ("local loss", self.local_loss),
            ("error-feedback residual norm", self.error_feedback_residual_l2_norm),
            ("error-feedback signal norm", self.error_feedback_signal_l2_norm),
            (
                "error-feedback residual-to-signal ratio",
                self.error_feedback_residual_to_signal_ratio,
            ),
        )
        for label, value in values:
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{label} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class AlgorithmEvaluation:
    """Immutable algorithm evaluation result."""

    loss: float
    accuracy: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.loss) or self.loss < 0:
            raise ValueError("evaluation loss must be finite and non-negative")
        if not math.isfinite(self.accuracy) or not 0 <= self.accuracy <= 1:
            raise ValueError("evaluation accuracy must be finite in [0, 1]")


def checksum_tensors(tensors: Mapping[str, np.ndarray]) -> str:
    """Hash named tensor values independently of the algorithm implementation."""
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = np.ascontiguousarray(tensors[name])
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tensor.shape).encode())
        digest.update(tensor.tobytes())
    return digest.hexdigest()


def _immutable_tensors(
    tensors: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    copied: dict[str, np.ndarray] = {}
    for name, value in tensors.items():
        tensor = np.ascontiguousarray(value).copy()
        tensor.flags.writeable = False
        copied[name] = tensor
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class AlgorithmSnapshot:
    round_id: RoundId
    phase: str
    weights: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", _immutable_tensors(self.weights))


@dataclass(frozen=True, slots=True)
class ValidatedUpdate:
    """Algorithm-owned peer update ready to apply."""

    round_id: RoundId
    tensors: Mapping[str, np.ndarray]
    checksum: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensors", _immutable_tensors(self.tensors))


@dataclass(frozen=True, slots=True)
class NamedValidatedUpdate:
    """Algorithm-owned named peer artifacts ready to apply."""

    round_id: RoundId
    artifacts: Mapping[str, Mapping[str, np.ndarray]]
    checksum: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifacts",
            MappingProxyType(
                {
                    name: _immutable_tensors(tensors)
                    for name, tensors in self.artifacts.items()
                }
            ),
        )


AlgorithmUpdate = ValidatedUpdate | NamedValidatedUpdate


def checksum_artifacts(
    artifacts: Mapping[str, Mapping[str, np.ndarray]],
) -> str:
    """Hash named tensor artifacts without flattening their public shape."""
    return checksum_tensors(
        {
            f"{artifact_name}.{tensor_name}": tensor
            for artifact_name, tensors in artifacts.items()
            for tensor_name, tensor in tensors.items()
        }
    )


@dataclass(frozen=True, slots=True)
class MaterializedArtifact:
    """Algorithm-produced bytes plus unchanged v1 transfer metadata."""

    path: Path
    transfer_codec_id: Identifier
    transfer_schema: TensorSchema


@dataclass(frozen=True, slots=True)
class UpdateBundle:
    """Materialized algorithm update whose contents are opaque to gossip."""

    metadata: OpaqueUpdateBundleMetadata
    artifacts: tuple[MaterializedArtifact, ...]

    def __post_init__(self) -> None:
        if len(self.artifacts) != len(self.metadata.artifacts):
            raise ValueError("materialized artifacts must match bundle metadata")

    @property
    def digest(self) -> str:
        return update_bundle_digest(self.metadata)

    def validate_materialized(self, max_bytes: int) -> None:
        """Validate materialized hashes plus aggregate encoded size."""
        declared = sum(
            artifact.size_bytes for artifact in self.metadata.artifacts
        )
        actual = sum(artifact.path.stat().st_size for artifact in self.artifacts)
        if actual != declared:
            raise ValueError("materialized update bundle size mismatch")
        if actual > max_bytes:
            raise ValueError("materialized update bundle exceeds payload limit")
        for metadata, materialized in zip(
            self.metadata.artifacts, self.artifacts, strict=True
        ):
            if file_sha256(materialized.path) != metadata.sha256:
                raise ValueError("materialized update bundle checksum mismatch")


__all__ = [
    "AlgorithmSnapshot",
    "AlgorithmUpdate",
    "MaterializedArtifact",
    "NamedValidatedUpdate",
    "SerializableState",
    "UpdateBundle",
    "ValidatedUpdate",
    "checksum_artifacts",
    "checksum_tensors",
]
