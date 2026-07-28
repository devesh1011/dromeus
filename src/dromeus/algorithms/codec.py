"""Update codecs and their serializable local state."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast
from uuid import uuid4

import numpy as np
from safetensors.numpy import (
    load_file as _load_file,  # pyright: ignore[reportUnknownVariableType]
)
from safetensors.numpy import (
    save_file as _save_file,  # pyright: ignore[reportUnknownVariableType]
)

from dromeus.algorithms.base import MaterializedArtifact, UpdateBundle
from dromeus.manifests.canonical import canonical_hash, file_sha256
from dromeus.manifests.models import (
    AlgorithmId,
    OpaqueArtifactMetadata,
    OpaqueUpdateBundleMetadata,
    PublicKey,
    RoundId,
    RunId,
    Sha256,
    TensorSchema,
)

_LoadSafetensors = Callable[[str], dict[str, np.ndarray]]
_SaveSafetensors = Callable[[dict[str, np.ndarray], str], None]
load_safetensors = cast(_LoadSafetensors, _load_file)
save_safetensors = cast(_SaveSafetensors, _save_file)

TensorMap = dict[str, np.ndarray]
StateMap = Mapping[str, object]


class UpdateCodec(Protocol):
    """Encode/decode an algorithm update without owning transport concerns."""

    @property
    def codec_id(self) -> str: ...

    def encode(self, tensors: Mapping[str, np.ndarray]) -> TensorMap: ...

    def decode(self, tensors: Mapping[str, np.ndarray]) -> TensorMap: ...

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: StateMap) -> None: ...


class UpdateBundleCodec(Protocol):
    def encode(
        self, *, round_id: RoundId, tensors: Mapping[str, np.ndarray]
    ) -> UpdateBundle: ...

    def decode(self, bundle: UpdateBundle) -> TensorMap: ...

    def release(self, bundle: UpdateBundle) -> None: ...

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: StateMap) -> None: ...


@dataclass(frozen=True, slots=True)
class IdentityCodec:
    """M1 codec: preserve named tensors and keep no codec state."""

    _codec_id: str = "safetensors-v1"

    @property
    def codec_id(self) -> str:
        return self._codec_id

    def encode(self, tensors: Mapping[str, np.ndarray]) -> TensorMap:
        return _copy_tensors(tensors)

    def decode(self, tensors: Mapping[str, np.ndarray]) -> TensorMap:
        return _copy_tensors(tensors)

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: StateMap) -> None:
        if state:
            raise ValueError("identity codec has no state")


def _copy_tensors(tensors: Mapping[str, np.ndarray]) -> TensorMap:
    return {
        name: np.ascontiguousarray(value).copy() for name, value in tensors.items()
    }


@dataclass(frozen=True, slots=True)
class SafetensorsUpdateBundleCodec:
    """Materialize and validate M1 identity updates as safetensors artifacts."""

    artifact_root: Path
    run_id: RunId
    manifest_hash: Sha256
    sender_public_key: PublicKey
    algorithm_id: AlgorithmId
    tensor_schema: TensorSchema

    def encode(
        self, *, round_id: RoundId, tensors: Mapping[str, np.ndarray]
    ) -> UpdateBundle:
        values = _copy_tensors(tensors)
        validate_tensor_map(values, self.tensor_schema)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        path = self.artifact_root / f"round-{round_id}-{uuid4().hex}.safetensors"
        try:
            save_safetensors(values, str(path))
            schema_hash = canonical_hash(self.tensor_schema)
            artifact = OpaqueArtifactMetadata(
                name="trained_weights",
                size_bytes=path.stat().st_size,
                sha256=file_sha256(path),
                codec_id="safetensors",
                codec_version=1,
                logical_schema_hash=schema_hash,
                encoded_schema_hash=schema_hash,
            )
            return UpdateBundle(
                metadata=OpaqueUpdateBundleMetadata(
                    run_id=self.run_id,
                    manifest_hash=self.manifest_hash,
                    sender_public_key=self.sender_public_key,
                    algorithm_id=self.algorithm_id,
                    round_id=round_id,
                    artifacts=(artifact,),
                ),
                artifacts=(
                    MaterializedArtifact(
                        path=path,
                        transfer_codec_id="safetensors-v1",
                        transfer_schema=self.tensor_schema,
                    ),
                ),
            )
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    def decode(self, bundle: UpdateBundle) -> TensorMap:
        self._validate_metadata(bundle)
        artifact = bundle.metadata.artifacts[0]
        path = bundle.artifacts[0].path
        if path.stat().st_size != artifact.size_bytes:
            raise ValueError("bundle artifact size mismatch")
        if file_sha256(path) != artifact.sha256:
            raise ValueError("bundle artifact checksum mismatch")
        values = {
            name: np.ascontiguousarray(value)
            for name, value in load_safetensors(str(path)).items()
        }
        validate_tensor_map(values, self.tensor_schema)
        return values

    def release(self, bundle: UpdateBundle) -> None:
        for artifact in bundle.artifacts:
            artifact.path.unlink(missing_ok=True)

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: StateMap) -> None:
        if state:
            raise ValueError("safetensors bundle codec has no state")

    def _validate_metadata(self, bundle: UpdateBundle) -> None:
        metadata = bundle.metadata
        if metadata.run_id != self.run_id:
            raise ValueError("bundle run does not match codec")
        if metadata.manifest_hash != self.manifest_hash:
            raise ValueError("bundle manifest does not match codec")
        if metadata.algorithm_id != self.algorithm_id:
            raise ValueError("bundle algorithm does not match codec")
        if len(metadata.artifacts) != 1:
            raise ValueError("M1 safetensors bundle requires one artifact")
        artifact = metadata.artifacts[0]
        schema_hash = canonical_hash(self.tensor_schema)
        if artifact.codec_id != "safetensors" or artifact.codec_version != 1:
            raise ValueError("unsupported bundle codec")
        if (
            artifact.logical_schema_hash != schema_hash
            or artifact.encoded_schema_hash != schema_hash
        ):
            raise ValueError("bundle schema does not match codec")


def validate_tensor_map(
    tensors: Mapping[str, np.ndarray], schema: TensorSchema
) -> None:
    expected = {tensor.name: tensor for tensor in schema.tensors}
    if set(tensors) != set(expected):
        raise ValueError("tensor names do not match schema")
    for name, value in tensors.items():
        tensor = np.ascontiguousarray(value)
        spec = expected[name]
        if str(tensor.dtype) != spec.dtype:
            raise ValueError(f"tensor {name} dtype does not match schema")
        if tensor.shape != spec.shape:
            raise ValueError(f"tensor {name} shape does not match schema")
        if not np.isfinite(tensor).all():
            raise ValueError(f"tensor {name} contains non-finite values")

__all__ = [
    "IdentityCodec",
    "SafetensorsUpdateBundleCodec",
    "StateMap",
    "TensorMap",
    "UpdateBundleCodec",
    "UpdateCodec",
    "validate_tensor_map",
]
