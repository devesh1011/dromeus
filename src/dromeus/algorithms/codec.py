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
    UpdateCodecBinding,
)

_LoadSafetensors = Callable[[str], dict[str, np.ndarray]]
_SaveSafetensors = Callable[[dict[str, np.ndarray], str], None]
load_safetensors = cast(_LoadSafetensors, _load_file)
save_safetensors = cast(_SaveSafetensors, _save_file)

TensorMap = dict[str, np.ndarray]
NamedTensorMap = dict[str, TensorMap]
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
    """Materialize codec-encoded tensors into an opaque update bundle."""

    def encode(
        self,
        *,
        round_id: RoundId,
        tensors: Mapping[str, np.ndarray],
        codec_binding: UpdateCodecBinding | None = None,
    ) -> UpdateBundle: ...

    def decode(
        self,
        bundle: UpdateBundle,
        codec_binding: UpdateCodecBinding | None = None,
    ) -> TensorMap: ...

    def release(self, bundle: UpdateBundle) -> None: ...

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: StateMap) -> None: ...


class NamedUpdateBundleCodec(Protocol):
    """Materialize independently encoded named tensor artifacts."""

    def encode(
        self,
        *,
        round_id: RoundId,
        artifacts: Mapping[str, Mapping[str, np.ndarray]],
        codec_bindings: Mapping[str, UpdateCodecBinding] | None = None,
    ) -> UpdateBundle: ...

    def decode(
        self,
        bundle: UpdateBundle,
        codec_bindings: Mapping[str, UpdateCodecBinding] | None = None,
    ) -> NamedTensorMap: ...

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
    """Materialize encoded tensors and bind their logical codec metadata."""

    artifact_root: Path
    run_id: RunId
    manifest_hash: Sha256
    sender_public_key: PublicKey
    algorithm_id: AlgorithmId
    tensor_schema: TensorSchema

    def encode(
        self,
        *,
        round_id: RoundId,
        tensors: Mapping[str, np.ndarray],
        codec_binding: UpdateCodecBinding | None = None,
    ) -> UpdateBundle:
        values = _copy_tensors(tensors)
        validate_tensor_map(values, self.tensor_schema)
        binding = _resolve_codec_binding(codec_binding, self.tensor_schema)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        path = self.artifact_root / f"round-{round_id}-{uuid4().hex}.safetensors"
        try:
            save_safetensors(values, str(path))
            artifact = OpaqueArtifactMetadata(
                name="trained_weights",
                size_bytes=path.stat().st_size,
                sha256=file_sha256(path),
                codec_id=binding.codec_id,
                codec_version=binding.codec_version,
                logical_schema_hash=canonical_hash(binding.logical_schema),
                encoded_schema_hash=canonical_hash(self.tensor_schema),
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

    def decode(
        self,
        bundle: UpdateBundle,
        codec_binding: UpdateCodecBinding | None = None,
    ) -> TensorMap:
        self._validate_metadata(bundle, codec_binding)
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

    def _validate_metadata(
        self,
        bundle: UpdateBundle,
        codec_binding: UpdateCodecBinding | None,
    ) -> None:
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
        if artifact.name != "trained_weights":
            raise ValueError("unsupported M1 bundle artifact")
        binding = _resolve_codec_binding(codec_binding, self.tensor_schema)
        if (
            artifact.codec_id != binding.codec_id
            or artifact.codec_version != binding.codec_version
        ):
            raise ValueError("unsupported bundle codec")
        if (
            artifact.logical_schema_hash != canonical_hash(binding.logical_schema)
            or artifact.encoded_schema_hash != canonical_hash(self.tensor_schema)
        ):
            raise ValueError("bundle schema does not match codec")


@dataclass(frozen=True, slots=True)
class NamedSafetensorsUpdateBundleCodec:
    """Materialize each named update artifact as one safetensors file."""

    artifact_root: Path
    run_id: RunId
    manifest_hash: Sha256
    sender_public_key: PublicKey
    algorithm_id: AlgorithmId
    artifact_schemas: Mapping[str, TensorSchema]

    def encode(
        self,
        *,
        round_id: RoundId,
        artifacts: Mapping[str, Mapping[str, np.ndarray]],
        codec_bindings: Mapping[str, UpdateCodecBinding] | None = None,
    ) -> UpdateBundle:
        names = tuple(sorted(self.artifact_schemas))
        if set(artifacts) != set(names):
            raise ValueError("named update artifacts do not match codec")
        bindings = self._bindings(codec_bindings)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        metadata: list[OpaqueArtifactMetadata] = []
        materialized: list[MaterializedArtifact] = []
        try:
            for name in names:
                values = _copy_tensors(artifacts[name])
                schema = self.artifact_schemas[name]
                validate_tensor_map(values, schema)
                path = (
                    self.artifact_root
                    / f"round-{round_id}-{name}-{uuid4().hex}.safetensors"
                )
                paths.append(path)
                save_safetensors(values, str(path))
                binding = bindings[name]
                metadata.append(
                    OpaqueArtifactMetadata(
                        name=name,
                        size_bytes=path.stat().st_size,
                        sha256=file_sha256(path),
                        codec_id=binding.codec_id,
                        codec_version=binding.codec_version,
                        logical_schema_hash=canonical_hash(
                            binding.logical_schema
                        ),
                        encoded_schema_hash=canonical_hash(schema),
                    )
                )
                materialized.append(
                    MaterializedArtifact(
                        path=path,
                        transfer_codec_id="safetensors-v1",
                        transfer_schema=schema,
                    )
                )
            return UpdateBundle(
                metadata=OpaqueUpdateBundleMetadata(
                    run_id=self.run_id,
                    manifest_hash=self.manifest_hash,
                    sender_public_key=self.sender_public_key,
                    algorithm_id=self.algorithm_id,
                    round_id=round_id,
                    artifacts=tuple(metadata),
                ),
                artifacts=tuple(materialized),
            )
        except BaseException:
            for path in paths:
                path.unlink(missing_ok=True)
            raise

    def decode(
        self,
        bundle: UpdateBundle,
        codec_bindings: Mapping[str, UpdateCodecBinding] | None = None,
    ) -> NamedTensorMap:
        metadata = bundle.metadata
        if metadata.run_id != self.run_id:
            raise ValueError("bundle run does not match codec")
        if metadata.manifest_hash != self.manifest_hash:
            raise ValueError("bundle manifest does not match codec")
        if metadata.algorithm_id != self.algorithm_id:
            raise ValueError("bundle algorithm does not match codec")
        names = set(self.artifact_schemas)
        metadata_by_name = {
            artifact.name: artifact for artifact in metadata.artifacts
        }
        materialized_by_name = {
            item.name: artifact
            for item, artifact in zip(
                metadata.artifacts, bundle.artifacts, strict=True
            )
        }
        if (
            set(metadata_by_name) != names
            or len(metadata_by_name) != len(metadata.artifacts)
            or set(materialized_by_name) != names
        ):
            raise ValueError("named update artifacts do not match codec")
        bindings = self._bindings(codec_bindings)
        decoded: NamedTensorMap = {}
        for name in sorted(names):
            item = metadata_by_name[name]
            artifact = materialized_by_name[name]
            schema = self.artifact_schemas[name]
            binding = bindings[name]
            if (
                item.codec_id != binding.codec_id
                or item.codec_version != binding.codec_version
                or item.logical_schema_hash
                != canonical_hash(binding.logical_schema)
                or item.encoded_schema_hash != canonical_hash(schema)
            ):
                raise ValueError("named update artifact binding does not match codec")
            if artifact.path.stat().st_size != item.size_bytes:
                raise ValueError("bundle artifact size mismatch")
            if file_sha256(artifact.path) != item.sha256:
                raise ValueError("bundle artifact checksum mismatch")
            values = {
                tensor_name: np.ascontiguousarray(value)
                for tensor_name, value in load_safetensors(
                    str(artifact.path)
                ).items()
            }
            validate_tensor_map(values, schema)
            decoded[name] = values
        return decoded

    def release(self, bundle: UpdateBundle) -> None:
        for artifact in bundle.artifacts:
            artifact.path.unlink(missing_ok=True)

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: StateMap) -> None:
        if state:
            raise ValueError("named safetensors bundle codec has no state")

    def _bindings(
        self,
        bindings: Mapping[str, UpdateCodecBinding] | None,
    ) -> dict[str, UpdateCodecBinding]:
        if bindings is None:
            return {
                name: _resolve_codec_binding(None, schema)
                for name, schema in self.artifact_schemas.items()
            }
        if set(bindings) != set(self.artifact_schemas):
            raise ValueError("named update codec bindings do not match artifacts")
        return dict(bindings)


def _resolve_codec_binding(
    binding: UpdateCodecBinding | None,
    encoded_schema: TensorSchema,
) -> UpdateCodecBinding:
    if binding is not None:
        return binding
    return UpdateCodecBinding(
        codec_id="safetensors-v1",
        codec_version=1,
        logical_schema=encoded_schema,
    )


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
    "NamedSafetensorsUpdateBundleCodec",
    "NamedTensorMap",
    "NamedUpdateBundleCodec",
    "SafetensorsUpdateBundleCodec",
    "StateMap",
    "TensorMap",
    "UpdateCodecBinding",
    "UpdateBundleCodec",
    "UpdateCodec",
    "validate_tensor_map",
]
