"""Update codecs and their serializable local state."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, cast, runtime_checkable
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
from dromeus.manifests.models import Tensor as TensorSpec

_LoadSafetensors = Callable[[str], dict[str, np.ndarray]]
_SaveSafetensors = Callable[[dict[str, np.ndarray], str], None]
load_safetensors = cast(_LoadSafetensors, _load_file)
save_safetensors = cast(_SaveSafetensors, _save_file)

TensorMap = dict[str, np.ndarray]
NamedTensorMap = dict[str, TensorMap]
StateMap = Mapping[str, object]


@runtime_checkable
class UpdateCodec(Protocol):
    """Encode/decode an algorithm update without owning transport concerns."""

    @property
    def codec_id(self) -> str: ...

    @property
    def codec_version(self) -> int: ...

    @property
    def lossy(self) -> bool: ...

    def encoded_schema_for(self, logical_schema: TensorSchema) -> TensorSchema:
        """Resolve the wire schema, rejecting an incompatible logical schema."""
        ...

    def encode(self, tensors: Mapping[str, np.ndarray]) -> TensorMap: ...

    def decode(self, tensors: Mapping[str, np.ndarray]) -> TensorMap: ...

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: StateMap) -> None: ...


@dataclass(frozen=True, slots=True)
class CodecDescription:
    """Validated capabilities bound to an algorithm's logical schema."""

    codec_id: str
    codec_version: int
    lossy: bool
    encoded_schema: TensorSchema


def describe_update_codec(
    codec: object, logical_schema: TensorSchema
) -> CodecDescription:
    """Require complete capabilities before an algorithm can use a codec."""
    if not isinstance(codec, UpdateCodec):
        raise TypeError("codec must implement the complete UpdateCodec interface")
    codec_id = cast(object, codec.codec_id)
    codec_version = cast(object, codec.codec_version)
    lossy = cast(object, codec.lossy)
    encoded_schema = cast(object, codec.encoded_schema_for(logical_schema))
    if not isinstance(codec_id, str) or not codec_id:
        raise TypeError("codec ID must be a nonempty string")
    if (
        not isinstance(codec_version, int)
        or isinstance(codec_version, bool)
        or codec_version <= 0
    ):
        raise TypeError("codec version must be a positive integer")
    if not isinstance(lossy, bool):
        raise TypeError("codec lossy marker must be boolean")
    if not isinstance(encoded_schema, TensorSchema):
        raise TypeError("codec must resolve an encoded tensor schema")
    return CodecDescription(codec_id, codec_version, lossy, encoded_schema)


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

    @property
    def codec_version(self) -> int:
        return 1

    @property
    def lossy(self) -> bool:
        return False

    def encoded_schema_for(self, logical_schema: TensorSchema) -> TensorSchema:
        """Identity uses the algorithm's unchanged logical tensor schema."""
        return logical_schema

    def encode(self, tensors: Mapping[str, np.ndarray]) -> TensorMap:
        return _copy_tensors(tensors)

    def decode(self, tensors: Mapping[str, np.ndarray]) -> TensorMap:
        return _copy_tensors(tensors)

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: StateMap) -> None:
        if state:
            raise ValueError("identity codec has no state")


@dataclass(frozen=True, slots=True)
class DenseInt8Codec:
    """Per-tensor symmetric signed-int8 quantization with explicit metadata."""

    logical_schema: TensorSchema
    _encoded_schema: TensorSchema = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_lossy_logical_schema(self.logical_schema)
        object.__setattr__(
            self,
            "_encoded_schema",
            _dense_encoded_schema(self.logical_schema),
        )

    @property
    def codec_id(self) -> str:
        return "dense-int8-v1"

    @property
    def codec_version(self) -> int:
        return 1

    @property
    def encoded_schema(self) -> TensorSchema:
        return self._encoded_schema

    @property
    def lossy(self) -> bool:
        return True

    def encoded_schema_for(self, logical_schema: TensorSchema) -> TensorSchema:
        _validate_codec_logical_schema(logical_schema, self.logical_schema)
        return self.encoded_schema

    def encode(self, tensors: Mapping[str, np.ndarray]) -> TensorMap:
        validate_tensor_map(tensors, self.logical_schema)
        encoded: TensorMap = {}
        for spec in self.logical_schema.tensors:
            quantized, scale = _quantize_symmetric(tensors[spec.name])
            encoded[f"{spec.name}.__q"] = quantized
            encoded[f"{spec.name}.__scale"] = np.array(
                [scale], dtype="<f4"
            )
            encoded[f"{spec.name}.__zero_point"] = np.array(
                [0], dtype="<i4"
            )
        validate_tensor_map(encoded, self.encoded_schema)
        return encoded

    def decode(self, tensors: Mapping[str, np.ndarray]) -> TensorMap:
        validate_tensor_map(tensors, self.encoded_schema)
        decoded: TensorMap = {}
        for spec in self.logical_schema.tensors:
            scale = _validated_quantization_metadata(tensors, spec.name)
            quantized = np.asarray(tensors[f"{spec.name}.__q"], dtype=np.int8)
            decoded[spec.name] = (
                quantized.astype(np.float32) * np.float32(scale)
            ).reshape(spec.shape)
        validate_tensor_map(decoded, self.logical_schema)
        return decoded

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: StateMap) -> None:
        if state:
            raise ValueError("dense int8 codec has no state")


@dataclass(frozen=True, slots=True)
class TopKInt8Codec:
    """Deterministic per-tensor top-k sparsification plus signed int8 values."""

    logical_schema: TensorSchema
    top_k_fraction: float
    _encoded_schema: TensorSchema = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_lossy_logical_schema(self.logical_schema)
        if not 0.0 < self.top_k_fraction <= 1.0 or not math.isfinite(
            self.top_k_fraction
        ):
            raise ValueError("top-k fraction must be finite in (0, 1]")
        object.__setattr__(
            self,
            "_encoded_schema",
            _topk_encoded_schema(self.logical_schema, self.top_k_fraction),
        )

    @property
    def codec_id(self) -> str:
        return "topk-int8-v1"

    @property
    def codec_version(self) -> int:
        return 1

    @property
    def encoded_schema(self) -> TensorSchema:
        return self._encoded_schema

    @property
    def lossy(self) -> bool:
        return True

    def encoded_schema_for(self, logical_schema: TensorSchema) -> TensorSchema:
        _validate_codec_logical_schema(logical_schema, self.logical_schema)
        return self.encoded_schema

    def encode(self, tensors: Mapping[str, np.ndarray]) -> TensorMap:
        validate_tensor_map(tensors, self.logical_schema)
        encoded: TensorMap = {}
        encoded_names = {tensor.name for tensor in self.encoded_schema.tensors}
        for spec in self.logical_schema.tensors:
            if f"{spec.name}.__q" in encoded_names:
                quantized, scale = _quantize_symmetric(tensors[spec.name])
                encoded[f"{spec.name}.__q"] = quantized
            else:
                flattened = np.asarray(tensors[spec.name], dtype=np.float32).reshape(-1)
                indices = _deterministic_topk_indices(
                    flattened,
                    _top_k_count(flattened.size, self.top_k_fraction),
                )
                quantized, scale = _quantize_symmetric(flattened[indices])
                encoded[f"{spec.name}.__indices"] = indices.astype(
                    "<i4", copy=False
                )
                encoded[f"{spec.name}.__values"] = quantized
            encoded[f"{spec.name}.__scale"] = np.array(
                [scale], dtype="<f4"
            )
            encoded[f"{spec.name}.__zero_point"] = np.array(
                [0], dtype="<i4"
            )
        validate_tensor_map(encoded, self.encoded_schema)
        return encoded

    def decode(self, tensors: Mapping[str, np.ndarray]) -> TensorMap:
        validate_tensor_map(tensors, self.encoded_schema)
        decoded: TensorMap = {}
        encoded_names = set(tensors)
        for spec in self.logical_schema.tensors:
            scale = _validated_quantization_metadata(tensors, spec.name)
            if f"{spec.name}.__q" in encoded_names:
                quantized = np.asarray(tensors[f"{spec.name}.__q"], dtype=np.int8)
                value = quantized.astype(np.float32) * np.float32(scale)
            else:
                indices = np.asarray(
                    tensors[f"{spec.name}.__indices"], dtype=np.int32
                )
                if indices.size > 1 and np.any(indices[1:] <= indices[:-1]):
                    raise ValueError("top-k indices must be strictly increasing")
                element_count = math.prod(spec.shape)
                if np.any(indices < 0) or np.any(indices >= element_count):
                    raise ValueError("top-k index is outside logical tensor range")
                quantized = np.asarray(
                    tensors[f"{spec.name}.__values"], dtype=np.int8
                )
                value = np.zeros(element_count, dtype=np.float32)
                value[indices] = quantized.astype(np.float32) * np.float32(scale)
            decoded[spec.name] = value.reshape(spec.shape)
        validate_tensor_map(decoded, self.logical_schema)
        return decoded

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: StateMap) -> None:
        if state:
            raise ValueError("top-k int8 codec has no state")


@dataclass(frozen=True, slots=True)
class BitmapTopKInt8Codec:
    """Deterministic top-k int8 values with one-bit membership indices."""

    logical_schema: TensorSchema
    top_k_fraction: float
    _encoded_schema: TensorSchema = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_lossy_logical_schema(self.logical_schema)
        if not 0.0 < self.top_k_fraction <= 0.5 or not math.isfinite(
            self.top_k_fraction
        ):
            raise ValueError("bitmap top-k fraction must be finite in (0, 0.5]")
        object.__setattr__(
            self,
            "_encoded_schema",
            _bitmap_topk_encoded_schema(
                self.logical_schema,
                self.top_k_fraction,
            ),
        )

    @property
    def codec_id(self) -> str:
        return "topk-bitmap-int8-v2"

    @property
    def codec_version(self) -> int:
        return 2

    @property
    def encoded_schema(self) -> TensorSchema:
        return self._encoded_schema

    @property
    def lossy(self) -> bool:
        return True

    def encoded_schema_for(self, logical_schema: TensorSchema) -> TensorSchema:
        _validate_codec_logical_schema(logical_schema, self.logical_schema)
        return self.encoded_schema

    def encode(self, tensors: Mapping[str, np.ndarray]) -> TensorMap:
        validate_tensor_map(tensors, self.logical_schema)
        encoded: TensorMap = {}
        for spec in self.logical_schema.tensors:
            flattened = np.asarray(tensors[spec.name], dtype=np.float32).reshape(-1)
            indices = _deterministic_topk_indices(
                flattened,
                _top_k_count(flattened.size, self.top_k_fraction),
            )
            quantized, scale = _quantize_symmetric(flattened[indices])
            membership = np.zeros(flattened.size, dtype=np.uint8)
            membership[indices] = np.uint8(1)
            bitmap = np.packbits(membership, bitorder="little").view(np.int8)
            encoded[f"{spec.name}.__bitmap"] = np.ascontiguousarray(bitmap)
            encoded[f"{spec.name}.__values"] = quantized
            encoded[f"{spec.name}.__scale"] = np.array([scale], dtype="<f4")
            encoded[f"{spec.name}.__zero_point"] = np.array([0], dtype="<i4")
        validate_tensor_map(encoded, self.encoded_schema)
        return encoded

    def decode(self, tensors: Mapping[str, np.ndarray]) -> TensorMap:
        validate_tensor_map(tensors, self.encoded_schema)
        decoded: TensorMap = {}
        for spec in self.logical_schema.tensors:
            element_count = math.prod(spec.shape)
            bitmap = np.asarray(
                tensors[f"{spec.name}.__bitmap"],
                dtype=np.int8,
            ).view(np.uint8)
            bits = np.unpackbits(bitmap, bitorder="little")
            if np.any(bits[element_count:]):
                raise ValueError("bitmap top-k padding bits must be zero")
            indices = np.flatnonzero(bits[:element_count])
            expected_count = _top_k_count(element_count, self.top_k_fraction)
            if indices.size != expected_count:
                raise ValueError("bitmap top-k selected count does not match schema")
            scale = _validated_quantization_metadata(tensors, spec.name)
            quantized = np.asarray(
                tensors[f"{spec.name}.__values"],
                dtype=np.int8,
            )
            value = np.zeros(element_count, dtype=np.float32)
            value[indices] = quantized.astype(np.float32) * np.float32(scale)
            decoded[spec.name] = value.reshape(spec.shape)
        validate_tensor_map(decoded, self.logical_schema)
        return decoded

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: StateMap) -> None:
        if state:
            raise ValueError("bitmap top-k int8 codec has no state")


def _validate_codec_logical_schema(
    actual: TensorSchema, expected: TensorSchema
) -> None:
    if actual != expected:
        raise ValueError("codec logical schema does not match algorithm")


def _validate_lossy_logical_schema(schema: TensorSchema) -> None:
    if any(tensor.dtype != "float32" for tensor in schema.tensors):
        raise ValueError("lossy codecs require FP32 logical tensors")


def _dense_encoded_schema(schema: TensorSchema) -> TensorSchema:
    tensors: list[TensorSpec] = []
    for spec in schema.tensors:
        tensors.extend(
            (
                TensorSpec(name=f"{spec.name}.__q", dtype="int8", shape=spec.shape),
                TensorSpec(
                    name=f"{spec.name}.__scale", dtype="float32", shape=(1,)
                ),
                TensorSpec(
                    name=f"{spec.name}.__zero_point", dtype="int32", shape=(1,)
                ),
            )
        )
    return TensorSchema(tensors=tuple(sorted(tensors, key=lambda item: item.name)))


def _topk_encoded_schema(
    schema: TensorSchema, top_k_fraction: float
) -> TensorSchema:
    tensors: list[TensorSpec] = []
    for spec in schema.tensors:
        element_count = math.prod(spec.shape)
        k = _top_k_count(element_count, top_k_fraction)
        if 5 * k <= element_count:
            tensors.extend(
                (
                    TensorSpec(
                        name=f"{spec.name}.__indices", dtype="int32", shape=(k,)
                    ),
                    TensorSpec(
                        name=f"{spec.name}.__values", dtype="int8", shape=(k,)
                    ),
                )
            )
        else:
            tensors.append(
                TensorSpec(name=f"{spec.name}.__q", dtype="int8", shape=spec.shape)
            )
        tensors.extend(
            (
                TensorSpec(
                    name=f"{spec.name}.__scale", dtype="float32", shape=(1,)
                ),
                TensorSpec(
                    name=f"{spec.name}.__zero_point", dtype="int32", shape=(1,)
                ),
            )
        )
    return TensorSchema(tensors=tuple(sorted(tensors, key=lambda item: item.name)))


def _bitmap_topk_encoded_schema(
    schema: TensorSchema,
    top_k_fraction: float,
) -> TensorSchema:
    tensors: list[TensorSpec] = []
    for spec in schema.tensors:
        element_count = math.prod(spec.shape)
        tensors.extend(
            (
                TensorSpec(
                    name=f"{spec.name}.__bitmap",
                    dtype="int8",
                    shape=(math.ceil(element_count / 8),),
                ),
                TensorSpec(
                    name=f"{spec.name}.__values",
                    dtype="int8",
                    shape=(_top_k_count(element_count, top_k_fraction),),
                ),
                TensorSpec(
                    name=f"{spec.name}.__scale",
                    dtype="float32",
                    shape=(1,),
                ),
                TensorSpec(
                    name=f"{spec.name}.__zero_point",
                    dtype="int32",
                    shape=(1,),
                ),
            )
        )
    return TensorSchema(tensors=tuple(sorted(tensors, key=lambda item: item.name)))


def _top_k_count(element_count: int, fraction: float) -> int:
    return min(element_count, max(1, math.ceil(element_count * fraction)))


def _deterministic_topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    magnitudes = np.abs(values)
    if k == values.size:
        return np.arange(values.size, dtype=np.int64)
    threshold = np.partition(magnitudes, values.size - k)[values.size - k]
    greater = np.flatnonzero(magnitudes > threshold)
    equal = np.flatnonzero(magnitudes == threshold)
    needed = k - greater.size
    selected = np.concatenate((greater, equal[:needed]))
    return np.sort(selected).astype(np.int64, copy=False)


def _quantize_symmetric(value: np.ndarray) -> tuple[np.ndarray, float]:
    source = np.asarray(value, dtype=np.float32)
    maximum = float(np.max(np.abs(source))) if source.size else 0.0
    scale = maximum / 127.0 if maximum > 0.0 else 1.0
    if np.float32(scale) == np.float32(0.0):
        scale = float(np.nextafter(np.float32(0.0), np.float32(1.0)))
    quantized = np.clip(np.rint(source / np.float32(scale)), -127, 127).astype(
        np.int8
    )
    return np.ascontiguousarray(quantized), scale


def _validated_quantization_metadata(
    tensors: Mapping[str, np.ndarray], name: str
) -> float:
    scale = float(np.asarray(tensors[f"{name}.__scale"])[0])
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("quantization scale must be positive and finite")
    zero_point = int(np.asarray(tensors[f"{name}.__zero_point"])[0])
    if zero_point != 0:
        raise ValueError("symmetric quantization zero point must equal zero")
    return scale


def _copy_tensors(tensors: Mapping[str, np.ndarray]) -> TensorMap:
    return {
        name: np.ascontiguousarray(value).copy() for name, value in tensors.items()
    }


@dataclass(frozen=True, slots=True)
class NamedSafetensorsUpdateBundleCodec:
    """Materialize each named update artifact as one safetensors file."""

    artifact_root: Path
    run_id: RunId
    manifest_hash: Sha256
    sender_public_key: PublicKey
    algorithm_id: AlgorithmId
    artifact_schemas: Mapping[str, TensorSchema]

    def __post_init__(self) -> None:
        schemas = dict(self.artifact_schemas)
        if not 1 <= len(schemas) <= 16:
            raise ValueError("named bundle codec requires 1 through 16 artifacts")
        object.__setattr__(self, "artifact_schemas", MappingProxyType(schemas))

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
    "BitmapTopKInt8Codec",
    "CodecDescription",
    "DenseInt8Codec",
    "IdentityCodec",
    "NamedSafetensorsUpdateBundleCodec",
    "NamedTensorMap",
    "NamedUpdateBundleCodec",
    "StateMap",
    "TensorMap",
    "TopKInt8Codec",
    "UpdateCodecBinding",
    "UpdateCodec",
    "describe_update_codec",
    "validate_tensor_map",
]
