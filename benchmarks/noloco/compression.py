"""Measure complete transfer-wire size for one materialized update bundle."""

from __future__ import annotations

import hashlib
import math
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dromeus.algorithms.base import UpdateBundle
from dromeus.manifests.canonical import (
    file_sha256,
    materialize_bundle_metadata,
)
from dromeus.manifests.models import TensorSchema, TransportLimits
from dromeus.protocol.codec import encode_envelope, encode_message
from dromeus.protocol.models import (
    Chunk,
    ChunkAck,
    MessageType,
    TransferBegin,
    TransferComplete,
    create_envelope,
)


@dataclass(frozen=True, slots=True)
class BundleWireMeasurement:
    raw_artifact_bytes: int
    encoded_artifact_bytes: int
    metadata_carrier_bytes: int
    wire_bytes: int
    chunk_count: int

    @property
    def protocol_overhead_bytes(self) -> int:
        """Return envelope, ACK, and chunking bytes beyond transferred files."""
        return (
            self.wire_bytes
            - self.metadata_carrier_bytes
            - self.encoded_artifact_bytes
        )


def measure_bundle_wire_bytes(
    bundle: UpdateBundle,
    *,
    transport: TransportLimits,
    logical_artifact_schemas: Mapping[str, TensorSchema],
) -> BundleWireMeasurement:
    """Encode transfer messages and account for raw and complete wire bytes."""
    bundle.validate_materialized(transport.max_update_bundle_bytes)
    expected_names = {metadata.name for metadata in bundle.metadata.artifacts}
    if set(logical_artifact_schemas) != expected_names:
        raise ValueError("logical artifact schemas do not match bundle artifacts")
    with tempfile.TemporaryDirectory(prefix="dromeus-wire-measure-") as temporary:
        carrier = materialize_bundle_metadata(bundle.metadata, Path(temporary))
        try:
            transfers = [
                (
                    "update-bundle-metadata",
                    carrier.path,
                    "safetensors-v1",
                    carrier.tensor_schema,
                ),
                *(
                    (
                        metadata.name,
                        artifact.path,
                        artifact.transfer_codec_id,
                        artifact.transfer_schema,
                    )
                    for metadata, artifact in zip(
                        bundle.metadata.artifacts,
                        bundle.artifacts,
                        strict=True,
                    )
                ),
            ]
            wire_bytes = 0
            chunk_count = 0
            raw_artifact_bytes = 0
            for metadata, artifact in zip(
                bundle.metadata.artifacts,
                bundle.artifacts,
                strict=True,
            ):
                del artifact
                raw_artifact_bytes += _schema_payload_bytes(
                    logical_artifact_schemas[metadata.name]
                )
            for transfer_index, (
                artifact_name,
                path,
                codec_id,
                tensor_schema,
            ) in enumerate(transfers):
                measured, chunks = _measure_transfer(
                    bundle=bundle,
                    transport=transport,
                    transfer_index=transfer_index,
                    artifact_name=artifact_name,
                    path=path,
                    codec_id=codec_id,
                    tensor_schema=tensor_schema,
                )
                wire_bytes += measured
                chunk_count += chunks
            return BundleWireMeasurement(
                raw_artifact_bytes=raw_artifact_bytes,
                encoded_artifact_bytes=sum(
                    artifact.path.stat().st_size for artifact in bundle.artifacts
                ),
                metadata_carrier_bytes=carrier.path.stat().st_size,
                wire_bytes=wire_bytes,
                chunk_count=chunk_count,
            )
        finally:
            carrier.path.unlink(missing_ok=True)


def _measure_transfer(
    *,
    bundle: UpdateBundle,
    transport: TransportLimits,
    transfer_index: int,
    artifact_name: str,
    path: Path,
    codec_id: str,
    tensor_schema: object,
) -> tuple[int, int]:
    from dromeus.manifests.models import TensorSchema

    if not isinstance(tensor_schema, TensorSchema):
        raise TypeError("wire measurement requires a tensor schema")
    size_bytes = path.stat().st_size
    transfer_id = f"measure-{transfer_index}"
    chunk_size = transport.effective_chunk_size
    chunk_count = math.ceil(size_bytes / chunk_size)
    total_sha256 = file_sha256(path)
    total = _envelope_size(
        bundle,
        message_type=MessageType.TRANSFER_BEGIN,
        message_id=f"{transfer_id}-begin",
        correlation_id=transfer_id,
        payload=encode_message(
            TransferBegin(
                transfer_id=transfer_id,
                artifact_name=artifact_name,
                total_size_bytes=size_bytes,
                total_sha256=total_sha256,
                chunk_count=chunk_count,
                codec_id=codec_id,
                tensor_schema=tensor_schema,
            )
        ),
    )
    with path.open("rb") as handle:
        for chunk_index in range(chunk_count):
            data = handle.read(chunk_size)
            digest = hashlib.sha256(data).hexdigest()
            chunk = Chunk(
                transfer_id=transfer_id,
                chunk_index=chunk_index,
                chunk_count=chunk_count,
                chunk_sha256=digest,
                data=data,
            )
            total += _envelope_size(
                bundle,
                message_type=MessageType.CHUNK,
                message_id=f"{transfer_id}-chunk-{chunk_index}",
                correlation_id=transfer_id,
                payload=encode_message(chunk),
            )
            total += _envelope_size(
                bundle,
                message_type=MessageType.CHUNK_ACK,
                message_id=f"{transfer_id}-ack-{chunk_index}",
                correlation_id=transfer_id,
                payload=encode_message(
                    ChunkAck(
                        transfer_id=transfer_id,
                        chunk_index=chunk_index,
                        chunk_sha256=digest,
                    )
                ),
                sender_public_key="measurement-peer",
            )
    total += _envelope_size(
        bundle,
        message_type=MessageType.TRANSFER_COMPLETE,
        message_id=f"{transfer_id}-complete",
        correlation_id=transfer_id,
        payload=encode_message(
            TransferComplete(
                transfer_id=transfer_id,
                total_sha256=total_sha256,
            )
        ),
    )
    return total, chunk_count


_DTYPE_BYTES = {
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "int32": 4,
    "int64": 8,
}


def _schema_payload_bytes(schema: TensorSchema) -> int:
    total = 0
    for tensor in schema.tensors:
        try:
            dtype_bytes = _DTYPE_BYTES[tensor.dtype]
        except KeyError as error:
            raise ValueError(f"unsupported tensor dtype: {tensor.dtype}") from error
        total += math.prod(tensor.shape) * dtype_bytes
    return total


def _envelope_size(
    bundle: UpdateBundle,
    *,
    message_type: MessageType,
    message_id: str,
    correlation_id: str,
    payload: bytes,
    sender_public_key: str | None = None,
) -> int:
    metadata = bundle.metadata
    envelope = create_envelope(
        message_type=message_type,
        message_id=message_id,
        run_id=metadata.run_id,
        manifest_hash=metadata.manifest_hash,
        sender_public_key=sender_public_key or metadata.sender_public_key,
        algorithm_id=metadata.algorithm_id,
        round_id=metadata.round_id,
        correlation_id=correlation_id,
        payload=payload,
    )
    return len(encode_envelope(envelope))


__all__ = ["BundleWireMeasurement", "measure_bundle_wire_bytes"]
