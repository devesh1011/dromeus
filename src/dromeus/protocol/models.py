"""Closed models carried by the Dromeus wire protocol."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from dromeus.protocol.version import PROTOCOL_VERSION

Identifier = Annotated[
    str, StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
]
RunId = Identifier
MessageId = Identifier
TransferId = Identifier
AlgorithmId = Identifier
RoundId = Annotated[int, Field(ge=0)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
PublicKey = Annotated[
    str, StringConstraints(min_length=1, max_length=512, pattern=r"^\S+$")
]

MAGIC: Final = b"DRM1"


class DomainModel(BaseModel):
    """Closed, immutable value object used at protocol seams."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Tensor(DomainModel):
    name: Identifier
    dtype: Literal["float16", "float32", "float64", "int8", "int32", "int64"]
    shape: tuple[Annotated[int, Field(gt=0)], ...]


class TensorSchema(DomainModel):
    tensors: tuple[Tensor, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_names(self) -> Self:
        names = [tensor.name for tensor in self.tensors]
        if len(names) != len(set(names)):
            raise ValueError("tensor names must be unique")
        return self


class MessageType(StrEnum):
    JOIN_REQUEST = "JOIN_REQUEST"
    JOIN_ACCEPTED = "JOIN_ACCEPTED"
    MANIFEST_SEALED = "MANIFEST_SEALED"
    READY = "READY"
    START = "START"
    START_ACK = "START_ACK"
    RUN_FAILED = "RUN_FAILED"
    RUN_COMPLETE = "RUN_COMPLETE"
    TRANSFER_BEGIN = "TRANSFER_BEGIN"
    CHUNK = "CHUNK"
    CHUNK_ACK = "CHUNK_ACK"
    TRANSFER_COMPLETE = "TRANSFER_COMPLETE"
    UPDATE_READY = "UPDATE_READY"
    ROUND_COMMITTED = "ROUND_COMMITTED"
    CONSENSUS_SKETCH = "CONSENSUS_SKETCH"


class Envelope(DomainModel):
    magic: Literal[b"DRM1"] = MAGIC
    protocol_version: Literal[1] = PROTOCOL_VERSION
    message_type: MessageType
    message_id: MessageId
    run_id: RunId
    manifest_hash: Sha256
    sender_public_key: PublicKey
    algorithm_id: AlgorithmId
    round_id: RoundId | None = None
    correlation_id: MessageId | None = None
    payload_length: Annotated[int, Field(ge=0)]
    payload_sha256: Sha256
    payload: bytes

    @model_validator(mode="after")
    def valid_payload(self) -> Envelope:
        if len(self.payload) != self.payload_length:
            raise ValueError("payload length does not match envelope")
        if hashlib.sha256(self.payload).hexdigest() != self.payload_sha256:
            raise ValueError("payload checksum does not match envelope")
        return self


class JoinRequest(DomainModel):
    draft_hash: Sha256


class JoinAccepted(DomainModel):
    draft_hash: Sha256


class ReadyMessage(DomainModel):
    manifest_hash: Sha256


class StartMessage(DomainModel):
    manifest_hash: Sha256


class TransferBegin(DomainModel):
    transfer_id: TransferId
    artifact_name: Identifier
    total_size_bytes: int = Field(gt=0)
    total_sha256: Sha256
    chunk_count: int = Field(gt=0)
    codec_id: Identifier
    tensor_schema: TensorSchema


class Chunk(DomainModel):
    transfer_id: TransferId
    chunk_index: int = Field(ge=0)
    chunk_count: int = Field(gt=0)
    chunk_sha256: Sha256
    data: bytes


class ChunkAck(DomainModel):
    transfer_id: TransferId
    chunk_index: int = Field(ge=0)
    chunk_sha256: Sha256


class TransferComplete(DomainModel):
    transfer_id: TransferId
    total_sha256: Sha256


class PairCommitMessage(DomainModel):
    round_id: RoundId
    checksum: Sha256


class RunFailedMessage(DomainModel):
    round_id: RoundId
    error_type: str
    reason: str


def create_envelope(
    *,
    message_type: MessageType,
    message_id: MessageId,
    run_id: RunId,
    manifest_hash: Sha256,
    sender_public_key: PublicKey,
    algorithm_id: AlgorithmId,
    payload: bytes,
    round_id: RoundId | None = None,
    correlation_id: MessageId | None = None,
) -> Envelope:
    return Envelope(
        message_type=message_type,
        message_id=message_id,
        run_id=run_id,
        manifest_hash=manifest_hash,
        sender_public_key=sender_public_key,
        algorithm_id=algorithm_id,
        round_id=round_id,
        correlation_id=correlation_id,
        payload_length=len(payload),
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        payload=payload,
    )


__all__ = [
    "AlgorithmId",
    "Chunk",
    "ChunkAck",
    "DomainModel",
    "Envelope",
    "Identifier",
    "JoinAccepted",
    "JoinRequest",
    "MAGIC",
    "MessageId",
    "MessageType",
    "PairCommitMessage",
    "PublicKey",
    "ReadyMessage",
    "RoundId",
    "RunFailedMessage",
    "RunId",
    "Sha256",
    "StartMessage",
    "Tensor",
    "TensorSchema",
    "TransferBegin",
    "TransferComplete",
    "TransferId",
    "create_envelope",
]
