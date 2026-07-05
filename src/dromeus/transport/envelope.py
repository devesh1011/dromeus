"""Validated MessagePack wire envelopes."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Annotated, Final, Literal, cast

import msgpack  # pyright: ignore[reportMissingTypeStubs]
from pydantic import Field, ValidationError, model_validator

from dromeus.manifests.models import (
    PROTOCOL_VERSION,
    AlgorithmId,
    DomainModel,
    MessageId,
    PublicKey,
    RoundId,
    RunId,
    Sha256,
)

MAGIC: Final = b"DRM1"
DEFAULT_MAX_PAYLOAD_BYTES: Final = 8 * 1024 * 1024


class EnvelopeError(ValueError):
    """The wire envelope is malformed or unauthorised."""


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


def encode_envelope(envelope: Envelope) -> bytes:
    value = envelope.model_dump(mode="python", exclude_none=True)
    return cast(bytes, msgpack.packb(value))  # pyright: ignore[reportUnknownMemberType]


def decode_envelope(
    data: bytes,
    *,
    authenticated_sender: str,
    participant_keys: frozenset[str] | None,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
) -> Envelope:
    """Decode only bounded messages from authenticated sealed participants."""
    if max_payload_bytes < 0:
        raise ValueError("max payload bytes must not be negative")
    if len(data) > max_payload_bytes + 4096:
        raise EnvelopeError("encoded envelope exceeds size limit")
    try:
        unpacked = cast(
            object,
            msgpack.unpackb(  # pyright: ignore[reportUnknownMemberType]
                data,
                raw=False,
                strict_map_key=True,
                max_bin_len=max_payload_bytes,
                max_str_len=4096,
                max_array_len=128,
                max_map_len=32,
                max_ext_len=0,
            ),
        )
        envelope = Envelope.model_validate(unpacked)
    except (ValueError, ValidationError, msgpack.UnpackException) as error:
        raise EnvelopeError("invalid envelope") from error
    if envelope.payload_length > max_payload_bytes:
        raise EnvelopeError("payload exceeds size limit")
    if envelope.sender_public_key != authenticated_sender:
        raise EnvelopeError("authenticated sender does not match envelope")
    if (
        participant_keys is not None
        and envelope.sender_public_key not in participant_keys
    ):
        raise EnvelopeError("sender is not a sealed participant")
    return envelope
