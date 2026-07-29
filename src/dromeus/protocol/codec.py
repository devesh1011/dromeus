"""Bounded MessagePack encoding for Dromeus protocol models."""

from __future__ import annotations

from typing import cast

import msgpack  # pyright: ignore[reportMissingTypeStubs]
from pydantic import ValidationError

from dromeus.protocol.models import DomainModel, Envelope

DEFAULT_MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
_ENVELOPE_OVERHEAD_BYTES = 4096
_MAX_ARRAY_LENGTH = 1024


class ProtocolDecodeError(ValueError):
    """Encoded protocol bytes are malformed or unsupported."""


class EnvelopeError(ProtocolDecodeError):
    """The wire envelope is malformed or unauthorised."""


def encode_message(model: DomainModel) -> bytes:
    value = model.model_dump(mode="python")
    return cast(bytes, msgpack.packb(value))  # pyright: ignore[reportUnknownMemberType]


def decode_message[T: DomainModel](
    data: bytes,
    model_type: type[T],
    *,
    max_bytes: int,
) -> T:
    try:
        return model_type.model_validate(_unpack(data, max_bytes=max_bytes))
    except (ValueError, ValidationError, msgpack.UnpackException) as error:
        raise ProtocolDecodeError("invalid protocol payload") from error


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
    envelope = _decode_envelope(data, max_payload_bytes=max_payload_bytes)
    if envelope.sender_public_key != authenticated_sender:
        raise EnvelopeError("authenticated sender does not match envelope")
    if (
        participant_keys is not None
        and envelope.sender_public_key not in participant_keys
    ):
        raise EnvelopeError("sender is not a sealed participant")
    return envelope


def decode_envelope_sender(
    data: bytes,
    *,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
) -> str:
    return _decode_envelope(
        data, max_payload_bytes=max_payload_bytes
    ).sender_public_key


def _decode_envelope(data: bytes, *, max_payload_bytes: int) -> Envelope:
    if max_payload_bytes < 0:
        raise ValueError("max payload bytes must not be negative")
    try:
        envelope = Envelope.model_validate(
            _unpack(
                data,
                max_bytes=max_payload_bytes + _ENVELOPE_OVERHEAD_BYTES,
                max_bin_len=max_payload_bytes,
            )
        )
    except (ValueError, ValidationError, msgpack.UnpackException) as error:
        raise EnvelopeError("invalid envelope") from error
    if envelope.payload_length > max_payload_bytes:
        raise EnvelopeError("payload exceeds size limit")
    return envelope


def _unpack(
    data: bytes,
    *,
    max_bytes: int,
    max_bin_len: int | None = None,
) -> object:
    if max_bytes < 0:
        raise ValueError("max bytes must not be negative")
    if len(data) > max_bytes:
        raise ProtocolDecodeError("encoded payload exceeds size limit")
    binary_limit = max_bytes if max_bin_len is None else max_bin_len
    return cast(
        object,
        msgpack.unpackb(  # pyright: ignore[reportUnknownMemberType]
            data,
            raw=False,
            strict_map_key=True,
            max_bin_len=binary_limit,
            max_str_len=4096,
            max_array_len=_MAX_ARRAY_LENGTH,
            max_map_len=32,
            max_ext_len=0,
        ),
    )


__all__ = [
    "DEFAULT_MAX_PAYLOAD_BYTES",
    "EnvelopeError",
    "ProtocolDecodeError",
    "decode_envelope",
    "decode_envelope_sender",
    "decode_message",
    "encode_envelope",
    "encode_message",
]
