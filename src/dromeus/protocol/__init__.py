"""Validated Dromeus wire protocol."""

from dromeus.protocol.codec import (
    EnvelopeError,
    ProtocolDecodeError,
    decode_envelope,
    decode_message,
    encode_envelope,
    encode_message,
)
from dromeus.protocol.models import Envelope, MessageType, create_envelope
from dromeus.protocol.version import PROTOCOL_VERSION

__all__ = [
    "Envelope",
    "EnvelopeError",
    "MessageType",
    "PROTOCOL_VERSION",
    "ProtocolDecodeError",
    "create_envelope",
    "decode_envelope",
    "decode_message",
    "encode_envelope",
    "encode_message",
]
