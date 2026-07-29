"""Compatibility imports for the protocol-owned wire envelope."""

from dromeus.protocol.codec import (
    DEFAULT_MAX_PAYLOAD_BYTES,
    EnvelopeError,
    decode_envelope,
    encode_envelope,
)
from dromeus.protocol.models import (
    MAGIC,
    Envelope,
    MessageType,
    create_envelope,
)

__all__ = [
    "DEFAULT_MAX_PAYLOAD_BYTES",
    "Envelope",
    "EnvelopeError",
    "MAGIC",
    "MessageType",
    "create_envelope",
    "decode_envelope",
    "encode_envelope",
]
