from __future__ import annotations

import pytest

from dromeus.protocol.codec import (
    EnvelopeError,
    decode_envelope,
    encode_envelope,
)
from dromeus.protocol.models import MessageType, create_envelope

HASH = "a" * 64
PEERS = frozenset({"peer-0", "peer-1", "peer-2", "peer-3"})


def envelope_bytes(payload: bytes = b"weights") -> bytes:
    envelope = create_envelope(
        message_type=MessageType.CHUNK,
        message_id="message-1",
        run_id="run-001",
        manifest_hash=HASH,
        sender_public_key="peer-0",
        algorithm_id="dpsgd",
        round_id=2,
        correlation_id="transfer-1",
        payload=payload,
    )
    return encode_envelope(envelope)


def test_envelope_round_trip() -> None:
    decoded = decode_envelope(
        envelope_bytes(), authenticated_sender="peer-0", participant_keys=PEERS
    )

    assert decoded.message_type is MessageType.CHUNK
    assert decoded.payload == b"weights"
    assert decoded.round_id == 2


def test_control_envelope_round_trip_without_round_id() -> None:
    encoded = encode_envelope(
        create_envelope(
            message_type=MessageType.READY,
            message_id="message-2",
            run_id="run-001",
            manifest_hash=HASH,
            sender_public_key="peer-1",
            algorithm_id="dpsgd",
            correlation_id="message-1",
            payload=b"ready",
        )
    )

    decoded = decode_envelope(
        encoded, authenticated_sender="peer-1", participant_keys=PEERS
    )

    assert decoded.message_type is MessageType.READY
    assert decoded.round_id is None
    assert decoded.correlation_id == "message-1"


@pytest.mark.parametrize("sender", ["peer-1", "outsider"])
def test_sender_must_match_authenticated_sealed_participant(sender: str) -> None:
    with pytest.raises(EnvelopeError):
        decode_envelope(
            envelope_bytes(), authenticated_sender=sender, participant_keys=PEERS
        )


def test_sender_prefix_does_not_authenticate_a_different_identity() -> None:
    sender = "0123456789abcdef000000000000000000000000000000000000000000000000"
    impostor = "0123456789abcdefffffffffffffffffffffffffffffffffffffffffffffffff"
    encoded = encode_envelope(
        create_envelope(
            message_type=MessageType.CHUNK,
            message_id="message-prefix",
            run_id="run-001",
            manifest_hash=HASH,
            sender_public_key=sender,
            algorithm_id="dpsgd",
            payload=b"weights",
        )
    )
    with pytest.raises(EnvelopeError, match="authenticated sender"):
        decode_envelope(
            encoded,
            authenticated_sender=impostor,
            participant_keys=frozenset({sender}),
        )


def test_payload_size_is_bounded_before_unpacking() -> None:
    with pytest.raises(EnvelopeError):
        decode_envelope(
            envelope_bytes(b"12345"),
            authenticated_sender="peer-0",
            participant_keys=PEERS,
            max_payload_bytes=4,
        )


def test_corrupt_payload_is_rejected() -> None:
    encoded = envelope_bytes()
    corrupted = replace_bytes(encoded, b"weights", b"xeights")

    with pytest.raises(EnvelopeError, match="invalid envelope"):
        decode_envelope(
            corrupted, authenticated_sender="peer-0", participant_keys=PEERS
        )


def replace_bytes(value: bytes, old: bytes, new: bytes) -> bytes:
    assert old in value
    return value.replace(old, new, 1)
