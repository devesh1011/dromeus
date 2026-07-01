from __future__ import annotations

import pytest

from dromeus.transport.envelope import (
    EnvelopeError,
    MessageType,
    create_envelope,
    decode_envelope,
    encode_envelope,
)

HASH = "a" * 64
PEERS = frozenset({"peer-0", "peer-1", "peer-2", "peer-3"})


def envelope_bytes(payload: bytes = b"weights") -> bytes:
    envelope = create_envelope(
        message_type=MessageType.CHUNK,
        message_id="message-1",
        run_id="run-001",
        manifest_hash=HASH,
        sender_public_key="peer-0",
        algorithm_id="dpsgd-v1",
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


@pytest.mark.parametrize("sender", ["peer-1", "outsider"])
def test_sender_must_match_authenticated_sealed_participant(sender: str) -> None:
    with pytest.raises(EnvelopeError):
        decode_envelope(
            envelope_bytes(), authenticated_sender=sender, participant_keys=PEERS
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
