import asyncio

import pytest

from dromeus.protocol.codec import encode_envelope
from dromeus.protocol.models import MessageType, create_envelope
from dromeus.transport.axl import (
    AXLBridgeConfig,
    AXLTransport,
    matches_yggdrasil_sender,
)


def test_yggdrasil_sender_identity_uses_known_bits_not_string_prefix() -> None:
    bridge_sender = "d5bcbd9608ae04b5a88dbafe46287" + "f" * 35
    actual_sender = "d5bcbd9608ae04b5a88dbafe4628672ed7d776c8fdef2857c3f09093d9ecbe0a"
    different_sender = (
        "d5bcbd9608ae04b5a88dbafe4628e72ed7d776c8fdef2857c3f09093d9ecbe0a"
    )

    assert matches_yggdrasil_sender(actual_sender, bridge_sender)
    assert not matches_yggdrasil_sender(different_sender, bridge_sender)


def test_yggdrasil_sender_accepts_valid_address_with_leading_zero_key_bits() -> None:
    actual_sender = "9c800afbc9caad915aaec9b2dd6be5bb53874a44e0993cfc325dacefed413bde"
    bridge_sender = "9c800afbc9caad915aaec9b2dd6b" + "f" * 36

    assert matches_yggdrasil_sender(actual_sender, bridge_sender)


def test_sender_resolution_retries_ambiguous_topology() -> None:
    bridge_sender = "d5bcbd9608ae04b5a88dbafe46287" + "f" * 35
    actual_sender = "d5bcbd9608ae04b5a88dbafe4628672ed7d776c8fdef2857c3f09093d9ecbe0a"
    ambiguous_sender = (
        "d5bcbd9608ae04b5a88dbafe4628672ed7d776c8fdef2857c3f09093d9ecbe0b"
    )
    transport = SequencedTopologyTransport(
        [
            {
                "peers": [
                    {"public_key": actual_sender},
                    {"public_key": ambiguous_sender},
                ]
            },
            {"peers": [{"public_key": actual_sender}]},
        ]
    )

    assert transport.resolve_sender(bridge_sender) == actual_sender


def test_sender_resolution_uses_claimed_sender_when_topology_stays_ambiguous() -> None:
    bridge_sender = "d5bcbd9608ae04b5a88dbafe46287" + "f" * 35
    actual_sender = "d5bcbd9608ae04b5a88dbafe4628672ed7d776c8fdef2857c3f09093d9ecbe0a"
    ambiguous_sender = (
        "d5bcbd9608ae04b5a88dbafe4628672ed7d776c8fdef2857c3f09093d9ecbe0b"
    )
    transport = SequencedTopologyTransport(
        [
            {
                "peers": [
                    {"public_key": actual_sender},
                    {"public_key": ambiguous_sender},
                ]
            },
        ]
    )
    payload = _envelope_from(actual_sender)

    assert transport.resolve_sender(bridge_sender, payload) == actual_sender


def test_topology_returns_raw_axl_snapshot() -> None:
    snapshot: dict[str, object] = {
        "our_public_key": "peer-0",
        "peers": [{"public_key": "peer-1"}],
    }
    transport = SequencedTopologyTransport([snapshot])

    assert asyncio.run(transport.topology()) == snapshot


def test_recv_socket_timeout_is_an_empty_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = AXLTransport(AXLBridgeConfig(base_url="http://127.0.0.1:0"))

    def time_out(*_args: object, **_kwargs: object) -> None:
        raise TimeoutError("timed out")

    monkeypatch.setattr("dromeus.transport.axl.urlopen", time_out)

    assert asyncio.run(transport.recv(0.1)) is None


def test_sender_resolution_uses_header_matching_claim_absent_from_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge_sender = "d5bcbd9608ae04b5a88dbafe46287" + "f" * 35
    actual_sender = "d5bcbd9608ae04b5a88dbafe4628672ed7d776c8fdef2857c3f09093d9ecbe0a"
    transport = SequencedTopologyTransport(
        [{"peers": [{"public_key": "a" * 64}]}]
    )
    payload = _envelope_from(actual_sender)

    def skip_sleep(_: float) -> None:
        return

    monkeypatch.setattr("dromeus.transport.axl.time.sleep", skip_sleep)

    assert transport.resolve_sender(bridge_sender, payload) == actual_sender


class SequencedTopologyTransport(AXLTransport):
    def __init__(self, topologies: list[dict[str, object]]) -> None:
        super().__init__(AXLBridgeConfig(base_url="http://127.0.0.1:0"))
        self._topologies = topologies

    def _load_topology(self) -> dict[str, object]:
        if len(self._topologies) > 1:
            return self._topologies.pop(0)
        return self._topologies[0]

    def resolve_sender(self, bridge_sender: str, payload: bytes | None = None) -> str:
        return self._resolve_sender(bridge_sender, payload)


def _envelope_from(sender: str) -> bytes:
    return encode_envelope(
        create_envelope(
            message_type=MessageType.READY,
            message_id="message-1",
            run_id="run-001",
            manifest_hash="a" * 64,
            sender_public_key=sender,
            algorithm_id="dpsgd",
            payload=b"",
        )
    )
