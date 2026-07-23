from typing import cast

import msgpack  # pyright: ignore[reportMissingTypeStubs]
import pytest

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
    payload = cast(
        bytes,
        msgpack.packb(  # pyright: ignore[reportUnknownMemberType]
            {"sender_public_key": actual_sender}
        ),
    )

    assert transport.resolve_sender(bridge_sender, payload) == actual_sender


def test_sender_resolution_uses_header_matching_claim_absent_from_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge_sender = "d5bcbd9608ae04b5a88dbafe46287" + "f" * 35
    actual_sender = "d5bcbd9608ae04b5a88dbafe4628672ed7d776c8fdef2857c3f09093d9ecbe0a"
    transport = SequencedTopologyTransport(
        [{"peers": [{"public_key": "a" * 64}]}]
    )
    payload = cast(
        bytes,
        msgpack.packb(  # pyright: ignore[reportUnknownMemberType]
            {"sender_public_key": actual_sender}
        ),
    )

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
