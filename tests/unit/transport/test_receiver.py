from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field

import pytest
from support.in_memory_transport import (
    InMemoryNetwork,
    InMemoryTransport,
)
from support.sample_manifest import manifest_data

from dromeus.manifests.models import SealedManifest
from dromeus.protocol.codec import encode_envelope, encode_message
from dromeus.protocol.models import (
    Chunk,
    ChunkAck,
    Envelope,
    MessageType,
    PairCommitMessage,
    ReadyMessage,
    RunFailedMessage,
    create_envelope,
)
from dromeus.transport.interface import ReceivedBytes
from dromeus.transport.receiver import MessageChannel, Receiver, ReceiverPolicy

_QUEUE_CAPACITY = 64
_REQUIRED_CHANNELS = (
    (MessageType.READY, MessageChannel.CONTROL),
    (MessageType.CHUNK, MessageChannel.TRANSFER),
    (MessageType.CHUNK_ACK, MessageChannel.ACKNOWLEDGMENT),
    (MessageType.UPDATE_READY, MessageChannel.PAIR_COMMIT),
)


class ObservedTransport(InMemoryTransport):
    """Observe delivery through the transport seam without inspecting inboxes."""

    def __init__(self, network: InMemoryNetwork) -> None:
        super().__init__(network=network, public_key="local")
        self.received_count = 0
        self.changed = asyncio.Event()

    async def recv(self, timeout_seconds: float) -> ReceivedBytes | None:
        message = await super().recv(timeout_seconds)
        if message is not None:
            self.received_count += 1
            self.changed.set()
        return message

    async def wait_for_received(self, count: int) -> None:
        async with asyncio.timeout(1):
            while self.received_count < count:
                self.changed.clear()
                await self.changed.wait()


@dataclass
class RecordingSink:
    records: list[dict[str, object]] = field(default_factory=list[dict[str, object]])

    def append(self, record: Mapping[str, object]) -> None:
        self.records.append(dict(record))


def _setup(
    *, world_size: int = 16, current_round: int = 4
) -> tuple[Receiver, ObservedTransport, tuple[InMemoryTransport, ...], RecordingSink]:
    network = InMemoryNetwork()
    inbound = ObservedTransport(network)
    keys = tuple(f"peer-{index}" for index in range(world_size - 1))
    senders = tuple(InMemoryTransport(network=network, public_key=key) for key in keys)
    sink = RecordingSink()
    receiver = Receiver(
        inbound,
        ReceiverPolicy(
            run_id="receiver-backpressure",
            manifest_hash="a" * 64,
            algorithm_id="noloco",
            participant_keys=frozenset(("local", *keys)),
            current_round=lambda: current_round,
        ),
        event_sink=sink,
    )
    return receiver, inbound, senders, sink


async def _send(
    sender: InMemoryTransport,
    message_type: MessageType,
    message_id: str,
    *,
    round_id: int = 4,
) -> None:
    digest = hashlib.sha256(b"chunk").hexdigest()
    payloads = {
        MessageType.CONSENSUS_SKETCH: bytes(4096 * 4),
        MessageType.READY: encode_message(ReadyMessage(manifest_hash="a" * 64)),
        MessageType.CHUNK: encode_message(
            Chunk(
                transfer_id="transfer",
                chunk_index=0,
                chunk_count=1,
                chunk_sha256=digest,
                data=b"chunk",
            )
        ),
        MessageType.CHUNK_ACK: encode_message(
            ChunkAck(transfer_id="transfer", chunk_index=0, chunk_sha256=digest)
        ),
        MessageType.UPDATE_READY: encode_message(
            PairCommitMessage(round_id=round_id, checksum="b" * 64)
        ),
        MessageType.RUN_FAILED: encode_message(
            RunFailedMessage(round_id=round_id, error_type="Test", reason="test")
        ),
    }
    await sender.send(
        "local",
        encode_envelope(
            create_envelope(
                message_type=message_type,
                message_id=message_id,
                run_id="receiver-backpressure",
                manifest_hash="a" * 64,
                sender_public_key=await sender.local_public_key(),
                algorithm_id="noloco",
                round_id=round_id,
                correlation_id="transfer",
                payload=payloads[message_type],
            )
        ),
    )


async def _cleanup(receiver: Receiver) -> None:
    # A failing pre-fix test must still terminate when its inbox is blocked.
    await asyncio.gather(
        asyncio.wait_for(receiver.stop(), timeout=0.2), return_exceptions=True
    )


@pytest.mark.parametrize("world_size", (4, 8, 16))
@pytest.mark.parametrize(
    ("message_type", "channel"),
    (
        (MessageType.CHUNK_ACK, MessageChannel.ACKNOWLEDGMENT),
        (MessageType.UPDATE_READY, MessageChannel.PAIR_COMMIT),
        (MessageType.RUN_FAILED, MessageChannel.CONTROL),
    ),
)
def test_full_telemetry_does_not_block_required_messages(
    world_size: int, message_type: MessageType, channel: MessageChannel
) -> None:
    async def run() -> None:
        current = _QUEUE_CAPACITY // (world_size - 1)
        receiver, _, senders, sink = _setup(
            world_size=world_size, current_round=current
        )
        await receiver.start()
        try:
            for index in range(_QUEUE_CAPACITY + 1):
                await _send(
                    senders[index % len(senders)],
                    MessageType.CONSENSUS_SKETCH,
                    f"sketch-{index}",
                    round_id=index // len(senders),
                )
            await _send(senders[0], message_type, "required", round_id=current)
            received = await receiver.receive(channel, timeout_seconds=1)
            assert received.message_id == "required"
            assert receiver.stats.dropped_telemetry_messages == 1
            assert receiver.stats.rejected_messages == 0
            assert receiver.stats.accepted_messages == _QUEUE_CAPACITY + 2
            queued = [
                await receiver.receive(MessageChannel.TELEMETRY, timeout_seconds=1)
                for _ in range(_QUEUE_CAPACITY)
            ]
            assert [item.message_id for item in queued] == [
                f"sketch-{index}" for index in range(_QUEUE_CAPACITY)
            ]
            with pytest.raises(TimeoutError):
                await receiver.receive(MessageChannel.TELEMETRY, timeout_seconds=0.01)
            await _send(
                senders[0],
                MessageType.CONSENSUS_SKETCH,
                "after-drain",
                round_id=current,
            )
            resumed = await receiver.receive(
                MessageChannel.TELEMETRY, timeout_seconds=1
            )
            assert resumed.message_id == "after-drain"
            dropped = [
                event
                for event in sink.records
                if event["event"] == "telemetry_message_dropped"
            ]
            assert len(dropped) == 1
            assert dropped[0]["message_id"] == "sketch-64"
            received_event = next(
                event
                for event in sink.records
                if event["event"] == "message_received"
                and event.get("message_id") == "sketch-64"
            )
            assert received_event["routed"] is False
        finally:
            await _cleanup(receiver)

    asyncio.run(run())


def test_future_telemetry_does_not_block_round_advancement() -> None:
    async def run() -> None:
        receiver, inbound, senders, _ = _setup()
        await receiver.start()
        try:
            for index in range(_QUEUE_CAPACITY):
                await _send(
                    senders[index % len(senders)],
                    MessageType.CONSENSUS_SKETCH,
                    f"sketch-{index}",
                    round_id=index // len(senders),
                )
            await _send(senders[0], MessageType.CONSENSUS_SKETCH, "future", round_id=5)
            await _send(senders[0], MessageType.READY, "received-future")
            await receiver.receive(MessageChannel.CONTROL, timeout_seconds=1)
            receiver.set_current_round(5)
            await asyncio.wait_for(receiver.advance_round(5), timeout=1)
            await _send(senders[0], MessageType.UPDATE_READY, "next-pair", round_id=5)
            await inbound.wait_for_received(_QUEUE_CAPACITY + 3)
            message = await receiver.receive(
                MessageChannel.PAIR_COMMIT, timeout_seconds=1
            )
            assert message.message_id == "next-pair"
            assert receiver.stats.dropped_telemetry_messages == 1
        finally:
            await _cleanup(receiver)

    asyncio.run(run())


@pytest.mark.parametrize(("message_type", "channel"), _REQUIRED_CHANNELS)
def test_required_channel_backpressure_preserves_messages(
    message_type: MessageType, channel: MessageChannel
) -> None:
    async def run() -> None:
        receiver, inbound, senders, _ = _setup()
        await receiver.start()
        try:
            for index in range(_QUEUE_CAPACITY + 1):
                await _send(senders[0], message_type, f"required-{index}")
            await inbound.wait_for_received(_QUEUE_CAPACITY + 1)
            assert receiver.stats.accepted_messages == _QUEUE_CAPACITY
            messages: list[Envelope] = [
                await receiver.receive(channel, timeout_seconds=1)
                for _ in range(_QUEUE_CAPACITY + 1)
            ]
            assert [item.message_id for item in messages] == [
                f"required-{index}" for index in range(_QUEUE_CAPACITY + 1)
            ]
        finally:
            await _cleanup(receiver)

    asyncio.run(run())


@pytest.mark.parametrize(("message_type", "channel"), _REQUIRED_CHANNELS)
def test_stop_terminates_with_a_full_required_channel(
    message_type: MessageType, channel: MessageChannel
) -> None:
    async def run() -> None:
        receiver, inbound, senders, _ = _setup()
        await receiver.start()
        try:
            for index in range(_QUEUE_CAPACITY + 1):
                await _send(senders[0], message_type, f"required-{index}")
            await inbound.wait_for_received(_QUEUE_CAPACITY + 1)
            await asyncio.wait_for(receiver.stop(), timeout=1)
            await asyncio.wait_for(receiver.stop(), timeout=1)
            retained = [
                await receiver.receive(channel, timeout_seconds=1)
                for _ in range(_QUEUE_CAPACITY)
            ]
            assert [item.message_id for item in retained] == [
                f"required-{index}" for index in range(_QUEUE_CAPACITY)
            ]
            with pytest.raises(TimeoutError):
                await receiver.receive(channel, timeout_seconds=0.01)
        finally:
            await _cleanup(receiver)

    asyncio.run(run())


def test_stop_preserves_an_unexpected_reader_failure() -> None:
    async def run() -> None:
        called = asyncio.Event()

        class FailedTransport(ObservedTransport):
            async def recv(self, timeout_seconds: float) -> ReceivedBytes | None:
                called.set()
                raise RuntimeError("reader failed")

        receiver = Receiver(FailedTransport(InMemoryNetwork()))
        await receiver.start()
        await asyncio.wait_for(called.wait(), timeout=1)
        with pytest.raises(RuntimeError, match="reader failed"):
            await receiver.stop()

    asyncio.run(run())


def test_receiver_logs_correlation_ids(
    capsys: pytest.CaptureFixture[str],
) -> None:
    asyncio.run(_test_receiver_logs_correlation_ids(capsys))


async def _test_receiver_logs_correlation_ids(
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    network = InMemoryNetwork()
    sender = InMemoryTransport(network=network, public_key="peer-0")
    transport = InMemoryTransport(network=network, public_key="peer-1")
    receiver = Receiver(
        transport,
        ReceiverPolicy(
            run_id=manifest.run_id,
            manifest_hash=manifest.draft_hash,
            algorithm_id=manifest.algorithm_id,
            participant_keys=frozenset({"peer-0", "peer-1"}),
        ),
    )
    await receiver.start()
    envelope = create_envelope(
        message_type=MessageType.READY,
        message_id="message-1",
        run_id=manifest.run_id,
        manifest_hash=manifest.draft_hash,
        sender_public_key="peer-0",
        algorithm_id=manifest.algorithm_id,
        round_id=2,
        correlation_id="transfer-1",
        payload=b"ready",
    )
    await sender.send("peer-1", encode_envelope(envelope))
    await receiver.receive(MessageChannel.CONTROL, timeout_seconds=0.1)
    await receiver.stop()

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    received = next(
        record for record in records if record["event"] == "message_received"
    )
    expected = {
        "run_id": manifest.run_id,
        "message_id": "message-1",
        "transfer_id": "transfer-1",
        "peer_id": "peer-0",
        "round_id": 2,
    }
    assert {key: received[key] for key in expected} == expected




def test_future_round_message_id_is_deduplicated_per_sender() -> None:
    asyncio.run(_test_future_round_message_id_is_deduplicated_per_sender())


def test_future_round_consensus_sketch_is_buffered_until_round_advances() -> None:
    asyncio.run(_test_future_round_consensus_sketch_is_buffered())


async def _test_future_round_consensus_sketch_is_buffered() -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    network = InMemoryNetwork()
    sender_transport = InMemoryTransport(network=network, public_key="peer-0")
    receiver_transport = InMemoryTransport(network=network, public_key="peer-1")
    current_round = 0
    receiver = Receiver(
        receiver_transport,
        ReceiverPolicy(
            run_id=manifest.run_id,
            manifest_hash=manifest.draft_hash,
            algorithm_id=manifest.algorithm_id,
            participant_keys=frozenset({"peer-0", "peer-1"}),
            current_round=lambda: current_round,
        ),
    )
    await receiver.start()
    try:
        envelope = create_envelope(
            message_type=MessageType.CONSENSUS_SKETCH,
            message_id="future-consensus",
            run_id=manifest.run_id,
            manifest_hash=manifest.draft_hash,
            sender_public_key="peer-0",
            algorithm_id=manifest.algorithm_id,
            round_id=1,
            payload=b"sketch",
        )
        await sender_transport.send("peer-1", encode_envelope(envelope))
        await asyncio.sleep(0.05)
        with pytest.raises(TimeoutError):
            await receiver.receive(MessageChannel.TELEMETRY, timeout_seconds=0.01)

        current_round = 1
        await receiver.advance_round(1)
        received = await receiver.receive(MessageChannel.TELEMETRY, timeout_seconds=0.1)
        assert received.message_id == "future-consensus"
        assert receiver.stats.rejected_messages == 0
    finally:
        await receiver.stop()


async def _test_future_round_message_id_is_deduplicated_per_sender() -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    network = InMemoryNetwork()
    sender_transport = InMemoryTransport(network=network, public_key="peer-0")
    second_sender_transport = InMemoryTransport(network=network, public_key="peer-2")
    receiver_transport = InMemoryTransport(network=network, public_key="peer-1")
    current_round = 0
    receiver = Receiver(
        receiver_transport,
        ReceiverPolicy(
            run_id=manifest.run_id,
            manifest_hash=manifest.draft_hash,
            algorithm_id=manifest.algorithm_id,
            participant_keys=frozenset({"peer-0", "peer-1", "peer-2"}),
            current_round=lambda: current_round,
        ),
    )
    await receiver.start()
    envelope = create_envelope(
        message_type=MessageType.UPDATE_READY,
        message_id="future-update",
        run_id=manifest.run_id,
        manifest_hash=manifest.draft_hash,
        sender_public_key="peer-0",
        algorithm_id=manifest.algorithm_id,
        round_id=1,
        payload=b"update",
    )
    await sender_transport.send("peer-1", encode_envelope(envelope))
    await sender_transport.send("peer-1", encode_envelope(envelope))
    await second_sender_transport.send(
        "peer-1",
        encode_envelope(envelope.model_copy(update={"sender_public_key": "peer-2"})),
    )
    await asyncio.sleep(0.05)
    with pytest.raises(TimeoutError):
        await receiver.receive(MessageChannel.PAIR_COMMIT, timeout_seconds=0.01)

    current_round = 1
    await receiver.advance_round(1)
    received = [
        await receiver.receive(MessageChannel.PAIR_COMMIT, timeout_seconds=0.1)
        for _ in range(2)
    ]
    assert {envelope.sender_public_key for envelope in received} == {
        "peer-0",
        "peer-2",
    }
    assert all(envelope.message_id == "future-update" for envelope in received)
    with pytest.raises(TimeoutError):
        await receiver.receive(MessageChannel.PAIR_COMMIT, timeout_seconds=0.01)
    await receiver.stop()


def test_future_round_transfer_sequence_is_buffered_until_round_advances() -> None:
    asyncio.run(_test_future_round_transfer_sequence_is_buffered())


async def _test_future_round_transfer_sequence_is_buffered() -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    network = InMemoryNetwork()
    sender_transport = InMemoryTransport(network=network, public_key="peer-0")
    receiver_transport = InMemoryTransport(network=network, public_key="peer-1")
    current_round = 0
    receiver = Receiver(
        receiver_transport,
        ReceiverPolicy(
            run_id=manifest.run_id,
            manifest_hash=manifest.draft_hash,
            algorithm_id=manifest.algorithm_id,
            participant_keys=frozenset({"peer-0", "peer-1"}),
            current_round=lambda: current_round,
        ),
    )
    envelopes = [
        create_envelope(
            message_type=message_type,
            message_id=f"future-transfer-{index}",
            correlation_id="future-transfer",
            run_id=manifest.run_id,
            manifest_hash=manifest.draft_hash,
            sender_public_key="peer-0",
            algorithm_id=manifest.algorithm_id,
            round_id=1,
            payload=b"transfer",
        )
        for index, message_type in enumerate(
            (
                MessageType.TRANSFER_BEGIN,
                MessageType.CHUNK,
                MessageType.TRANSFER_COMPLETE,
            )
        )
    ]
    await receiver.start()
    try:
        for envelope in envelopes:
            await sender_transport.send("peer-1", encode_envelope(envelope))
        await asyncio.sleep(0.05)
        assert receiver.stats.rejected_messages == 0
        second_update = create_envelope(
            message_type=MessageType.TRANSFER_BEGIN,
            message_id="second-future-transfer",
            correlation_id="second-future-transfer",
            run_id=manifest.run_id,
            manifest_hash=manifest.draft_hash,
            sender_public_key="peer-0",
            algorithm_id=manifest.algorithm_id,
            round_id=1,
            payload=b"second-transfer",
        )
        await sender_transport.send("peer-1", encode_envelope(second_update))
        await asyncio.sleep(0.05)
        assert receiver.stats.rejected_messages == 1

        current_round = 1
        await receiver.advance_round(1)
        received = [
            await receiver.receive(
                MessageChannel.TRANSFER,
                timeout_seconds=0.1,
            )
            for _ in envelopes
        ]
        assert [envelope.message_id for envelope in received] == [
            envelope.message_id for envelope in envelopes
        ]
    finally:
        await receiver.stop()
