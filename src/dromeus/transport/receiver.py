"""Inbound message routing."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from dromeus.manifests.models import (
    AlgorithmId,
    MessageId,
    PublicKey,
    RoundId,
    RunId,
    Sha256,
)
from dromeus.protocol.codec import EnvelopeError, decode_envelope
from dromeus.protocol.models import Envelope, MessageType
from dromeus.telemetry.events import EventSink, emit_event
from dromeus.transport.interface import AsyncTransport, TransportError

CONTROL_TYPES = {
    MessageType.JOIN_REQUEST,
    MessageType.JOIN_ACCEPTED,
    MessageType.MANIFEST_SEALED,
    MessageType.READY,
    MessageType.START,
    MessageType.START_ACK,
    MessageType.RUN_FAILED,
    MessageType.RUN_COMPLETE,
}
TRANSFER_TYPES = {
    MessageType.TRANSFER_BEGIN,
    MessageType.CHUNK,
    MessageType.TRANSFER_COMPLETE,
}
ACK_TYPES = {MessageType.CHUNK_ACK}
TELEMETRY_TYPES = {MessageType.CONSENSUS_SKETCH}
ROUND_TRACKED_TYPES = TRANSFER_TYPES | {
    MessageType.UPDATE_READY,
    MessageType.ROUND_COMMITTED,
    MessageType.CONSENSUS_SKETCH,
}
IDEMPOTENT_TYPES = {MessageType.UPDATE_READY, MessageType.ROUND_COMMITTED}
MAX_FUTURE_ROUND_MESSAGES = 64
MAX_TELEMETRY_ROUND_AGE = 64


class ReceiverError(RuntimeError):
    """Inbound traffic violated routing policy."""


class MessageChannel(StrEnum):
    CONTROL = "control"
    TRANSFER = "transfer"
    ACKNOWLEDGMENT = "acknowledgment"
    TELEMETRY = "telemetry"
    PAIR_COMMIT = "pair_commit"


@dataclass
class ReceiverPolicy:
    run_id: RunId | None = None
    manifest_hash: Sha256 | None = None
    algorithm_id: AlgorithmId | None = None
    participant_keys: frozenset[PublicKey] | None = None
    current_round: Callable[[], int] = lambda: 0
    max_payload_bytes: int = 8 * 1024 * 1024


@dataclass
class ReceiverStats:
    accepted_messages: int = 0
    rejected_messages: int = 0


@dataclass
class Receiver:
    transport: AsyncTransport
    _policy: ReceiverPolicy = field(default_factory=ReceiverPolicy)
    stats: ReceiverStats = field(default_factory=ReceiverStats)
    event_sink: EventSink | None = None

    def __post_init__(self) -> None:
        self._queues = {
            channel: asyncio.Queue[Envelope](maxsize=64) for channel in MessageChannel
        }
        self._seen_messages: set[MessageId] = set()
        self._future_round_messages: list[Envelope] = []
        self._future_round_transfer_by_sender: dict[PublicKey, MessageId] = {}
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="dromeus-recv")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                inbound = await self.transport.recv(timeout_seconds=0.1)
            except TransportError as error:
                emit_event(
                    "transport_receive_failed",
                    run_id=self._policy.run_id,
                    error=str(error),
                    sink=self.event_sink,
                )
                continue
            if inbound is None:
                continue
            try:
                envelope = decode_envelope(
                    inbound.payload,
                    authenticated_sender=inbound.sender_public_key,
                    participant_keys=self._policy.participant_keys,
                    max_payload_bytes=self._policy.max_payload_bytes,
                )
                should_route = self._validate_envelope(envelope)
            except (EnvelopeError, ReceiverError, TransportError) as error:
                self.stats.rejected_messages += 1
                emit_event(
                    "message_rejected",
                    run_id=self._policy.run_id,
                    peer_id=inbound.sender_public_key,
                    error=str(error),
                    sink=self.event_sink,
                )
                continue
            if should_route:
                await self._route(envelope)
            self.stats.accepted_messages += 1
            emit_event(
                "message_received",
                run_id=envelope.run_id,
                message_id=envelope.message_id,
                transfer_id=envelope.correlation_id,
                peer_id=envelope.sender_public_key,
                round_id=envelope.round_id,
                message_type=envelope.message_type,
                routed=should_route,
                sink=self.event_sink,
            )

    def _validate_envelope(self, envelope: Envelope) -> bool:
        if self._policy.run_id is not None and envelope.run_id != self._policy.run_id:
            raise ReceiverError("unexpected run id")
        if (
            self._policy.manifest_hash is not None
            and envelope.manifest_hash != self._policy.manifest_hash
        ):
            raise ReceiverError("unexpected manifest hash")
        if (
            self._policy.algorithm_id is not None
            and envelope.algorithm_id != self._policy.algorithm_id
        ):
            raise ReceiverError("unexpected algorithm id")
        if envelope.message_type not in TRANSFER_TYPES | ACK_TYPES | IDEMPOTENT_TYPES:
            if envelope.message_id in self._seen_messages:
                raise ReceiverError("replayed message id")
            self._seen_messages.add(envelope.message_id)
        if (
            envelope.message_type in ROUND_TRACKED_TYPES
            and envelope.round_id is not None
        ):
            return self._validate_round_window(envelope)
        return True

    def _validate_round_window(self, envelope: Envelope) -> bool:
        current_round = self._policy.current_round()
        assert envelope.round_id is not None
        if envelope.message_type is MessageType.CONSENSUS_SKETCH:
            if envelope.round_id <= current_round:
                if current_round - envelope.round_id > MAX_TELEMETRY_ROUND_AGE:
                    raise ReceiverError("stale telemetry message")
                return True
        if envelope.round_id < current_round:
            raise ReceiverError("stale message")
        if envelope.round_id > current_round + 1:
            raise ReceiverError("message too far in future")
        if envelope.round_id == current_round + 1:
            if any(
                buffered.sender_public_key == envelope.sender_public_key
                and buffered.message_id == envelope.message_id
                for buffered in self._future_round_messages
            ):
                return False
            if len(self._future_round_messages) >= MAX_FUTURE_ROUND_MESSAGES:
                raise ReceiverError("future-round message buffer is full")
            if envelope.message_type in TRANSFER_TYPES:
                transfer_id = envelope.correlation_id
                if transfer_id is None:
                    raise ReceiverError("future-round transfer has no correlation id")
                existing_transfer = self._future_round_transfer_by_sender.get(
                    envelope.sender_public_key
                )
                if existing_transfer is not None and existing_transfer != transfer_id:
                    raise ReceiverError("too many future-round updates from sender")
                self._future_round_transfer_by_sender[envelope.sender_public_key] = (
                    transfer_id
                )
            self._future_round_messages.append(envelope)
            return False
        return True

    async def _route(self, envelope: Envelope) -> None:
        if envelope.message_type in CONTROL_TYPES:
            await self._queues[MessageChannel.CONTROL].put(envelope)
            return
        if envelope.message_type in TRANSFER_TYPES:
            await self._queues[MessageChannel.TRANSFER].put(envelope)
            return
        if envelope.message_type in ACK_TYPES:
            await self._queues[MessageChannel.ACKNOWLEDGMENT].put(envelope)
            return
        if envelope.message_type in TELEMETRY_TYPES:
            await self._queues[MessageChannel.TELEMETRY].put(envelope)
            return
        await self._queues[MessageChannel.PAIR_COMMIT].put(envelope)

    async def receive(
        self,
        channel: MessageChannel,
        *,
        timeout_seconds: float | None = None,
    ) -> Envelope:
        """Return the next validated message for a protocol channel."""
        queue = self._queues[channel]
        if timeout_seconds is None:
            return await queue.get()
        return await asyncio.wait_for(queue.get(), timeout=timeout_seconds)

    def configure_sealed_run(
        self,
        *,
        manifest_hash: Sha256,
        participant_keys: frozenset[PublicKey],
    ) -> None:
        self._policy.manifest_hash = manifest_hash
        self._policy.participant_keys = participant_keys

    def set_current_round(self, round_id: RoundId) -> None:
        """Advance round validation after a pair commit is durable."""
        self._policy.current_round = lambda: round_id

    async def advance_round(self, next_round: RoundId) -> None:
        ready = [
            envelope
            for envelope in self._future_round_messages
            if envelope.round_id == next_round
        ]
        self._future_round_messages = [
            envelope
            for envelope in self._future_round_messages
            if envelope.round_id != next_round
        ]
        self._future_round_transfer_by_sender = {
            envelope.sender_public_key: envelope.correlation_id
            for envelope in self._future_round_messages
            if envelope.message_type in TRANSFER_TYPES
            and envelope.correlation_id is not None
        }
        for envelope in ready:
            await self._route(envelope)
