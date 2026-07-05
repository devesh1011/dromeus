"""Inbound message routing."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

from dromeus.manifests.models import (
    AlgorithmId,
    MessageId,
    PublicKey,
    RoundId,
    RunId,
    Sha256,
)
from dromeus.transport.base import AsyncTransport, TransportError
from dromeus.transport.envelope import (
    Envelope,
    EnvelopeError,
    MessageType,
    decode_envelope,
)

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
}


class ReceiverError(RuntimeError):
    """Inbound traffic violated routing policy."""


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
    policy: ReceiverPolicy = field(default_factory=ReceiverPolicy)
    control_queue: asyncio.Queue[Envelope] = field(
        default_factory=lambda: asyncio.Queue(maxsize=64)
    )
    transfer_queue: asyncio.Queue[Envelope] = field(
        default_factory=lambda: asyncio.Queue(maxsize=64)
    )
    acknowledgment_queue: asyncio.Queue[Envelope] = field(
        default_factory=lambda: asyncio.Queue(maxsize=64)
    )
    telemetry_queue: asyncio.Queue[Envelope] = field(
        default_factory=lambda: asyncio.Queue(maxsize=64)
    )
    pair_commit_queue: asyncio.Queue[Envelope] = field(
        default_factory=lambda: asyncio.Queue(maxsize=64)
    )
    stats: ReceiverStats = field(default_factory=ReceiverStats)

    def __post_init__(self) -> None:
        self._seen_messages: set[MessageId] = set()
        self._future_round_by_sender: dict[PublicKey, Envelope] = {}
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
            inbound = await self.transport.recv(timeout_seconds=0.1)
            if inbound is None:
                continue
            try:
                envelope = decode_envelope(
                    inbound.payload,
                    authenticated_sender=inbound.sender_public_key,
                    participant_keys=self.policy.participant_keys,
                    max_payload_bytes=self.policy.max_payload_bytes,
                )
                should_route = self._validate_envelope(envelope)
            except (EnvelopeError, ReceiverError, TransportError):
                self.stats.rejected_messages += 1
                continue
            if should_route:
                await self._route(envelope)
            self.stats.accepted_messages += 1

    def _validate_envelope(self, envelope: Envelope) -> bool:
        if self.policy.run_id is not None and envelope.run_id != self.policy.run_id:
            raise ReceiverError("unexpected run id")
        if (
            self.policy.manifest_hash is not None
            and envelope.manifest_hash != self.policy.manifest_hash
        ):
            raise ReceiverError("unexpected manifest hash")
        if (
            self.policy.algorithm_id is not None
            and envelope.algorithm_id != self.policy.algorithm_id
        ):
            raise ReceiverError("unexpected algorithm id")
        if envelope.message_type not in TRANSFER_TYPES | ACK_TYPES:
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
        current_round = self.policy.current_round()
        assert envelope.round_id is not None
        if envelope.round_id < current_round:
            raise ReceiverError("stale message")
        if envelope.round_id > current_round + 1:
            raise ReceiverError("message too far in future")
        if envelope.round_id == current_round + 1:
            existing = self._future_round_by_sender.get(envelope.sender_public_key)
            if existing is not None:
                raise ReceiverError("too many future-round messages from sender")
            self._future_round_by_sender[envelope.sender_public_key] = envelope
            return False
        return True

    async def _route(self, envelope: Envelope) -> None:
        if envelope.message_type in CONTROL_TYPES:
            await self.control_queue.put(envelope)
            return
        if envelope.message_type in TRANSFER_TYPES:
            await self.transfer_queue.put(envelope)
            return
        if envelope.message_type in ACK_TYPES:
            await self.acknowledgment_queue.put(envelope)
            return
        if envelope.message_type in TELEMETRY_TYPES:
            await self.telemetry_queue.put(envelope)
            return
        await self.pair_commit_queue.put(envelope)

    async def advance_round(self, next_round: RoundId) -> None:
        ready = [
            sender
            for sender, envelope in self._future_round_by_sender.items()
            if envelope.round_id == next_round
        ]
        for sender in ready:
            envelope = self._future_round_by_sender.pop(sender)
            await self._route(envelope)
