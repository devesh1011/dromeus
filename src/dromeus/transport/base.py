"""Transport interface and deterministic in-memory adapter."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from dromeus.manifests.models import PublicKey


class TransportError(RuntimeError):
    """The underlying byte transport failed."""


@dataclass(frozen=True)
class ReceivedBytes:
    """One authenticated inbound byte payload from the transport."""

    sender_public_key: PublicKey
    payload: bytes


class AsyncTransport(Protocol):
    """Small async seam used by the receiver and sender."""

    async def local_public_key(self) -> PublicKey: ...

    async def send(self, destination: PublicKey, payload: bytes) -> None: ...

    async def recv(self, timeout_seconds: float) -> ReceivedBytes | None: ...


@dataclass(frozen=True)
class InMemoryFaults:
    """Deterministic fault injection keyed by send-call index."""

    drop_send_calls: frozenset[int] = frozenset()
    duplicate_send_calls: frozenset[int] = frozenset()
    corrupt_send_calls: frozenset[int] = frozenset()
    delay_send_calls: Mapping[int, float] = field(default_factory=lambda: {})


class InMemoryNetwork:
    """Shared message router for tests."""

    def __init__(self) -> None:
        self._queues: dict[PublicKey, asyncio.Queue[ReceivedBytes]] = {}

    def register(self, public_key: PublicKey) -> asyncio.Queue[ReceivedBytes]:
        queue = asyncio.Queue[ReceivedBytes]()
        self._queues[public_key] = queue
        return queue

    async def deliver(self, destination: PublicKey, message: ReceivedBytes) -> None:
        queue = self._queues.get(destination)
        if queue is None:
            raise TransportError(f"unknown destination peer: {destination}")
        await queue.put(message)


class InMemoryTransport:
    """Test-only adapter with deterministic drop/dup/corrupt/delay hooks."""

    def __init__(
        self,
        *,
        network: InMemoryNetwork,
        public_key: PublicKey,
        faults: InMemoryFaults | None = None,
    ) -> None:
        self._public_key = public_key
        self._network = network
        self._queue = network.register(public_key)
        self._faults = faults or InMemoryFaults()
        self._send_calls = 0

    async def local_public_key(self) -> PublicKey:
        return self._public_key

    async def send(self, destination: PublicKey, payload: bytes) -> None:
        self._send_calls += 1
        call_index = self._send_calls
        delay = self._faults.delay_send_calls.get(call_index)
        if delay is not None:
            await asyncio.sleep(delay)
        if call_index in self._faults.drop_send_calls:
            return
        delivered = payload
        if call_index in self._faults.corrupt_send_calls and payload:
            delivered = payload[:-1] + bytes([payload[-1] ^ 0xFF])
        message = ReceivedBytes(self._public_key, delivered)
        await self._network.deliver(destination, message)
        if call_index in self._faults.duplicate_send_calls:
            await self._network.deliver(destination, message)

    async def recv(self, timeout_seconds: float) -> ReceivedBytes | None:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout_seconds)
        except TimeoutError:
            return None
