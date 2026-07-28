"""Outbound message scheduling."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import IntEnum

from dromeus.manifests.models import PublicKey
from dromeus.transport.base import AsyncTransport


class Priority(IntEnum):
    ACK = 0
    CONTROL = 10
    DATA = 20
    TELEMETRY = 30


@dataclass(frozen=True)
class SendTiming:
    queue_seconds: float
    send_seconds: float
    retry_count: int
    completion_seconds: float


@dataclass(order=True)
class _ScheduledSend:
    priority: int
    sequence: int
    destination: PublicKey = field(compare=False)
    payload: bytes = field(compare=False)
    retries: int = field(compare=False)
    retry_delay_seconds: float = field(compare=False)
    result: asyncio.Future[SendTiming] = field(compare=False)
    enqueued_at: float = field(compare=False)


class OutboundScheduler:
    """Single bounded sender with per-peer in-flight limits."""

    def __init__(
        self,
        transport: AsyncTransport,
        *,
        max_queue_size: int = 256,
        per_peer_in_flight: int = 1,
    ) -> None:
        self._transport = transport
        self._queue = asyncio.PriorityQueue[_ScheduledSend](maxsize=max_queue_size)
        self._per_peer_limit = per_peer_in_flight
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._sequence = 0
        self._peer_semaphores: dict[PublicKey, asyncio.Semaphore] = {}

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="dromeus-send")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task

    async def send(
        self,
        destination: PublicKey,
        payload: bytes,
        *,
        priority: Priority,
        retries: int = 0,
        retry_delay_seconds: float = 0.1,
    ) -> SendTiming:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[SendTiming] = loop.create_future()
        self._sequence += 1
        await self._queue.put(
            _ScheduledSend(
                priority=int(priority),
                sequence=self._sequence,
                destination=destination,
                payload=payload,
                retries=retries,
                retry_delay_seconds=retry_delay_seconds,
                result=future,
                enqueued_at=time.monotonic(),
            )
        )
        return await future

    async def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                scheduled = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            except TimeoutError:
                continue
            semaphore = self._peer_semaphores.setdefault(
                scheduled.destination, asyncio.Semaphore(self._per_peer_limit)
            )
            async with semaphore:
                await self._deliver(scheduled)
            self._queue.task_done()

    async def _deliver(self, scheduled: _ScheduledSend) -> None:
        started = time.monotonic()
        retry_count = 0
        while True:
            if scheduled.result.done():
                return
            send_started = time.monotonic()
            try:
                await self._transport.send(scheduled.destination, scheduled.payload)
            except Exception as error:
                if scheduled.result.done():
                    return
                if retry_count >= scheduled.retries:
                    scheduled.result.set_exception(error)
                    return
                retry_count += 1
                await asyncio.sleep(scheduled.retry_delay_seconds)
                continue
            completed = time.monotonic()
            if not scheduled.result.done():
                scheduled.result.set_result(
                    SendTiming(
                        queue_seconds=send_started - scheduled.enqueued_at,
                        send_seconds=completed - send_started,
                        retry_count=retry_count,
                        completion_seconds=completed - started,
                    )
                )
            return
