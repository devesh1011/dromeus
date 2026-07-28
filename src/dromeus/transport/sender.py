"""Outbound message scheduling."""

from __future__ import annotations

import asyncio
import heapq
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
    admission: asyncio.Semaphore = field(compare=False)
    retry_count: int = field(default=0, compare=False)
    first_attempt_at: float | None = field(default=None, compare=False)
    send_seconds: float = field(default=0.0, compare=False)

    @property
    def is_control(self) -> bool:
        return self.priority <= Priority.CONTROL


class OutboundScheduler:
    """Bounded priority sender with independent per-peer control and bulk lanes."""

    def __init__(
        self,
        transport: AsyncTransport,
        *,
        max_queue_size: int = 256,
        max_control_queue_size: int = 64,
        per_peer_in_flight: int = 1,
    ) -> None:
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        if max_control_queue_size <= 0:
            raise ValueError("max_control_queue_size must be positive")
        if per_peer_in_flight <= 0:
            raise ValueError("per_peer_in_flight must be positive")
        self._transport = transport
        self._queue = asyncio.PriorityQueue[_ScheduledSend]()
        self._control_admission = asyncio.Semaphore(max_control_queue_size)
        self._bulk_admission = asyncio.Semaphore(max_queue_size)
        self._per_peer_limit = per_peer_in_flight
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._wake = asyncio.Event()
        self._sequence = 0
        self._delay_sequence = 0
        self._delayed: list[tuple[float, int, _ScheduledSend]] = []
        self._peer_control_active: set[PublicKey] = set()
        self._peer_bulk_active: dict[PublicKey, int] = {}
        self._active_tasks: set[asyncio.Task[None]] = set()
        self._admission_waiters: set[asyncio.Future[None]] = set()

    async def start(self) -> None:
        if self._stop.is_set():
            raise RuntimeError("outbound scheduler is stopped")
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="dromeus-send")

    async def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._admission_waiters:
            await asyncio.gather(*tuple(self._admission_waiters))
        if self._task is not None:
            await self._task
        else:
            self._fail_queued()
        if self._active_tasks:
            await asyncio.gather(*tuple(self._active_tasks))

    async def send(
        self,
        destination: PublicKey,
        payload: bytes,
        *,
        priority: Priority,
        retries: int = 0,
        retry_delay_seconds: float = 0.1,
    ) -> SendTiming:
        if self._stop.is_set():
            raise RuntimeError("outbound scheduler is stopped")
        loop = asyncio.get_running_loop()
        admission = (
            self._control_admission
            if priority <= Priority.CONTROL
            else self._bulk_admission
        )
        if admission.locked():
            admission_waiter: asyncio.Future[None] = loop.create_future()
            self._admission_waiters.add(admission_waiter)
            try:
                await self._acquire_admission(admission)
            finally:
                self._admission_waiters.discard(admission_waiter)
                admission_waiter.set_result(None)
        else:
            await admission.acquire()
        enqueued_at = time.monotonic()
        future: asyncio.Future[SendTiming] = loop.create_future()
        self._sequence += 1
        scheduled = _ScheduledSend(
            priority=int(priority),
            sequence=self._sequence,
            destination=destination,
            payload=payload,
            retries=retries,
            retry_delay_seconds=retry_delay_seconds,
            result=future,
            enqueued_at=enqueued_at,
            admission=admission,
        )
        future.add_done_callback(lambda _: self._wake.set())
        await self._queue.put(scheduled)
        self._wake.set()
        return await future

    async def _acquire_admission(self, admission: asyncio.Semaphore) -> None:
        acquire = asyncio.create_task(admission.acquire())
        stopping = asyncio.create_task(self._stop.wait())
        acquired = False
        try:
            done, _ = await asyncio.wait(
                (acquire, stopping), return_when=asyncio.FIRST_COMPLETED
            )
            if stopping in done:
                raise RuntimeError("outbound scheduler is stopped")
            acquired = acquire.result()
            if self._stop.is_set():
                raise RuntimeError("outbound scheduler is stopped")
        except BaseException:
            if (
                not acquired
                and acquire.done()
                and not acquire.cancelled()
                and acquire.exception() is None
            ):
                acquired = acquire.result()
            if acquired:
                admission.release()
            raise
        finally:
            for task in (acquire, stopping):
                if not task.done():
                    task.cancel()

    async def _run(self) -> None:
        while True:
            self._wake.clear()
            if self._stop.is_set():
                self._fail_queued()
                return
            self._promote_delayed()
            self._dispatch_ready()
            timeout = (
                max(0.0, self._delayed[0][0] - time.monotonic())
                if self._delayed
                else None
            )
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=timeout)
            except TimeoutError:
                pass

    def _fail_queued(self) -> None:
        while not self._queue.empty():
            scheduled = self._queue.get_nowait()
            self._queue.task_done()
            if not scheduled.result.done():
                scheduled.result.set_exception(
                    RuntimeError("outbound scheduler is stopped")
                )
            scheduled.admission.release()
        while self._delayed:
            _, _, scheduled = heapq.heappop(self._delayed)
            if not scheduled.result.done():
                scheduled.result.set_exception(
                    RuntimeError("outbound scheduler is stopped")
                )
            scheduled.admission.release()

    def _promote_delayed(self) -> None:
        pending: list[tuple[float, int, _ScheduledSend]] = []
        now = time.monotonic()
        while self._delayed:
            ready_at, delay_sequence, scheduled = heapq.heappop(self._delayed)
            if scheduled.result.done():
                scheduled.admission.release()
            elif ready_at <= now:
                self._sequence += 1
                scheduled.sequence = self._sequence
                self._queue.put_nowait(scheduled)
            else:
                pending.append((ready_at, delay_sequence, scheduled))
        self._delayed = pending
        heapq.heapify(self._delayed)

    def _dispatch_ready(self) -> None:
        blocked: list[_ScheduledSend] = []
        for _ in range(self._queue.qsize()):
            scheduled = self._queue.get_nowait()
            self._queue.task_done()
            if scheduled.result.done():
                scheduled.admission.release()
                continue
            if not self._lane_available(scheduled):
                blocked.append(scheduled)
                continue
            self._reserve_lane(scheduled)
            task = asyncio.create_task(
                self._run_attempt(scheduled), name="dromeus-send-attempt"
            )
            self._active_tasks.add(task)
            task.add_done_callback(self._active_tasks.discard)
        for scheduled in blocked:
            self._queue.put_nowait(scheduled)

    def _lane_available(self, scheduled: _ScheduledSend) -> bool:
        if scheduled.is_control:
            return scheduled.destination not in self._peer_control_active
        return (
            self._peer_bulk_active.get(scheduled.destination, 0)
            < self._per_peer_limit
        )

    def _reserve_lane(self, scheduled: _ScheduledSend) -> None:
        if scheduled.is_control:
            self._peer_control_active.add(scheduled.destination)
            return
        self._peer_bulk_active[scheduled.destination] = (
            self._peer_bulk_active.get(scheduled.destination, 0) + 1
        )

    def _release_lane(self, scheduled: _ScheduledSend) -> None:
        if scheduled.is_control:
            self._peer_control_active.remove(scheduled.destination)
        else:
            remaining = self._peer_bulk_active[scheduled.destination] - 1
            if remaining:
                self._peer_bulk_active[scheduled.destination] = remaining
            else:
                del self._peer_bulk_active[scheduled.destination]
        self._wake.set()

    async def _run_attempt(self, scheduled: _ScheduledSend) -> None:
        try:
            should_retry = await self._attempt(scheduled)
        finally:
            self._release_lane(scheduled)
        if should_retry:
            if not self._stop.is_set() and not scheduled.result.done():
                self._delay_sequence += 1
                heapq.heappush(
                    self._delayed,
                    (
                        time.monotonic() + scheduled.retry_delay_seconds,
                        self._delay_sequence,
                        scheduled,
                    ),
                )
                self._wake.set()
                return
            if not scheduled.result.done():
                scheduled.result.set_exception(
                    RuntimeError("outbound scheduler is stopped")
                )
        scheduled.admission.release()

    async def _attempt(self, scheduled: _ScheduledSend) -> bool:
        if scheduled.result.done():
            return False
        send_started = time.monotonic()
        if scheduled.first_attempt_at is None:
            scheduled.first_attempt_at = send_started
        try:
            await self._transport.send(scheduled.destination, scheduled.payload)
        except Exception as error:
            scheduled.send_seconds += time.monotonic() - send_started
            if scheduled.result.done():
                return False
            if self._stop.is_set() or scheduled.retry_count >= scheduled.retries:
                scheduled.result.set_exception(error)
                return False
            scheduled.retry_count += 1
            return True
        completed = time.monotonic()
        scheduled.send_seconds += completed - send_started
        if not scheduled.result.done():
            scheduled.result.set_result(
                SendTiming(
                    queue_seconds=scheduled.first_attempt_at - scheduled.enqueued_at,
                    send_seconds=scheduled.send_seconds,
                    retry_count=scheduled.retry_count,
                    completion_seconds=completed - scheduled.enqueued_at,
                )
            )
        return False
