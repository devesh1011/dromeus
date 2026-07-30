from __future__ import annotations

import asyncio

import pytest

from dromeus.transport.outbound_scheduler import OutboundScheduler, Priority


class ControlledTransport:
    def __init__(self) -> None:
        self.data_started = asyncio.Event()
        self.release_data = asyncio.Event()
        self.sent: list[bytes] = []

    async def local_public_key(self) -> str:
        return "peer-0"

    async def send(self, destination: str, payload: bytes) -> None:
        self.sent.append(payload)
        if payload == b"data":
            self.data_started.set()
            await self.release_data.wait()

    async def recv(self, timeout_seconds: float) -> None:
        return None


class RetryTransport:
    def __init__(self) -> None:
        self.first_attempt_failed = asyncio.Event()
        self.sent: list[bytes] = []

    async def local_public_key(self) -> str:
        return "peer-0"

    async def send(self, destination: str, payload: bytes) -> None:
        self.sent.append(payload)
        if payload == b"retry" and self.sent.count(payload) == 1:
            self.first_attempt_failed.set()
            raise OSError("temporary failure")

    async def recv(self, timeout_seconds: float) -> None:
        return None


class FairRetryTransport:
    def __init__(self) -> None:
        self.first_attempt_failed = asyncio.Event()
        self.block_started = asyncio.Event()
        self.release_block = asyncio.Event()
        self.sent: list[bytes] = []

    async def local_public_key(self) -> str:
        return "peer-0"

    async def send(self, destination: str, payload: bytes) -> None:
        self.sent.append(payload)
        if payload == b"retry" and self.sent.count(payload) == 1:
            self.first_attempt_failed.set()
            raise OSError("temporary failure")
        if payload == b"block":
            self.block_started.set()
            await self.release_block.wait()

    async def recv(self, timeout_seconds: float) -> None:
        return None


class PriorityTransport:
    def __init__(self) -> None:
        self.block_started = asyncio.Event()
        self.release_block = asyncio.Event()
        self.sent: list[bytes] = []

    async def local_public_key(self) -> str:
        return "peer-0"

    async def send(self, destination: str, payload: bytes) -> None:
        self.sent.append(payload)
        if payload == b"block":
            self.block_started.set()
            await self.release_block.wait()

    async def recv(self, timeout_seconds: float) -> None:
        return None


class TimedRetryTransport:
    def __init__(self) -> None:
        self.attempts = 0

    async def local_public_key(self) -> str:
        return "peer-0"

    async def send(self, destination: str, payload: bytes) -> None:
        self.attempts += 1
        await asyncio.sleep(0.02)
        if self.attempts == 1:
            raise OSError("temporary failure")

    async def recv(self, timeout_seconds: float) -> None:
        return None


class BlockingFailingTransport:
    def __init__(self) -> None:
        self.attempt_started = asyncio.Event()
        self.release_attempt = asyncio.Event()
        self.failed_attempts = 0

    async def local_public_key(self) -> str:
        return "peer-0"

    async def send(self, destination: str, payload: bytes) -> None:
        if payload != b"payload":
            return
        self.failed_attempts += 1
        self.attempt_started.set()
        await self.release_attempt.wait()
        raise OSError("send failed")

    async def recv(self, timeout_seconds: float) -> None:
        return None


class ConcurrencyTransport:
    def __init__(self) -> None:
        self.two_started = asyncio.Event()
        self.release = asyncio.Event()
        self.active = 0
        self.max_active = 0

    async def local_public_key(self) -> str:
        return "peer-0"

    async def send(self, destination: str, payload: bytes) -> None:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.active == 2:
            self.two_started.set()
        await self.release.wait()
        self.active -= 1

    async def recv(self, timeout_seconds: float) -> None:
        return None


def test_scheduler_rejects_nonpositive_bounds() -> None:
    transport = ControlledTransport()
    with pytest.raises(ValueError, match="max_queue_size"):
        OutboundScheduler(transport, max_queue_size=0)
    with pytest.raises(ValueError, match="max_control_queue_size"):
        OutboundScheduler(transport, max_control_queue_size=0)
    with pytest.raises(ValueError, match="per_peer_in_flight"):
        OutboundScheduler(transport, per_peer_in_flight=0)


def test_control_send_progresses_while_same_peer_data_is_active() -> None:
    asyncio.run(_test_control_send_progresses_while_same_peer_data_is_active())


async def _test_control_send_progresses_while_same_peer_data_is_active() -> None:
    transport = ControlledTransport()
    scheduler = OutboundScheduler(transport)
    await scheduler.start()
    data = asyncio.create_task(
        scheduler.send("peer-1", b"data", priority=Priority.DATA)
    )
    await transport.data_started.wait()

    control = asyncio.create_task(
        scheduler.send("peer-1", b"control", priority=Priority.CONTROL)
    )
    await asyncio.wait_for(control, timeout=0.1)

    assert transport.sent == [b"data", b"control"]

    transport.release_data.set()
    await data
    await scheduler.stop()


def test_bulk_sends_to_different_peers_progress_independently() -> None:
    asyncio.run(_test_bulk_sends_to_different_peers_progress_independently())


async def _test_bulk_sends_to_different_peers_progress_independently() -> None:
    transport = ControlledTransport()
    scheduler = OutboundScheduler(transport)
    await scheduler.start()
    blocked = asyncio.create_task(
        scheduler.send("peer-1", b"data", priority=Priority.DATA)
    )
    await transport.data_started.wait()

    await asyncio.wait_for(
        scheduler.send("peer-2", b"other", priority=Priority.DATA),
        timeout=0.1,
    )

    transport.release_data.set()
    await blocked
    await scheduler.stop()


def test_bulk_lane_honors_configured_per_peer_limit() -> None:
    asyncio.run(_test_bulk_lane_honors_configured_per_peer_limit())


async def _test_bulk_lane_honors_configured_per_peer_limit() -> None:
    transport = ConcurrencyTransport()
    scheduler = OutboundScheduler(transport, per_peer_in_flight=2)
    await scheduler.start()
    sends = [
        asyncio.create_task(
            scheduler.send("peer-1", bytes([index]), priority=Priority.DATA)
        )
        for index in range(3)
    ]
    await transport.two_started.wait()

    assert transport.active == 2
    assert transport.max_active == 2

    transport.release.set()
    await asyncio.gather(*sends)
    assert transport.max_active == 2
    await scheduler.stop()


def test_control_send_is_admitted_while_bulk_capacity_is_full() -> None:
    asyncio.run(_test_control_send_is_admitted_while_bulk_capacity_is_full())


async def _test_control_send_is_admitted_while_bulk_capacity_is_full() -> None:
    transport = ControlledTransport()
    scheduler = OutboundScheduler(
        transport, max_queue_size=1, max_control_queue_size=1
    )
    await scheduler.start()
    data = asyncio.create_task(
        scheduler.send("peer-1", b"data", priority=Priority.DATA)
    )
    await transport.data_started.wait()

    waiting_data = asyncio.create_task(
        scheduler.send("peer-2", b"waiting-data", priority=Priority.DATA)
    )
    await asyncio.sleep(0)
    assert not waiting_data.done()

    await asyncio.wait_for(
        scheduler.send("peer-1", b"control", priority=Priority.CONTROL),
        timeout=0.1,
    )

    transport.release_data.set()
    await data
    await waiting_data
    await scheduler.stop()


def test_control_admission_applies_backpressure_across_peers() -> None:
    asyncio.run(_test_control_admission_applies_backpressure_across_peers())


async def _test_control_admission_applies_backpressure_across_peers() -> None:
    transport = PriorityTransport()
    scheduler = OutboundScheduler(transport, max_control_queue_size=1)
    await scheduler.start()
    blocking = asyncio.create_task(
        scheduler.send("peer-1", b"block", priority=Priority.CONTROL)
    )
    await transport.block_started.wait()

    waiting = asyncio.create_task(
        scheduler.send("peer-2", b"control", priority=Priority.CONTROL)
    )
    await asyncio.sleep(0)
    assert transport.sent == [b"block"]
    assert not waiting.done()

    transport.release_block.set()
    await blocking
    await waiting
    await scheduler.stop()


def test_retry_backoff_releases_lane_and_requeues_behind_ready_work() -> None:
    asyncio.run(_test_retry_backoff_releases_lane_and_requeues_behind_ready_work())


async def _test_retry_backoff_releases_lane_and_requeues_behind_ready_work() -> None:
    transport = FairRetryTransport()
    scheduler = OutboundScheduler(transport)
    await scheduler.start()
    retrying = asyncio.create_task(
        scheduler.send(
            "peer-1",
            b"retry",
            priority=Priority.DATA,
            retries=1,
            retry_delay_seconds=0.01,
        )
    )
    await transport.first_attempt_failed.wait()

    blocking = asyncio.create_task(
        scheduler.send("peer-1", b"block", priority=Priority.DATA)
    )
    await transport.block_started.wait()
    older_ready = asyncio.create_task(
        scheduler.send("peer-1", b"older-ready", priority=Priority.DATA)
    )
    await asyncio.sleep(0.02)
    transport.release_block.set()

    await blocking
    await older_ready
    timing = await retrying

    assert transport.sent == [b"retry", b"block", b"older-ready", b"retry"]
    assert timing.retry_count == 1
    await scheduler.stop()


def test_terminal_failure_does_not_stop_unrelated_sends() -> None:
    asyncio.run(_test_terminal_failure_does_not_stop_unrelated_sends())


async def _test_terminal_failure_does_not_stop_unrelated_sends() -> None:
    transport = RetryTransport()
    scheduler = OutboundScheduler(transport)
    await scheduler.start()

    with pytest.raises(OSError, match="temporary failure"):
        await scheduler.send("peer-1", b"retry", priority=Priority.DATA)
    await scheduler.send("peer-2", b"fresh", priority=Priority.DATA)

    assert transport.sent == [b"retry", b"fresh"]
    await scheduler.stop()


def test_send_timing_includes_all_attempts_and_retry_backoff() -> None:
    asyncio.run(_test_send_timing_includes_all_attempts_and_retry_backoff())


async def _test_send_timing_includes_all_attempts_and_retry_backoff() -> None:
    scheduler = OutboundScheduler(TimedRetryTransport())
    await scheduler.start()

    timing = await scheduler.send(
        "peer-1",
        b"payload",
        priority=Priority.DATA,
        retries=1,
        retry_delay_seconds=0.02,
    )

    assert timing.queue_seconds >= 0
    assert timing.send_seconds >= 0.03
    assert timing.completion_seconds >= (
        timing.queue_seconds + timing.send_seconds + 0.01
    )
    assert timing.retry_count == 1
    await scheduler.stop()


def test_queue_timing_includes_wait_for_peer_lane() -> None:
    asyncio.run(_test_queue_timing_includes_wait_for_peer_lane())


async def _test_queue_timing_includes_wait_for_peer_lane() -> None:
    transport = ControlledTransport()
    scheduler = OutboundScheduler(transport)
    await scheduler.start()
    blocked = asyncio.create_task(
        scheduler.send("peer-1", b"data", priority=Priority.DATA)
    )
    await transport.data_started.wait()
    queued = asyncio.create_task(
        scheduler.send("peer-1", b"queued", priority=Priority.DATA)
    )
    await asyncio.sleep(0.02)

    transport.release_data.set()
    timing = await queued

    assert timing.queue_seconds >= 0.015
    await blocked
    await scheduler.stop()


def test_ack_overtakes_control_waiting_for_same_peer_lane() -> None:
    asyncio.run(_test_ack_overtakes_control_waiting_for_same_peer_lane())


async def _test_ack_overtakes_control_waiting_for_same_peer_lane() -> None:
    transport = PriorityTransport()
    scheduler = OutboundScheduler(transport)
    await scheduler.start()
    blocking = asyncio.create_task(
        scheduler.send("peer-1", b"block", priority=Priority.CONTROL)
    )
    await transport.block_started.wait()

    control = asyncio.create_task(
        scheduler.send("peer-1", b"control", priority=Priority.CONTROL)
    )
    await asyncio.sleep(0)
    ack = asyncio.create_task(
        scheduler.send("peer-1", b"ack", priority=Priority.ACK)
    )
    await asyncio.sleep(0)
    transport.release_block.set()

    await asyncio.gather(blocking, control, ack)
    assert transport.sent == [b"block", b"ack", b"control"]
    await scheduler.stop()


def test_data_overtakes_telemetry_and_preserves_fifo() -> None:
    asyncio.run(_test_data_overtakes_telemetry_and_preserves_fifo())


async def _test_data_overtakes_telemetry_and_preserves_fifo() -> None:
    transport = PriorityTransport()
    scheduler = OutboundScheduler(transport)
    await scheduler.start()
    blocking = asyncio.create_task(
        scheduler.send("peer-1", b"block", priority=Priority.DATA)
    )
    await transport.block_started.wait()

    telemetry = asyncio.create_task(
        scheduler.send("peer-1", b"telemetry", priority=Priority.TELEMETRY)
    )
    await asyncio.sleep(0)
    first_data = asyncio.create_task(
        scheduler.send("peer-1", b"data-1", priority=Priority.DATA)
    )
    second_data = asyncio.create_task(
        scheduler.send("peer-1", b"data-2", priority=Priority.DATA)
    )
    await asyncio.sleep(0)
    transport.release_block.set()

    await asyncio.gather(blocking, telemetry, first_data, second_data)
    assert transport.sent == [b"block", b"data-1", b"data-2", b"telemetry"]
    await scheduler.stop()


def test_stop_fails_queued_and_new_sends_but_finishes_active_attempt() -> None:
    asyncio.run(_test_stop_fails_queued_and_new_sends_but_finishes_active_attempt())


async def _test_stop_fails_queued_and_new_sends_but_finishes_active_attempt() -> None:
    transport = ControlledTransport()
    scheduler = OutboundScheduler(transport)
    await scheduler.start()
    active = asyncio.create_task(
        scheduler.send("peer-1", b"data", priority=Priority.DATA)
    )
    await transport.data_started.wait()
    queued = asyncio.create_task(
        scheduler.send("peer-1", b"queued", priority=Priority.DATA)
    )
    await asyncio.sleep(0)

    stopping = asyncio.create_task(scheduler.stop())
    try:
        with pytest.raises(RuntimeError, match="stopped"):
            await asyncio.wait_for(asyncio.shield(queued), timeout=0.1)
        with pytest.raises(RuntimeError, match="stopped"):
            await asyncio.wait_for(
                scheduler.send("peer-2", b"new", priority=Priority.DATA),
                timeout=0.1,
            )
        assert not active.done()
        assert not stopping.done()
    finally:
        transport.release_data.set()
        await active
        await stopping


def test_stop_releases_all_sends_waiting_for_admission() -> None:
    asyncio.run(_test_stop_releases_all_sends_waiting_for_admission())


async def _test_stop_releases_all_sends_waiting_for_admission() -> None:
    transport = ControlledTransport()
    scheduler = OutboundScheduler(transport, max_queue_size=1)
    await scheduler.start()
    active = asyncio.create_task(
        scheduler.send("peer-1", b"data", priority=Priority.DATA)
    )
    await transport.data_started.wait()
    waiters = [
        asyncio.create_task(
            scheduler.send(f"peer-{index}", b"waiting", priority=Priority.DATA)
        )
        for index in (2, 3)
    ]
    await asyncio.sleep(0)

    stopping = asyncio.create_task(scheduler.stop())
    for waiter in waiters:
        with pytest.raises(RuntimeError, match="stopped"):
            await asyncio.wait_for(waiter, timeout=0.1)
    assert not stopping.done()

    transport.release_data.set()
    await active
    await stopping


def test_cancel_queued_send_releases_admission_immediately() -> None:
    asyncio.run(_test_cancel_queued_send_releases_admission_immediately())


async def _test_cancel_queued_send_releases_admission_immediately() -> None:
    transport = ControlledTransport()
    scheduler = OutboundScheduler(transport, max_queue_size=2)
    await scheduler.start()
    active = asyncio.create_task(
        scheduler.send("peer-1", b"data", priority=Priority.DATA)
    )
    await transport.data_started.wait()
    queued = asyncio.create_task(
        scheduler.send("peer-1", b"queued", priority=Priority.DATA)
    )
    await asyncio.sleep(0)

    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    await asyncio.wait_for(
        scheduler.send("peer-2", b"fresh", priority=Priority.DATA),
        timeout=0.1,
    )

    transport.release_data.set()
    await active
    await scheduler.stop()


def test_cancel_during_retry_backoff_releases_admission_immediately() -> None:
    asyncio.run(_test_cancel_during_retry_backoff_releases_admission_immediately())


async def _test_cancel_during_retry_backoff_releases_admission_immediately() -> None:
    transport = RetryTransport()
    scheduler = OutboundScheduler(transport, max_queue_size=1)
    await scheduler.start()
    retrying = asyncio.create_task(
        scheduler.send(
            "peer-1",
            b"retry",
            priority=Priority.DATA,
            retries=1,
            retry_delay_seconds=10,
        )
    )
    await transport.first_attempt_failed.wait()
    retrying.cancel()
    with pytest.raises(asyncio.CancelledError):
        await retrying

    try:
        await asyncio.wait_for(
            scheduler.send("peer-2", b"fresh", priority=Priority.DATA),
            timeout=0.1,
        )
    finally:
        await scheduler.stop()


def test_stop_fails_delayed_retry_without_another_attempt() -> None:
    asyncio.run(_test_stop_fails_delayed_retry_without_another_attempt())


async def _test_stop_fails_delayed_retry_without_another_attempt() -> None:
    transport = RetryTransport()
    scheduler = OutboundScheduler(transport)
    await scheduler.start()
    retrying = asyncio.create_task(
        scheduler.send(
            "peer-1",
            b"retry",
            priority=Priority.DATA,
            retries=1,
            retry_delay_seconds=10,
        )
    )
    await transport.first_attempt_failed.wait()

    await scheduler.stop()

    with pytest.raises(RuntimeError, match="stopped"):
        await retrying
    assert transport.sent == [b"retry"]


def test_cancel_active_send_suppresses_retry_after_attempt_fails() -> None:
    asyncio.run(_test_cancel_active_send_suppresses_retry_after_attempt_fails())


async def _test_cancel_active_send_suppresses_retry_after_attempt_fails() -> None:
    transport = BlockingFailingTransport()
    scheduler = OutboundScheduler(transport)
    await scheduler.start()
    sending = asyncio.create_task(
        scheduler.send(
            "peer-1",
            b"payload",
            priority=Priority.DATA,
            retries=1,
            retry_delay_seconds=0,
        )
    )
    await transport.attempt_started.wait()

    sending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sending
    transport.release_attempt.set()
    await scheduler.send("peer-2", b"fresh", priority=Priority.DATA)

    assert transport.failed_attempts == 1
    await scheduler.stop()


def test_active_failure_is_preserved_during_shutdown_without_retry() -> None:
    asyncio.run(_test_active_failure_is_preserved_during_shutdown_without_retry())


async def _test_active_failure_is_preserved_during_shutdown_without_retry() -> None:
    transport = BlockingFailingTransport()
    scheduler = OutboundScheduler(transport)
    await scheduler.start()
    sending = asyncio.create_task(
        scheduler.send(
            "peer-1",
            b"payload",
            priority=Priority.DATA,
            retries=1,
            retry_delay_seconds=0,
        )
    )
    await transport.attempt_started.wait()

    stopping = asyncio.create_task(scheduler.stop())
    transport.release_attempt.set()

    with pytest.raises(OSError, match="send failed"):
        await sending
    await stopping
    assert transport.failed_attempts == 1
