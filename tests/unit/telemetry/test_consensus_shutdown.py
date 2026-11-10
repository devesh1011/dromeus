"""Final-round telemetry regressions through real engine and Runtime cleanup.

Public scalar fixtures only. Time bounds are scaled to keep the exact timeout
and peer-ordering failures deterministic; no language model or private data.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import numpy as np
import pytest
from support.gossip_fakes import (
    InMemoryPairTransport,
    LinearTrainer,
    SharedPairChannel,
    make_algorithm,
)

from dromeus.gossip.engine import GossipEngine
from dromeus.gossip.peer_scheduler import PeerScheduler
from dromeus.manifests.models import ConsensusSketchMessage, Tensor, TensorSchema
from dromeus.runtime import (
    ConsensusCompletionPolicy,
    ConsensusCompletionReceipt,
    NodeRuntime,
)
from dromeus.telemetry import consensus as module
from dromeus.telemetry.consensus import LiveConsensusTelemetry, encode_sketch


def cleanup_runtime(
    telemetry: LiveConsensusTelemetry,
    close: Any,
    *,
    key: str = "peer-0",
    keys: tuple[str, ...] = ("peer-0", "peer-1"),
    policy: ConsensusCompletionPolicy | None = None,
) -> NodeRuntime:
    """Keep real Runtime cleanup/stop-consensus; unrelated subsystems are absent."""
    runtime = cast(Any, object.__new__(NodeRuntime))
    runtime._consensus_telemetry = telemetry
    runtime._consensus_completion_tasks = set()
    runtime._result = (
        SimpleNamespace(
            manifest=SimpleNamespace(
                run_id="test-run",
                participants=[SimpleNamespace(public_key=k) for k in keys],
            ),
            manifest_hash="0" * 64,
        )
        if policy
        else None
    )
    runtime._local_public_key = key
    runtime._training = (
        SimpleNamespace(consensus_completion=policy, metrics_publisher=None)
        if policy
        else None
    )
    runtime._divergence_monitor = None
    runtime._control_task = None
    runtime._formation = type("Formation", (), {"stop": staticmethod(close)})()
    return cast(NodeRuntime, runtime)


async def one_round_engines(
    root: Path, telemetry: list[LiveConsensusTelemetry]
) -> None:
    keys = [f"peer-{i}" for i in range(len(telemetry))]
    channel = SharedPairChannel.create()
    schema = TensorSchema(tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),))
    engines = [
        GossipEngine(
            local_public_key=key,
            round_count=1,
            scheduler=PeerScheduler(keys, seed=17),
            algorithm=make_algorithm(
                key=key,
                trainer=LinearTrainer(float(index)),
                schema=schema,
                artifact_root=root / str(index),
            ),
            transport=InMemoryPairTransport(key, channel),
            commit_callback=lambda _commit: None,
            consensus_publisher=telemetry[index],
        )
        for index, key in enumerate(keys)
    ]
    commits = await asyncio.gather(*(engine.run() for engine in engines))
    assert all(len(value) == 1 for value in commits)


@pytest.mark.parametrize("opt_in", [False, True])
def test_final_projection_must_finish_before_runtime_closes_receiver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, opt_in: bool
) -> None:
    """Reproduces full-size projection exceeding stop budget after final commit."""
    release = threading.Event()
    entered = threading.Event()
    events: list[str] = []
    original = module.count_sketch

    def blocked_projection(*args: Any, **kwargs: Any) -> np.ndarray:
        entered.set()
        assert release.wait(2), "fixture did not release projection thread"
        events.append("projection_finished")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "count_sketch", blocked_projection)

    async def exercise() -> None:
        inboxes: list[asyncio.Queue[ConsensusSketchMessage]] = [
            asyncio.Queue(),
            asyncio.Queue(),
        ]

        async def close() -> None:
            events.append("receiver_closed")

        telemetry: list[LiveConsensusTelemetry] = []
        for i in range(2):

            async def receive(timeout: float, i: int = i) -> ConsensusSketchMessage:
                return await asyncio.wait_for(inboxes[i].get(), timeout)

            async def publish(round_id: int, sketch: np.ndarray, i: int = i) -> None:
                events.append("published")
                await inboxes[1 - i].put(
                    ConsensusSketchMessage(
                        sender_public_key=f"peer-{i}",
                        round_id=round_id,
                        payload=encode_sketch(sketch),
                    )
                )

            telemetry.append(
                LiveConsensusTelemetry(
                    local_public_key=f"peer-{i}",
                    participant_keys=["peer-0", "peer-1"],
                    seed=17,
                    receive=receive,
                    publish=publish,
                    drain_timeout_seconds=0.15,
                )
            )

        async def barrier(
            _receipt: ConsensusCompletionReceipt, _deadline: float
        ) -> None:
            # This case isolates local in-flight completion.
            pass

        for value in telemetry:
            await value.start()
        try:
            await one_round_engines(tmp_path, telemetry)
            assert await asyncio.to_thread(entered.wait, 1)
            # Exact observed symptom: queued pending can be zero while projection
            # is in flight. Cancellation of to_thread does not stop its thread.
            assert all(value.pending == 0 for value in telemetry)
            runtimes = [
                cleanup_runtime(
                    value,
                    close,
                    key=f"peer-{i}",
                    policy=ConsensusCompletionPolicy(0.8, barrier) if opt_in else None,
                )
                for i, value in enumerate(telemetry)
            ]
            if opt_in:
                asyncio.get_running_loop().call_later(0.25, release.set)
                await asyncio.gather(
                    *(cast(Any, r)._complete_consensus(0) for r in runtimes)
                )
            await asyncio.gather(*(cast(Any, r)._cleanup() for r in runtimes))
            assert [value.dropped for value in telemetry] == (
                [0, 0] if opt_in else [1, 1]
            ), events
            assert events.count("published") == (2 if opt_in else 0), events
            assert (
                (events.index("projection_finished") < events.index("receiver_closed"))
                if opt_in
                else "projection_finished" not in events
            )
        finally:
            release.set()
            for value in telemetry:
                await value.stop()

    asyncio.run(exercise())


@pytest.mark.parametrize("opt_in", [False, True])
def test_local_complete_buffer_is_not_all_peer_delivery(
    tmp_path: Path, opt_in: bool
) -> None:
    """Best-effort bridge send completion cannot serve as a peer receipt barrier."""

    async def exercise() -> None:
        keys = [f"peer-{i}" for i in range(4)]
        inboxes: list[asyncio.Queue[ConsensusSketchMessage]] = [
            asyncio.Queue() for _ in keys
        ]
        delayed: list[tuple[int, ConsensusSketchMessage]] = []
        actual_received: set[tuple[int, int]] = set()
        open_receivers = set(range(4))
        telemetry: list[LiveConsensusTelemetry] = []
        closed_with_complete_local: list[bool] = []

        for index in range(4):

            async def receive(
                timeout: float, index: int = index
            ) -> ConsensusSketchMessage:
                message = await asyncio.wait_for(inboxes[index].get(), timeout)
                actual_received.add((keys.index(message.sender_public_key), index))
                return message

            async def publish(
                round_id: int, sketch: np.ndarray, index: int = index
            ) -> None:
                message = ConsensusSketchMessage(
                    sender_public_key=keys[index],
                    round_id=round_id,
                    payload=encode_sketch(sketch),
                )
                for target in range(4):
                    if target == index:
                        continue
                    if index == 0:
                        delayed.append(
                            (target, message)
                        )  # accepted send, not delivered
                    elif target in open_receivers:
                        await inboxes[target].put(message)

            telemetry.append(
                LiveConsensusTelemetry(
                    local_public_key=keys[index],
                    participant_keys=keys,
                    seed=17,
                    receive=receive,
                    publish=publish,
                    drain_timeout_seconds=0.2,
                )
            )
        for value in telemetry:
            await value.start()

        async def close(index: int) -> None:
            open_receivers.remove(index)
            if index == 0:
                closed_with_complete_local.append(telemetry[0].result(0) is not None)
                # Closing the transport discards pending outbound best-effort sends.
                delayed.clear()

        receipts: dict[str, ConsensusCompletionReceipt] = {}
        all_complete = asyncio.Event()

        async def collective(
            receipt: ConsensusCompletionReceipt, deadline: float
        ) -> None:
            assert receipt.received_member_keys == tuple(keys)
            receipts[receipt.node_public_key] = receipt
            # Transport catches up only while the first complete node remains alive.
            if receipt.node_public_key == keys[0]:
                for target, message in delayed:
                    assert target in open_receivers
                    await inboxes[target].put(message)
                delayed.clear()
            if len(receipts) == len(keys):
                all_complete.set()
            async with asyncio.timeout_at(deadline):
                await all_complete.wait()

        async def finish(i: int, value: LiveConsensusTelemetry) -> None:
            runtime = cast(
                Any,
                cleanup_runtime(
                    value,
                    lambda: close(i),
                    key=keys[i],
                    keys=tuple(keys),
                    policy=ConsensusCompletionPolicy(1.0, collective)
                    if opt_in
                    else None,
                ),
            )
            try:
                await runtime._complete_consensus(0)
            finally:
                await runtime._cleanup()

        try:
            await one_round_engines(tmp_path, telemetry)
            await asyncio.gather(
                *(finish(i, value) for i, value in enumerate(telemetry))
            )
            assert closed_with_complete_local == [True]
            assert all(value.pending == 0 and value.dropped == 0 for value in telemetry)
            assert len(actual_received) == (12 if opt_in else 9), sorted(
                actual_received
            )
            assert all(value.result(0) is not None for value in telemetry) == opt_in
            assert len(receipts) == (4 if opt_in else 0)
        finally:
            for value in telemetry:
                await value.stop()

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["report", "publish"])
def test_completion_rejects_failed_callback(failure: str) -> None:
    async def exercise() -> None:
        async def receive(timeout: float) -> ConsensusSketchMessage:
            await asyncio.sleep(timeout)
            raise TimeoutError()

        async def report(_distance: Any) -> None:
            if failure == "report":
                raise RuntimeError("fixture report failure")

        async def publish(_round: int, _sketch: np.ndarray) -> None:
            if failure == "publish":
                raise RuntimeError("fixture publish failure")

        telemetry = LiveConsensusTelemetry(
            local_public_key="only",
            participant_keys=["only"],
            seed=17,
            receive=receive,
            publish=publish,
            on_distance=report,
        )
        await telemetry.start()
        try:
            assert telemetry.submit(
                round_id=0, weights={"weight": np.ones(1, dtype=np.float32)}
            )
            result = await telemetry.wait_complete(
                0, deadline=asyncio.get_running_loop().time() + 0.5
            )
            assert result.status == "dropped" and result.dropped == 1
            assert result.report_completed == (failure != "report")
        finally:
            await telemetry.stop()

    asyncio.run(exercise())


def test_report_callback_timeout_keeps_receivers_alive() -> None:
    async def exercise() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        async def receive(timeout: float) -> ConsensusSketchMessage:
            await asyncio.sleep(timeout)
            raise TimeoutError()

        async def publish(_round: int, _sketch: np.ndarray) -> None:
            pass

        async def report(_distance: Any) -> None:
            entered.set()
            await release.wait()

        telemetry = LiveConsensusTelemetry(
            local_public_key="only",
            participant_keys=["only"],
            seed=17,
            receive=receive,
            publish=publish,
            on_distance=report,
        )
        await telemetry.start()
        try:
            assert telemetry.submit(
                round_id=0, weights={"weight": np.ones(1, dtype=np.float32)}
            )
            await asyncio.wait_for(entered.wait(), 0.5)
            assert telemetry.result(0) is not None
            timed = await telemetry.wait_complete(
                0, deadline=asyncio.get_running_loop().time() + 0.02
            )
            assert (
                timed.status == "timeout"
                and not timed.report_completed
                and telemetry.dropped == 0
            )
            assert not cast(Any, telemetry)._task.done()
            release.set()
            complete = await telemetry.wait_complete(
                0, deadline=asyncio.get_running_loop().time() + 0.5
            )
            assert complete.status == "complete" and complete.report_completed
        finally:
            release.set()
            await telemetry.stop()

    asyncio.run(exercise())


def test_missing_peer_is_bounded_without_false_completion() -> None:
    async def exercise() -> None:
        async def receive(timeout: float) -> ConsensusSketchMessage:
            await asyncio.sleep(timeout)
            raise TimeoutError()

        async def publish(_round: int, _sketch: np.ndarray) -> None:
            pass

        telemetry = LiveConsensusTelemetry(
            local_public_key="a",
            participant_keys=["a", "missing"],
            seed=17,
            receive=receive,
            publish=publish,
            drain_timeout_seconds=0.15,
        )
        await telemetry.start()
        try:
            assert telemetry.submit(
                round_id=0, weights={"w": np.ones(1, dtype=np.float32)}
            )
            result = await asyncio.wait_for(
                telemetry.wait_complete(
                    0, deadline=asyncio.get_running_loop().time() + 0.05
                ),
                0.5,
            )
            assert result.status == "timeout" and not result.report_completed
            assert result.received_member_keys == () and telemetry.dropped == 0
        finally:
            await telemetry.stop()

    asyncio.run(exercise())


def test_runtime_barrier_uses_remaining_deadline_and_cancellation_propagates() -> None:
    async def exercise() -> None:
        calls: list[float] = []

        async def barrier(receipt: ConsensusCompletionReceipt, deadline: float) -> None:
            assert receipt.received_member_keys == ("peer-0", "peer-1")
            calls.append(deadline)
            await asyncio.sleep(1)

        runtime = cast(
            Any,
            cleanup_runtime(
                cast(Any, None),
                AsyncMock(),
                policy=ConsensusCompletionPolicy(0.08, barrier),
            ),
        )

        async def local(
            _round: int, *, deadline: float
        ) -> module.ConsensusCompletionResult:
            calls.append(deadline)
            await asyncio.sleep(0.04)
            return module.ConsensusCompletionResult(
                0, "complete", ("peer-0", "peer-1"), True, 0
            )

        runtime._consensus_telemetry = SimpleNamespace(wait_complete=local, dropped=0)
        start = asyncio.get_running_loop().time()
        with pytest.raises(TimeoutError):
            await runtime._complete_consensus(0)
        assert len(calls) == 2 and calls[0] == calls[1]
        assert asyncio.get_running_loop().time() - start < 0.2
        task = asyncio.create_task(runtime._complete_consensus(0))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())


def test_completion_policy_default_is_legacy_and_invalid_bounds_reject() -> None:
    from dromeus.runtime import TrainingConfig

    assert TrainingConfig.__dataclass_fields__["consensus_completion"].default is None

    async def barrier(_receipt: ConsensusCompletionReceipt, _deadline: float) -> None:
        pass

    for limit in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            ConsensusCompletionPolicy(limit, barrier)


@pytest.mark.parametrize("ignore_first_cancel", [False, True])
@pytest.mark.parametrize("cancel_wait", [False, True])
def test_opt_in_failed_report_cleanup_returns_before_callback_is_released(
    ignore_first_cancel: bool,
    cancel_wait: bool,
) -> None:
    async def exercise() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        closed = asyncio.Event()
        cancelled = asyncio.Event()

        async def receive(timeout: float) -> ConsensusSketchMessage:
            await asyncio.sleep(timeout)
            raise TimeoutError()

        async def publish(_round: int, _sketch: np.ndarray) -> None:
            pass

        async def report(_value: Any) -> None:
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                if not ignore_first_cancel:
                    raise
                await release.wait()

        async def barrier(
            _receipt: ConsensusCompletionReceipt, _deadline: float
        ) -> None:
            pytest.fail("Incomplete report cannot enter collective barrier")

        telemetry = LiveConsensusTelemetry(
            local_public_key="only",
            participant_keys=["only"],
            seed=17,
            receive=receive,
            publish=publish,
            on_distance=report,
        )
        await telemetry.start()
        assert telemetry.submit(round_id=0, weights={"w": np.ones(1, dtype=np.float32)})
        await asyncio.wait_for(entered.wait(), 0.5)
        runtime = cast(
            Any,
            cleanup_runtime(
                telemetry,
                AsyncMock(side_effect=closed.set),
                key="only",
                keys=("only",),
                policy=ConsensusCompletionPolicy(0.03, barrier),
            ),
        )
        task: asyncio.Task[None] | None = None
        try:
            if cancel_wait:
                waiting = asyncio.create_task(runtime._complete_consensus(0))
                await asyncio.sleep(0.01)
                waiting.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await waiting
            else:
                with pytest.raises(Exception, match="consensus completion incomplete"):
                    await runtime._complete_consensus(0)
            task = asyncio.create_task(runtime._cleanup())
            done, _ = await asyncio.wait({task}, timeout=0.25)
            assert task in done, "opt-in cleanup waited indefinitely for blocked report"
            assert closed.is_set() and not release.is_set()
            assert telemetry.dropped >= 1 and telemetry.shutdown_incomplete
            await asyncio.wait_for(cancelled.wait(), 0.5)
            if ignore_first_cancel:
                assert telemetry.pending_shutdown_tasks >= 1
            before = telemetry.dropped
            await telemetry.stop(deadline=asyncio.get_running_loop().time())
            assert telemetry.dropped == before
            if ignore_first_cancel:
                assert telemetry.pending_shutdown_tasks >= 1
        finally:
            release.set()
            if task is not None:
                await asyncio.wait_for(task, 1)
            await telemetry.stop()

            async def settled() -> None:
                while telemetry.pending_shutdown_tasks:
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(settled(), 0.5)

    asyncio.run(exercise())


def test_bounded_cleanup_reports_native_projection_may_outlive_async_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    original = module.count_sketch

    def projection(*args: Any, **kwargs: Any) -> np.ndarray:
        entered.set()
        try:
            assert release.wait(2)
            return original(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(module, "count_sketch", projection)

    async def exercise() -> None:
        async def receive(timeout: float) -> ConsensusSketchMessage:
            await asyncio.sleep(timeout)
            raise TimeoutError()

        async def publish(_round: int, _sketch: np.ndarray) -> None:
            pass

        async def barrier(
            _receipt: ConsensusCompletionReceipt, _deadline: float
        ) -> None:
            pytest.fail("projection incomplete")

        telemetry = LiveConsensusTelemetry(
            local_public_key="only",
            participant_keys=["only"],
            seed=17,
            receive=receive,
            publish=publish,
        )
        await telemetry.start()
        assert telemetry.submit(round_id=0, weights={"w": np.ones(1, dtype=np.float32)})
        runtime = cast(
            Any,
            cleanup_runtime(
                telemetry,
                AsyncMock(),
                key="only",
                keys=("only",),
                policy=ConsensusCompletionPolicy(0.03, barrier),
            ),
        )
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            with pytest.raises(Exception, match="consensus completion incomplete"):
                await runtime._complete_consensus(0)
            await asyncio.wait_for(runtime._cleanup(), 0.25)
            assert (
                telemetry.shutdown_incomplete
                and telemetry.projection_thread_may_continue
            )
            assert not finished.is_set()
            # Finishing/cancelling asyncio tasks cannot prove their native worker ended.
            await asyncio.sleep(0.01)
            assert telemetry.pending_shutdown_tasks == 0 and not finished.is_set()
        finally:
            release.set()
            assert await asyncio.to_thread(finished.wait, 1)
            await telemetry.stop(deadline=asyncio.get_running_loop().time())

    asyncio.run(exercise())


@pytest.mark.parametrize("caller_cancel", [False, True])
def test_barrier_ignoring_cancel_cannot_hold_runtime_completion_forever(
    caller_cancel: bool,
) -> None:
    async def exercise() -> None:
        entered = asyncio.Event()
        cancelled = asyncio.Event()
        release = asyncio.Event()

        async def barrier(
            _receipt: ConsensusCompletionReceipt, _deadline: float
        ) -> None:
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()

        runtime = cast(
            Any,
            cleanup_runtime(
                cast(Any, None),
                AsyncMock(),
                policy=ConsensusCompletionPolicy(0.03, barrier),
            ),
        )

        async def local(
            round_id: int, *, deadline: float
        ) -> module.ConsensusCompletionResult:
            return module.ConsensusCompletionResult(
                round_id, "complete", ("peer-0", "peer-1"), True, 0
            )

        runtime._consensus_telemetry = SimpleNamespace(wait_complete=local, dropped=0)
        task = asyncio.create_task(runtime._complete_consensus(0))
        try:
            await asyncio.wait_for(entered.wait(), 0.5)
            if caller_cancel:
                task.cancel()
            await asyncio.wait_for(cancelled.wait(), 0.5)
            done, _ = await asyncio.wait({task}, timeout=0.15)
            assert task in done, (
                "noncooperative barrier held runtime beyond absolute deadline"
            )
            assert runtime.pending_consensus_completion_tasks == 1
            with pytest.raises(
                asyncio.CancelledError if caller_cancel else TimeoutError
            ):
                await task
        finally:
            release.set()
            try:
                await asyncio.wait_for(task, 0.5)
            except (asyncio.CancelledError, TimeoutError, RuntimeError):
                pass

            async def settled() -> None:
                while runtime.pending_consensus_completion_tasks:
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(settled(), 0.5)

    asyncio.run(exercise())
