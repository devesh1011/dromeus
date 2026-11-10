"""Deterministic live and exact consensus distance calculations."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import math
from collections import deque
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Literal

import numpy as np

from dromeus.manifests.models import ConsensusSketchMessage, PublicKey, RoundId

COUNT_SKETCH_SIZE = 4096
# Keep integer draw boundaries aligned across named tensors. NumPy's small
# integer draws can pack several values into a generator word.
_COUNT_SKETCH_CHUNK_VALUES = 65536


class ConsensusSketchError(ValueError):
    """A consensus sketch was malformed or arrived from an invalid member."""


@dataclass(frozen=True, slots=True)
class MemberDistance:
    """A sealed member's distance from the complete group's mean sketch."""

    public_key: PublicKey
    absolute_distance: float
    normalized_distance: float

    def __post_init__(self) -> None:
        if not self.public_key or any(
            not math.isfinite(value) or value < 0
            for value in (self.absolute_distance, self.normalized_distance)
        ):
            raise ConsensusSketchError("invalid member distance")


@dataclass(frozen=True, slots=True)
class ConsensusObservation:
    """One complete round's group spread, scale and per-member distances."""

    round_id: RoundId
    normalized_rms: float
    sketch_count: int
    absolute_rms_spread: float
    mean_sketch_norm: float
    denominator_degenerate: bool
    member_distances: tuple[MemberDistance, ...] = ()

    def __post_init__(self) -> None:
        if (
            self.round_id < 0
            or self.sketch_count < 1
            or any(
                not math.isfinite(v) or v < 0
                for v in (
                    self.normalized_rms,
                    self.absolute_rms_spread,
                    self.mean_sketch_norm,
                )
            )
        ):
            raise ConsensusSketchError("invalid consensus observation")
        if self.denominator_degenerate != (self.mean_sketch_norm <= 1e-12):
            raise ConsensusSketchError("inconsistent denominator scale flag")
        if not math.isclose(
            self.normalized_rms,
            self.absolute_rms_spread / (self.mean_sketch_norm + 1e-12),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ConsensusSketchError("inconsistent consensus scale measurements")
        if self.member_distances:
            if (
                len(self.member_distances) != self.sketch_count
                or len({m.public_key for m in self.member_distances})
                != self.sketch_count
                or any(
                    not math.isclose(
                        m.normalized_distance,
                        m.absolute_distance / (self.mean_sketch_norm + 1e-12),
                        rel_tol=1e-9,
                        abs_tol=1e-12,
                    )
                    for m in self.member_distances
                )
                or not math.isclose(
                    math.hypot(*(m.absolute_distance for m in self.member_distances))
                    / math.sqrt(self.sketch_count),
                    self.absolute_rms_spread,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
            ):
                raise ConsensusSketchError("inconsistent member distances")


def consensus_observation(
    sketches: Sequence[np.ndarray],
    *,
    round_id: int,
    participant_keys: Sequence[PublicKey] | None = None,
) -> ConsensusObservation:
    """Measure spread and mean norm with a 1e-12 denominator stabilizer."""
    if not sketches:
        raise ConsensusSketchError("at least one sketch is required")
    first = np.asarray(sketches[0])
    if first.ndim != 1 or not first.size:
        raise ConsensusSketchError("sketch must be a nonempty vector")
    values = np.stack([_validate_sketch(x, size=first.size) for x in sketches]).astype(
        np.float64
    )
    return _measure_consensus(
        values, round_id=round_id, participant_keys=participant_keys
    )


def exact_consensus_observation(
    models: Sequence[Mapping[str, np.ndarray]], *, round_id: int
) -> ConsensusObservation:
    """Use the same scale definitions for exact public model snapshots."""
    if not models or any(model.keys() != models[0].keys() for model in models):
        raise ValueError("exact models must share their named schema")
    if any(
        model[name].shape != models[0][name].shape for model in models for name in model
    ):
        raise ValueError("exact model shapes differ")
    values = np.stack([_flatten_weights(model) for model in models]).astype(np.float64)
    return _measure_consensus(values, round_id=round_id)


def _measure_consensus(
    values: np.ndarray,
    *,
    round_id: int,
    participant_keys: Sequence[PublicKey] | None = None,
) -> ConsensusObservation:
    if participant_keys is not None and (
        len(participant_keys) != len(values)
        or len(set(participant_keys)) != len(values)
        or any(not key for key in participant_keys)
    ):
        raise ConsensusSketchError("participant keys must match sketches and be unique")
    mean = values.mean(axis=0)
    squared_distances = np.sum((values - mean) ** 2, axis=1)
    spread = float(np.sqrt(np.mean(squared_distances)))
    norm = float(np.linalg.norm(mean))
    members = (
        tuple(
            sorted(
                (
                    MemberDistance(
                        key, float(distance), float(distance) / (norm + 1e-12)
                    )
                    for key, distance in zip(
                        participant_keys, np.sqrt(squared_distances), strict=True
                    )
                ),
                key=lambda member: member.public_key,
            )
        )
        if participant_keys is not None
        else ()
    )
    return ConsensusObservation(
        round_id=round_id,
        normalized_rms=spread / (norm + 1e-12),
        sketch_count=len(values),
        absolute_rms_spread=spread,
        mean_sketch_norm=norm,
        denominator_degenerate=norm <= 1e-12,
        member_distances=members,
    )


SketchReceiver = Callable[[float], Awaitable[ConsensusSketchMessage]]
SketchPublisher = Callable[[RoundId, np.ndarray], Awaitable[None]]
DistanceCallback = Callable[[ConsensusObservation], None | Awaitable[None]]


def count_sketch(
    weights: Mapping[str, np.ndarray],
    *,
    seed: int,
    size: int = COUNT_SKETCH_SIZE,
) -> np.ndarray:
    """Return a deterministic signed CountSketch of named finite tensors."""
    if size <= 0:
        raise ValueError("sketch size must be positive")
    arrays = _validated_sketch_arrays(weights)
    element_count = sum(value.size for value in arrays)
    normalised_seed = _normalise_seed(seed)
    bucket_rng = np.random.default_rng(normalised_seed)
    sign_rng = np.random.default_rng(normalised_seed)
    # The projection draws ALL buckets before ANY signs. Replay that
    # first segment in bounded chunks for the sign stream instead of changing
    # the seed/distribution or assuming a raw bit-generator advance count.
    for start in range(0, element_count, _COUNT_SKETCH_CHUNK_VALUES):
        sign_rng.integers(
            0, size, size=min(_COUNT_SKETCH_CHUNK_VALUES, element_count - start)
        )
    sketch = np.zeros(size, dtype=np.float32)
    for values in _sketch_value_chunks(arrays):
        buckets = bucket_rng.integers(0, size, size=values.size)
        signs = np.where(
            sign_rng.integers(0, 2, size=values.size, dtype=np.int8) == 0,
            -1.0,
            1.0,
        )
        np.add.at(sketch, buckets, values * signs)
    return sketch


def _validated_sketch_arrays(
    weights: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, ...]:
    """Validate in projection name order without allocating model-sized copies."""
    if not weights:
        raise ValueError("weights must contain at least one tensor")
    arrays: list[np.ndarray] = []
    for name in sorted(weights):
        value = np.asarray(weights[name])
        if not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"tensor {name} must be floating point")
        for start in range(0, value.size, _COUNT_SKETCH_CHUNK_VALUES):
            if not np.isfinite(
                value.flat[start : start + _COUNT_SKETCH_CHUNK_VALUES]
            ).all():
                raise ValueError(f"tensor {name} contains non-finite values")
        arrays.append(value)
    return tuple(arrays)


def _sketch_value_chunks(arrays: Sequence[np.ndarray]) -> Iterator[np.ndarray]:
    """Flatten globally in C order, keeping per-name RNG boundaries."""
    buffer = np.empty(_COUNT_SKETCH_CHUNK_VALUES, dtype=np.float64)
    used = 0
    for value in arrays:
        position = 0
        while position < value.size:
            count = min(_COUNT_SKETCH_CHUNK_VALUES - used, value.size - position)
            buffer[used : used + count] = value.flat[position : position + count]
            used += count
            position += count
            if used == _COUNT_SKETCH_CHUNK_VALUES:
                yield buffer
                used = 0
    if used:
        yield buffer[:used]


def encode_sketch(sketch: np.ndarray, *, size: int = COUNT_SKETCH_SIZE) -> bytes:
    """Encode exactly one FP32 sketch without a JSON or MessagePack expansion."""
    value = _validate_sketch(sketch, size=size)
    return value.astype("<f4", copy=False).tobytes()


def decode_sketch(data: bytes, *, size: int = COUNT_SKETCH_SIZE) -> np.ndarray:
    """Decode a bounded FP32 sketch payload."""
    if len(data) != size * np.dtype(np.float32).itemsize:
        raise ConsensusSketchError("sketch payload has an unexpected size")
    value = np.frombuffer(data, dtype="<f4").copy()
    return _validate_sketch(value, size=size)


class ConsensusSketchBuffer:
    """Bounded round-keyed sketch buffer that never waits on training."""

    def __init__(
        self,
        *,
        participant_keys: Sequence[PublicKey],
        size: int = COUNT_SKETCH_SIZE,
        max_pending_rounds: int = 64,
    ) -> None:
        if not participant_keys or len(set(participant_keys)) != len(participant_keys):
            raise ValueError("participant keys must be non-empty and unique")
        if size <= 0:
            raise ValueError("sketch size must be positive")
        if max_pending_rounds <= 0:
            raise ValueError("max_pending_rounds must be positive")
        self._participant_keys = frozenset(participant_keys)
        self._size = size
        self._max_pending_rounds = max_pending_rounds
        self._lock = RLock()
        self._pending: dict[RoundId, dict[PublicKey, np.ndarray]] = {}
        self._results: dict[RoundId, ConsensusObservation] = {}
        self._dropped = 0

    def add(
        self,
        *,
        round_id: RoundId,
        sender_public_key: PublicKey,
        sketch: np.ndarray,
    ) -> ConsensusObservation | None:
        with self._lock:
            if sender_public_key not in self._participant_keys:
                raise ConsensusSketchError("sender is not a sealed participant")
            value = _validate_sketch(sketch, size=self._size)
            existing_result = self._results.get(round_id)
            if existing_result is not None:
                existing = self._pending[round_id][sender_public_key]
                if not np.array_equal(existing, value):
                    raise ConsensusSketchError("duplicate sketch does not match")
                return existing_result
            if (
                round_id not in self._pending
                and len(self.pending_rounds()) >= self._max_pending_rounds
            ):
                oldest_round = min(self.pending_rounds())
                self._pending.pop(oldest_round, None)
                self._dropped += 1
            round_sketches = self._pending.setdefault(round_id, {})
            existing = round_sketches.get(sender_public_key)
            if existing is not None:
                if not np.array_equal(existing, value):
                    raise ConsensusSketchError("duplicate sketch does not match")
            else:
                round_sketches[sender_public_key] = value
            if len(round_sketches) != len(self._participant_keys):
                return None
            result = consensus_observation(
                [round_sketches[key] for key in sorted(round_sketches)],
                round_id=round_id,
                participant_keys=sorted(round_sketches),
            )
            self._results[round_id] = result
            completed_rounds = sorted(self._results)
            for old_round in completed_rounds[: -self._max_pending_rounds]:
                self._results.pop(old_round, None)
                self._pending.pop(old_round, None)
            return result

    def result(self, round_id: RoundId) -> ConsensusObservation | None:
        with self._lock:
            return self._results.get(round_id)

    @property
    def dropped(self) -> int:
        """Return incomplete rounds evicted to keep the buffer bounded."""
        return self._dropped

    def pending_rounds(self) -> tuple[RoundId, ...]:
        with self._lock:
            return tuple(
                sorted(
                    round_id
                    for round_id in self._pending
                    if round_id not in self._results
                )
            )


@dataclass(frozen=True, slots=True)
class ConsensusCompletionResult:
    """Local final-round evidence; never an all-peer delivery acknowledgement."""

    round_id: RoundId
    status: Literal["complete", "timeout", "dropped", "stopped"]
    received_member_keys: tuple[PublicKey, ...]
    report_completed: bool
    dropped: int


class ConsensusSketchPublisher:
    """Bounded background sketch computation and publication queue."""

    def __init__(
        self,
        *,
        seed: int,
        publish: Callable[[RoundId, np.ndarray], Awaitable[None]],
        size: int = COUNT_SKETCH_SIZE,
        max_queue_size: int = 64,
        on_sketch: Callable[[RoundId, np.ndarray], None | Awaitable[None]]
        | None = None,
    ) -> None:
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        self._seed = seed
        self._publish = publish
        self._on_sketch = on_sketch
        self._size = size
        self._dropped = 0
        self._queue: asyncio.Queue[tuple[RoundId, Mapping[str, np.ndarray]]] = (
            asyncio.Queue(maxsize=max_queue_size)
        )
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._inflight = False
        self._thread_call_active = False

    def submit(self, *, round_id: RoundId, weights: Mapping[str, np.ndarray]) -> bool:
        """Queue a model snapshot, returning false when telemetry is full."""
        if self._queue.full():
            self._dropped += 1
            return False
        try:
            self._queue.put_nowait((round_id, weights))
        except asyncio.QueueFull:
            self._dropped += 1
            return False
        return True

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="dromeus-consensus")

    async def stop(self, *, timeout_seconds: float | None = None) -> bool:
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("stop timeout must be positive")
        self._stop.set()
        task = self._task
        if task is None:
            return True
        try:
            if timeout_seconds is None:
                await task
            else:
                await asyncio.wait_for(task, timeout=timeout_seconds)
            return True
        except TimeoutError:
            self._dropped += 1 + self._queue.qsize()
            return False
        finally:
            self._task = None

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    @property
    def dropped(self) -> int:
        """Return the number of snapshots rejected by the bounded queue."""
        return self._dropped

    async def wait_idle(self) -> None:
        """Wait for queued AND in-flight publication without stopping its worker.

        The caller must first finish submissions. Cancellation affects this wait,
        not the background projection thread or publisher task.
        """
        await self._queue.join()

    async def finish(self) -> None:
        """Finish accepted publication without cascading caller cancellation."""
        await self.wait_idle()
        self._stop.set()
        if self._task is not None:
            await asyncio.shield(self._task)

    def abort(self) -> tuple[asyncio.Task[None] | None, bool]:
        """Cancel async publication once; native projection may continue."""
        thread_may_continue = self._thread_call_active
        self._dropped += self.pending + int(self._inflight)
        self._stop.set()
        while not self._queue.empty():
            self._queue.get_nowait()
            self._queue.task_done()
        if self._task is not None and not self._task.done():
            self._task.cancel()
        return self._task, thread_may_continue

    async def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                round_id, weights = await asyncio.wait_for(
                    self._queue.get(), timeout=0.1
                )
            except TimeoutError:
                continue
            self._inflight = True
            try:
                self._thread_call_active = True
                snapshot = await asyncio.to_thread(_copy_weights, weights)
                sketch = await asyncio.to_thread(
                    count_sketch,
                    snapshot,
                    seed=self._seed,
                    size=self._size,
                )
                self._thread_call_active = False
                if self._on_sketch is not None:
                    try:
                        result = self._on_sketch(round_id, sketch)
                        if inspect.isawaitable(result):
                            await result
                    except Exception:
                        self._dropped += 1
                        pass
                try:
                    await self._publish(round_id, sketch)
                except Exception:
                    self._dropped += 1
                    pass
            except Exception:
                self._dropped += 1
                pass
            finally:
                # A live publisher can wait indefinitely for the next item.
                # Do not keep the last model and defensive copy in its frame.
                self._inflight = False
                self._thread_call_active = False
                weights = None
                snapshot = None
                self._queue.task_done()


class LiveConsensusTelemetry:
    """Publish local sketches and consume remote sketches off training path."""

    def __init__(
        self,
        *,
        local_public_key: PublicKey,
        participant_keys: Sequence[PublicKey],
        seed: int,
        receive: SketchReceiver,
        publish: SketchPublisher,
        on_distance: DistanceCallback | None = None,
        size: int = COUNT_SKETCH_SIZE,
        max_pending_rounds: int = 64,
        max_queue_size: int = 64,
        drain_timeout_seconds: float = 5.0,
    ) -> None:
        if drain_timeout_seconds <= 0:
            raise ValueError("drain timeout must be positive")
        self._local_public_key = local_public_key
        self._receive = receive
        self._on_distance = on_distance
        self._buffer = ConsensusSketchBuffer(
            participant_keys=participant_keys,
            size=size,
            max_pending_rounds=max_pending_rounds,
        )
        self._publisher = ConsensusSketchPublisher(
            seed=seed,
            publish=publish,
            size=size,
            max_queue_size=max_queue_size,
            on_sketch=self._accept_local,
        )
        self._size = size
        self._drain_timeout_seconds = drain_timeout_seconds
        self._submitted_rounds: deque[RoundId] = deque()
        self._reported_rounds: set[RoundId] = set()
        self._completed_reports: set[RoundId] = set()
        self._reported_round_order: deque[RoundId] = deque()
        self._max_pending_rounds = max_pending_rounds
        self._max_reported_rounds = max_pending_rounds
        self._distance_queue: asyncio.Queue[ConsensusObservation] = asyncio.Queue(
            maxsize=max_pending_rounds
        )
        self._dropped_distance_reports = 0
        self._stop = asyncio.Event()
        self._report_stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._report_task: asyncio.Task[None] | None = None
        self._report_inflight = False
        self._shutdown_waiter: asyncio.Task[None] | None = None
        self._shutdown_tasks: set[asyncio.Task[None]] = set()
        self._shutdown_incomplete = False
        self._projection_thread_may_continue = False

    @property
    def pending(self) -> int:
        """Return queued local sketches awaiting background publication."""
        return self._publisher.pending

    def submit(self, *, round_id: RoundId, weights: Mapping[str, np.ndarray]) -> bool:
        """Queue one committed local model without waiting on telemetry."""
        submitted = self._publisher.submit(round_id=round_id, weights=weights)
        if submitted:
            self._submitted_rounds.append(round_id)
            while len(self._submitted_rounds) > self._max_pending_rounds:
                self._submitted_rounds.popleft()
        return submitted

    def result(self, round_id: RoundId) -> ConsensusObservation | None:
        """Return completed live consensus for one round, when available."""
        return self._buffer.result(round_id)

    @property
    def dropped(self) -> int:
        """Return local sketch and distance-report drops."""
        return (
            self._publisher.dropped
            + self._buffer.dropped
            + self._dropped_distance_reports
        )

    async def start(self) -> None:
        if self._task is None:
            self._stop.clear()
            self._report_stop.clear()
            self._task = asyncio.create_task(
                self._consume(), name="dromeus-consensus-recv"
            )
            self._report_task = asyncio.create_task(
                self._report_distances(), name="dromeus-consensus-report"
            )
            await self._publisher.start()

    @property
    def shutdown_incomplete(self) -> bool:
        """Whether opted-in cleanup exhausted its deadline or was cancelled."""
        return self._shutdown_incomplete

    @property
    def pending_shutdown_tasks(self) -> int:
        """Unfinished asyncio tasks, NOT proof that native threads have stopped."""
        return sum(not task.done() for task in self._shutdown_tasks)

    @property
    def projection_thread_may_continue(self) -> bool:
        """Conservative flag: cancelling to_thread cannot kill its native thread."""
        return self._projection_thread_may_continue

    async def stop(self, *, deadline: float | None = None) -> None:
        if self._shutdown_incomplete:
            return  # Preserve bounded/idempotent shutdown after an opted-in abort.
        if deadline is None:
            await self._stop_without_deadline()
            return
        if not math.isfinite(deadline):
            raise ValueError("finite absolute shutdown deadline required")
        if self._shutdown_incomplete:
            return  # Idempotent; never repeatedly cancel an uncooperative callback.
        if self._task is None:
            return
        if self._shutdown_waiter is None:
            self._shutdown_waiter = asyncio.create_task(
                self._finish_without_cancelling_workers(), name="consensus-finish"
            )
        waiter = self._shutdown_waiter
        try:
            done, _ = await asyncio.wait(
                {waiter}, timeout=max(0.0, deadline - asyncio.get_running_loop().time())
            )
            if waiter in done:
                waiter.result()
                return
        except BaseException:
            self._abort_bounded_shutdown()
            raise
        self._abort_bounded_shutdown()

    async def _finish_without_cancelling_workers(self) -> None:
        await self._publisher.finish()
        await self._drain_submitted()
        self._stop.set()
        if self._task is not None:
            await asyncio.shield(self._task)
        self._report_stop.set()
        await self._distance_queue.join()
        if self._report_task is not None:
            await asyncio.shield(self._report_task)
        self._task = None
        self._report_task = None

    def _abort_bounded_shutdown(self) -> None:
        if self._shutdown_incomplete:
            return
        self._shutdown_incomplete = True
        publisher_task, self._projection_thread_may_continue = self._publisher.abort()
        self._dropped_distance_reports += self._distance_queue.qsize() + int(
            self._report_inflight
        )
        self._stop.set()
        self._report_stop.set()
        self._submitted_rounds.clear()
        # Release queued models/reports. In-flight frames are released by their
        # cooperative finally blocks; an ignoring callback stays referenced.
        while not self._distance_queue.empty():
            self._distance_queue.get_nowait()
            self._distance_queue.task_done()
        for task in (
            self._shutdown_waiter,
            publisher_task,
            self._task,
            self._report_task,
        ):
            if task is None or task.done():
                continue
            self._shutdown_tasks.add(task)
            task.add_done_callback(self._observe_shutdown_task)
            if task is not publisher_task:
                task.cancel()

    def _observe_shutdown_task(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()  # Consume failures without an unbounded await.
        self._shutdown_tasks.discard(task)

    async def _stop_without_deadline(self) -> None:
        """Default shutdown: drain and report without an absolute deadline."""
        if self._task is None:
            return
        publisher_drained = await self._publisher.stop(
            timeout_seconds=self._drain_timeout_seconds
        )
        if publisher_drained:
            try:
                await asyncio.wait_for(
                    self._drain_submitted(), timeout=self._drain_timeout_seconds
                )
            except TimeoutError:
                pass
        else:
            self._submitted_rounds.clear()
        self._stop.set()
        await self._task
        self._report_stop.set()
        if self._report_task is not None:
            await self._report_task
        self._task = None
        self._report_task = None

    async def wait_complete(
        self, round_id: RoundId, *, deadline: float
    ) -> ConsensusCompletionResult:
        """Observe final publication, all local members and the report callback.

        This is opt-in, does not close receivers, and does not assert remote
        receipt of our outgoing sketch. Use a separate application barrier for
        that. The engine must have finished submitting before calling this API.
        """
        if round_id < 0 or not math.isfinite(deadline):
            raise ValueError("valid final round and absolute deadline required")

        def result(
            status: Literal["complete", "timeout", "dropped", "stopped"],
        ) -> ConsensusCompletionResult:
            observation = self.result(round_id)
            keys = (
                tuple(member.public_key for member in observation.member_distances)
                if isinstance(observation, ConsensusObservation)
                else ()
            )
            return ConsensusCompletionResult(
                round_id,
                status,
                keys,
                round_id in self._completed_reports,
                self.dropped,
            )

        if asyncio.get_running_loop().time() >= deadline:
            return result("timeout")
        if self._task is None or self._task.done():
            return result("stopped")
        try:
            async with asyncio.timeout_at(deadline):
                await self._publisher.wait_idle()
                while self.result(round_id) is None:
                    if self.dropped:
                        return result("dropped")
                    if self._task.done():
                        return result("stopped")
                    await asyncio.sleep(0.01)
                await self._distance_queue.join()
                # A result can precede its callback's queue admission. Checking
                # the completed callback marker closes that scheduling seam.
                while round_id not in self._completed_reports:
                    if self.dropped:
                        return result("dropped")
                    if self._report_task is None or self._report_task.done():
                        return result("stopped")
                    await asyncio.sleep(0.01)
                return result("dropped" if self.dropped else "complete")
        except TimeoutError:
            return result("timeout")

    async def _accept_local(self, round_id: RoundId, sketch: np.ndarray) -> None:
        await self._accept(
            ConsensusSketchMessage(
                sender_public_key=self._local_public_key,
                round_id=round_id,
                payload=encode_sketch(sketch, size=self._size),
            ),
            sketch,
        )

    async def _consume(self) -> None:
        while not self._stop.is_set():
            try:
                message = await self._receive(0.1)
            except TimeoutError:
                continue
            try:
                sketch = decode_sketch(message.payload, size=self._size)
                await self._accept(message, sketch)
            except (ConsensusSketchError, TypeError, ValueError):
                continue

    async def _accept(
        self, message: ConsensusSketchMessage, sketch: np.ndarray
    ) -> None:
        try:
            distance = await asyncio.to_thread(
                self._buffer.add,
                round_id=message.round_id,
                sender_public_key=message.sender_public_key,
                sketch=sketch,
            )
        except (ConsensusSketchError, TypeError, ValueError):
            return
        if distance is None or distance.round_id in self._reported_rounds:
            return
        try:
            self._submitted_rounds.remove(distance.round_id)
        except ValueError:
            pass
        try:
            self._distance_queue.put_nowait(distance)
        except asyncio.QueueFull:
            self._dropped_distance_reports += 1
            return
        self._reported_rounds.add(distance.round_id)
        self._reported_round_order.append(distance.round_id)
        while len(self._reported_round_order) > self._max_reported_rounds:
            expired = self._reported_round_order.popleft()
            self._reported_rounds.remove(expired)
            self._completed_reports.discard(expired)

    async def _report_distances(self) -> None:
        while not self._report_stop.is_set() or not self._distance_queue.empty():
            try:
                distance = await asyncio.wait_for(
                    self._distance_queue.get(), timeout=0.1
                )
            except TimeoutError:
                continue
            self._report_inflight = True
            try:
                if self._on_distance is not None:
                    result = self._on_distance(distance)
                    if inspect.isawaitable(result):
                        await result
                if distance.round_id in self._reported_rounds:
                    self._completed_reports.add(distance.round_id)
            except Exception:
                self._dropped_distance_reports += 1
                pass
            finally:
                self._report_inflight = False
                self._distance_queue.task_done()

    async def _drain_submitted(self) -> None:
        while self._submitted_rounds:
            await asyncio.sleep(0.01)


def _copy_weights(weights: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.ascontiguousarray(value).copy() for name, value in weights.items()}


def _flatten_weights(weights: Mapping[str, np.ndarray]) -> np.ndarray:
    if not weights:
        raise ValueError("weights must contain at least one tensor")
    flattened: list[np.ndarray] = []
    for name in sorted(weights):
        value = np.asarray(weights[name])
        if not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"tensor {name} must be floating point")
        if not np.isfinite(value).all():
            raise ValueError(f"tensor {name} contains non-finite values")
        flattened.append(value.astype(np.float64, copy=False).reshape(-1))
    return np.concatenate(flattened)


def _validate_sketch(
    sketch: np.ndarray, *, size: int = COUNT_SKETCH_SIZE
) -> np.ndarray:
    value = np.asarray(sketch)
    if value.shape != (size,) or value.dtype != np.float32:
        raise ConsensusSketchError("sketch must be a one-dimensional FP32 vector")
    if not np.isfinite(value).all():
        raise ConsensusSketchError("sketch contains non-finite values")
    return value.copy()


def _normalise_seed(seed: int) -> int:
    digest = hashlib.sha256(str(seed).encode()).digest()
    return int.from_bytes(digest[:8], "little")


__all__ = [
    "COUNT_SKETCH_SIZE",
    "ConsensusObservation",
    "ConsensusSketchBuffer",
    "ConsensusSketchError",
    "ConsensusSketchMessage",
    "ConsensusSketchPublisher",
    "LiveConsensusTelemetry",
    "MemberDistance",
    "count_sketch",
    "decode_sketch",
    "encode_sketch",
]
