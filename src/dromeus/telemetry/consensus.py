"""Deterministic live and exact consensus distance calculations."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from threading import RLock

import numpy as np

from dromeus.manifests.models import ConsensusSketchMessage, PublicKey, RoundId

COUNT_SKETCH_SIZE = 4096


class ConsensusSketchError(ValueError):
    """A consensus sketch was malformed or arrived from an invalid member."""


@dataclass(frozen=True, slots=True)
class ConsensusDistance:
    round_id: RoundId
    normalized_rms: float
    sketch_count: int


SketchReceiver = Callable[[float], Awaitable[ConsensusSketchMessage]]
SketchPublisher = Callable[[RoundId, np.ndarray], Awaitable[None]]
DistanceCallback = Callable[[ConsensusDistance], None | Awaitable[None]]


def count_sketch(
    weights: Mapping[str, np.ndarray],
    *,
    seed: int,
    size: int = COUNT_SKETCH_SIZE,
) -> np.ndarray:
    """Return a deterministic signed CountSketch of named finite tensors."""
    if size <= 0:
        raise ValueError("sketch size must be positive")
    values = _flatten_weights(weights)
    rng = np.random.default_rng(_normalise_seed(seed))
    buckets = rng.integers(0, size, size=values.size)
    signs = np.where(
        rng.integers(0, 2, size=values.size, dtype=np.int8) == 0,
        -1.0,
        1.0,
    )
    sketch = np.zeros(size, dtype=np.float32)
    np.add.at(sketch, buckets, values * signs)
    return sketch


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


def normalized_rms_consensus_distance(
    sketches: Sequence[np.ndarray], *, epsilon: float = 1e-12
) -> float:
    """Compute normalized RMS distance in sketch space."""
    if not sketches:
        raise ValueError("at least one sketch is required")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    first = np.asarray(sketches[0])
    if first.ndim != 1:
        raise ConsensusSketchError("sketch must be a one-dimensional FP32 vector")
    size = first.shape[0]
    values = np.stack(
        [_validate_sketch(sketch, size=size) for sketch in sketches]
    ).astype(np.float64)
    mean = values.mean(axis=0)
    numerator = np.sqrt(np.mean(np.sum((values - mean) ** 2, axis=1)))
    denominator = np.linalg.norm(mean) + epsilon
    return float(numerator / denominator)


def exact_normalized_rms_distance(
    models: Sequence[Mapping[str, np.ndarray]], *, epsilon: float = 1e-12
) -> float:
    """Compute normalized RMS distance from archived model tensors."""
    if not models:
        raise ValueError("at least one model is required")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    values = np.stack([_flatten_weights(model) for model in models]).astype(np.float64)
    mean = values.mean(axis=0)
    numerator = np.sqrt(np.mean(np.sum((values - mean) ** 2, axis=1)))
    denominator = np.linalg.norm(mean) + epsilon
    return float(numerator / denominator)


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
        self._results: dict[RoundId, ConsensusDistance] = {}
        self._dropped = 0

    def add(
        self,
        *,
        round_id: RoundId,
        sender_public_key: PublicKey,
        sketch: np.ndarray,
    ) -> ConsensusDistance | None:
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
            result = ConsensusDistance(
                round_id=round_id,
                normalized_rms=normalized_rms_consensus_distance(
                    [round_sketches[key] for key in sorted(round_sketches)]
                ),
                sketch_count=len(round_sketches),
            )
            self._results[round_id] = result
            completed_rounds = sorted(self._results)
            for old_round in completed_rounds[: -self._max_pending_rounds]:
                self._results.pop(old_round, None)
                self._pending.pop(old_round, None)
            return result

    def result(self, round_id: RoundId) -> ConsensusDistance | None:
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

    def submit(
        self, *, round_id: RoundId, weights: Mapping[str, np.ndarray]
    ) -> bool:
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

    async def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                round_id, weights = await asyncio.wait_for(
                    self._queue.get(), timeout=0.1
                )
            except TimeoutError:
                continue
            try:
                snapshot = await asyncio.to_thread(_copy_weights, weights)
                sketch = await asyncio.to_thread(
                    count_sketch,
                    snapshot,
                    seed=self._seed,
                    size=self._size,
                )
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
        self._reported_round_order: deque[RoundId] = deque()
        self._max_pending_rounds = max_pending_rounds
        self._max_reported_rounds = max_pending_rounds
        self._distance_queue: asyncio.Queue[ConsensusDistance] = asyncio.Queue(
            maxsize=max_pending_rounds
        )
        self._dropped_distance_reports = 0
        self._stop = asyncio.Event()
        self._report_stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._report_task: asyncio.Task[None] | None = None

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

    def result(self, round_id: RoundId) -> ConsensusDistance | None:
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

    async def stop(self) -> None:
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
            self._reported_rounds.remove(self._reported_round_order.popleft())

    async def _report_distances(self) -> None:
        while not self._report_stop.is_set() or not self._distance_queue.empty():
            try:
                distance = await asyncio.wait_for(
                    self._distance_queue.get(), timeout=0.1
                )
            except TimeoutError:
                continue
            try:
                if self._on_distance is not None:
                    result = self._on_distance(distance)
                    if inspect.isawaitable(result):
                        await result
            except Exception:
                self._dropped_distance_reports += 1
                pass
            finally:
                self._distance_queue.task_done()

    async def _drain_submitted(self) -> None:
        while self._submitted_rounds:
            await asyncio.sleep(0.01)


def _copy_weights(weights: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name: np.ascontiguousarray(value).copy() for name, value in weights.items()
    }


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
    "ConsensusDistance",
    "ConsensusSketchBuffer",
    "ConsensusSketchError",
    "ConsensusSketchMessage",
    "ConsensusSketchPublisher",
    "LiveConsensusTelemetry",
    "count_sketch",
    "decode_sketch",
    "encode_sketch",
    "exact_normalized_rms_distance",
    "normalized_rms_consensus_distance",
]
