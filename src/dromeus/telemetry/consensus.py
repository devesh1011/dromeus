"""Deterministic live and exact consensus distance calculations."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from dromeus.manifests.models import PublicKey, RoundId

COUNT_SKETCH_SIZE = 4096
class ConsensusSketchError(ValueError):
    """A consensus sketch was malformed or arrived from an invalid member."""


@dataclass(frozen=True, slots=True)
class ConsensusDistance:
    round_id: RoundId
    normalized_rms: float
    sketch_count: int


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
        seed: int,
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
        self._seed = seed
        self._size = size
        self._max_pending_rounds = max_pending_rounds
        self._pending: dict[RoundId, dict[PublicKey, np.ndarray]] = {}
        self._results: dict[RoundId, ConsensusDistance] = {}

    def add(
        self,
        *,
        round_id: RoundId,
        sender_public_key: PublicKey,
        sketch: np.ndarray,
    ) -> ConsensusDistance | None:
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
            raise ConsensusSketchError("consensus sketch buffer is full")
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
        return result

    def result(self, round_id: RoundId) -> ConsensusDistance | None:
        return self._results.get(round_id)

    def pending_rounds(self) -> tuple[RoundId, ...]:
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
    ) -> None:
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        self._seed = seed
        self._publish = publish
        self._size = size
        self._queue: asyncio.Queue[tuple[RoundId, dict[str, np.ndarray]]] = (
            asyncio.Queue(maxsize=max_queue_size)
        )
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def submit(
        self, *, round_id: RoundId, weights: Mapping[str, np.ndarray]
    ) -> bool:
        """Queue a copied model snapshot, returning false when telemetry is full."""
        if self._queue.full():
            return False
        snapshot = {
            name: np.ascontiguousarray(value).copy() for name, value in weights.items()
        }
        try:
            self._queue.put_nowait((round_id, snapshot))
        except asyncio.QueueFull:
            return False
        return True

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="dromeus-consensus")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    async def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                round_id, weights = await asyncio.wait_for(
                    self._queue.get(), timeout=0.1
                )
            except TimeoutError:
                continue
            try:
                sketch = await asyncio.to_thread(
                    count_sketch,
                    weights,
                    seed=self._seed,
                    size=self._size,
                )
                await self._publish(round_id, sketch)
            except Exception:
                pass
            finally:
                self._queue.task_done()


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
    "ConsensusSketchPublisher",
    "count_sketch",
    "decode_sketch",
    "encode_sketch",
    "exact_normalized_rms_distance",
    "normalized_rms_consensus_distance",
]
