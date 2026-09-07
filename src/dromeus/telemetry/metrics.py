"""Bounded, append-only node metric publication."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from dromeus.telemetry.events import EventSink
from dromeus.telemetry.evidence import (
    EvidenceRecord,
    RoundMetricsEvidence,
    RunFailedEvidence,
    TaskRoundMetricsEvidence,
    append_evidence,
)


@dataclass(frozen=True, slots=True)
class RoundTiming:
    """Timing and scalar metrics captured for one committed local round."""

    round_id: int
    peer_id: str
    local_compute_seconds: float
    peer_wait_seconds: float
    transfer_seconds: float
    mixing_seconds: float
    evaluation_seconds: float
    retries: int = 0
    local_loss: float | None = None
    error_feedback_residual_l2_norm: float | None = None
    error_feedback_signal_l2_norm: float | None = None
    error_feedback_residual_to_signal_ratio: float | None = None
    evaluation_loss: float | None = None
    evaluation_accuracy: float | None = None
    evaluation_metrics: Mapping[str, float] | None = None
    transfer_id: str | None = None
    encoded_artifact_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.round_id < 0:
            raise ValueError("round_id must be non-negative")
        if not self.peer_id:
            raise ValueError("peer_id must not be empty")
        durations = (
            self.local_compute_seconds,
            self.peer_wait_seconds,
            self.transfer_seconds,
            self.mixing_seconds,
            self.evaluation_seconds,
        )
        if any(not math.isfinite(value) or value < 0 for value in durations):
            raise ValueError("metric durations must be finite and non-negative")
        if self.retries < 0:
            raise ValueError("retries must be non-negative")
        if self.encoded_artifact_bytes is not None and self.encoded_artifact_bytes <= 0:
            raise ValueError("encoded artifact bytes must be positive")
        if self.local_loss is not None and (not math.isfinite(self.local_loss)):
            raise ValueError("local_loss must be finite")
        observations = (
            self.error_feedback_residual_l2_norm,
            self.error_feedback_signal_l2_norm,
            self.error_feedback_residual_to_signal_ratio,
        )
        if any(
            value is not None and (not math.isfinite(value) or value < 0)
            for value in observations
        ):
            raise ValueError(
                "error-feedback observations must be finite and non-negative"
            )
        if self.evaluation_loss is not None and (
            not math.isfinite(self.evaluation_loss) or self.evaluation_loss < 0
        ):
            raise ValueError("evaluation_loss must be finite and non-negative")
        if self.evaluation_accuracy is not None and (
            not math.isfinite(self.evaluation_accuracy)
            or not 0 <= self.evaluation_accuracy <= 1
        ):
            raise ValueError("evaluation_accuracy must be finite in [0, 1]")
        if self.evaluation_metrics is not None:
            if any(
                not name.strip() or not math.isfinite(value)
                for name, value in self.evaluation_metrics.items()
            ):
                raise ValueError(
                    "task metrics require nonblank names and finite values"
                )
            object.__setattr__(
                self,
                "evaluation_metrics",
                MappingProxyType(dict(self.evaluation_metrics)),
            )


class MetricsPublisher(Protocol):
    """Protocol-like base for non-blocking metric publication."""

    def submit(self, timing: RoundTiming) -> bool:
        raise NotImplementedError

    def submit_failure(self, *, round_id: int, error_type: str, reason: str) -> bool:
        raise NotImplementedError


class JsonlMetricsPublisher(MetricsPublisher):
    """Publish correlated metrics to one node's JSONL sink off the event loop."""

    def __init__(
        self,
        *,
        sink: EventSink,
        run_id: str,
        manifest_hash: str,
        node_id: str,
        max_queue_size: int = 256,
    ) -> None:
        if not run_id or not manifest_hash or not node_id:
            raise ValueError("metric context identifiers must not be empty")
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        self._sink = sink
        self._run_id = run_id
        self._manifest_hash = manifest_hash
        self._node_id = node_id
        self._queue: asyncio.Queue[EvidenceRecord] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._dropped = 0

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    @property
    def dropped(self) -> int:
        return self._dropped

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="dromeus-metrics")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop.set()
        await self._task
        self._task = None

    def submit(self, timing: RoundTiming) -> bool:
        # Copy fields individually: mappingproxy is deliberately not pickleable.
        payload: dict[str, object] = {
            name: getattr(timing, name)
            for name in timing.__dataclass_fields__
            if name != "evaluation_metrics"
        }
        payload.update(
            run_id=self._run_id,
            manifest_hash=self._manifest_hash,
            node_id=self._node_id,
            message_id=f"metric-{self._node_id[:8]}-{timing.round_id}",
        )
        record: EvidenceRecord
        if timing.evaluation_metrics is not None or (
            timing.local_loss is not None and timing.local_loss < 0
        ):
            payload["evaluation_metrics"] = dict(timing.evaluation_metrics or {})
            record = TaskRoundMetricsEvidence.model_validate(payload)
        else:
            record = RoundMetricsEvidence.model_validate(payload)
        return self._enqueue(record)

    def submit_failure(self, *, round_id: int, error_type: str, reason: str) -> bool:
        return self._enqueue(
            RunFailedEvidence(
                run_id=self._run_id,
                manifest_hash=self._manifest_hash,
                node_id=self._node_id,
                message_id=f"run-failure-{self._node_id[:8]}-{round_id}",
                round_id=round_id,
                error_type=error_type[:128],
                error=reason[:1024] or "unknown failure",
            )
        )

    def _enqueue(self, record: EvidenceRecord) -> bool:
        if self._stop.is_set():
            self._dropped += 1
            return False
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            self._dropped += 1
            return False
        return True

    async def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                record = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            except TimeoutError:
                continue
            try:
                written = await asyncio.to_thread(append_evidence, self._sink, record)
                if not written:
                    self._dropped += 1
            except Exception:
                self._dropped += 1
            finally:
                self._queue.task_done()


__all__ = ["JsonlMetricsPublisher", "MetricsPublisher", "RoundTiming"]
