"""Compose pure detector state with bounded best-effort evidence delivery."""

from __future__ import annotations

from dataclasses import asdict
from threading import RLock

from dromeus.manifests.canonical import canonical_hash
from dromeus.manifests.models import DivergencePolicy
from dromeus.telemetry.consensus import ConsensusObservation
from dromeus.telemetry.divergence import (
    DetectorSnapshot,
    DetectorTransition,
    DivergenceDetector,
)
from dromeus.telemetry.events import BoundedEventSink, EventSink
from dromeus.telemetry.evidence import (
    ConsensusObservationEvidence,
    DivergenceStatusEvidence,
    EvidenceRecord,
    append_evidence,
)


class DivergenceMonitor:
    """A single daemon writer isolates even a blocked synchronous sink.

    Closing is bounded. An in-flight blocking sink cannot be killed by Python;
    it is counted as a drop on timeout and may finish later. No new writes follow
    that timeout. Status never depends on successful evidence delivery.
    """

    def __init__(
        self,
        *,
        policy: DivergencePolicy,
        participant_count: int,
        run_id: str,
        manifest_hash: str,
        node_id: str,
        sink: EventSink | None,
        queue_size: int = 64,
    ) -> None:
        if queue_size <= 0:
            raise ValueError("warning queue size must be positive")
        self.detector = DivergenceDetector(policy, participant_count=participant_count)
        self._identity = dict(
            run_id=run_id, manifest_hash=manifest_hash, node_id=node_id
        )
        self._policy_hash = canonical_hash(policy)
        self._owns_delivery = not isinstance(sink, BoundedEventSink)
        self._delivery = (
            sink
            if isinstance(sink, BoundedEventSink)
            else BoundedEventSink(sink, capacity=queue_size)
        )
        self._lock = RLock()

    @property
    def snapshot(self) -> DetectorSnapshot:
        return self.detector.snapshot

    @property
    def dropped(self) -> int:
        return self._delivery.dropped

    def start(self) -> None:
        self._delivery.start()

    def _enqueue(self, record: EvidenceRecord) -> None:
        append_evidence(self._delivery, record)

    def observe(self, observation: ConsensusObservation) -> None:
        with self._lock:
            ignored = (
                self.detector.ignored_observations + self.detector.invalid_observations
            )
            changes = self.detector.observe(observation)
            if (
                ignored
                == self.detector.ignored_observations
                + self.detector.invalid_observations
            ):
                self._enqueue(
                    ConsensusObservationEvidence(
                        run_id=self._identity["run_id"],
                        manifest_hash=self._identity["manifest_hash"],
                        node_id=self._identity["node_id"],
                        message_id=f"consensus-observation-{observation.round_id}",
                        round_id=observation.round_id,
                        normalized_rms=observation.normalized_rms,
                        absolute_rms_spread=observation.absolute_rms_spread,
                        mean_sketch_norm=observation.mean_sketch_norm,
                        denominator_degenerate=observation.denominator_degenerate,
                        sketch_count=observation.sketch_count,
                    )
                )
            self._publish(changes)

    def advance(self, committed_round: int) -> None:
        with self._lock:
            self._publish(self.detector.advance(committed_round))

    def _publish(self, transitions: tuple[DetectorTransition, ...]) -> None:
        for change in transitions:
            value = {
                **self._identity,
                **asdict(change.snapshot),
                "state": change.snapshot.state.value,
                "previous_state": change.previous_state.value,
                "transition_id": change.sequence,
                "policy_hash": self._policy_hash,
                "threshold_set_id": self.detector.policy.threshold_set_id,
            }
            self._enqueue(DivergenceStatusEvidence.model_validate(value))

    def stop(self, *, timeout_seconds: float = 0.2) -> bool:
        if self._owns_delivery:
            return self._delivery.stop(timeout_seconds=timeout_seconds)
        return True
