"""Warning delivery must not block the training or detector state paths."""

import threading
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from dromeus.manifests.models import DivergencePolicy
from dromeus.telemetry.consensus import consensus_observation
from dromeus.telemetry.divergence import DetectorState
from dromeus.telemetry.divergence_monitor import DivergenceMonitor


def test_blocked_sink_does_not_block_progress_and_reports_drops() -> None:
    entered, release = threading.Event(), threading.Event()

    class SlowSink:
        def append(self, record: Mapping[str, object]) -> None:
            entered.set()
            release.wait(5)

    policy = DivergencePolicy(
        threshold_set_id="synthetic",
        warmup_rounds=0,
        window_rounds=4,
        patience_windows=2,
        distance_floor=0.01,
        growth_ratio=1.5,
        min_log_slope=0.1,
        severe_distance=0.5,
        severe_patience=2,
        recovery_windows=2,
        recovery_distance=0.1,
        max_lag_rounds=1,
    )
    monitor = DivergenceMonitor(
        policy=policy,
        participant_count=4,
        run_id="test",
        manifest_hash="a" * 64,
        node_id="node",
        sink=SlowSink(),
        queue_size=2,
    )
    monitor.start()
    try:
        # Alternating spread produces many status transitions; the monitor
        # emits only transitions, so this is what fills the blocked queue.
        for i in range(40):
            monitor.advance(i)
            spread = [1, 3] * 2 if (i // 2) % 2 == 0 else [1, 1] * 2
            monitor.observe(
                consensus_observation(
                    [np.array([x], dtype=np.float32) for x in spread],
                    round_id=i,
                )
            )
        assert entered.wait(1)
        assert monitor.snapshot.state == DetectorState.HEALTHY
        assert monitor.dropped > 0
        assert not monitor.stop(timeout_seconds=0.01)
        assert monitor.dropped > 0
    finally:
        release.set()
        monitor.stop(timeout_seconds=1)


def test_duplicate_callbacks_emit_only_transitions_not_observations(
    tmp_path: Path,
) -> None:

    from dromeus.telemetry.events import JsonlEventSink
    from dromeus.telemetry.evidence import (
        ConsensusObservationEvidence,
        DivergenceStatusEvidence,
        EvidenceLog,
    )

    policy = DivergencePolicy(
        threshold_set_id="synthetic",
        warmup_rounds=0,
        window_rounds=4,
        patience_windows=2,
        distance_floor=0.01,
        growth_ratio=1.5,
        min_log_slope=0.1,
        severe_distance=0.1,
        severe_patience=2,
        recovery_windows=2,
        recovery_distance=0.05,
        max_lag_rounds=1,
    )
    sink = JsonlEventSink(tmp_path / "events.jsonl")
    monitor = DivergenceMonitor(
        policy=policy,
        participant_count=4,
        run_id="test",
        manifest_hash="a" * 64,
        node_id="node",
        sink=sink,
    )
    monitor.start()
    for i in range(5):
        monitor.advance(i)
        observation = consensus_observation(
            [np.array([1], dtype=np.float32), np.array([3], dtype=np.float32)] * 2,
            round_id=i,
        )
        monitor.observe(observation)
        monitor.observe(observation)
        monitor.advance(i)
    assert monitor.stop(timeout_seconds=1)
    log = EvidenceLog.open(sink.path, run_id="test", manifest_hash="a" * 64)
    changes = [r for r in log.records if isinstance(r, DivergenceStatusEvidence)]
    assert [r.state for r in changes] == ["suspect", "diverging"]
    # The runtime, not the monitor, records per-round observations.
    assert not any(
        isinstance(r, ConsensusObservationEvidence) for r in log.records
    )
    assert len(log.records) == 2
    with sink.path.open("a") as f:
        from dromeus.telemetry.evidence import encode_evidence

        f.write(encode_evidence(changes[-1]).decode() + "\n")
    import pytest

    from dromeus.telemetry.evidence import EvidenceError

    with pytest.raises(EvidenceError, match="transition"):
        EvidenceLog.open(sink.path, run_id="test", manifest_hash="a" * 64)
