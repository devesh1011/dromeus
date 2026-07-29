from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path

from dromeus.telemetry.events import EventSink, JsonlEventSink
from dromeus.telemetry.metrics import JsonlMetricsPublisher, RoundTiming


class _FailingSink:
    def append(self, record: Mapping[str, object]) -> None:
        del record
        raise OSError("sink unavailable")


def test_jsonl_metrics_publisher_writes_correlated_round_and_failure_records(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        path = tmp_path / "node.jsonl"
        publisher = JsonlMetricsPublisher(
            sink=JsonlEventSink(path),
            run_id="run-1",
            manifest_hash="a" * 64,
            node_id="peer-0",
        )
        await publisher.start()
        assert publisher.submit(
            RoundTiming(
                round_id=3,
                peer_id="peer-1",
                local_compute_seconds=1.25,
                peer_wait_seconds=0.5,
                transfer_seconds=0.2,
                mixing_seconds=0.01,
                evaluation_seconds=0.3,
                retries=2,
                local_loss=0.75,
                evaluation_loss=0.6,
                evaluation_accuracy=0.8,
                transfer_id="transfer-3",
            )
        )
        assert publisher.submit_failure(
            round_id=4,
            error_type="PairCommitError",
            reason="peer timeout",
        )
        await publisher.stop()

        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert records[0] == {
            "evidence_version": 1,
            "event": "round_metrics",
            "evaluation_accuracy": 0.8,
            "evaluation_loss": 0.6,
            "evaluation_seconds": 0.3,
            "local_compute_seconds": 1.25,
            "local_loss": 0.75,
            "manifest_hash": "a" * 64,
            "message_id": "metric-peer-0-3",
            "mixing_seconds": 0.01,
            "node_id": "peer-0",
            "peer_id": "peer-1",
            "peer_wait_seconds": 0.5,
            "retries": 2,
            "round_id": 3,
            "run_id": "run-1",
            "transfer_id": "transfer-3",
            "transfer_seconds": 0.2,
            "timestamp": records[0]["timestamp"],
        }
        assert records[1]["event"] == "run_failed"
        assert records[1]["message_id"] == "run-failure-peer-0-4"
        assert records[1]["round_id"] == 4
        assert records[1]["error_type"] == "PairCommitError"

    asyncio.run(run())


def test_round_timing_rejects_invalid_metric_values() -> None:
    try:
        RoundTiming(
            round_id=0,
            peer_id="peer-1",
            local_compute_seconds=-1,
            peer_wait_seconds=0,
            transfer_seconds=0,
            mixing_seconds=0,
            evaluation_seconds=0,
        )
    except ValueError as error:
        assert str(error) == "metric durations must be finite and non-negative"
    else:
        raise AssertionError("invalid metric duration was accepted")


def test_metrics_queue_full_and_sink_failure_are_best_effort() -> None:
    timing = RoundTiming(
        round_id=0,
        peer_id="peer-1",
        local_compute_seconds=0,
        peer_wait_seconds=0,
        transfer_seconds=0,
        mixing_seconds=0,
        evaluation_seconds=0,
    )
    queued = JsonlMetricsPublisher(
        sink=_FailingSink(),
        run_id="run-1",
        manifest_hash="a" * 64,
        node_id="peer-0",
        max_queue_size=1,
    )
    assert queued.submit(timing)
    assert not queued.submit(timing)
    assert queued.dropped == 1

    async def run_sink_failure(sink: EventSink) -> None:
        publisher = JsonlMetricsPublisher(
            sink=sink,
            run_id="run-1",
            manifest_hash="a" * 64,
            node_id="peer-0",
        )
        await publisher.start()
        assert publisher.submit(timing)
        await publisher.stop()
        assert publisher.dropped == 1

    asyncio.run(run_sink_failure(_FailingSink()))
