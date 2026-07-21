from __future__ import annotations

import json
from pathlib import Path

from dromeus.telemetry.events import JsonlEventSink, emit_event


def test_jsonl_event_sink_appends_correlated_records(tmp_path: Path) -> None:
    path = tmp_path / "node.jsonl"
    sink = JsonlEventSink(path)

    emit_event(
        "round_metric",
        sink=sink,
        run_id="run-1",
        manifest_hash="a" * 64,
        node_id="peer-0",
        message_id="message-1",
        transfer_id="transfer-1",
        peer_id="peer-1",
        round_id=3,
        loss=0.25,
    )
    emit_event("round_complete", sink=sink, run_id="run-1", node_id="peer-0")

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["manifest_hash"] == "a" * 64
    assert records[0]["node_id"] == "peer-0"
    assert records[0]["peer_id"] == "peer-1"
    assert records[0]["loss"] == 0.25
