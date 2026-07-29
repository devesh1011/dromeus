from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from dromeus.telemetry.events import JsonlEventSink, emit_event
from dromeus.telemetry.evidence import (
    BenchmarkNodeReadyEvidence,
    EvidenceError,
    EvidenceLog,
    RoundMetricsEvidence,
    append_evidence,
    decode_evidence,
    encode_evidence,
    evidence_record,
)


class _FailingSink:
    def append(self, record: Mapping[str, object]) -> None:
        del record
        raise OSError("sink unavailable")


def _round_metrics(**updates: object) -> RoundMetricsEvidence:
    values: dict[str, object] = {
        "run_id": "run-1",
        "manifest_hash": "a" * 64,
        "node_id": "peer-0",
        "message_id": "metric-peer-0-3",
        "transfer_id": "transfer-3",
        "peer_id": "peer-1",
        "round_id": 3,
        "local_loss": 0.75,
        "evaluation_loss": 0.6,
        "evaluation_accuracy": 0.8,
        "local_compute_seconds": 1.25,
        "peer_wait_seconds": 0.5,
        "transfer_seconds": 0.2,
        "mixing_seconds": 0.01,
        "evaluation_seconds": 0.3,
        "retries": 2,
    }
    values.update(updates)
    return RoundMetricsEvidence.model_validate(values)


def test_typed_evidence_jsonl_round_trip_ignores_flexible_diagnostics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "node.jsonl"
    sink = JsonlEventSink(path)
    emit_event("debug_detail", sink=sink, run_id="run-1", detail="ignored")
    expected = _round_metrics()

    assert append_evidence(sink, expected)
    log = EvidenceLog.open(
        path,
        run_id="run-1",
        manifest_hash="a" * 64,
    )

    assert log.node_id == "peer-0"
    assert log.records == (expected,)
    assert decode_evidence(encode_evidence(expected)) == expected


def _invalid_records() -> tuple[dict[str, object], ...]:
    missing = evidence_record(_round_metrics())
    del missing["round_id"]
    return (
        missing,
        {**evidence_record(_round_metrics()), "unexpected": True},
        {**evidence_record(_round_metrics()), "round_id": "3"},
        {**evidence_record(_round_metrics()), "local_loss": float("nan")},
        {**evidence_record(_round_metrics()), "evaluation_loss": float("inf")},
        {**evidence_record(_round_metrics()), "evidence_version": 2},
    )


@pytest.mark.parametrize("value", _invalid_records())
def test_strict_evidence_rejects_invalid_records(
    value: dict[str, object],
) -> None:
    with pytest.raises(EvidenceError):
        decode_evidence(json.dumps(value))


def test_known_evidence_event_without_version_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "node.jsonl"
    value = evidence_record(_round_metrics())
    del value["evidence_version"]
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(EvidenceError, match="invalid evidence"):
        EvidenceLog.open(
            path,
            run_id="run-1",
            manifest_hash="a" * 64,
        )


def test_evidence_log_rejects_identity_changes(tmp_path: Path) -> None:
    path = tmp_path / "node.jsonl"
    sink = JsonlEventSink(path)
    assert append_evidence(
        sink,
        BenchmarkNodeReadyEvidence(
            run_id="run-1",
            manifest_hash="a" * 64,
            node_id="peer-0",
            benchmark_seed=17,
            transport="axl",
        ),
    )
    assert append_evidence(
        sink,
        _round_metrics(node_id="peer-1"),
    )

    with pytest.raises(EvidenceError, match="node id mismatch"):
        EvidenceLog.open(
            path,
            run_id="run-1",
            manifest_hash="a" * 64,
        )
    with pytest.raises(EvidenceError, match="run id mismatch"):
        EvidenceLog.open(
            path,
            run_id="other-run",
            manifest_hash="a" * 64,
        )


def test_evidence_log_rejects_duplicate_round_identity(tmp_path: Path) -> None:
    path = tmp_path / "node.jsonl"
    sink = JsonlEventSink(path)
    assert append_evidence(sink, _round_metrics())
    assert append_evidence(
        sink,
        _round_metrics(message_id="metric-peer-0-3-duplicate"),
    )

    with pytest.raises(EvidenceError, match="duplicate evidence round"):
        EvidenceLog.open(
            path,
            run_id="run-1",
            manifest_hash="a" * 64,
        )


def test_best_effort_evidence_write_contains_sink_failure() -> None:
    assert not append_evidence(_FailingSink(), _round_metrics())
    assert not append_evidence(None, _round_metrics())
