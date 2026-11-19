from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from pathlib import Path

import pytest

from dromeus.errors import ConfigError
from dromeus.telemetry.events import event_record
from dromeus.telemetry.evidence import (
    ConsensusObservationEvidence,
    RoundMetricsEvidence,
    RunFailedEvidence,
    evidence_record,
)
from dromeus.telemetry.progress import (
    ConsensusMeasured,
    NodeReady,
    NodeStarted,
    Progress,
    RoundCommitted,
    RunCompleted,
    RunFailed,
    parse_progress,
    read_progress,
)

RUN_ID = "run-1"
MANIFEST = "a" * 64


def _round_record(**updates: object) -> dict[str, object]:
    values: dict[str, object] = {
        "run_id": RUN_ID,
        "manifest_hash": MANIFEST,
        "node_id": "peer-0",
        "message_id": "metric-3",
        "peer_id": "peer-1",
        "round_id": 3,
        "local_loss": 0.75,
        "evaluation_metrics": {"loss": 0.6, "accuracy": 0.8},
        "local_compute_seconds": 1.25,
        "peer_wait_seconds": 0.5,
        "transfer_seconds": 0.2,
        "mixing_seconds": 0.01,
        "evaluation_seconds": 0.3,
        "retries": 2,
        "encoded_artifact_bytes": 4096,
    }
    values.update(updates)
    return evidence_record(RoundMetricsEvidence.model_validate(values))


def _without(record: Mapping[str, object], key: str) -> dict[str, object]:
    return {name: value for name, value in record.items() if name != key}


def _consensus_record() -> dict[str, object]:
    return evidence_record(
        ConsensusObservationEvidence.model_validate(
            {
                "run_id": RUN_ID,
                "manifest_hash": MANIFEST,
                "node_id": "peer-0",
                "message_id": "consensus-3",
                "round_id": 3,
                "normalized_rms": 0.25,
                "absolute_rms_spread": 0.5,
                "mean_sketch_norm": 2.0,
                "denominator_degenerate": False,
                "sketch_count": 3,
            }
        )
    )


def _line(record: Mapping[str, object]) -> bytes:
    return (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _log_path(run_root: Path) -> Path:
    return run_root / "logs" / "dromeus.jsonl"


def test_parse_maps_lifecycle_events() -> None:
    assert parse_progress(
        event_record("node_start", run_id=RUN_ID, role="peer")
    ) == NodeStarted(run_id=RUN_ID, role="peer")
    assert parse_progress(
        event_record("node_ready", run_id=RUN_ID, manifest_hash=MANIFEST, node_id="p0")
    ) == NodeReady(run_id=RUN_ID, manifest_hash=MANIFEST)
    assert parse_progress(
        event_record(
            "node_complete",
            run_id=RUN_ID,
            manifest_hash=MANIFEST,
            node_id="p0",
            committed_rounds=3,
        )
    ) == RunCompleted(committed_rounds=3)


def test_parse_round_metrics_sums_timing_and_keeps_fields() -> None:
    item = parse_progress(_round_record())

    assert isinstance(item, RoundCommitted)
    assert item.round_id == 3
    assert item.peer_id == "peer-1"
    assert item.local_loss == 0.75
    assert item.metrics == {"loss": 0.6, "accuracy": 0.8}
    assert item.round_seconds == pytest.approx(2.26)
    assert item.encoded_bytes == 4096


def test_parse_round_metrics_allows_absent_loss_and_bytes() -> None:
    record = _round_record(local_loss=None, encoded_artifact_bytes=None)
    item = parse_progress(record)

    assert isinstance(item, RoundCommitted)
    assert item.local_loss is None
    assert item.encoded_bytes is None
    assert item.metrics == {"loss": 0.6, "accuracy": 0.8}


def test_parse_accepts_integer_loss_as_float() -> None:
    record = _round_record()
    record["local_loss"] = 1

    item = parse_progress(record)

    assert isinstance(item, RoundCommitted)
    assert item.local_loss == 1.0
    assert isinstance(item.local_loss, float)


def test_parse_consensus_measured() -> None:
    assert parse_progress(_consensus_record()) == ConsensusMeasured(
        round_id=3, normalized_rms=0.25
    )


def test_parse_run_failed_from_run_or_formation_failure() -> None:
    run_failed = evidence_record(
        RunFailedEvidence.model_validate(
            {
                "run_id": RUN_ID,
                "manifest_hash": MANIFEST,
                "node_id": "peer-0",
                "message_id": "failure-3",
                "round_id": 3,
                "error_type": "PairCommitError",
                "error": "peer timed out",
            }
        )
    )
    formation_failed = event_record(
        "formation_failed",
        run_id=RUN_ID,
        error="formation control deadline exceeded",
    )

    assert parse_progress(run_failed) == RunFailed(error="peer timed out", round_id=3)
    assert parse_progress(formation_failed) == RunFailed(
        error="formation control deadline exceeded", round_id=None
    )


@pytest.mark.parametrize(
    "record",
    [
        pytest.param(event_record("divergence_status", run_id=RUN_ID), id="noise"),
        pytest.param(event_record("round_metric", run_id=RUN_ID), id="typo-event"),
        pytest.param({"timestamp": "t"}, id="no-event"),
        pytest.param(
            event_record("node_start", run_id=RUN_ID), id="start-missing-role"
        ),
        pytest.param({**_round_record(), "round_id": True}, id="round-bool-is-not-int"),
        pytest.param({**_round_record(), "round_id": 3.0}, id="round-float"),
        pytest.param(
            _without(_round_record(), "local_compute_seconds"),
            id="round-missing-timing",
        ),
        pytest.param({**_round_record(), "local_loss": "0.7"}, id="loss-string"),
        pytest.param(
            {**_round_record(), "evaluation_metrics": {"loss": True}},
            id="metric-bool-value",
        ),
        pytest.param(
            {**_round_record(), "evaluation_metrics": ["loss"]},
            id="metrics-not-mapping",
        ),
        pytest.param(
            {**_round_record(), "encoded_artifact_bytes": "4096"},
            id="bytes-string",
        ),
        pytest.param({**_round_record(), "peer_id": None}, id="round-null-peer"),
        pytest.param({**_round_record(), "round_id": None}, id="round-null-id"),
        pytest.param(
            event_record("node_complete", run_id=RUN_ID, committed_rounds="3"),
            id="complete-string-count",
        ),
        pytest.param(
            _without(_consensus_record(), "normalized_rms"),
            id="consensus-missing-rms",
        ),
        pytest.param(
            event_record("formation_failed", run_id=RUN_ID, error=7),
            id="failure-error-not-string",
        ),
        pytest.param(
            {
                **evidence_record(
                    RunFailedEvidence.model_validate(
                        {
                            "run_id": RUN_ID,
                            "manifest_hash": MANIFEST,
                            "node_id": "peer-0",
                            "message_id": "failure-3",
                            "round_id": 3,
                            "error_type": "PairCommitError",
                            "error": "peer timed out",
                        }
                    )
                ),
                "round_id": "3",
            },
            id="failure-round-string",
        ),
    ],
)
def test_parse_ignores_unknown_and_malformed_records(
    record: Mapping[str, object],
) -> None:
    assert parse_progress(record) is None


def test_non_follow_read_stops_at_terminal_event_and_skips_noise(
    tmp_path: Path,
) -> None:
    path = _log_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(
        b"".join(
            [
                _line(event_record("node_start", run_id=RUN_ID, role="peer")),
                _line(event_record("divergence_status", run_id=RUN_ID)),
                b"not json\n",
                b"[1, 2]\n",
                b"\n",
                _line(_round_record()),
                _line(event_record("node_complete", run_id=RUN_ID, committed_rounds=1)),
                _line(_round_record(round_id=4)),
            ]
        )
    )

    items = list(read_progress(path, follow=False))

    assert items == [
        NodeStarted(run_id=RUN_ID, role="peer"),
        RoundCommitted(
            round_id=3,
            peer_id="peer-1",
            local_loss=0.75,
            metrics={"loss": 0.6, "accuracy": 0.8},
            round_seconds=1.25 + 0.5 + 0.2 + 0.01 + 0.3,
            encoded_bytes=4096,
        ),
        RunCompleted(committed_rounds=1),
    ]


def test_non_follow_read_reaches_eof_without_terminal_event(tmp_path: Path) -> None:
    path = _log_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(
        _line(event_record("node_ready", run_id=RUN_ID, manifest_hash=MANIFEST))
    )

    assert list(read_progress(path, follow=False)) == [
        NodeReady(run_id=RUN_ID, manifest_hash=MANIFEST)
    ]


def test_non_follow_read_leaves_unterminated_last_line_unparsed(
    tmp_path: Path,
) -> None:
    path = _log_path(tmp_path)
    path.parent.mkdir(parents=True)
    complete = _line(event_record("node_start", run_id=RUN_ID, role="peer"))
    partial = _line(_consensus_record())[:-1]
    path.write_bytes(complete + partial)

    assert list(read_progress(path, follow=False)) == [
        NodeStarted(run_id=RUN_ID, role="peer")
    ]


def test_missing_log_raises_config_error_naming_path(tmp_path: Path) -> None:
    path = _log_path(tmp_path / "absent-run")

    with pytest.raises(ConfigError, match="node event log not found") as info:
        read_progress(path, follow=False)

    assert str(path) in str(info.value)


def test_follow_waits_for_late_file_and_yields_items_split_across_writes(
    tmp_path: Path,
) -> None:
    path = _log_path(tmp_path)
    lines = [
        _line(event_record("node_start", run_id=RUN_ID, role="répliques")),
        b"not json\n",
        _line(event_record("node_ready", run_id=RUN_ID, manifest_hash=MANIFEST)),
        _line(_round_record()),
        _line(_consensus_record()),
        _line(event_record("node_complete", run_id=RUN_ID, committed_rounds=1)),
        _line(_round_record(round_id=4)),
    ]
    payload = b"".join(lines)
    # Cut inside the JSON, inside the two-byte "é", and inside the second line.
    utf8_cut = payload.index("é".encode("utf-8")) + 1
    cuts = [10, utf8_cut, len(lines[0]) + 3]
    pieces = [
        payload[start:end]
        for start, end in zip([0, *cuts], [*cuts, len(payload)], strict=True)
    ]

    received: list[Progress] = []
    errors: list[Exception] = []

    def write_late() -> None:
        try:
            time.sleep(0.05)
            path.parent.mkdir(parents=True)
            with path.open("ab") as handle:
                for piece in pieces:
                    handle.write(piece)
                    handle.flush()
                    time.sleep(0.02)
        except Exception as error:
            errors.append(error)

    def follow() -> None:
        try:
            received.extend(read_progress(path, follow=True, poll_seconds=0.01))
        except Exception as error:
            errors.append(error)

    writer = threading.Thread(target=write_late)
    reader = threading.Thread(target=follow)
    reader.start()
    writer.start()
    writer.join(timeout=5)
    reader.join(timeout=5)

    assert not writer.is_alive()
    assert not reader.is_alive(), "follow mode did not stop at node_complete"
    assert errors == []
    assert received == [
        NodeStarted(run_id=RUN_ID, role="répliques"),
        NodeReady(run_id=RUN_ID, manifest_hash=MANIFEST),
        RoundCommitted(
            round_id=3,
            peer_id="peer-1",
            local_loss=0.75,
            metrics={"loss": 0.6, "accuracy": 0.8},
            round_seconds=1.25 + 0.5 + 0.2 + 0.01 + 0.3,
            encoded_bytes=4096,
        ),
        ConsensusMeasured(round_id=3, normalized_rms=0.25),
        RunCompleted(committed_rounds=1),
    ]
