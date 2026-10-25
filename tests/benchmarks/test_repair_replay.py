"""Historical public traces expose false alerts and pre-active fault masking."""

import json
from pathlib import Path

import pytest

from benchmarks.m3.repair_replay import main, replay_fixture
from dromeus.telemetry.evidence import EvidenceError

FIXTURE = Path(__file__).parents[1] / "fixtures/divergence_repair/historical.json"


def test_historical_minimal_traces_reproduce_the_three_warning_outcomes() -> None:
    report = replay_fixture(FIXTURE)
    cases = {case["run_id"]: case for case in report["cases"]}
    healthy = cases["final-baseline-s29"]
    assert healthy["alerts"] == [
        {"source_round": 19, "detection_round": 19, "reason": "sustained_growth"}
    ]
    masked = cases["final-fault-s29-retry1"]
    assert masked["false_alerts"] == 1
    assert masked["missed_fault"] is True
    assert masked["warning_active_at_fault"] is True
    detected = cases["final-fault-s41"]
    assert detected["delay_rounds"] == 2
    assert detected["alerts"] == [
        {"source_round": 27, "detection_round": 27, "reason": "severe_distance"}
    ]
    assert report["readiness"]["passed"] is False
    assert healthy["rounds"][-1]["window_start"] == 12
    assert healthy["rounds"][-1]["log_slope"] is not None


def test_development_candidate_does_not_rewrite_the_original_failure() -> None:
    report = replay_fixture(FIXTURE, overrides={"growth_ratio": 3.0})
    assert report["mode"] == "historical_development_replay"
    assert report["metrics"] == {
        "false_alerts": 0,
        "missed_fault_nodes": 0,
        "detected_fault_nodes": 2,
        "max_delay_rounds": 2,
    }
    assert report["readiness"]["passed"] is True
    assert report["original_policy_hash"] != report["policy_hash"]
    assert replay_fixture(FIXTURE)["readiness"]["passed"] is False


def test_changed_public_measurement_is_rejected(tmp_path: Path) -> None:
    content = json.loads(FIXTURE.read_text())
    content["cases"][0]["observations"][19]["normalized_rms"] = 0.0
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(content))
    with pytest.raises(EvidenceError, match="content hash"):
        replay_fixture(changed)


def test_cli_reports_failed_original_and_labels_development_override(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["repair-replay", "--fixture", str(FIXTURE)])
    with pytest.raises(SystemExit) as original:
        main()
    assert original.value.code == 1
    assert json.loads(capsys.readouterr().out)["readiness"]["passed"] is False
    monkeypatch.setattr(
        "sys.argv",
        ["repair-replay", "--fixture", str(FIXTURE), "--probe", "growth_ratio=3"],
    )
    with pytest.raises(SystemExit) as candidate:
        main()
    assert candidate.value.code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["readiness"]["passed"] is True
    assert output["mode"] == "historical_development_replay"
    assert output["overrides"] == {"growth_ratio": 3}


def test_incomplete_regression_suite_cannot_report_readiness(tmp_path: Path) -> None:
    import hashlib

    document = json.loads(FIXTURE.read_text())
    healthy = next(
        case for case in document["cases"] if case["fault_start_round"] is None
    )
    # A deliberately permissive valid development policy removes the healthy
    # false entry; lack of fault evidence must still prevent a passing verdict.
    document["cases"] = [healthy]
    unsigned = {
        key: value for key, value in document.items() if key != "content_sha256"
    }
    document["content_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    healthy_only = tmp_path / "healthy-only.json"
    healthy_only.write_text(json.dumps(document))
    report = replay_fixture(healthy_only, overrides={"growth_ratio": 3})
    assert report["readiness"]["passed"] is False
    assert "insufficient_fault_evidence" in report["readiness"]["failures"]

    document["cases"] = []
    unsigned = {
        key: value for key, value in document.items() if key != "content_sha256"
    }
    document["content_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps(document))
    with pytest.raises(EvidenceError, match="nonempty"):
        replay_fixture(empty)
