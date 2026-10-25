"""Read-only repair diagnostics; historical failed policies exit with status one.

Full campaigns reuse the strict M3 loader and assessment. Small public fixtures
retain measured scale channels and provenance, but are regression/development
evidence, never replacement live or full-horizon validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Retain the historical loader's exact identity/scale checks; do not fork it.
from benchmarks.m3.calibrate_detector import (
    _load,  # pyright: ignore[reportPrivateUsage]
    evaluate,
)
from dromeus.manifests.canonical import canonical_hash, file_sha256
from dromeus.manifests.models import DivergencePolicy
from dromeus.telemetry.consensus import ConsensusObservation
from dromeus.telemetry.divergence import DivergenceDetector
from dromeus.telemetry.evidence import EvidenceError


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def readiness(metrics: dict[str, Any]) -> dict[str, Any]:
    """Explicit engineering verdict, separate from successfully parsing/replaying."""
    failures: list[str] = []
    if metrics["detected_fault_nodes"] + metrics["missed_fault_nodes"] == 0:
        failures.append("insufficient_fault_evidence")
    if metrics["false_alerts"]:
        failures.append("false_warning_entries")
    if metrics["missed_fault_nodes"]:
        failures.append("missed_faults")
    if metrics["max_delay_rounds"] is not None and metrics["max_delay_rounds"] > 10:
        failures.append("detection_delay_above_10_rounds")
    return {
        "passed": not failures,
        "failures": failures,
        "scope": "declared replay suite; not production certification",
    }


def replay_case(case: dict[str, Any], policy: DivergencePolicy) -> dict[str, Any]:
    """Replay full recorded observations and expose every contributing window."""
    detector = DivergenceDetector(policy, participant_count=case["participant_count"])
    alerts: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    fault_start = case["fault_start_round"]
    active = False
    active_at_fault = False
    expected_round = 0
    for raw in case["observations"]:
        point = ConsensusObservation(**raw)
        if point.round_id != expected_round:
            raise EvidenceError(
                "repair fixture observations must be consecutive from zero"
            )
        expected_round += 1
        if point.round_id == fault_start:
            active_at_fault = detector.snapshot.warning_active
        changes = (*detector.advance(point.round_id), *detector.observe(point))
        for change in changes:
            snapshot = change.snapshot
            if snapshot.warning_active and not active:
                alerts.append(
                    {
                        "source_round": snapshot.source_round,
                        "detection_round": snapshot.detection_round,
                        "reason": snapshot.reason,
                    }
                )
            active = snapshot.warning_active
        rounds.append(asdict(detector.snapshot))
    if not rounds or (fault_start is not None and expected_round <= fault_start):
        raise EvidenceError("repair fixture must cover its declared fault onset")
    false = [
        a for a in alerts if fault_start is None or a["source_round"] < fault_start
    ]
    true = [a for a in alerts if a not in false]
    return {
        "run_id": case["run_id"],
        "node_id": case["node_id"],
        "fault_start_round": fault_start,
        "alerts": alerts,
        "false_alerts": len(false),
        "missed_fault": fault_start is not None and not true,
        "delay_rounds": true[0]["detection_round"] - fault_start if true else None,
        "warning_active_at_fault": active_at_fault,
        "rounds": rounds,
        "source": case["source"],
    }


def _development_policy(
    policy: DivergencePolicy, overrides: dict[str, Any] | None
) -> DivergencePolicy:
    if not overrides:
        return policy
    if "version" in overrides or "threshold_set_id" in overrides:
        raise EvidenceError("development probes cannot replace policy version/identity")
    values = policy.model_dump(mode="json") | overrides
    values["threshold_set_id"] = "development-" + _digest(overrides)[:16]
    return DivergencePolicy.model_validate(values)


def _metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    delays = [
        case["delay_rounds"] for case in cases if case["delay_rounds"] is not None
    ]
    return {
        "false_alerts": sum(case["false_alerts"] for case in cases),
        "missed_fault_nodes": sum(case["missed_fault"] for case in cases),
        "detected_fault_nodes": len(delays),
        "max_delay_rounds": max(delays, default=None),
    }


def replay_fixture(
    path: Path, *, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Verify a portable public trace capture, then replay without large archives."""
    document = json.loads(path.read_text())
    expected = document.get("content_sha256")
    if document.get("version") != 1 or expected != _digest(
        {k: v for k, v in document.items() if k != "content_sha256"}
    ):
        raise EvidenceError("repair fixture content hash/version mismatch")
    policy = DivergencePolicy.model_validate(document["policy"])
    if canonical_hash(policy) != document["policy_hash"]:
        raise EvidenceError("repair fixture policy hash mismatch")
    if not document["cases"]:
        raise EvidenceError("repair fixture requires a nonempty case set")
    policy = _development_policy(policy, overrides)
    cases = [replay_case(case, policy) for case in document["cases"]]
    metrics = _metrics(cases)
    return {
        "version": 1,
        "mode": "historical_development_replay"
        if overrides
        else "minimal_public_regression",
        "original_policy_hash": document["policy_hash"],
        "overrides": overrides or {},
        "fixture_sha256": file_sha256(path),
        "provenance": document["provenance"],
        "policy_hash": canonical_hash(policy),
        "metrics": metrics,
        "cases": cases,
        "readiness": readiness(metrics),
        "limitation": (
            "Truncated single-monitor traces; not full campaign or live validation."
        ),
    }


def replay_campaign(
    index: Path, frozen_path: Path, *, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Retain existing assessment scoring and add window-level production snapshots."""
    frozen = json.loads(frozen_path.read_text())
    assessed = evaluate(index, frozen, require_live=True)
    policy = _development_policy(
        DivergencePolicy.model_validate(frozen["policy"]), overrides
    )
    runs = _load(index, "validation")
    cases = [
        replay_case(
            {
                "run_id": run.spec.run_id,
                "node_id": node.node_id,
                "participant_count": run.manifest.participant_count,
                "fault_start_round": run.spec.fault.start_round
                if run.spec.fault
                else None,
                "observations": [asdict(point) for point in node.observations],
                "source": {
                    "log_sha256": node.sha256,
                    "manifest_sha256": run.manifest_sha256,
                },
            },
            policy,
        )
        for run in runs
        for node in run.nodes
    ]
    metrics = _metrics(cases) if overrides else assessed["metrics"]
    return {
        "version": 1,
        "mode": "historical_development_replay"
        if overrides
        else "full_historical_assessment",
        "original_policy_hash": frozen["policy_hash"],
        "policy_hash": canonical_hash(policy),
        "overrides": overrides or {},
        "metrics": metrics,
        "source_index_sha256": file_sha256(index),
        "frozen_policy_sha256": file_sha256(frozen_path),
        "assessment": assessed,
        "cases": cases,
        "readiness": readiness(metrics),
        "limitation": (
            "Overrides are historical development replay, not revised live validation."
        ),
    }


def capture_fixture(index: Path, frozen_path: Path, output: Path) -> None:
    """Capture the shortest prefixes establishing the three recorded symptoms.

    The full loader verifies identities, scales and sealed fault metadata first.
    Prefix endpoints are the false entry, post-fault successful detection, and
    the same post-fault round for the already-active case. No observations change.
    """
    frozen = json.loads(frozen_path.read_text())
    assessed = evaluate(index, frozen, require_live=True)
    selected = {
        "final-baseline-s29": 19,
        "final-fault-s29-retry1": 27,
        "final-fault-s41": 27,
    }
    cases: list[dict[str, Any]] = []
    for run in _load(index, "validation"):
        if run.spec.run_id not in selected:
            continue
        node = run.nodes[0]
        cases.append(
            {
                "run_id": run.spec.run_id,
                "node_id": node.node_id,
                "participant_count": run.manifest.participant_count,
                "fault_start_round": run.spec.fault.start_round
                if run.spec.fault
                else None,
                "observations": [
                    asdict(point)
                    for point in node.observations
                    if point.round_id <= selected[run.spec.run_id]
                ],
                "source": {
                    "log": str(
                        Path(node.path)
                        .resolve()
                        .relative_to(index.resolve().parents[1])
                    ),
                    "log_sha256": node.sha256,
                    "manifest_sha256": run.manifest_sha256,
                    "source_observation_count": len(node.observations),
                    "capture": (
                        "exact loader observations, prefix only; "
                        "scale values not synthesized"
                    ),
                },
            }
        )
    if len(cases) != len(selected):
        raise EvidenceError("historical capture is missing a required symptom run")
    document: dict[str, Any] = {
        "version": 1,
        "policy": frozen["policy"],
        "policy_hash": frozen["policy_hash"],
        "provenance": {
            "index_sha256": file_sha256(index),
            "frozen_policy_file_sha256": file_sha256(frozen_path),
            "freeze_hash": frozen["freeze_hash"],
            "original_metrics": assessed["metrics"],
            "purpose": "public development regression; not untouched assessment",
        },
        "cases": cases,
    }
    document["content_sha256"] = _digest(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fixture", type=Path)
    source.add_argument("--index", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--capture-fixture", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--probe",
        action="append",
        default=[],
        metavar="FIELD=JSON",
        help="Development-only policy override; repeat for multiple fields",
    )
    args = parser.parse_args()
    if args.index is not None and args.frozen is None:
        parser.error("--index requires --frozen")
    overrides: dict[str, Any] = {}
    for probe in args.probe:
        try:
            field, raw = probe.split("=", 1)
            if field in overrides:
                parser.error("duplicate development probe field")
            overrides[field] = json.loads(raw)
        except ValueError:
            parser.error("--probe requires FIELD=JSON")
    if args.capture_fixture:
        if args.index is None:
            parser.error("--capture-fixture requires --index and --frozen")
        capture_fixture(args.index, args.frozen, args.capture_fixture)
        report = replay_fixture(args.capture_fixture, overrides=overrides)
    elif args.fixture:
        report = replay_fixture(args.fixture, overrides=overrides)
    else:
        assert args.index is not None and args.frozen is not None
        report = replay_campaign(args.index, args.frozen, overrides=overrides)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    metrics = report.get("metrics", report.get("assessment", {}).get("metrics"))
    print(
        json.dumps(
            {
                "mode": report["mode"],
                "overrides": report.get("overrides", {}),
                "metrics": metrics,
                "readiness": report["readiness"],
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0 if report["readiness"]["passed"] else 1)


if __name__ == "__main__":
    main()
