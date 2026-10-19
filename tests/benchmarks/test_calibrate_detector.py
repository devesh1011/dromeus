"""Detector calibration is separate from held-out assessment and live evidence."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from support.sample_manifest import manifest_data

from benchmarks.m3.calibrate_detector import calibrate, evaluate
from benchmarks.m3.final_report import expected_drafts
from benchmarks.m3.final_runner import FrozenRunSpec, PublicFault
from benchmarks.m3.heterogeneous_data import SOURCES
from dromeus.manifests.canonical import canonical_hash
from dromeus.manifests.models import DraftRunSpec, SealedManifest
from dromeus.telemetry.evidence import EvidenceError


def write_index(root: Path, *, seed: int = 17, split: str = "calibration") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for label in ("healthy", "faulted"):
        data = manifest_data()
        run_id = f"{split}-{seed}-{label}"
        configuration = FrozenRunSpec(
            run_id=run_id,
            source_commit="a" * 40,
            torch_version=str(torch.__version__),
            arm="private",
            split_manifest_sha256="b" * 64,
            input_sha256={
                f"{source}-{partition}.npz": "c" * 64
                for source in SOURCES
                for partition in ("train", "calibration")
            },
            evaluation_partition="calibration",
            seed=seed,
            steps=500,
            fault=PublicFault() if label == "faulted" else None,
        )
        _, draft = expected_drafts(configuration)
        data.update(draft.model_dump(mode="json"))
        data["draft_hash"] = canonical_hash(draft)
        manifest = SealedManifest.model_validate(data)
        manifest_path = root / f"{label}-manifest.json"
        manifest_path.write_text(manifest.model_dump_json())
        logs: list[str] = []
        for rank in range(4):
            rows: list[dict[str, object]] = []
            for round_id in range(50):
                # Exact synthetic scalar-sketch geometry is immaterial to replay;
                # each scale-aware observation is the measured group statistic.
                value = 0.01 if label == "healthy" or round_id < 25 else 0.2
                rows.append(
                    {
                        "timestamp": "2026-10-21T00:00:00Z",
                        "evidence_version": 2,
                        "event": "consensus_observation",
                        "run_id": run_id,
                        "manifest_hash": canonical_hash(manifest),
                        "node_id": f"peer-{rank}",
                        "message_id": f"observation-{round_id}",
                        "round_id": round_id,
                        "normalized_rms": value,
                        "absolute_rms_spread": value * (1 + 1e-12),
                        "mean_sketch_norm": 1,
                        "denominator_degenerate": False,
                        "sketch_count": 4,
                    }
                )
            log = root / f"{label}-{rank}.jsonl"
            log.write_text("".join(json.dumps(row) + "\n" for row in rows))
            logs.append(log.name)
        runs.append(
            {
                "run_id": run_id,
                "split": split,
                "seed": seed,
                "label": label,
                "manifest": manifest_path.name,
                "logs": logs,
                "configuration": configuration.model_dump(mode="json"),
                "fault": {
                    "kind": "public_post_local_offset",
                    "start_round": 25,
                    "metadata": {"rank": 0, "scale": 0.05, "growth": 1.15},
                }
                if label == "faulted"
                else None,
            }
        )
    path = root / "index.json"
    path.write_text(json.dumps({"version": 1, "runs": runs}))
    return path


def test_freeze_uses_calibration_only_and_scores_heldout_without_refitting(
    tmp_path: Path,
) -> None:
    calibration = write_index(tmp_path / "calibration")
    frozen = calibrate(calibration)
    assert frozen["status"] == "frozen"
    assert frozen["metrics"]["healthy_run_false_warning_rate"] == 0
    assert frozen["metrics"]["fault_node_recall"] == 1
    assert frozen["metrics"]["delay_rounds"] == [1, 1, 1, 1]
    assert frozen["policy"]["severe_patience"] == 2
    assert frozen["source_index_sha256"]
    heldout = write_index(tmp_path / "heldout", seed=29, split="validation")
    report = evaluate(heldout, frozen)
    assert report["policy_hash"] == frozen["policy_hash"]
    assert report["metrics"]["fault_node_recall"] == 1
    assert report["live_status"] == "unavailable"
    assert report["live_metrics"] is None


def rewrite_rows(path: Path, change: Callable[[dict[str, Any]], None]) -> None:
    rows = [json.loads(row) for row in path.read_text().splitlines()]
    for row in rows:
        change(row)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_unobservable_fault_is_reported_as_missed_and_does_not_freeze(
    tmp_path: Path,
) -> None:
    index = write_index(tmp_path)
    for path in tmp_path.glob("faulted-*.jsonl"):
        rewrite_rows(
            path,
            lambda row: row.update(
                normalized_rms=0.01, absolute_rms_spread=0.01 * (1 + 1e-12)
            ),
        )
    report = calibrate(index)
    assert report["status"] == "blocked"
    assert report["policy"] is None
    assert report["metrics"]["missed_fault_nodes"] == 4
    assert report["metrics"]["alert_precision"] is None
    with pytest.raises(EvidenceError, match="passing calibration freeze"):
        evaluate(index, report)


@pytest.mark.parametrize(
    "problem", ["missing", "identity", "coverage", "scale", "legacy"]
)
def test_incomplete_or_inconsistent_evidence_cannot_calibrate(
    tmp_path: Path, problem: str
) -> None:
    index = write_index(tmp_path)
    path = tmp_path / "healthy-0.jsonl"
    if problem == "missing":
        path.write_text("\n".join(path.read_text().splitlines()[1:]) + "\n")
    elif problem == "identity":
        rewrite_rows(path, lambda row: row.update(node_id="unknown-node"))
    elif problem == "coverage":
        rewrite_rows(path, lambda row: row.update(sketch_count=3))
    elif problem == "scale":
        rewrite_rows(
            path,
            lambda row: row.update(
                normalized_rms=0.02, absolute_rms_spread=0.02 * (1 + 1e-12)
            ),
        )
    else:

        def legacy(row: dict[str, Any]) -> None:
            row.update(event="consensus_distance", evidence_version=1)
            for field in (
                "absolute_rms_spread",
                "mean_sketch_norm",
                "denominator_degenerate",
            ):
                del row[field]

        for legacy_path in tmp_path.glob("*.jsonl"):
            rewrite_rows(legacy_path, legacy)
    with pytest.raises(EvidenceError):
        calibrate(index)


@pytest.mark.parametrize(
    "seed,split", [(29, "calibration"), (17, "validation"), (41, "validation")]
)
def test_calibration_rejects_heldout_seeds_and_validation_labels(
    tmp_path: Path, seed: int, split: str
) -> None:
    index = write_index(tmp_path, seed=seed, split=split)
    with pytest.raises(EvidenceError, match="wrong split or forbidden seed"):
        calibrate(index)


def test_assessment_preserves_false_alerts_and_rejects_modified_freeze(
    tmp_path: Path,
) -> None:
    frozen = calibrate(write_index(tmp_path / "calibration"))
    heldout = write_index(tmp_path / "heldout", seed=41, split="validation")
    for path in heldout.parent.glob("healthy-*.jsonl"):
        rewrite_rows(
            path,
            lambda row: row.update(
                normalized_rms=0.2, absolute_rms_spread=0.2 * (1 + 1e-12)
            ),
        )
    report = evaluate(heldout, frozen)
    assert report["metrics"]["healthy_run_false_warning_rate"] == 1
    assert report["metrics"]["false_alerts"] == 4
    assert report["metrics"]["alert_precision"] == 0.5
    with pytest.raises(EvidenceError, match="live warning evidence is missing"):
        evaluate(heldout, frozen, require_live=True)
    frozen["policy"]["severe_distance"] = 1
    with pytest.raises(EvidenceError, match="intact passing calibration freeze"):
        evaluate(heldout, frozen)


def test_node_measurements_supply_saved_group_scales_without_inventing_curves(
    tmp_path: Path,
) -> None:
    index = write_index(tmp_path)
    for path in tmp_path.glob("*.jsonl"):

        def convert(row: dict[str, Any]) -> None:
            row.update(
                event="node_consensus_distance",
                group_normalized_rms=row.pop("normalized_rms"),
                absolute_distance=row.pop("absolute_rms_spread"),
            )
            # Equal individual distances reconcile to this synthetic group RMS.
            row["normalized_distance"] = row["group_normalized_rms"]

        rewrite_rows(path, convert)
    report = calibrate(index)
    assert report["status"] == "frozen"
    assert report["metrics"]["delay_rounds"] == [1, 1, 1, 1]


def add_live_statuses(index: Path, frozen: dict[str, Any]) -> None:
    """Write explicitly synthetic logged transitions with a three-round delay."""
    from dromeus.manifests.models import DivergencePolicy
    from dromeus.telemetry.evidence import DivergenceStatusEvidence

    policy = DivergencePolicy.model_validate(frozen["policy"])
    data = json.loads(index.read_text())
    for run in data["runs"]:
        run["summaries"] = []
        run["configuration"]["detector"] = frozen["policy"]
        manifest_path = index.parent / run["manifest"]
        manifest_data = json.loads(manifest_path.read_text())
        manifest_data["divergence_detection"] = policy.model_dump(mode="json")
        draft = DraftRunSpec.model_validate(
            {k: v for k, v in manifest_data.items() if k in DraftRunSpec.model_fields}
        )
        manifest_data["draft_hash"] = canonical_hash(draft)
        manifest = SealedManifest.model_validate(manifest_data)
        manifest_path.write_text(manifest.model_dump_json())
        for rank, log in enumerate(run["logs"]):
            summary_path = index.parent / f"{run['label']}-{rank}-summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "rank": rank,
                        "source": ("mnist", "usps", "svhn", "optdigits")[rank],
                        "node_public_key": f"peer-{rank}",
                        "mode": "node",
                        "draft_hash": manifest.draft_hash,
                        "event_delivery_dropped": 0,
                        "run_config": run["configuration"],
                    }
                )
            )
            run["summaries"].append(summary_path.name)
            path = index.parent / log
            rewrite_rows(
                path, lambda row: row.update(manifest_hash=canonical_hash(manifest))
            )
            record = DivergenceStatusEvidence(
                run_id=run["run_id"],
                manifest_hash=canonical_hash(manifest),
                node_id=f"peer-{rank}",
                policy_hash=canonical_hash(policy),
                threshold_set_id=policy.threshold_set_id,
                transition_id=1,
                previous_state="warming_up",
                state="healthy",
                warning_active=False,
                scale_warning=False,
                reason="stable_window",
                source_round=17,
                detection_round=17,
                window_start=10,
                window_end=17,
                window_count=8,
                expected_participants=4,
                observed_participants=4,
                age_rounds=0,
                missing_rounds=0,
                measurement_channel="normalized",
            )
            with path.open("a") as stream:
                stream.write(record.model_dump_json() + "\n")
                if run["label"] == "faulted":
                    warning = record.model_copy(
                        update={
                            "transition_id": 2,
                            "previous_state": "healthy",
                            "state": "diverging",
                            "warning_active": True,
                            "reason": "severe_distance",
                            "source_round": 26,
                            "detection_round": 28,
                            "age_rounds": 2,
                            "window_start": 19,
                            "window_end": 26,
                        }
                    )
                    stream.write(warning.model_dump_json() + "\n")
    index.write_text(json.dumps(data))


def test_live_delays_are_separate_from_round_order_replay_and_policy_is_bound(
    tmp_path: Path,
) -> None:
    frozen = calibrate(write_index(tmp_path / "calibration"))
    heldout = write_index(tmp_path / "heldout", seed=29, split="validation")
    add_live_statuses(heldout, frozen)
    report = evaluate(heldout, frozen, require_live=True)
    assert report["metrics"]["delay_rounds"] == [1, 1, 1, 1]
    assert report["live_status"] == "available"
    assert report["live_metrics"]["delay_rounds"] == [3, 3, 3, 3]
    assert report["live_nodes"][-1]["alerts"][0]["source_round"] == 26
    assert report["live_nodes"][-1]["alerts"][0]["detection_round"] == 28

    def corrupt(row: dict[str, Any]) -> None:
        if row["event"] == "divergence_status":
            row["policy_hash"] = "0" * 64

    rewrite_rows(heldout.parent / "faulted-0.jsonl", corrupt)
    with pytest.raises(EvidenceError, match="live warning identity, policy or round"):
        evaluate(heldout, frozen, require_live=True)


@pytest.mark.parametrize(
    "problem",
    ["dropped", "missing", "rank", "draft", "policy", "key", "duplicate_rank"],
)
def test_live_assessment_requires_bound_terminal_zero_drop_summaries(
    tmp_path: Path, problem: str
) -> None:
    frozen = calibrate(write_index(tmp_path / "calibration"))
    heldout = write_index(tmp_path / "heldout", seed=29, split="validation")
    add_live_statuses(heldout, frozen)
    path = heldout.parent / "faulted-0-summary.json"
    summary = json.loads(path.read_text())
    if problem == "dropped":
        summary["event_delivery_dropped"] = 1
    elif problem == "missing":
        index = json.loads(heldout.read_text())
        for run in index["runs"]:
            del run["summaries"]
        heldout.write_text(json.dumps(index))
        report = evaluate(heldout, frozen)
        assert report["live_status"] == "unverified_delivery"
        assert report["live_delivery_verified"] is False
    elif problem == "rank":
        summary["rank"] = 1
    elif problem == "key":
        summary["node_public_key"] = "peer-1"
    elif problem == "duplicate_rank":
        summary["rank"] = 1
        summary["source"] = "usps"
    elif problem == "draft":
        summary["draft_hash"] = "0" * 64
    else:
        summary["run_config"]["detector"] = None
    path.write_text(json.dumps(summary))
    with pytest.raises(EvidenceError):
        evaluate(heldout, frozen, require_live=True)


def test_source_rank_can_differ_from_sealed_participant_index(tmp_path: Path) -> None:
    frozen = calibrate(write_index(tmp_path / "calibration"))
    heldout = write_index(tmp_path / "heldout", seed=29, split="validation")
    add_live_statuses(heldout, frozen)
    for path in heldout.parent.glob("*-summary.json"):
        summary = json.loads(path.read_text())
        summary["rank"] = (summary["rank"] + 1) % 4
        summary["source"] = ("mnist", "usps", "svhn", "optdigits")[summary["rank"]]
        path.write_text(json.dumps(summary))
    report = evaluate(heldout, frozen, require_live=True)
    assert report["live_delivery_verified"] is True
    assert report["live_metrics"]["delay_rounds"] == [3, 3, 3, 3]
    assert report["sources"][0]["nodes"][0]["node_id"] == "peer-0"
    assert report["sources"][0]["nodes"][0]["summary"]["source_rank"] == 1


def test_relabeling_index_seed_cannot_turn_training_seed_into_heldout(
    tmp_path: Path,
) -> None:
    frozen = calibrate(write_index(tmp_path / "calibration"))
    heldout = write_index(tmp_path / "heldout", seed=17, split="validation")
    index = json.loads(heldout.read_text())
    for run in index["runs"]:
        run["seed"] = 29
    heldout.write_text(json.dumps(index))
    with pytest.raises(EvidenceError, match="seed"):
        evaluate(heldout, frozen)


def test_configuration_seed_must_match_declared_and_sealed_seed(tmp_path: Path) -> None:
    index = write_index(tmp_path)
    data = json.loads(index.read_text())
    data["runs"][0]["configuration"]["seed"] = 29
    index.write_text(json.dumps(data))
    with pytest.raises(EvidenceError, match="seed"):
        calibrate(index)


@pytest.mark.parametrize(
    "field,value",
    [
        ("start_round", 26),
        ("kind", "other_fault"),
        ("rank", 1),
        ("scale", 0.1),
        ("growth", 1.2),
    ],
)
def test_fault_ground_truth_cannot_change_without_sealed_configuration(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    index = write_index(tmp_path)
    data = json.loads(index.read_text())
    fault = data["runs"][1]["fault"]
    target = fault if field in ("kind", "start_round") else fault["metadata"]
    target[field] = value
    index.write_text(json.dumps(data))
    with pytest.raises(EvidenceError, match="fault.*configuration"):
        calibrate(index)


@pytest.mark.parametrize(
    "field,value",
    [
        ("learning_rate", 0.003),
        ("sigma", 2.0),
        ("split_manifest_sha256", "d" * 64),
    ],
)
def test_index_configuration_must_reconstruct_sealed_draft(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    index = write_index(tmp_path)
    data = json.loads(index.read_text())
    data["runs"][0]["configuration"][field] = value
    index.write_text(json.dumps(data))
    with pytest.raises(EvidenceError, match="configuration.*sealed"):
        calibrate(index)
