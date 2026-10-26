"""Fresh warning campaigns keep development and untouched evidence separate."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch
from support.sample_manifest import manifest_data

from benchmarks.m3 import detector_campaign
from benchmarks.m3.detector_campaign import (
    calibrate,
    evaluate,
    freeze_spec,
    template,
)
from benchmarks.m3.final_report import expected_drafts
from benchmarks.m3.final_runner import FrozenRunSpec, PublicFault
from benchmarks.m3.heterogeneous_data import SOURCES
from dromeus.manifests.canonical import canonical_hash, file_sha256
from dromeus.manifests.models import DivergencePolicy, DraftRunSpec, SealedManifest
from dromeus.telemetry.consensus import ConsensusObservation
from dromeus.telemetry.divergence import DivergenceDetector
from dromeus.telemetry.evidence import DivergenceStatusEvidence, EvidenceError


def recipe(seed: int, split: str, label: str) -> FrozenRunSpec:
    return FrozenRunSpec(
        run_id=f"repair-{split}-{seed}-{label}",
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


def spec_file(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    spec = {
        "version": 1,
        "campaign_id": "repair-fixture",
        "scoring_version": "warning-entry-v1",
        "calibration_seeds": [101],
        "validation_seeds": [103],
        "runs": [
            {
                "split": split,
                "configuration": recipe(seed, split, label).model_dump(mode="json"),
                "participant_count": 4,
                "participant_keys": [f"peer-{rank}" for rank in range(4)],
            }
            for seed, split in ((101, "calibration"), (103, "validation"))
            for label in ("healthy", "faulted")
        ],
        "candidates": [
            DivergencePolicy(
                threshold_set_id="repair-candidate-0",
                warmup_rounds=2,
                window_rounds=4,
                patience_windows=2,
                distance_floor=0.02,
                growth_ratio=2,
                min_log_slope=0.02,
                severe_distance=0.1,
                severe_patience=2,
                recovery_windows=2,
                recovery_distance=0.05,
                max_lag_rounds=8,
                absolute_distance_floor=0.02,
                absolute_severe_distance=0.1,
                absolute_recovery_distance=0.05,
            ).model_dump(mode="json")
        ],
        "criteria": {"max_delay_rounds": 10},
    }
    path = root / "campaign.json"
    path.write_text(json.dumps(spec))
    return path


def evidence(root: Path, seed: int, split: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for label in ("healthy", "faulted"):
        configuration = recipe(seed, split, label)
        _, draft = expected_drafts(configuration)
        data = manifest_data()
        data.update(draft.model_dump(mode="json"))
        data["draft_hash"] = canonical_hash(draft)
        manifest = SealedManifest.model_validate(data)
        manifest_path = root / f"{label}-manifest.json"
        manifest_path.write_text(manifest.model_dump_json())
        logs: list[str] = []
        for rank in range(4):
            rows: list[dict[str, Any]] = []
            for round_id in range(50):
                value = 0.01 if label == "healthy" or round_id < 25 else 0.2
                rows.append(
                    {
                        "timestamp": "2026-10-25T00:00:00Z",
                        "evidence_version": 2,
                        "event": "consensus_observation",
                        "run_id": configuration.run_id,
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
                "run_id": configuration.run_id,
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


def test_fresh_spec_freeze_calibration_and_untouched_replay(tmp_path: Path) -> None:
    predeclared = freeze_spec(spec_file(tmp_path))
    # Validation paths do not exist during calibration: no held-out access needed.
    frozen = calibrate(evidence(tmp_path / "cal", 101, "calibration"), predeclared)
    assert frozen["status"] == "frozen"
    assert frozen["candidate_count"] == 1
    assert frozen["metrics"]["missed_fault_nodes"] == 0
    index = evidence(tmp_path / "final", 103, "validation")
    report = evaluate(index, predeclared, frozen)
    assert report["replay_gate_passed"]
    assert not report["readiness_passed"]  # Replay cannot certify live delivery.
    assert report["group_metrics"]["fault_groups_detected_on_all_nodes"] == 1
    assert report["seed_metrics"]["103"]["healthy_groups"] == 1
    assert report["metrics"]["delay_rounds"] == [1] * 4
    with pytest.raises(EvidenceError, match="live"):
        evaluate(index, predeclared, frozen, require_live=True)


def test_frozen_json_subset_recipe_does_not_block_calibration(tmp_path: Path) -> None:
    path = spec_file(tmp_path)
    specification = json.loads(path.read_text())
    subset = next(
        run
        for run in specification["runs"]
        if run["split"] == "validation" and run["configuration"]["fault"] is None
    )
    subset["configuration"]["train_counts"] = [3000, 2400, 1800, 1200]
    path.write_text(json.dumps(specification))
    frozen_path = tmp_path / "predeclared.json"
    frozen_path.write_text(json.dumps(freeze_spec(path)))
    before = frozen_path.read_bytes()
    predeclared = json.loads(before)

    result = calibrate(evidence(tmp_path / "cal", 101, "calibration"), predeclared)

    assert result["status"] == "frozen"
    assert result["group_metrics"]["groups"] == 2
    assert frozen_path.read_bytes() == before
    assert predeclared == json.loads(before)
    assert not (tmp_path / "final").exists()


@pytest.mark.parametrize(
    "problem", ["historical", "overlap", "unused_seed", "duplicate"]
)
def test_spec_rejects_unfresh_or_incomplete_split_contract(
    tmp_path: Path, problem: str
) -> None:
    path = spec_file(tmp_path)
    raw = json.loads(path.read_text())
    if problem == "historical":
        raw["calibration_seeds"] = [17]
    elif problem == "overlap":
        raw["validation_seeds"] = [101]
    elif problem == "unused_seed":
        raw["calibration_seeds"].append(107)
    else:
        raw["runs"].append(raw["runs"][0])
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        freeze_spec(path)


@pytest.mark.parametrize("problem", ["missing_run", "recipe", "round", "duplicate"])
def test_calibration_rejects_missing_tampered_or_duplicate_evidence(
    tmp_path: Path, problem: str
) -> None:
    predeclared = freeze_spec(spec_file(tmp_path))
    index = evidence(tmp_path / "cal", 101, "calibration")
    raw = json.loads(index.read_text())
    if problem == "missing_run":
        raw["runs"].pop()
    elif problem == "recipe":
        raw["runs"][0]["configuration"]["learning_rate"] = 0.9
    elif problem == "duplicate":
        raw["runs"].append(raw["runs"][0])
    else:
        log = index.parent / "healthy-0.jsonl"
        log.write_text("\n".join(log.read_text().splitlines()[1:]) + "\n")
    index.write_text(json.dumps(raw))
    with pytest.raises(EvidenceError):
        calibrate(index, predeclared)


def test_changed_freeze_or_scorer_rejected(tmp_path: Path) -> None:
    predeclared = freeze_spec(spec_file(tmp_path))
    frozen = calibrate(evidence(tmp_path / "cal", 101, "calibration"), predeclared)
    index = evidence(tmp_path / "final", 103, "validation")
    frozen["policy"]["distance_floor"] = 0.04
    with pytest.raises(EvidenceError, match="freeze"):
        evaluate(index, predeclared, frozen)
    predeclared["source_hashes"]["benchmarks/m3/calibrate_detector.py"] = "0" * 64
    with pytest.raises(EvidenceError, match="freeze|source"):
        calibrate(index, predeclared)


def test_preactive_false_warning_is_not_new_fault_detection(tmp_path: Path) -> None:
    predeclared = freeze_spec(spec_file(tmp_path))
    frozen = calibrate(evidence(tmp_path / "cal", 101, "calibration"), predeclared)
    index = evidence(tmp_path / "final", 103, "validation")
    for path in index.parent.glob("faulted-*.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row["round_id"] >= 10:
                row["normalized_rms"] = 0.2
                row["absolute_rms_spread"] = 0.2 * (1 + 1e-12)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = evaluate(index, predeclared, frozen)
    assert not report["replay_gate_passed"]
    assert report["metrics"]["false_alerts"] == 4
    assert report["metrics"]["missed_fault_nodes"] == 4
    assert report["group_metrics"]["fault_groups_detected_on_all_nodes"] == 0


def test_template_is_explicitly_unready() -> None:
    example = template()
    assert example["status"] == "template_requires_measured_metadata"
    assert "freeze-spec" in example["instructions"]


def add_live(index: Path, frozen: dict[str, Any], *, delay: int = 0) -> None:
    policy = DivergencePolicy.model_validate(frozen["policy"])
    data = json.loads(index.read_text())
    for run in data["runs"]:
        run["configuration"]["detector"] = policy.model_dump(mode="json")
        raw = json.loads((index.parent / run["manifest"]).read_text())
        raw["divergence_detection"] = policy.model_dump(mode="json")
        draft = DraftRunSpec.model_validate(
            {k: v for k, v in raw.items() if k in DraftRunSpec.model_fields}
        )
        raw["draft_hash"] = canonical_hash(draft)
        manifest = SealedManifest.model_validate(raw)
        (index.parent / run["manifest"]).write_text(manifest.model_dump_json())
        run["summaries"] = []
        for rank, relative in enumerate(run["logs"]):
            path = index.parent / relative
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            detector = DivergenceDetector(policy, participant_count=4)
            statuses: list[dict[str, Any]] = []
            for row in rows:
                row["manifest_hash"] = canonical_hash(manifest)
                point = ConsensusObservation(
                    **{
                        key: row[key]
                        for key in (
                            "round_id",
                            "normalized_rms",
                            "absolute_rms_spread",
                            "mean_sketch_norm",
                            "denominator_degenerate",
                            "sketch_count",
                        )
                    }
                )
                for transition in (
                    *detector.advance(point.round_id),
                    *detector.observe(point),
                ):
                    status = DivergenceStatusEvidence.model_validate(
                        {
                            "run_id": run["run_id"],
                            "manifest_hash": canonical_hash(manifest),
                            "node_id": f"peer-{rank}",
                            "policy_hash": canonical_hash(policy),
                            "threshold_set_id": policy.threshold_set_id,
                            "transition_id": transition.sequence,
                            "previous_state": transition.previous_state,
                            **asdict(transition.snapshot),
                        }
                    ).model_dump(mode="json")
                    status["detection_round"] += delay
                    status["age_rounds"] += delay
                    statuses.append(status)
            path.write_text("".join(json.dumps(r) + "\n" for r in [*rows, *statuses]))
            summary = index.parent / f"{run['label']}-{rank}-summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "rank": rank,
                        "source": SOURCES[rank],
                        "mode": "node",
                        "node_public_key": f"peer-{rank}",
                        "draft_hash": manifest.draft_hash,
                        "event_delivery_dropped": 0,
                        "run_config": run["configuration"],
                    }
                )
            )
            run["summaries"].append(summary.name)
    index.write_text(json.dumps(data))


def test_live_gate_requires_all_semantic_transitions_not_only_alert_entries(
    tmp_path: Path,
) -> None:
    predeclared = freeze_spec(spec_file(tmp_path))
    frozen = calibrate(evidence(tmp_path / "cal", 101, "calibration"), predeclared)
    index = evidence(tmp_path / "final", 103, "validation")
    add_live(index, frozen)
    report = evaluate(index, predeclared, frozen)
    assert not report["readiness_passed"]
    assert report["live_warning_gate_passed"]
    assert not report["runtime_proof_verified"]
    with pytest.raises(EvidenceError, match="runtime proof"):
        evaluate(index, predeclared, frozen, require_live=True)
    path = index.parent / "healthy-0.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        if row["event"] == "divergence_status":
            row["reason"] = "warmup"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(EvidenceError, match="live/replay"):
        evaluate(index, predeclared, frozen)


def test_live_delay_gates_readiness_separately_from_canonical_replay(
    tmp_path: Path,
) -> None:
    predeclared = freeze_spec(spec_file(tmp_path))
    frozen = calibrate(evidence(tmp_path / "cal", 101, "calibration"), predeclared)
    index = evidence(tmp_path / "final", 103, "validation")
    add_live(index, frozen, delay=11)
    report = evaluate(index, predeclared, frozen)
    assert report["replay_gate_passed"]
    assert report["live_metrics"]["delay_rounds"] == [12] * 4
    assert not report["readiness_passed"]


def test_duplicate_round_measurements_are_not_extra_evidence(tmp_path: Path) -> None:
    predeclared = freeze_spec(spec_file(tmp_path))
    index = evidence(tmp_path / "cal", 101, "calibration")
    log = index.parent / "healthy-0.jsonl"
    duplicate = json.loads(log.read_text().splitlines()[0])
    duplicate["message_id"] = "extra-copy"
    with log.open("a") as stream:
        stream.write(json.dumps(duplicate) + "\n")
    with pytest.raises(EvidenceError, match="duplicate"):
        calibrate(index, predeclared)


def runtime_spec(root: Path) -> Path:
    path = spec_file(root)
    raw = json.loads(path.read_text())
    raw["runtime_contract"] = {
        "source_hashes": {"src/dromeus/__init__.py": "a" * 64},
        "environment": {
            "nodes": 4,
            "torch": str(torch.__version__),
            "axl_binary_sha256": "b" * 64,
            "gpu_profile": "NVIDIA L4-3Q",
            "grid_driver": "580.159.03",
        },
    }
    path.write_text(json.dumps(raw))
    return path


def test_statuses_and_minimal_summaries_cannot_replace_runtime_artifacts(
    tmp_path: Path,
) -> None:
    predeclared = freeze_spec(runtime_spec(tmp_path))
    frozen = calibrate(evidence(tmp_path / "cal", 101, "calibration"), predeclared)
    index = evidence(tmp_path / "final", 103, "validation")
    add_live(index, frozen)
    with pytest.raises(EvidenceError, match="runtime proof failed"):
        evaluate(
            index,
            predeclared,
            frozen,
            require_live=True,
            runtime_root=tmp_path / "missing-execution-archive",
        )


def test_full_runtime_audit_adapter_binds_exact_assessed_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mock only the independently tested full auditor, not warning assessment."""
    from benchmarks.m3.final_report import DeclaredRun

    predeclared = freeze_spec(runtime_spec(tmp_path))
    frozen = calibrate(evidence(tmp_path / "cal", 101, "calibration"), predeclared)
    index = evidence(tmp_path / "final", 103, "validation")
    add_live(index, frozen)
    archive = tmp_path / "archive"
    runs = {run["run_id"]: run for run in json.loads(index.read_text())["runs"]}
    calls: list[str] = []
    tampered = False

    def audited(
        root: Path,
        declared: DeclaredRun,
        hashes: dict[str, Any],
        environment: dict[str, Any],
    ) -> dict[str, Any]:
        assert root == archive
        assert hashes == predeclared["spec"]["runtime_contract"]["source_hashes"]
        assert environment == predeclared["spec"]["runtime_contract"]["environment"]
        calls.append(declared.config.run_id)
        run = runs[declared.config.run_id]
        assert declared.config.model_dump(mode="json") == run["configuration"]
        assert (declared.role == "validation_fault") == (run["fault"] is not None)
        manifest = SealedManifest.model_validate_json(
            (index.parent / run["manifest"]).read_bytes()
        )
        return {
            "run_id": run["run_id"],
            "manifest_hash": canonical_hash(manifest),
            "config_hash": canonical_hash(declared.config),
            "nodes": [
                {
                    "node_id": f"peer-{rank}",
                    "rank": rank,
                    "event_delivery_dropped": 0,
                    "runtime_provenance": {},
                    "source_hashes": {
                        "logs/dromeus.jsonl": "0" * 64
                        if tampered
                        else file_sha256(index.parent / log),
                        "summary.json": file_sha256(
                            index.parent / run["summaries"][rank]
                        ),
                        "run-store/manifest.json": file_sha256(
                            index.parent / run["manifest"]
                        ),
                    },
                }
                for rank, log in enumerate(run["logs"])
            ],
            "final_consensus": {},
        }

    monkeypatch.setattr(detector_campaign, "load_run", audited)
    report = evaluate(
        index, predeclared, frozen, require_live=True, runtime_root=archive
    )
    assert report["runtime_proof_verified"]
    assert report["readiness_passed"]
    assert set(calls) == set(runs)
    tampered = True
    with pytest.raises(EvidenceError, match="runtime proof artifacts"):
        evaluate(index, predeclared, frozen, require_live=True, runtime_root=archive)
