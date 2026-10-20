"""Final reports treat seeds as repetitions and reject incomplete evidence."""

from pathlib import Path
from typing import Any

import pytest

from benchmarks.m3.final_report import summarize_setting


def test_uncertainty_is_sample_sd_of_seed_means_not_individual_nodes() -> None:
    runs: list[dict[str, Any]] = [
        {
            "run_id": f"run-{seed}",
            "seed": seed,
            "nodes": [
                {"accuracy": [mean - 0.05] * 4},
                {"accuracy": [mean - 0.05] * 4},
                {"accuracy": [mean + 0.05] * 4},
                {"accuracy": [mean + 0.05] * 4},
            ],
        }
        for seed, mean in zip((17, 29, 41), (0.2, 0.4, 0.6), strict=True)
    ]
    result = summarize_setting(runs, seeds=(17, 29, 41))
    assert result["mean_accuracy"] == pytest.approx(0.4)
    assert result["sample_sd_seed_means"] == pytest.approx(0.2)
    assert result["repetitions"] == 3
    assert result["seed_means"] == pytest.approx([0.2, 0.4, 0.6])


def test_summary_rejects_missing_repetition_and_nonfinite_accuracy() -> None:
    with pytest.raises(ValueError, match="missing or duplicate"):
        summarize_setting([], seeds=(17, 29, 41))
    runs: list[dict[str, Any]] = [
        {
            "run_id": str(seed),
            "seed": seed,
            "nodes": [{"accuracy": [0.5] * 4} for _ in range(4)],
        }
        for seed in (17, 29, 41)
    ]
    runs[0]["nodes"][0]["accuracy"][0] = float("nan")
    with pytest.raises(ValueError, match="finite four-node"):
        summarize_setting(runs, seeds=(17, 29, 41))


def test_frozen_matrix_requires_every_predeclared_noise_seed_slot() -> None:
    from benchmarks.m3.final_report import FrozenExperiment

    with pytest.raises(ValueError, match="utility matrix"):
        FrozenExperiment.model_validate(
            {
                "version": 1,
                "data_manifest": {"path": "data.json", "sha256": "a" * 64},
                "source_artifacts": {"source.tar.gz": "b" * 64},
                "runtime_source_hashes": {"path": "runtime.json", "sha256": "d" * 64},
                "execution_environment": {
                    "path": "environment.json",
                    "sha256": "e" * 64,
                },
                "detector_policy_hash": "c" * 64,
                "utility_seeds": [17, 29, 41],
                "utility_sigmas": [0.5, 1.0, 2.0, 4.0],
                "runs": [],
            }
        )


def _spec() -> Any:
    import torch

    from benchmarks.m3.final_runner import FrozenRunSpec
    from benchmarks.m3.heterogeneous_data import SOURCES
    from dromeus.manifests.models import DivergencePolicy

    policy = DivergencePolicy(
        threshold_set_id="test-frozen",
        warmup_rounds=10,
        window_rounds=20,
        patience_windows=3,
        distance_floor=0.01,
        growth_ratio=2.0,
        min_log_slope=0.05,
        severe_distance=0.5,
        severe_patience=3,
        recovery_windows=3,
        recovery_distance=0.1,
        max_lag_rounds=4,
    )
    return FrozenRunSpec(
        run_id="final-private-s17",
        source_commit="a" * 40,
        torch_version=str(torch.__version__),
        arm="private",
        split_manifest_sha256="b" * 64,
        input_sha256={
            f"{s}-{p}.npz": "c" * 64 for s in SOURCES for p in ("train", "final")
        },
        evaluation_partition="final",
        evaluation_count=470,
        detector=policy,
    )


def _experiment() -> Any:
    from benchmarks.m3.final_report import ArtifactRef, DeclaredRun, FrozenExperiment
    from dromeus.manifests.canonical import canonical_hash

    base = _spec()
    runs = [
        DeclaredRun(
            role="utility",
            config=base.model_copy(
                update={
                    "run_id": f"final-{seed}-{sigma}",
                    "seed": seed,
                    "arm": "baseline" if sigma is None else "private",
                    "sigma": 1.0 if sigma is None else sigma,
                }
            ),
        )
        for seed in (17, 29, 41)
        for sigma in (None, 0.5, 1.0, 2.0, 4.0)
    ]
    return FrozenExperiment(
        version=1,
        data_manifest=ArtifactRef(path="data.json", sha256="b" * 64),
        source_artifacts={"source.tar.gz": "d" * 64},
        runtime_source_hashes=ArtifactRef(path="runtime.json", sha256="e" * 64),
        execution_environment=ArtifactRef(path="environment.json", sha256="f" * 64),
        detector_policy_hash=canonical_hash(base.detector),
        utility_seeds=(17, 29, 41),
        utility_sigmas=(0.5, 1.0, 2.0, 4.0),
        runs=tuple(runs),
    )


def test_freeze_rejects_calibration_reuse_unmatched_lr_and_duplicate_slot() -> None:
    from benchmarks.m3.final_report import FrozenExperiment

    freeze = _experiment()
    assert len(freeze.runs) == 15
    for field, value, match in (
        ("evaluation_count", 500, "final comparison"),
        ("learning_rate", 0.1, "unmatched"),
        ("seed", 29, "utility matrix"),
    ):
        payload = freeze.model_dump(mode="json")
        payload["runs"][0]["config"][field] = value
        with pytest.raises(ValueError, match=match):
            FrozenExperiment.model_validate(payload)


def test_reconstructed_private_contract_matches_runner_and_binds_gpu_torch() -> None:
    from benchmarks.m3.final_report import expected_drafts
    from benchmarks.m3.final_runner import reference_draft

    spec = _spec()
    reference, draft = expected_drafts(spec)
    assert reference == reference_draft(spec)
    assert draft.run_id == spec.run_id
    gpu_spec = spec.model_copy(update={"torch_version": "2.13.0+cu130"})
    gpu_reference, gpu_draft = expected_drafts(gpu_spec)
    assert gpu_reference.environment.pytorch_version == "2.13.0+cu130"
    assert gpu_draft.private_training is not None
    assert gpu_draft.private_training.optimizer.torch_version == "2.13.0+cu130"


def test_reservation_chain_cannot_fork_or_drop_earlier_expenditure() -> None:
    from benchmarks.m3.final_report import verify_reservation_chains

    def run(ids: list[str]) -> dict[str, Any]:
        return {
            "arm": "private",
            "nodes": [
                {"reservations": [{"run_id": identity} for identity in ids]}
                for _ in range(4)
            ],
        }

    verify_reservation_chains([run(["earlier", "one", "two"]), run(["earlier", "one"])])
    with pytest.raises(ValueError, match="fork or omit"):
        verify_reservation_chains([run(["earlier", "one"]), run(["earlier", "two"])])


def test_artifact_validation_rejects_changed_bytes_and_path_escape(
    tmp_path: Path,
) -> None:
    from benchmarks.m3.final_report import ArtifactRef, checked_artifact
    from dromeus.manifests.canonical import file_sha256

    item = tmp_path / "source.bin"
    item.write_bytes(b"original")
    ref = ArtifactRef(path="source.bin", sha256=file_sha256(item))
    assert checked_artifact(tmp_path, ref) == item
    item.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        checked_artifact(tmp_path, ref)
    with pytest.raises(ValueError, match="escapes"):
        checked_artifact(tmp_path, ArtifactRef(path="../source.bin", sha256="a" * 64))


def _history_fixture() -> tuple[Any, list[dict[str, Any]], list[Any]]:
    from datetime import UTC, datetime, timedelta

    from benchmarks.m3.heterogeneous_data import SOURCES
    from dromeus.telemetry.evidence import TaskRoundMetricsEvidence

    spec = _spec().model_copy(update={"steps": 20, "evaluation_interval": 1})

    def point(step: int, value: float) -> dict[str, Any]:
        return {
            "logical_steps": step,
            "accuracy_macro_source": value,
            "accuracy_worst_source": value,
            **{f"accuracy_{source}": value for source in SOURCES},
        }

    history = [point(0, 0.2), point(10, 0.5), point(20, 0.6), point(20, 0.6)]
    metrics = [
        TaskRoundMetricsEvidence(
            timestamp=datetime(2026, 10, 21, tzinfo=UTC) + timedelta(seconds=i),
            run_id=spec.run_id,
            manifest_hash="a" * 64,
            node_id="node-0",
            peer_id="node-1",
            round_id=i,
            message_id=f"metric-{i}",
            local_compute_seconds=0.1,
            peer_wait_seconds=0.1,
            transfer_seconds=0.1,
            mixing_seconds=0.1,
            evaluation_seconds=0.1,
            retries=0,
            evaluation_metrics={f"accuracy_{source}": value for source in SOURCES},
        )
        for i, value in enumerate((0.5, 0.6))
    ]
    return spec, history, metrics


def test_public_history_reconciles_typed_rounds_and_deduplicates_final_evaluation() -> (
    None
):
    from benchmarks.m3.final_report import validate_history

    spec, history, metrics = _history_fixture()
    points = validate_history(history, metrics, spec)
    assert [p["logical_steps"] for p in points] == [0, 10, 20]
    assert points[-1]["seconds_since_first_round_evidence"] == 1
    with pytest.raises(ValueError, match="incomplete or duplicate"):
        validate_history(history, metrics[:-1], spec)
    history[-1]["accuracy_mnist"] = 0.7
    history[-1]["accuracy_macro_source"] = 0.625
    with pytest.raises(ValueError, match="differs from typed"):
        validate_history(history, metrics, spec)


def test_private_evaluation_rejects_raw_loss_and_off_count_accuracy() -> None:
    from benchmarks.m3.final_report import validate_history

    spec, history, metrics = _history_fixture()
    metrics[0] = metrics[0].model_copy(update={"local_loss": 1.0})
    with pytest.raises(ValueError, match="raw loss"):
        validate_history(history, metrics, spec)
    spec, history, metrics = _history_fixture()
    history[0]["accuracy_mnist"] = 0.33333
    with pytest.raises(ValueError, match="count over frozen holdout"):
        validate_history(history, metrics, spec)


def test_missing_detector_assessment_remains_explicitly_unverified(
    tmp_path: Path,
) -> None:
    from benchmarks.m3.final_report import detector_evidence

    assert (
        detector_evidence(tmp_path, _experiment(), [], None)["status"] == "unverified"
    )


def test_report_refuses_incomplete_campaign_without_writing_success(
    tmp_path: Path,
) -> None:
    import json

    from benchmarks.m3.final_report import build_report
    from dromeus.manifests.canonical import file_sha256

    (tmp_path / "data.json").write_text("{}")
    (tmp_path / "source.tar.gz").write_bytes(b"frozen source")
    (tmp_path / "runs").mkdir()
    freeze = _experiment().model_dump(mode="json")
    freeze["data_manifest"]["sha256"] = file_sha256(tmp_path / "data.json")
    freeze["source_artifacts"]["source.tar.gz"] = file_sha256(
        tmp_path / "source.tar.gz"
    )
    for declared in freeze["runs"]:
        declared["config"]["split_manifest_sha256"] = freeze["data_manifest"]["sha256"]
    (tmp_path / "frozen-experiment.json").write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match="run directories differ"):
        build_report(tmp_path)
    assert not (tmp_path / "report").exists()


def test_export_audit_is_unverified_without_retained_evidence(tmp_path: Path) -> None:
    from benchmarks.m3.final_report import export_audit_evidence

    assert (
        export_audit_evidence(tmp_path, _experiment(), [], "a" * 64, None)["status"]
        == "unverified"
    )


def test_cpu_audit_discrepancies_are_retained_and_mislabelled_success_rejected(
    tmp_path: Path,
) -> None:
    import json

    from benchmarks.m3.final_report import ArtifactRef, export_audit_evidence
    from benchmarks.m3.heterogeneous_data import SOURCES
    from dromeus.manifests.canonical import canonical_hash, file_sha256

    freeze = _experiment()
    rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for declared in freeze.runs:
        config = declared.config
        nodes = [
            {
                "accuracy": [0.5] * 4,
                "source_hashes": {
                    "model.safetensors": "d" * 64,
                    "summary.json": "e" * 64,
                },
            }
            for _ in range(4)
        ]
        runs.append({"run_id": config.run_id, "nodes": nodes})
        for rank, source in enumerate(SOURCES):
            rows.append(
                {
                    "run_id": config.run_id,
                    "rank": rank,
                    "source": source,
                    "config_hash": canonical_hash(config),
                    "model_sha256": "d" * 64,
                    "summary_sha256": "e" * 64,
                    "evaluation_count": 470,
                    "gpu_correct": dict.fromkeys(SOURCES, 235),
                    "cpu_correct": dict.fromkeys(SOURCES, 235),
                    "all_match": True,
                }
            )
    rows[0]["cpu_correct"]["mnist"] = 234
    rows[0]["all_match"] = False
    discrepancy = {
        "run_id": rows[0]["run_id"],
        "rank": 0,
        "holdout_source": "mnist",
        "gpu_correct": 235,
        "cpu_correct": 234,
        "difference": -1,
    }
    audit = {
        "version": 1,
        "device": "cpu",
        "experiment_sha256": "a" * 64,
        "data_manifest_sha256": freeze.data_manifest.sha256,
        "input_sha256": {f"{source}-final.npz": "c" * 64 for source in SOURCES},
        "rows": rows,
        "all_matches": False,
        "discrepancies": [discrepancy],
        "skipped_nonfinal_run_ids": [],
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit))
    reference = ArtifactRef(path=path.name, sha256=file_sha256(path))
    checked = export_audit_evidence(tmp_path, freeze, runs, "a" * 64, reference)
    assert checked["status"] == "discrepancies"
    assert checked["audit"]["discrepancies"] == [discrepancy]
    assert runs[0]["nodes"][0]["accuracy"] == [0.5] * 4
    audit["all_matches"] = True
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="discrepancy summary"):
        export_audit_evidence(
            tmp_path,
            freeze,
            runs,
            "a" * 64,
            ArtifactRef(path=path.name, sha256=file_sha256(path)),
        )


def test_utility_forbids_unequal_training_counts() -> None:
    from benchmarks.m3.final_report import FrozenExperiment

    payload = _experiment().model_dump(mode="json")
    payload["runs"][0]["config"]["train_counts"] = (3000, 2400, 1800, 1200)
    with pytest.raises(ValueError, match="final comparison contract"):
        FrozenExperiment.model_validate(payload)


def test_robustness_baseline_accounting_uses_source_rank_local_count() -> None:
    from benchmarks.m3.final_report import expected_drafts, reconcile_accounting
    from dromeus.manifests.canonical import canonical_hash
    from dromeus.manifests.models import SealedManifest

    spec = _spec().model_copy(
        update={"arm": "baseline", "train_counts": (3000, 2400, 1800, 1200)}
    )
    _, draft = expected_drafts(spec)
    manifest = SealedManifest.model_validate(
        {
            **draft.model_dump(),
            "draft_hash": canonical_hash(draft),
            "participants": [
                {"public_key": f"peer-{i}", "node_index": i} for i in range(4)
            ],
            "initial_checkpoint_hash": "d" * 64,
            "tensor_schema": {
                "tensors": [{"name": "weight", "dtype": "float32", "shape": [1]}]
            },
        }
    )
    result: dict[str, Any] = {
        "rank": 1,
        "training_records": 2400,
        "privacy_claim": False,
        "statistics": {
            "epsilon": None,
            "logical_steps": 750,
            "sample_rate": 1 / 38,
            "expected_batch_size": 63,
            "physical_steps": 1500,
        },
    }
    assert reconcile_accounting(result, manifest, spec)["sample_rate"] == 1 / 38
    result["statistics"]["sample_rate"] = 1 / 47
    with pytest.raises(ValueError, match="sampling/horizon"):
        reconcile_accounting(result, manifest, spec)


@pytest.mark.parametrize("role", ["compression", "robustness"])
def test_role_name_cannot_certify_an_ordinary_balanced_identity_run(role: str) -> None:
    from benchmarks.m3.final_report import FrozenExperiment

    payload = _experiment().model_dump(mode="python")
    control = dict(payload["runs"][0])
    control["role"] = role
    control["config"] = {**control["config"], "run_id": "mislabeled-control"}
    payload["runs"] = (*payload["runs"], control)
    with pytest.raises(ValueError, match=role):
        FrozenExperiment.model_validate(payload)


def _compression_experiment() -> Any:
    from benchmarks.m3.final_report import FrozenExperiment

    payload = _experiment().model_dump(mode="python")
    compressed = dict(payload["runs"][2])  # private sigma1 seed17
    compressed["role"] = "compression"
    compressed["config"] = {
        **compressed["config"],
        "run_id": "compressed-s17",
        "codecs": "lossy",
    }
    payload["runs"] = (*payload["runs"], compressed)
    return FrozenExperiment.model_validate(payload)


def test_compression_requires_exactly_matching_private_identity_recipe() -> None:
    from benchmarks.m3.final_report import FrozenExperiment

    payload = _compression_experiment().model_dump(mode="python")
    payload["runs"][-1]["config"]["learning_rate"] = 0.25
    with pytest.raises(ValueError, match="exactly matched identity"):
        FrozenExperiment.model_validate(payload)


def test_compression_reports_paired_effect_and_flags_only_observed_seed_variation() -> (
    None
):
    from benchmarks.m3.final_report import compression_comparisons

    freeze = _compression_experiment()
    runs: list[dict[str, Any]] = []
    for declared in freeze.runs:
        config = declared.config
        score = (
            0.65
            if declared.role == "compression"
            else {17: 0.8, 29: 0.9, 41: 1.0}[config.seed]
        )
        runs.append(
            {
                "run_id": config.run_id,
                "role": declared.role,
                "seed": config.seed,
                "sigma": config.sigma if config.arm == "private" else None,
                "nodes": [
                    {
                        "node_id": f"peer-{i}",
                        "initial_sha256": str(config.seed),
                        "accuracy": [score] * 4,
                        "wall_seconds": 10.0,
                        "encoded_artifact_bytes": 50
                        if declared.role == "compression"
                        else 100,
                    }
                    for i in range(4)
                ],
            }
        )
    result = compression_comparisons(freeze, runs)
    assert len(result) == 1
    assert result[0]["accuracy_delta_percentage_points"] == pytest.approx(-15)
    assert result[0]["identity_sample_sd_percentage_points"] == pytest.approx(10)
    assert result[0]["difference_exceeds_identity_seed_sd"] is True
    assert result[0]["encoded_bytes_ratio"] == 0.5
    runs[-1]["nodes"][0]["initial_sha256"] = "different initialization"
    with pytest.raises(ValueError, match="initialization"):
        compression_comparisons(freeze, runs)


def test_robustness_provenance_requires_actual_skew_counts_and_parent_positions(
    tmp_path: Path,
) -> None:
    import json

    from benchmarks.m3.final_report import ArtifactRef, DeclaredRun, run_data_manifest
    from benchmarks.m3.heterogeneous_data import SOURCES
    from dromeus.manifests.canonical import file_sha256

    parent = {
        "subsets": {
            source: {
                "train": {
                    "sha256": "a" * 64,
                    "indices": list(range(3000)),
                    "selected": 3000,
                },
                "final": {
                    "sha256": "b" * 64,
                    "indices": list(range(470)),
                    "selected": 470,
                    "class_counts": [47] * 10,
                    "image_hashes": [str(i) for i in range(470)],
                },
            }
            for source in SOURCES
        }
    }
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(parent))
    frozen = _experiment().model_copy(
        update={
            "data_manifest": ArtifactRef(
                path=parent_path.name, sha256=file_sha256(parent_path)
            )
        }
    )
    counts = (3000, 2400, 1800, 1200)
    derived: dict[str, Any] = {
        "parent_manifest_sha256": file_sha256(parent_path),
        "train_counts": list(counts),
        "subsets": {
            source: {
                "train": {
                    "parent_sha256": "a" * 64,
                    "selected": counts[rank],
                    "indices": list(range(counts[rank])),
                    "parent_positions": list(range(counts[rank])),
                    "allowed_labels": list(range(10 - rank * 2)),
                    "class_counts": [300] * (10 - rank * 2) + [0] * (rank * 2),
                },
                "final": parent["subsets"][source]["final"],
            }
            for rank, source in enumerate(SOURCES)
        },
    }
    derived_path = tmp_path / "derived.json"

    def declare() -> Any:
        derived_path.write_text(json.dumps(derived))
        digest = file_sha256(derived_path)
        return DeclaredRun(
            role="robustness",
            data_manifest=ArtifactRef(path=derived_path.name, sha256=digest),
            config=_spec().model_copy(
                update={"train_counts": counts, "split_manifest_sha256": digest}
            ),
        )

    assert run_data_manifest(tmp_path, frozen, declare())["train_counts"] == list(
        counts
    )
    derived["subsets"]["usps"]["train"]["class_counts"] = [240] * 10
    with pytest.raises(ValueError, match="class counts"):
        run_data_manifest(tmp_path, frozen, declare())
    derived["subsets"]["usps"]["train"]["class_counts"] = [300] * 8 + [0] * 2
    derived["subsets"]["usps"]["train"]["parent_positions"][1] = 2
    with pytest.raises(ValueError, match="parent positions"):
        run_data_manifest(tmp_path, frozen, declare())


def test_detector_assessment_binds_fault_start_metadata_and_full_configuration(
    tmp_path: Path,
) -> None:
    import hashlib
    import json

    from benchmarks.m3.final_report import (
        ArtifactRef,
        FrozenExperiment,
        detector_evidence,
    )
    from benchmarks.m3.final_runner import PublicFault
    from dromeus.manifests.canonical import file_sha256

    def digest(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(
                value, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()

    payload = _experiment().model_dump(mode="python")
    healthy = payload["runs"][7]["config"]  # sigma1 seed29
    faulty = {
        **healthy,
        "run_id": "fault-s29",
        "fault": PublicFault(start_round=25).model_dump(),
    }
    payload["runs"] = (*payload["runs"], {"role": "validation_fault", "config": faulty})
    freeze = FrozenExperiment.model_validate(payload)
    detector = freeze.runs[0].config.detector
    assert detector is not None
    policy = {
        "status": "frozen",
        "policy": detector.model_dump(mode="json"),
        "policy_hash": freeze.detector_policy_hash,
    }
    policy["freeze_hash"] = digest(policy)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    freeze = freeze.model_copy(
        update={
            "detector_freeze": ArtifactRef(
                path=policy_path.name, sha256=file_sha256(policy_path)
            )
        }
    )
    configs = [freeze.runs[7], freeze.runs[-1]]
    runs: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for declared in configs:
        spec = declared.config
        configuration = spec.model_dump(mode="json")
        nodes = [
            {
                "node_id": f"peer-{rank}",
                "event_delivery_dropped": 0,
                "source_hashes": {
                    "logs/dromeus.jsonl": "d" * 64,
                    "run-store/manifest.json": "e" * 64,
                },
            }
            for rank in range(4)
        ]
        runs.append(
            {
                "run_id": spec.run_id,
                "role": declared.role,
                "seed": 29,
                "manifest_hash": "f" * 64,
                "nodes": nodes,
            }
        )
        fault = (
            None
            if spec.fault is None
            else {
                "kind": spec.fault.kind,
                "start_round": spec.fault.start_round,
                "metadata": {
                    "rank": spec.fault.rank,
                    "scale": spec.fault.scale,
                    "growth": spec.fault.growth,
                },
            }
        )
        sources.append(
            {
                "run_id": spec.run_id,
                "seed": 29,
                "split": "validation",
                "label": "healthy" if fault is None else "faulted",
                "fault": fault,
                "configuration": configuration,
                "configuration_hash": digest(configuration),
                "manifest_hash": "f" * 64,
                "manifest_sha256": "e" * 64,
                "expected_rounds": 75,
                "nodes": [
                    {"node_id": node["node_id"], "sha256": "d" * 64} for node in nodes
                ],
            }
        )
    assessment = {
        "policy_hash": freeze.detector_policy_hash,
        "freeze_hash": policy["freeze_hash"],
        "sources": sources,
        "live_status": "available",
    }
    path = tmp_path / "validation.json"

    def reference() -> Any:
        path.write_text(json.dumps(assessment))
        return ArtifactRef(path=path.name, sha256=file_sha256(path))

    assert (
        detector_evidence(tmp_path, freeze, runs, reference())["status"]
        == "verified_evidence"
    )
    sources[-1]["fault"]["start_round"] = 70
    with pytest.raises(ValueError, match="fault metadata"):
        detector_evidence(tmp_path, freeze, runs, reference())
    sources[-1]["fault"]["start_round"] = 25
    sources[-1]["configuration"]["learning_rate"] = 0.5
    sources[-1]["configuration_hash"] = digest(sources[-1]["configuration"])
    with pytest.raises(ValueError, match="configuration"):
        detector_evidence(tmp_path, freeze, runs, reference())


def _runtime_files(root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    import json

    source_hashes = {"src/dromeus/__init__.py": "a" * 64}
    environment = {
        "nodes": 4,
        "torch": _spec().torch_version,
        "axl_binary_sha256": "b" * 64,
        "gpu_profile": "NVIDIA L4-3Q",
        "grid_driver": "580.159.03",
    }
    (root / "source-provenance.json").write_text(
        json.dumps(
            {
                "rank": 0,
                "source": "mnist",
                "public_key": "peer-0",
                "instance_id": "i-0123456789abcdef0",
                "source_hashes": source_hashes,
                "torch": environment["torch"],
                "axl_sha256": environment["axl_binary_sha256"],
                "training_files": ["mnist-train.npz"],
            }
        )
    )
    (root / "gpu-identity.txt").write_text(
        "name, uuid, driver_version, memory.total [MiB]\n"
        "NVIDIA L4-3Q, GPU-01234567-0123-0123-0123-0123456789ab, 580.159.03, 3072 MiB\n"
    )
    return source_hashes, environment


@pytest.mark.parametrize(
    "field,value",
    [
        ("rank", 1),
        ("source", "usps"),
        ("public_key", "peer-1"),
        ("source_hashes", {"src/dromeus/__init__.py": "c" * 64}),
        ("axl_sha256", "d" * 64),
        ("torch", "different"),
        ("training_files", ["usps-train.npz"]),
        ("instance_id", "not-an-instance"),
    ],
)
def test_runtime_provenance_rejects_source_binary_and_worker_identity_changes(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    import json

    from benchmarks.m3.final_report import reconcile_runtime_provenance

    hashes, environment = _runtime_files(tmp_path)
    path = tmp_path / "source-provenance.json"
    changed = json.loads(path.read_text())
    changed[field] = value
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        reconcile_runtime_provenance(
            tmp_path, _spec(), 0, "peer-0", hashes, environment
        )


def test_runtime_provenance_requires_collected_files_and_gpu_profile(
    tmp_path: Path,
) -> None:
    from benchmarks.m3.final_report import reconcile_runtime_provenance

    hashes, environment = _runtime_files(tmp_path)
    actual = reconcile_runtime_provenance(
        tmp_path, _spec(), 0, "peer-0", hashes, environment
    )
    assert actual["instance_id"] == "i-0123456789abcdef0"
    assert actual["gpu_uuid"] == "GPU-01234567-0123-0123-0123-0123456789ab"
    assert set(actual["artifact_hashes"]) == {
        "source-provenance.json",
        "gpu-identity.txt",
    }
    path = tmp_path / "gpu-identity.txt"
    path.write_text(path.read_text().replace("580.159.03", "other"))
    with pytest.raises(ValueError):
        reconcile_runtime_provenance(
            tmp_path, _spec(), 0, "peer-0", hashes, environment
        )
    path.unlink()
    with pytest.raises(FileNotFoundError):
        reconcile_runtime_provenance(
            tmp_path, _spec(), 0, "peer-0", hashes, environment
        )


@pytest.mark.parametrize("field", ["instance_id", "gpu_uuid"])
def test_runtime_workers_must_be_distinct(field: str) -> None:
    from benchmarks.m3.final_report import verify_worker_identities

    nodes = [
        {"runtime_provenance": {"instance_id": f"instance-{i}", "gpu_uuid": f"gpu-{i}"}}
        for i in range(4)
    ]
    verify_worker_identities(nodes)
    nodes[1]["runtime_provenance"][field] = nodes[0]["runtime_provenance"][field]
    with pytest.raises(ValueError, match="distinct"):
        verify_worker_identities(nodes)


def _private_accounting_fixture() -> tuple[dict[str, Any], Any, Any]:
    from benchmarks.m3.final_report import expected_drafts
    from dromeus.manifests.canonical import canonical_hash
    from dromeus.manifests.models import PrivacyReservation, SealedManifest
    from dromeus.training.privacy import privacy_epsilon

    spec = _spec()
    _, draft = expected_drafts(spec)
    manifest = SealedManifest.model_validate(
        {
            **draft.model_dump(),
            "draft_hash": canonical_hash(draft),
            "participants": [
                {"public_key": f"peer-{i}", "node_index": i} for i in range(4)
            ],
            "initial_checkpoint_hash": "d" * 64,
            "tensor_schema": {
                "tensors": [{"name": "weight", "dtype": "float32", "shape": [1]}]
            },
        }
    )
    policy = manifest.privacy
    assert policy is not None
    q = 1 / 47
    history = ((1.0, q, 750),)
    prv = privacy_epsilon(history, policy)
    rdp = privacy_epsilon(history, policy, accountant="rdp")
    reservation = PrivacyReservation(
        run_id=spec.run_id,
        policy_hash=canonical_hash(policy),
        sample_rate=q,
        noise_multiplier=1.0,
        logical_steps=750,
    )
    result = {
        "rank": 0,
        "training_records": 3000,
        "sample_rate": q,
        "expected_batch_size": 63,
        "physical_steps": 1500,
        "accounting_history": [[1.0, q, 750]],
        "accounting": {
            "privacy_epsilon": prv,
            "rdp_epsilon": rdp,
            "logical_steps": 750,
            "reserved_steps": 750,
            "delta": policy.delta,
            "randomness_profile": policy.randomness_profile,
        },
        "prior_reservations": [],
        "reservations": [reservation.model_dump(mode="json")],
        "cumulative_reserved_prv_epsilon": prv,
        "cumulative_reserved_rdp_epsilon": rdp,
    }
    return result, manifest, spec


def test_prv_reporting_roundoff_is_explicit_and_preserves_recorded_values() -> None:
    from benchmarks.m3.final_report import reconcile_accounting

    result, manifest, spec = _private_accounting_fixture()
    computed = result["cumulative_reserved_prv_epsilon"]
    result["cumulative_reserved_prv_epsilon"] += 1.5e-7
    actual = reconcile_accounting(result, manifest, spec)
    assert (
        actual["cumulative_reserved_prv_epsilon"]
        == result["cumulative_reserved_prv_epsilon"]
    )
    check = actual["accounting_reconciliation"]["cumulative_reserved_prv_epsilon"]
    assert check["recorded"] == result["cumulative_reserved_prv_epsilon"]
    assert check["recomputed"] == computed
    assert check["absolute_difference"] == pytest.approx(1.5e-7)
    assert check["absolute_tolerance"] == 1e-6
    assert check["relative_tolerance"] == 1e-9
    assert len(actual["accounting_reconciliation"]) == 4
    result["cumulative_reserved_prv_epsilon"] = computed + 2e-6
    with pytest.raises(ValueError, match="recomputed cumulative_reserved_prv_epsilon"):
        reconcile_accounting(result, manifest, spec)


def test_rdp_roundoff_and_exact_mechanism_checks_remain_strict() -> None:
    from benchmarks.m3.final_report import reconcile_accounting

    result, manifest, spec = _private_accounting_fixture()
    result["cumulative_reserved_rdp_epsilon"] += 1.5e-7
    with pytest.raises(ValueError, match="recomputed cumulative_reserved_rdp_epsilon"):
        reconcile_accounting(result, manifest, spec)
    result, manifest, spec = _private_accounting_fixture()
    result["accounting_history"][0][1] += 1e-12
    with pytest.raises(ValueError, match="sampling or mechanism history"):
        reconcile_accounting(result, manifest, spec)
