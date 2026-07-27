from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from benchmarks.cifar10.runner import (
    DatasetArtifact,
    ReportInput,
    create_draft,
    create_quality_draft,
    write_frozen_plan,
    write_node_configs,
    write_pilot_evidence,
)
from dromeus.manifests.canonical import canonical_hash
from dromeus.manifests.models import SealedManifest
from dromeus.persistence.run_store import RunStore
from dromeus.training.pytorch import (
    CIFAR10_DATASET_VERSION,
    CIFAR_RESNET32_MODEL_DEFINITION_HASH,
    CIFAR_RESNET32_MODEL_ID,
    CIFAR_RESNET32_PREPROCESSING_HASH,
    MODEL_DEFINITION_HASH,
    PREPROCESSING_HASH,
)


def _draft():
    return create_draft(
        run_id="pilot-001",
        benchmark_seed=17,
        dromeus_commit="a" * 40,
        image_digest=f"sha256:{'b' * 64}",
        pytorch_version="2.13.0+cpu",
        round_count=2,
        local_steps=1,
    )


def _write_completed_roots(root: Path) -> tuple[Path, Path, Path, Path]:
    draft = _draft()
    sealed = SealedManifest.model_validate(
        {
            **draft.model_dump(mode="json"),
            "draft_hash": canonical_hash(draft),
            "participants": [
                {"public_key": f"peer-{index}", "node_index": index}
                for index in range(4)
            ],
            "initial_checkpoint_hash": "c" * 64,
            "tensor_schema": {
                "tensors": [
                    {"name": "weight", "dtype": "float32", "shape": [1]}
                ]
            },
        }
    )
    roots: list[Path] = []
    for index in range(4):
        run_root = root / f"node-{index}"
        store = RunStore(run_root / "run-store")
        store.initialize(sealed)
        for round_id in range(draft.round_count):
            store.persist_commit(
                committed_round=round_id,
                algorithm_state={"weight": np.array([1], dtype=np.float32)},
                pre_mix_state={"weight": np.array([0], dtype=np.float32)},
                post_mix_state={"weight": np.array([1], dtype=np.float32)},
                state_checksum=f"{index * 2 + round_id + 1:064x}",
                schedule={"round_id": round_id, "peer": f"peer-{index}"},
            )
        store.record_terminal("complete", {"committed_rounds": draft.round_count})
        roots.append(run_root)
    return roots[0], roots[1], roots[2], roots[3]


def _write_pilot_inputs(
    root: Path,
    run_roots: tuple[Path, Path, Path, Path],
) -> tuple[tuple[Path, Path, Path, Path], tuple[Path, Path, Path, Path]]:
    manifest = SealedManifest.model_validate_json(
        (run_roots[0] / "run-store" / "manifest.json").read_text(encoding="utf-8")
    )
    manifest_hash = canonical_hash(manifest)
    logs: list[Path] = []
    artifacts: list[Path] = []
    for index in range(4):
        log = root / f"node-{index}.jsonl"
        log.write_text(
            json.dumps(
                {
                    "event": "benchmark_node_ready",
                    "run_id": manifest.run_id,
                    "manifest_hash": manifest_hash,
                    "node_id": f"peer-{index}",
                    "benchmark_seed": 17,
                    "transport": "axl",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        artifact = root / f"data-{index}.json"
        artifact.write_text(
            DatasetArtifact(
                data_source="torchvision-cifar10",
                archive_md5="c58f30108f718f92721af3b95e74349a",
                dataset_version=CIFAR10_DATASET_VERSION,
                preprocessing_hash=PREPROCESSING_HASH,
                train_sample_count=50_000,
                test_sample_count=10_000,
            ).model_dump_json(),
            encoding="utf-8",
        )
        logs.append(log)
        artifacts.append(artifact)
    return (
        (logs[0], logs[1], logs[2], logs[3]),
        (artifacts[0], artifacts[1], artifacts[2], artifacts[3]),
    )


def test_create_draft_uses_canonical_production_values() -> None:
    draft = _draft()

    assert draft.model_definition_hash == MODEL_DEFINITION_HASH
    assert draft.dataset.version == CIFAR10_DATASET_VERSION
    assert draft.dataset.preprocessing_hash == PREPROCESSING_HASH
    assert draft.environment.pytorch_version == "2.13.0+cpu"
    assert draft.transport.max_payload_bytes == 16 * 1024 * 1024
    assert draft.peer_scheduler_seed == 17


def test_quality_draft_freezes_a_160_epoch_resnet_recipe() -> None:
    draft = create_quality_draft(
        run_id="quality-001",
        benchmark_seed=17,
        dromeus_commit="a" * 40,
        image_digest=f"sha256:{'b' * 64}",
        pytorch_version="2.13.0+cpu",
    )

    assert draft.algorithm_id == "dpsgd-v2"
    assert draft.manifest_version == 2
    assert draft.model_id == CIFAR_RESNET32_MODEL_ID
    assert draft.model_definition_hash == CIFAR_RESNET32_MODEL_DEFINITION_HASH
    assert draft.dataset.preprocessing_hash == CIFAR_RESNET32_PREPROCESSING_HASH
    assert draft.local_steps * draft.round_count == 16_000
    assert draft.training is not None
    assert draft.training.batch_size == 128
    assert draft.training.momentum == 0.9
    assert draft.training.weight_decay == 1e-4
    assert draft.training.learning_rate_milestones == (8_000, 12_000)
    assert draft.training.final_consensus_rounds == 2


def test_completed_pilot_can_freeze_one_plan(tmp_path: Path) -> None:
    draft_path = tmp_path / "draft.yaml"
    draft_path.write_text(
        json.dumps(_draft().model_dump(mode="json")),
        encoding="utf-8",
    )
    evidence_path = tmp_path / "pilot.json"
    run_roots = _write_completed_roots(tmp_path)
    event_logs, data_artifacts = _write_pilot_inputs(tmp_path, run_roots)
    evidence = write_pilot_evidence(
        draft_path=draft_path,
        run_roots=run_roots,
        event_logs=event_logs,
        data_artifacts=data_artifacts,
        output=evidence_path,
    )

    plan = write_frozen_plan(
        draft_path=draft_path,
        pilot_artifact=evidence_path,
        benchmark_seeds=(17, 29, 41),
        worker_instance_type="c7i.xlarge",
        worker_regions=("us-east-1", "us-east-1", "us-east-2", "us-east-2"),
        bootstrap_region="us-east-1",
        worker_root_volume_gib=40,
        output=tmp_path / "plan.yaml",
    )

    assert evidence.status == "complete"
    assert plan.benchmark_seeds == (17, 29, 41)
    assert plan.dataset.preprocessing_hash == PREPROCESSING_HASH
    assert len(set(evidence.node_ids)) == 4


def test_pilot_rejects_reused_node_root(tmp_path: Path) -> None:
    draft_path = tmp_path / "draft.yaml"
    draft_path.write_text(
        json.dumps(_draft().model_dump(mode="json")),
        encoding="utf-8",
    )
    run_roots = _write_completed_roots(tmp_path)
    event_logs, data_artifacts = _write_pilot_inputs(tmp_path, run_roots)

    with pytest.raises(ValueError, match="run roots must be distinct"):
        write_pilot_evidence(
            draft_path=draft_path,
            run_roots=(run_roots[0],) * 4,
            event_logs=event_logs,
            data_artifacts=data_artifacts,
            output=tmp_path / "pilot.json",
        )


def test_pilot_node_configs_do_not_require_a_frozen_plan(tmp_path: Path) -> None:
    draft_path = tmp_path / "draft.yaml"
    draft_path.write_text(
        json.dumps(_draft().model_dump(mode="json")),
        encoding="utf-8",
    )

    paths = write_node_configs(
        plan_path=None,
        draft_path=draft_path,
        benchmark_seed=17,
        bootstrap_uri="tls://bootstrap.example:9300",
        output_dir=tmp_path / "configs",
    )

    assert len(paths) == 4
    assert all(path.is_file() for path in paths)


def test_report_input_requires_three_seed_archives() -> None:
    with pytest.raises(ValidationError):
        ReportInput.model_validate({"seeds": []})
