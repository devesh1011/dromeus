from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from support.sample_manifest import manifest_data

from benchmarks.cifar10.official import (
    OfficialBenchmarkError,
    PilotEvidence,
    load_frozen_benchmark_plan,
    prepare_dpsgd_node_configs,
)
from benchmarks.cifar10.runner import create_quality_draft, write_frozen_plan
from dromeus.manifests.models import DraftRunSpec


def _write_plan(root: Path) -> Path:
    data = manifest_data()
    pilot = root / "pilot.json"
    pilot.write_text(
        json.dumps(
            {
                "status": "complete",
                "model_definition_hash": "0" * 64,
                "dataset": data["dataset"],
                "data_source": "torchvision-cifar10",
                "local_steps": 5,
                "round_count": 100,
                "learning_rate": 0.1,
                "node_ids": ["node-0", "node-1", "node-2", "node-3"],
                "data_artifact_sha256": [
                    "1" * 64,
                    "2" * 64,
                    "3" * 64,
                    "4" * 64,
                ],
            }
        ),
        encoding="utf-8",
    )
    path = root / "benchmark.yaml"
    path.write_text(
        "\n".join(
            (
                "benchmark_seeds: [17, 23, 29]",
                "local_steps: 5",
                "round_count: 100",
                "learning_rate: 0.1",
                "model_id: cifar-cnn-v1",
                f'model_definition_hash: "{"0" * 64}"',
                f"dataset: {json.dumps(data['dataset'])}",
                f"environment: {json.dumps(data['environment'])}",
                "data_source: torchvision-cifar10",
                "max_payload_bytes: 8388608",
                "max_retries: 3",
                "retry_timeout_seconds: 5.0",
                f"pilot_artifact: {pilot}",
                "cloud_provider: aws",
                "worker_instance_type: c7i.xlarge",
                "worker_regions: [us-east-1, us-east-1, us-east-2, us-east-2]",
                "bootstrap_region: us-east-1",
                "worker_root_volume_gib: 40",
            )
        ),
        encoding="utf-8",
    )
    return path


def test_frozen_plan_requires_pilot_and_builds_three_fedavg_configs(
    tmp_path: Path,
) -> None:
    path = _write_plan(tmp_path)
    plan = load_frozen_benchmark_plan(path)

    assert plan.benchmark_seeds == (17, 23, 29)
    assert [config.trainer_seed for config in plan.fedavg_configs()] == [17, 23, 29]
    assert all(config.dataset == plan.dataset for config in plan.fedavg_configs())
    assert all(
        config.environment == plan.environment for config in plan.fedavg_configs()
    )
    assert plan.worker_regions == (
        "us-east-1",
        "us-east-1",
        "us-east-2",
        "us-east-2",
    )

    plan.pilot_artifact.unlink()
    with pytest.raises(OfficialBenchmarkError, match="pilot artifact"):
        load_frozen_benchmark_plan(path)


def test_frozen_plan_rejects_duplicate_seeds(tmp_path: Path) -> None:
    path = _write_plan(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace("[17, 23, 29]", "[17, 17, 29]"),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="distinct"):
        load_frozen_benchmark_plan(path)


def test_frozen_plan_rejects_empty_pilot_marker(tmp_path: Path) -> None:
    path = _write_plan(tmp_path)
    (tmp_path / "pilot.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(OfficialBenchmarkError, match="pilot artifact is invalid"):
        load_frozen_benchmark_plan(path)


def test_quality_plan_requires_a_90_percent_pilot(tmp_path: Path) -> None:
    draft = create_quality_draft(
        run_id="quality-001",
        benchmark_seed=17,
        dromeus_commit="a" * 40,
        image_digest=f"sha256:{'b' * 64}",
        pytorch_version="2.13.0+cpu",
    )
    draft_path = tmp_path / "draft.json"
    draft_path.write_text(draft.model_dump_json(), encoding="utf-8")
    pilot_path = tmp_path / "pilot.json"
    pilot = PilotEvidence(
        status="complete",
        model_definition_hash=draft.model_definition_hash,
        dataset=draft.dataset,
        data_source="torchvision-cifar10",
        local_steps=draft.local_steps,
        round_count=draft.round_count,
        learning_rate=draft.learning_rate,
        training=draft.training,
        node_ids=("peer-0", "peer-1", "peer-2", "peer-3"),
        data_artifact_sha256=("1" * 64, "2" * 64, "3" * 64, "4" * 64),
        final_node_accuracies=(0.91, 0.92, 0.93, 0.94),
    )
    pilot_path.write_text(pilot.model_dump_json(), encoding="utf-8")
    plan_path = tmp_path / "plan.yaml"

    plan = write_frozen_plan(
        draft_path=draft_path,
        pilot_artifact=pilot_path,
        benchmark_seeds=(17, 29, 41),
        worker_instance_type="g5.xlarge",
        worker_regions=("us-east-1", "us-east-1", "us-east-2", "us-east-2"),
        bootstrap_region="us-east-1",
        worker_root_volume_gib=80,
        output=plan_path,
    )

    assert plan.parameter_count >= 450_000
    assert plan.learning_rate_schedule == "multistep"
    assert plan.batch_size == 128

    contradictory = plan.model_copy(update={"batch_size": 32})
    with pytest.raises(ValidationError, match="projected training settings"):
        type(plan).model_validate(contradictory.model_dump(mode="python"))

    pilot_path.write_text(
        pilot.model_copy(
            update={"final_node_accuracies": (0.89, 0.92, 0.93, 0.94)}
        ).model_dump_json(),
        encoding="utf-8",
    )
    with pytest.raises(OfficialBenchmarkError, match="at least 90%"):
        load_frozen_benchmark_plan(plan_path)


def test_frozen_plan_rejects_mismatched_dpsgd_draft(tmp_path: Path) -> None:
    plan = load_frozen_benchmark_plan(_write_plan(tmp_path))
    data = manifest_data()
    for field in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del data[field]
    data["peer_scheduler_seed"] = 17
    draft = DraftRunSpec.model_validate(data)

    plan.validate_dpsgd_draft(draft, seed=17)

    mismatched = draft.model_copy(update={"round_count": 99})
    with pytest.raises(OfficialBenchmarkError, match="frozen configuration"):
        plan.validate_dpsgd_draft(mismatched, seed=17)

    mismatched_dataset = draft.model_copy(
        update={
            "dataset": draft.dataset.model_copy(
                update={"partition_sample_counts": (12499, 12501, 12500, 12500)}
            )
        }
    )
    with pytest.raises(OfficialBenchmarkError, match="frozen configuration"):
        plan.validate_dpsgd_draft(mismatched_dataset, seed=17)


def test_prepare_dpsgd_node_configs_connects_frozen_plan_to_four_nodes(
    tmp_path: Path,
) -> None:
    plan_path = _write_plan(tmp_path)
    data = manifest_data()
    for field in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del data[field]
    data["peer_scheduler_seed"] = 17
    draft_path = tmp_path / "draft.yaml"
    draft_path.write_text(json.dumps(data), encoding="utf-8")
    config_paths: list[Path] = []
    for index in range(4):
        path = tmp_path / f"node-{index}.yaml"
        path.write_text(
            "\n".join(
                (
                    f"role: {'initiator' if index == 0 else 'participant'}",
                    f"draft_path: {draft_path}",
                    f"axl_bridge_url: http://127.0.0.1:{9002 + index}",
                    f"run_root: {tmp_path / f'run-{index}'}",
                    f"cifar_root: {tmp_path / 'cifar'}",
                    f"invitation_path: {tmp_path / 'invitation.json'}",
                    "bootstrap_uri: tls://bootstrap.example:9000",
                    "benchmark_seed: 17",
                )
            ),
            encoding="utf-8",
        )
        config_paths.append(path)

    configs = prepare_dpsgd_node_configs(
        plan_path=plan_path,
        draft_path=draft_path,
        seed=17,
        node_config_paths=config_paths,
    )

    assert len(configs) == 4
    assert sum(config.role == "initiator" for config in configs) == 1

    for path in config_paths:
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                str(draft_path),
                "/run/dromeus/draft.yaml",
            ),
            encoding="utf-8",
        )
    deployed = prepare_dpsgd_node_configs(
        plan_path=plan_path,
        draft_path=draft_path,
        seed=17,
        node_config_paths=config_paths,
        deployed_draft_path=Path("/run/dromeus/draft.yaml"),
    )

    assert len(deployed) == 4
