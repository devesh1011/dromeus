from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from support.sample_manifest import manifest_data

from benchmarks.cifar10.official import (
    OfficialBenchmarkError,
    load_frozen_benchmark_plan,
    prepare_dpsgd_node_configs,
)
from dromeus.manifests.models import DraftRunSpec


def _write_plan(root: Path) -> Path:
    pilot = root / "pilot.json"
    pilot.write_text(
        json.dumps(
            {
                "status": "complete",
                "model_definition_hash": "0" * 64,
                "partition_seed": 7,
                "local_steps": 5,
                "round_count": 100,
                "learning_rate": 0.1,
            }
        ),
        encoding="utf-8",
    )
    path = root / "benchmark.yaml"
    path.write_text(
        "\n".join(
            (
                "benchmark_seeds: [17, 23, 29]",
                "partition_seed: 7",
                "local_steps: 5",
                "round_count: 100",
                "learning_rate: 0.1",
                "model_id: cifar-cnn-v1",
                f"model_definition_hash: \"{'0' * 64}\"",
                "dataset_version: '1'",
                f"preprocessing_hash: \"{'1' * 64}\"",
                "max_payload_bytes: 8388608",
                "max_retries: 3",
                "retry_timeout_seconds: 5.0",
                f"pilot_artifact: {pilot}",
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
    assert all(
        config.training_signature() == (5, 100, 0.1, 32, "cpu", True)
        for config in plan.fedavg_configs()
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
