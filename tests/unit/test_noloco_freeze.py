from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from benchmarks.noloco.experiment import (
    RunSelector,
    load_frozen_experiment,
)
from benchmarks.noloco.freeze import (
    build_frozen_experiment_document,
    validate_freeze_runtime,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _copy_pilot_report(destination: Path) -> None:
    shutil.copyfile(
        REPO_ROOT
        / "benchmarks"
        / "results"
        / "m2"
        / "compression-calibration"
        / "pilot-report.json",
        destination,
    )


def test_build_frozen_experiment_document_closes_complete_matrix(
    tmp_path: Path,
) -> None:
    pilot = tmp_path / "pilot-report.json"
    _copy_pilot_report(pilot)
    checkpoints: dict[int, Path] = {}
    for seed in (17, 29, 41):
        checkpoint = tmp_path / f"checkpoint-{seed}.safetensors"
        checkpoint.write_bytes(f"checkpoint:{seed}".encode())
        checkpoints[seed] = checkpoint
    identities = tuple(f"{index + 1:064x}" for index in reversed(range(16)))

    document = build_frozen_experiment_document(
        artifact_root=tmp_path,
        identity_public_keys=identities,
        pilot_report=pilot,
        checkpoints=checkpoints,
    )
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        yaml.safe_dump(document, sort_keys=True),
        encoding="utf-8",
    )
    experiment = load_frozen_experiment(experiment_path)
    trajectory = experiment.resolve(
        RunSelector(profile="trajectory", world_size=4, benchmark_seed=17)
    )

    assert len(experiment.runs) == 10
    assert trajectory.run.round_count == 2
    assert tuple(item.public_key for item in trajectory.run.participants) == tuple(
        sorted(identities)[:4]
    )
    assert experiment.source.commit == (
        "a1b4a425bdc4050a356cf9f4bae7c383419703ab"
    )
    assert experiment.hardware.container_image_digest == (
        "sha256:1b40d3774dc864f1f4450720f91243f2e379de8ec7d12805d120ec1ef1419a3c"
    )
    assert experiment.hardware.nccl_version == "2.29.7"
    assert experiment.hardware.accelerator_class == "NVIDIA-A10G"
    assert experiment.hardware.instance_type == "g5.xlarge"
    assert trajectory.run.evidence.evaluation_interval_rounds == 25
    assert trajectory.run.evidence.countsketch_interval_rounds == 1
    assert trajectory.run.evidence.analysis_checkpoint_interval_rounds == 50
    assert trajectory.run.evidence.analysis_storage_budget_bytes_per_node == (
        512 * 1024 * 1024
    )
    assert not trajectory.run.evidence.overlap_enabled
    assert [item.codec_id for item in experiment.ablations[1].artifact_codecs] == [
        "topk-bitmap-int8-v2",
        "dense-int8-v1",
    ]
    assert experiment.ablations[1].artifact_codecs[0].top_k_fraction == 0.45


@pytest.mark.parametrize(
    "identities",
    (
        tuple(f"{index:064x}" for index in range(15)),
        tuple("1" * 64 for _ in range(16)),
        tuple(f"key-{index}" for index in range(16)),
    ),
)
def test_build_frozen_experiment_rejects_invalid_identity_pool(
    tmp_path: Path,
    identities: tuple[str, ...],
) -> None:
    pilot = tmp_path / "pilot-report.json"
    _copy_pilot_report(pilot)
    checkpoints: dict[int, Path] = {}
    for seed in (17, 29, 41):
        checkpoint = tmp_path / f"checkpoint-{seed}.safetensors"
        checkpoint.write_bytes(b"checkpoint")
        checkpoints[seed] = checkpoint

    with pytest.raises(ValueError, match="identity"):
        build_frozen_experiment_document(
            artifact_root=tmp_path,
            identity_public_keys=identities,
            pilot_report=pilot,
            checkpoints=checkpoints,
        )


def test_build_frozen_experiment_rejects_unbound_pilot_report(
    tmp_path: Path,
) -> None:
    pilot = tmp_path / "pilot-report.json"
    pilot.write_text("{}\n", encoding="utf-8")
    checkpoints: dict[int, Path] = {}
    for seed in (17, 29, 41):
        checkpoint = tmp_path / f"checkpoint-{seed}.safetensors"
        checkpoint.write_bytes(b"checkpoint")
        checkpoints[seed] = checkpoint

    with pytest.raises(ValueError, match="pilot report hash"):
        build_frozen_experiment_document(
            artifact_root=tmp_path,
            identity_public_keys=tuple(f"{index + 1:064x}" for index in range(16)),
            pilot_report=pilot,
            checkpoints=checkpoints,
        )


def test_freeze_runtime_requires_final_x86_container_versions() -> None:
    validate_freeze_runtime(
        machine="x86_64",
        python_version="3.12.11",
        torch_version="2.12.1+cu130",
    )

    with pytest.raises(RuntimeError, match="x86_64"):
        validate_freeze_runtime(
            machine="arm64",
            python_version="3.12.11",
            torch_version="2.12.1+cu130",
        )
