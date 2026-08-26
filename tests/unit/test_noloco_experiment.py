from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest
import yaml

from benchmarks.noloco.dromeus_adapter import (
    DromeusComparabilityError,
    DromeusExperimentAdapter,
)
from benchmarks.noloco.dromeus_official_runner import (
    create_analysis_training_decorator,
)
from benchmarks.noloco.experiment import (
    ArtifactIntegrityError,
    ArtifactValidationError,
    ComparabilityError,
    RunSelectionError,
    RunSelector,
    load_frozen_experiment,
)
from benchmarks.noloco.official_launch import materialize_official_launch
from benchmarks.noloco.pilot import PilotCandidate
from benchmarks.noloco.trajectory_launch import materialize_trajectory_launch
from dromeus.gossip.peer_scheduler import PeerScheduler
from dromeus.manifests.canonical import canonical_hash, file_sha256
from dromeus.manifests.models import (
    EnvironmentFingerprint,
    SealedManifest,
    TransportLimits,
)
from dromeus.training.data import iid_partition_index_hashes
from dromeus.training.resnet18_groupnorm import (
    MODEL_DEFINITION_HASH as RESNET18_DEFINITION_HASH,
)
from dromeus.training.resnet18_groupnorm import build_model as build_resnet18
from dromeus.training.resnet18_groupnorm import tensor_schema_for_model
from dromeus.training.trainer import derive_benchmark_seed

UPSTREAM_COMMIT = "a1b4a425bdc4050a356cf9f4bae7c383419703ab"
TEST_SCHEMA = tensor_schema_for_model(build_resnet18(seed=0))


def _pairing_digest(members: tuple[str, ...], *, seed: int, rounds: int) -> str:
    scheduler = PeerScheduler(
        members,
        seed=seed,
        training_round_count=rounds,
        final_consensus_rounds=0,
    )
    value = [
        {
            "round_id": round_id,
            "pairs": scheduler.schedule(round_id).pairs,
        }
        for round_id in range(rounds)
    ]
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _run(
    root: Path,
    *,
    profile: str,
    world_size: int,
    seed: int,
) -> dict[str, object]:
    checkpoint = root / f"checkpoint-{seed}.safetensors"
    if not checkpoint.exists():
        checkpoint.write_bytes(f"checkpoint:{seed}".encode())
    members = tuple(
        f"axl-key-{world_size}-{index:02d}" for index in range(world_size)
    )
    trainer_seed = derive_benchmark_seed(seed, "local-training")
    round_count = 2
    return {
        "selector": {
            "profile": profile,
            "world_size": world_size,
            "benchmark_seed": seed,
        },
        "checkpoint": {
            "path": checkpoint.name,
            "sha256": file_sha256(checkpoint),
        },
        "model_seed": derive_benchmark_seed(seed, "model-initialization"),
        "scheduler_seed": seed,
        "rank_seeds": [
            {
                "rank": rank,
                "trainer_seed": trainer_seed + rank,
                "augmentation_seed": trainer_seed + rank + 1,
                "loader_seed": trainer_seed + rank + 2,
            }
            for rank in range(world_size)
        ],
        "participants": [
            {"rank": rank, "node_index": rank, "public_key": member}
            for rank, member in enumerate(members)
        ],
        "partition_index_hashes": iid_partition_index_hashes(
            source_sample_count=50_000,
            participant_count=world_size,
            seed=7,
        ),
        "round_count": round_count,
        "transfer": {
            "chunk_size_bytes": 1024 * 1024,
            "window_size": 4,
        },
        "schedule": {
            "schedule_id": "linear-warmup-cosine-v1",
            "total_inner_steps": round_count * 50,
            "warmup_inner_steps": 10,
            "start_learning_rate": 0.0001,
            "peak_learning_rate": 0.001,
            "final_learning_rate": 0.0001,
        },
        "pairing_digest": _pairing_digest(
            members,
            seed=seed,
            rounds=round_count,
        ),
        "evidence": {
            "trajectory_interval": 1 if profile == "trajectory" else 2,
            "absolute_tolerance": 1e-6,
            "relative_tolerance": 1e-6,
            "evaluation_interval_rounds": 25,
            "countsketch_interval_rounds": 1,
            "analysis_checkpoint_interval_rounds": 50,
            "analysis_storage_budget_bytes_per_node": 512 * 1024 * 1024,
            "residual_l2_bound": 3.6,
            "residual_to_signal_ratio_bound": 0.35,
            "overlap_enabled": False,
        },
    }


def _write_experiment(root: Path) -> tuple[Path, dict[str, object]]:
    pilot = root / "pilot.json"
    pilot.write_text('{"status":"complete"}\n', encoding="utf-8")
    runs = [
        _run(root, profile="official", world_size=world_size, seed=seed)
        for world_size in (4, 8, 16)
        for seed in (17, 29, 41)
    ]
    runs.append(_run(root, profile="trajectory", world_size=4, seed=17))
    value: dict[str, object] = {
        "schema_version": 2,
        "status": "frozen",
        "source": {
            "repository": "gensyn-ai/noloco",
            "commit": UPSTREAM_COMMIT,
            "files": ["src/noloco/sparse_optimizer_c.py"],
            "outer_gradient_convention": "phi-minus-theta-v1",
        },
        "model": {
            "model_id": "resnet18-groupnorm-cifar10-v1",
            "definition_hash": RESNET18_DEFINITION_HASH,
            "tensor_schema_hash": canonical_hash(TEST_SCHEMA),
            "parameter_count": 11_173_962,
            "dtype": "float32",
        },
        "dataset": {
            "dataset_id": "cifar10",
            "source": "huggingface-uoft-cs-cifar10",
            "revision": "0b2714987fa478483af9968de7c934580d0bb9a2",
            "preprocessing_hash": "3" * 64,
            "partition_seed": 7,
            "sample_count": 50_000,
        },
        "workload": {
            "batch_size": 128,
            "crop_padding": 4,
            "normalize": True,
            "augment": True,
            "weight_decay": 0.0,
        },
        "algorithm": {
            "alpha": 0.5,
            "beta": 0.7,
            "gamma": 0.7,
            "inner_steps": 50,
            "adam_learning_rate": 0.001,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "gradient_clip_norm": 1.0,
        },
        "ablations": [
            {
                "ablation_id": "identity",
                "artifact_codecs": [
                    {
                        "artifact_name": "outer_gradient",
                        "codec_id": "identity-v1",
                    },
                    {
                        "artifact_name": "slow_weights",
                        "codec_id": "identity-v1",
                    },
                ],
            },
            {
                "ablation_id": "compressed",
                "artifact_codecs": [
                    {
                        "artifact_name": "outer_gradient",
                        "codec_id": "topk-int8-v1",
                        "top_k_fraction": 0.01,
                        "lossy_allowed": True,
                    },
                    {
                        "artifact_name": "slow_weights",
                        "codec_id": "dense-int8-v1",
                        "lossy_allowed": True,
                    },
                ],
            },
        ],
        "pilot": {"path": pilot.name, "sha256": file_sha256(pilot)},
        "hardware": {
            "accelerator_class": "nvidia-a10g",
            "instance_type": "g5.xlarge",
            "container_image_digest": f"sha256:{'b' * 64}",
            "python_version": "3.12.11",
            "pytorch_version": "2.12.1+cu130",
            "cuda_version": "13.0",
            "cudnn_version": 92000,
            "nccl_version": "2.29.7",
            "driver_version": "595.91.07",
            "axl_commit": "628e28ace077f26dfe8d0259009b357216a9d8d4",
        },
        "runs": runs,
    }
    path = root / "experiment.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")
    return path, value


def test_load_and_resolve_frozen_experiment(tmp_path: Path) -> None:
    path, _ = _write_experiment(tmp_path)

    experiment = load_frozen_experiment(path)
    resolved = experiment.resolve(
        RunSelector(profile="official", world_size=8, benchmark_seed=29)
    )

    assert resolved.selector.world_size == 8
    assert resolved.model.model_id == "resnet18-groupnorm-cifar10-v1"
    assert resolved.algorithm.inner_steps == 50
    assert resolved.run.checkpoint.path.is_absolute()
    assert resolved.run.schedule.learning_rate(0) == pytest.approx(0.0001)
    assert resolved.run.schedule.learning_rate(9) == pytest.approx(0.001)
    assert resolved.run.schedule.learning_rate(99) == pytest.approx(0.0001)
    assert len(resolved.run.participants) == 8
    assert resolved.run.transfer.chunk_size_bytes == 1024 * 1024
    assert resolved.run.transfer.window_size == 4
    assert resolved.run.evidence.evaluation_interval_rounds == 25
    assert resolved.run.evidence.countsketch_interval_rounds == 1
    assert resolved.run.evidence.analysis_checkpoint_interval_rounds == 50
    assert resolved.run.evidence.analysis_storage_budget_bytes_per_node == (
        512 * 1024 * 1024
    )
    assert not resolved.run.evidence.overlap_enabled
    assert {item.ablation_id for item in resolved.ablations} == {
        "identity",
        "compressed",
    }
    assert len(resolved.experiment_sha256) == 64
    assert len(resolved.run_config_sha256) == 64
    assert "backend" not in resolved.model_dump()

    again = load_frozen_experiment(path).resolve(resolved.selector)
    assert again.run_config_sha256 == resolved.run_config_sha256


def test_experiment_requires_closed_frozen_schema_and_complete_matrix(
    tmp_path: Path,
) -> None:
    path, value = _write_experiment(tmp_path)
    value["unexpected"] = True
    path.write_text(yaml.safe_dump(value), encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="invalid experiment artifact"):
        load_frozen_experiment(path)

    path, value = _write_experiment(tmp_path)
    del value["ablations"]
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="invalid experiment artifact"):
        load_frozen_experiment(path)

    path, value = _write_experiment(tmp_path)
    value["status"] = "draft"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="invalid experiment artifact"):
        load_frozen_experiment(path)

    path, value = _write_experiment(tmp_path)
    runs = cast(list[object], value["runs"])
    value["runs"] = runs[:-1]
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="trajectory profile"):
        load_frozen_experiment(path)

    path, value = _write_experiment(tmp_path)
    runs = cast(list[object], value["runs"])
    first = cast(dict[str, object], runs[0])
    participants = cast(list[object], first["participants"])
    left = cast(dict[str, object], participants[0])
    right = cast(dict[str, object], participants[1])
    left["public_key"], right["public_key"] = (
        right["public_key"],
        left["public_key"],
    )
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="formation sort order"):
        load_frozen_experiment(path)


def test_experiment_verifies_pilot_and_checkpoint_bytes(tmp_path: Path) -> None:
    path, _ = _write_experiment(tmp_path)
    experiment = load_frozen_experiment(path)
    experiment.pilot.path.write_text("changed\n", encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="pilot artifact hash"):
        load_frozen_experiment(path)

    path, _ = _write_experiment(tmp_path)
    experiment = load_frozen_experiment(path)
    experiment.runs[0].checkpoint.path.write_bytes(b"changed")
    with pytest.raises(ArtifactIntegrityError, match="checkpoint hash"):
        load_frozen_experiment(path)


def test_experiment_rejects_partition_or_pairing_drift(tmp_path: Path) -> None:
    path, value = _write_experiment(tmp_path)
    runs = cast(list[object], value["runs"])
    first = cast(dict[str, object], runs[0])
    first["partition_index_hashes"] = ["f" * 64] * 4
    path.write_text(yaml.safe_dump(value), encoding="utf-8")

    with pytest.raises(ComparabilityError, match="partition hashes"):
        load_frozen_experiment(path)

    path, value = _write_experiment(tmp_path)
    runs = cast(list[object], value["runs"])
    first = cast(dict[str, object], runs[0])
    participants = cast(list[object], first["participants"])
    participant = cast(dict[str, object], participants[0])
    participant["public_key"] = "axl-key-4-00x"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ComparabilityError, match="pairing digest"):
        load_frozen_experiment(path)


def test_experiment_rejects_seed_derivation_drift_and_unknown_run(
    tmp_path: Path,
) -> None:
    path, value = _write_experiment(tmp_path)
    runs = cast(list[object], value["runs"])
    first = cast(dict[str, object], runs[0])
    first["model_seed"] = 123
    path.write_text(yaml.safe_dump(value), encoding="utf-8")

    with pytest.raises(ComparabilityError, match="model seed"):
        load_frozen_experiment(path)

    path, _ = _write_experiment(tmp_path)
    experiment = load_frozen_experiment(path)
    with pytest.raises(RunSelectionError, match="not frozen"):
        experiment.resolve(
            RunSelector(profile="smoke", world_size=4, benchmark_seed=17)
        )


def test_experiment_rejects_paths_outside_artifact_directory(tmp_path: Path) -> None:
    path, value = _write_experiment(tmp_path)
    pilot = cast(dict[str, object], value["pilot"])
    pilot["path"] = "../pilot.json"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="escapes"):
        load_frozen_experiment(path)


def test_semantic_changes_change_experiment_and_resolved_hashes(
    tmp_path: Path,
) -> None:
    path, value = _write_experiment(tmp_path)
    selector = RunSelector(profile="official", world_size=4, benchmark_seed=17)
    original = load_frozen_experiment(path).resolve(selector)

    hardware = cast(dict[str, object], value["hardware"])
    hardware["accelerator_class"] = "nvidia-h100"
    path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")
    changed = load_frozen_experiment(path).resolve(selector)

    assert changed.experiment_sha256 != original.experiment_sha256
    assert changed.run_config_sha256 != original.run_config_sha256


def _environment(model_hash: str) -> EnvironmentFingerprint:
    return EnvironmentFingerprint(
        dromeus_version="0.2.0",
        dromeus_commit="a" * 40,
        pytorch_version="2.12.1+cu130",
        axl_version="628e28ace077f26dfe8d0259009b357216a9d8d4",
        model_definition_hash=model_hash,
        container_image_digest=f"sha256:{'b' * 64}",
    )


def test_pilot_candidate_builds_matched_frozen_ablation_inputs() -> None:
    candidate = PilotCandidate()
    environment = _environment(RESNET18_DEFINITION_HASH)

    identity = candidate.build_draft(
        run_id="pilot-identity",
        ablation_id="identity",
        environment=environment,
    )
    compressed = candidate.build_draft(
        run_id="pilot-compressed",
        ablation_id="compressed",
        environment=environment,
    )

    identity_common = identity.model_dump(
        mode="json",
        exclude={"run_id", "artifact_codecs"},
    )
    compressed_common = compressed.model_dump(
        mode="json",
        exclude={"run_id", "artifact_codecs"},
    )
    assert identity_common == compressed_common
    assert candidate.round_count == 500
    assert identity.transport.chunk_size_bytes == 1024 * 1024
    assert identity.transport.window_size == 4
    assert identity.training is not None
    assert identity.training.learning_rate_schedule is not None
    assert identity.training.learning_rate_schedule.total_inner_steps == 25_000
    assert [item.codec_id for item in compressed.artifact_codecs or ()] == [
        "topk-bitmap-int8-v2",
        "dense-int8-v1",
    ]


def test_dromeus_adapter_projects_one_exact_noloco_draft(tmp_path: Path) -> None:
    path, _ = _write_experiment(tmp_path)
    resolved = load_frozen_experiment(path).resolve(
        RunSelector(profile="official", world_size=8, benchmark_seed=29)
    )
    adapter = DromeusExperimentAdapter(resolved)

    draft = adapter.build_draft(
        run_id="noloco-w8-s29",
        environment=_environment(resolved.model.definition_hash),
        transport=TransportLimits(
            max_payload_bytes=64 * 1024 * 1024,
            max_retries=3,
            retry_timeout_seconds=10.0,
            chunk_size_bytes=1024 * 1024,
            window_size=4,
        ),
    )

    assert draft.algorithm_id == "noloco"
    assert draft.optimizer == "adam"
    assert draft.local_steps == 50
    assert draft.round_count == resolved.run.round_count
    assert draft.learning_rate == resolved.algorithm.adam_learning_rate
    assert draft.peer_scheduler_seed == resolved.run.scheduler_seed
    assert draft.dataset.partition_sample_counts == (6_250,) * 8
    assert draft.dataset.node_index_partitions == tuple(range(8))
    assert draft.training is not None
    assert draft.training.batch_size == 128
    assert draft.training.learning_rate_schedule == resolved.run.schedule
    assert draft.algorithm_config is not None
    assert draft.algorithm_config.inner_steps == 50
    assert draft.artifact_codecs is not None
    assert {item.codec_id for item in draft.artifact_codecs} == {"identity-v1"}

    compressed = adapter.build_draft(
        run_id="noloco-w8-s29-compressed",
        environment=_environment(resolved.model.definition_hash),
        transport=draft.transport,
        ablation_id="compressed",
    )
    assert compressed.artifact_codecs is not None
    assert [item.codec_id for item in compressed.artifact_codecs] == [
        "topk-int8-v1",
        "dense-int8-v1",
    ]


def test_dromeus_adapter_rejects_transport_values_not_frozen_in_run(
    tmp_path: Path,
) -> None:
    path, _ = _write_experiment(tmp_path)
    resolved = load_frozen_experiment(path).resolve(
        RunSelector(profile="official", world_size=4, benchmark_seed=17)
    )

    with pytest.raises(DromeusComparabilityError, match="chunk/window"):
        DromeusExperimentAdapter(resolved).build_draft(
            run_id="wrong-transfer",
            environment=_environment(resolved.model.definition_hash),
            transport=TransportLimits(
                max_payload_bytes=16 * 1024 * 1024,
                max_retries=3,
                retry_timeout_seconds=10.0,
                chunk_size_bytes=1024,
                window_size=4,
            ),
        )


def test_dromeus_adapter_validates_formed_manifest(tmp_path: Path) -> None:
    path, _ = _write_experiment(tmp_path)
    resolved = load_frozen_experiment(path).resolve(
        RunSelector(profile="trajectory", world_size=4, benchmark_seed=17)
    )
    adapter = DromeusExperimentAdapter(resolved)
    draft = adapter.build_draft(
        run_id="trajectory-w4-s17",
        environment=_environment(resolved.model.definition_hash),
        transport=TransportLimits(
            max_payload_bytes=16 * 1024 * 1024,
            max_retries=3,
            retry_timeout_seconds=10.0,
            chunk_size_bytes=1024 * 1024,
            window_size=4,
        ),
    )
    manifest = SealedManifest.model_validate(
        {
            **draft.model_dump(mode="json"),
            "draft_hash": canonical_hash(draft),
            "participants": [
                {
                    "public_key": item.public_key,
                    "node_index": item.node_index,
                }
                for item in resolved.run.participants
            ],
            "initial_checkpoint_hash": resolved.run.checkpoint.sha256,
            "tensor_schema": TEST_SCHEMA.model_dump(mode="json"),
        }
    )

    adapter.validate_sealed_manifest(manifest, draft=draft)

    changed_member = manifest.model_copy(
        update={
            "participants": (
                manifest.participants[0].model_copy(
                    update={"public_key": "different-key"}
                ),
                *manifest.participants[1:],
            )
        }
    )
    with pytest.raises(DromeusComparabilityError, match="membership"):
        adapter.validate_sealed_manifest(changed_member, draft=draft)

    changed_checkpoint = manifest.model_copy(
        update={"initial_checkpoint_hash": "f" * 64}
    )
    with pytest.raises(DromeusComparabilityError, match="checkpoint"):
        adapter.validate_sealed_manifest(changed_checkpoint, draft=draft)


def test_dromeus_adapter_rejects_environment_model_mismatch(tmp_path: Path) -> None:
    path, _ = _write_experiment(tmp_path)
    resolved = load_frozen_experiment(path).resolve(
        RunSelector(profile="official", world_size=4, benchmark_seed=17)
    )

    with pytest.raises(DromeusComparabilityError, match="model definition"):
        DromeusExperimentAdapter(resolved).build_draft(
            run_id="bad-environment",
            environment=_environment("f" * 64),
            transport=TransportLimits(
                max_payload_bytes=16 * 1024 * 1024,
                max_retries=3,
                retry_timeout_seconds=10.0,
                chunk_size_bytes=1024,
                window_size=4,
            ),
        )


def test_dromeus_adapter_rejects_frozen_runtime_mismatch(tmp_path: Path) -> None:
    path, _ = _write_experiment(tmp_path)
    resolved = load_frozen_experiment(path).resolve(
        RunSelector(profile="official", world_size=4, benchmark_seed=17)
    )
    environment = _environment(resolved.model.definition_hash).model_copy(
        update={"container_image_digest": f"sha256:{'c' * 64}"}
    )

    with pytest.raises(DromeusComparabilityError, match="runtime"):
        DromeusExperimentAdapter(resolved).build_draft(
            run_id="bad-runtime",
            environment=environment,
            transport=TransportLimits(
                max_payload_bytes=16 * 1024 * 1024,
                max_retries=3,
                retry_timeout_seconds=10.0,
                chunk_size_bytes=1024 * 1024,
                window_size=4,
            ),
        )


def test_materialize_trajectory_launch_writes_bound_node_configs(
    tmp_path: Path,
) -> None:
    frozen_root = tmp_path / "frozen"
    frozen_root.mkdir()
    experiment_path, _ = _write_experiment(frozen_root)
    run = load_frozen_experiment(experiment_path).resolve(
        RunSelector(profile="trajectory", world_size=4, benchmark_seed=17)
    )
    output_root = tmp_path / "launch"
    runtime_root = Path("/opt/dromeus/trajectory")

    result = materialize_trajectory_launch(
        run=run,
        output_root=output_root,
        runtime_root=runtime_root,
        run_id="trajectory-w4-s17",
        environment=_environment(run.model.definition_hash),
        bootstrap_uri="tls://203.0.113.10:9300",
    )

    assert result.draft.round_count == 2
    assert [item.codec_id for item in result.draft.artifact_codecs or ()] == [
        "identity-v1",
        "identity-v1",
    ]
    assert len(result.node_configs) == 4
    assert result.node_configs[0].role.value == "initiator"
    assert all(
        item.manifest_expectation
        == DromeusExperimentAdapter(run).manifest_expectation(draft=result.draft)
        for item in result.node_configs
    )
    assert result.node_configs[0].draft_path == runtime_root / "draft.yaml"
    assert result.node_configs[3].run_root == runtime_root / "rank-3" / "run"
    assert (output_root / "draft.yaml").is_file()
    assert len(tuple(output_root.glob("node-*.yaml"))) == 4


def test_official_analysis_decorator_preflights_storage(tmp_path: Path) -> None:
    experiment_path, _ = _write_experiment(tmp_path)
    experiment = load_frozen_experiment(experiment_path)
    official = experiment.resolve(
        RunSelector(profile="official", world_size=16, benchmark_seed=17)
    )
    trajectory = experiment.resolve(
        RunSelector(profile="trajectory", world_size=4, benchmark_seed=17)
    )

    decorator = create_analysis_training_decorator(
        run=official,
        output_root=tmp_path / "analysis",
        available_bytes=512 * 1024 * 1024,
    )

    assert callable(decorator)
    with pytest.raises(ValueError, match="available"):
        create_analysis_training_decorator(
            run=official,
            output_root=tmp_path / "analysis",
            available_bytes=1,
        )
    with pytest.raises(ValueError, match="official"):
        create_analysis_training_decorator(
            run=trajectory,
            output_root=tmp_path / "analysis",
            available_bytes=512 * 1024 * 1024,
        )


def test_materialize_official_launch_closes_dromeus_and_nccl_inputs(
    tmp_path: Path,
) -> None:
    frozen_root = tmp_path / "frozen"
    frozen_root.mkdir()
    experiment_path, _ = _write_experiment(frozen_root)
    run = load_frozen_experiment(experiment_path).resolve(
        RunSelector(profile="official", world_size=8, benchmark_seed=29)
    )
    output_root = tmp_path / "launch"
    runtime_root = Path("/opt/dromeus/official")

    launch = materialize_official_launch(
        run=run,
        ablation_id="compressed",
        output_root=output_root,
        runtime_root=runtime_root,
        run_id="official-w8-s29-compressed",
        environment=_environment(run.model.definition_hash),
        bootstrap_uri="tls://203.0.113.10:9300",
    )

    assert len(launch.node_configs) == 8
    assert launch.node_configs[0].role.value == "initiator"
    assert launch.node_configs[-1].run_root == runtime_root / "rank-7" / "run"
    assert launch.draft.artifact_codecs is not None
    assert [item.codec_id for item in launch.draft.artifact_codecs] == [
        "topk-int8-v1",
        "dense-int8-v1",
    ]
    metadata = json.loads(launch.metadata_path.read_text(encoding="utf-8"))
    assert metadata["docker_shm_size_bytes"] == 1024 * 1024 * 1024
    assert metadata["nccl_socket_interface"] == "ens5"
    assert metadata["dromeus_runner_module"] == (
        "benchmarks.noloco.dromeus_official_runner"
    )
    assert len(tuple(output_root.glob("node-*.yaml"))) == 8
