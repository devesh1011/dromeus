from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from support.sample_manifest import manifest_data

from benchmarks.noloco.dromeus_trajectory_runner import (
    create_trajectory_training_decorator,
)
from benchmarks.noloco.trajectory import (
    DromeusTrajectoryContext,
    DromeusTrajectoryWriter,
    TrajectoryCapturingAlgorithm,
    compare_trajectories,
)
from benchmarks.noloco.trajectory_compare import main as compare_main
from benchmarks.noloco_reference.trajectory import (
    TrajectoryContext,
    TrajectoryWriter,
)
from dromeus.algorithms.base import UpdateBundle
from dromeus.algorithms.codec import NamedSafetensorsUpdateBundleCodec
from dromeus.algorithms.noloco import NoLoCoAlgorithm
from dromeus.gossip.peer_scheduler import PeerScheduler
from dromeus.manifests.canonical import canonical_hash
from dromeus.manifests.models import (
    AdamSettings,
    DraftRunSpec,
    NoLoCoConfig,
    SealedManifest,
    Tensor,
    TensorSchema,
)
from dromeus.membership.formation import FormationResult
from dromeus.persistence.run_store import RunStore
from dromeus.runtime import TrainingConfig


class _Trainer:
    def __init__(self, value: float) -> None:
        self._weights = {"weight": np.array([value], dtype=np.float32)}

    def train_local_steps(self, step_count: int) -> None:
        assert step_count == 50
        self._weights["weight"] *= np.float32(0.5)

    def weights(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self._weights.items()}

    def load_weights(self, weights: dict[str, np.ndarray]) -> None:
        self._weights = {name: value.copy() for name, value in weights.items()}

    @property
    def local_loss(self) -> float:
        return float(np.square(self._weights["weight"][0]))

    def evaluate(self) -> tuple[float, float]:
        return self.local_loss, 0.0


class _MatrixTrainer:
    def __init__(self) -> None:
        self._weights = {"layer.weight": np.zeros((2, 2), dtype=np.float32)}

    def train_local_steps(self, step_count: int) -> None:
        self._weights["layer.weight"] += np.float32(step_count * 0.001)

    def weights(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self._weights.items()}

    def load_weights(self, weights: dict[str, np.ndarray]) -> None:
        self._weights = {name: value.copy() for name, value in weights.items()}

    @property
    def local_loss(self) -> float:
        return float(np.mean(np.square(self._weights["layer.weight"])))

    def evaluate(self) -> tuple[float, float]:
        return self.local_loss, 0.0

    def checkpoint_tensors(self) -> dict[str, np.ndarray]:
        return self.weights()

    def load_checkpoint_tensors(self, state: dict[str, np.ndarray]) -> None:
        self.load_weights(state)


@pytest.mark.parametrize(
    ("mismatch", "expected_pass"),
    ((False, True), (True, False)),
)
def test_trajectories_write_report_and_delete_full_tensors(
    tmp_path: Path,
    mismatch: bool,
    expected_pass: bool,
) -> None:
    dromeus_root = tmp_path / "dromeus"
    reference_root = tmp_path / "reference"
    members = tuple(f"key-{rank}" for rank in range(4))
    for rank, node_id in enumerate(members):
        dromeus = DromeusTrajectoryWriter(
            root=dromeus_root,
            context=DromeusTrajectoryContext(
                experiment_sha256="1" * 64,
                run_config_sha256="2" * 64,
                initial_checkpoint_sha256="3" * 64,
                tensor_schema_hash="4" * 64,
                world_size=4,
                benchmark_seed=17,
                rank=rank,
                node_id=node_id,
                participants=members,
                scheduler_seed=17,
                total_outer_steps=2,
                source_repository="gensyn-ai/noloco",
                source_commit="a1b4a425bdc4050a356cf9f4bae7c383419703ab",
                source_path="src/noloco/sparse_optimizer_c.py",
                outer_gradient_convention="phi-minus-theta-v1",
                pairing_convention="dromeus-peer-scheduler-v1",
            ),
        )
        reference = TrajectoryWriter(
            root=reference_root,
            context=TrajectoryContext(
                experiment_sha256="1" * 64,
                run_config_sha256="2" * 64,
                initial_checkpoint_sha256="3" * 64,
                tensor_schema_hash="4" * 64,
                world_size=4,
                benchmark_seed=17,
                rank=rank,
                node_id=node_id,
                interval=1,
                total_outer_steps=2,
                backend="nccl",
                acceptance_eligible=True,
                source_repository="gensyn-ai/noloco",
                source_commit="a1b4a425bdc4050a356cf9f4bae7c383419703ab",
                source_path="src/noloco/sparse_optimizer_c.py",
                outer_gradient_convention="phi-minus-theta-v1",
                pairing_convention="dromeus-peer-scheduler-v1",
            ),
        )
        for completed in range(3):
            peer_rank = None if completed == 0 else (rank + 1) % 4
            values = np.array([rank, completed, -completed], dtype=np.float32)
            reference_values = values.copy()
            if mismatch and rank == 3 and completed == 2:
                reference_values[2] += np.float32(1e-4)
            dromeus.write(
                completed_outer_steps=completed,
                peer_rank=peer_rank,
                slow_weights={"weight": values},
            )
            reference.write(
                completed_outer_steps=completed,
                peer_rank=peer_rank,
                slow_weights={
                    "weight": torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
                        reference_values
                    )
                },
            )

    report_path = tmp_path / "comparison.json"
    report = compare_trajectories(
        dromeus_root=dromeus_root,
        reference_root=reference_root,
        report_path=report_path,
        absolute_tolerance=1e-6,
        relative_tolerance=1e-6,
    )

    assert report.passed is expected_pass
    assert len(report.snapshots) == 12
    assert report_path.is_file()
    assert (
        json.loads(report_path.read_text(encoding="utf-8"))["passed"]
        is expected_pass
    )
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_payload["source_commit"] == (
        "a1b4a425bdc4050a356cf9f4bae7c383419703ab"
    )
    assert report_payload["source_path"] == "src/noloco/sparse_optimizer_c.py"
    assert report_payload["outer_gradient_convention"] == "phi-minus-theta-v1"
    assert report_payload["pairing_convention"] == "dromeus-peer-scheduler-v1"
    assert not tuple(dromeus_root.rglob("*.safetensors"))
    assert not tuple(reference_root.rglob("*.safetensors"))


def test_trajectory_comparator_cli_returns_nonzero_for_mismatch(
    tmp_path: Path,
) -> None:
    dromeus_root = tmp_path / "dromeus"
    reference_root = tmp_path / "reference"
    members = tuple(f"key-{rank}" for rank in range(4))
    for rank, node_id in enumerate(members):
        dromeus = DromeusTrajectoryWriter(
            root=dromeus_root,
            context=_dromeus_context(rank=rank, node_id=node_id, members=members),
        )
        reference = TrajectoryWriter(
            root=reference_root,
            context=_reference_context(rank=rank, node_id=node_id),
        )
        for completed in range(3):
            peer_rank = None if completed == 0 else (rank + 1) % 4
            value = np.array([rank + completed], dtype=np.float32)
            reference_value = value.copy()
            if rank == 0 and completed == 2:
                reference_value += np.float32(1e-4)
            dromeus.write(
                completed_outer_steps=completed,
                peer_rank=peer_rank,
                slow_weights={"weight": value},
            )
            reference.write(
                completed_outer_steps=completed,
                peer_rank=peer_rank,
                slow_weights={
                    "weight": torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
                        reference_value
                    )
                },
            )

    assert compare_main(
        [
            "--dromeus-root",
            str(dromeus_root),
            "--reference-root",
            str(reference_root),
            "--report",
            str(tmp_path / "report.json"),
            "--absolute-tolerance",
            "1e-6",
            "--relative-tolerance",
            "1e-6",
        ]
    ) == 1
    assert not tuple(tmp_path.rglob("*.safetensors"))


def test_dromeus_wrapper_captures_initialization_and_two_outer_steps(
    tmp_path: Path,
) -> None:
    members = tuple(f"key-{rank}" for rank in range(4))
    schema = TensorSchema(
        tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),)
    )
    config = NoLoCoConfig(
        alpha=0.5,
        beta=0.7,
        gamma=0.7,
        inner_steps=50,
        adam=AdamSettings(
            learning_rate=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            gradient_clip_norm=1.0,
        ),
    )
    algorithms: list[TrajectoryCapturingAlgorithm] = []
    for rank, node_id in enumerate(members):
        delegate = NoLoCoAlgorithm(
            trainer=_Trainer(float(rank + 1)),
            tensor_schema=schema,
            config=config,
            bundle_codec=NamedSafetensorsUpdateBundleCodec(
                artifact_root=tmp_path / "bundles" / str(rank),
                run_id="trajectory-test",
                manifest_hash="0" * 64,
                sender_public_key=node_id,
                algorithm_id="noloco",
                artifact_schemas={
                    "outer_gradient": schema,
                    "slow_weights": schema,
                },
            ),
        )
        algorithms.append(
            TrajectoryCapturingAlgorithm(
                delegate=delegate,
                writer=DromeusTrajectoryWriter(
                    root=tmp_path / "trajectory",
                    context=DromeusTrajectoryContext(
                        experiment_sha256="1" * 64,
                        run_config_sha256="2" * 64,
                        initial_checkpoint_sha256="3" * 64,
                        tensor_schema_hash="4" * 64,
                        world_size=4,
                        benchmark_seed=17,
                        rank=rank,
                        node_id=node_id,
                        participants=members,
                        scheduler_seed=17,
                        total_outer_steps=2,
                        source_repository="gensyn-ai/noloco",
                        source_commit=(
                            "a1b4a425bdc4050a356cf9f4bae7c383419703ab"
                        ),
                        source_path="src/noloco/sparse_optimizer_c.py",
                        outer_gradient_convention="phi-minus-theta-v1",
                        pairing_convention="dromeus-peer-scheduler-v1",
                    ),
                ),
            )
        )
    scheduler = PeerScheduler(
        members,
        seed=17,
        training_round_count=2,
        final_consensus_rounds=0,
    )
    for round_id in range(2):
        bundles: list[UpdateBundle] = []
        for algorithm in algorithms:
            algorithm.pre_local(round_id)
            algorithm.local_training()
            bundles.append(algorithm.post_local_bundle())
        for left, right in scheduler.schedule(round_id).pairs:
            left_rank = members.index(left)
            right_rank = members.index(right)
            left_update = algorithms[left_rank].validate_peer(bundles[right_rank])
            right_update = algorithms[right_rank].validate_peer(bundles[left_rank])
            algorithms[left_rank].peer_apply(left_update)
            algorithms[right_rank].peer_apply(right_update)
        for algorithm, bundle in zip(algorithms, bundles, strict=True):
            algorithm.release_bundle(bundle)

    for rank in range(4):
        records = [
            json.loads(line)
            for line in (tmp_path / "trajectory" / f"rank-{rank}" / "trajectory.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert [item["completed_outer_steps"] for item in records] == [0, 1, 2]
        assert records[0]["peer_rank"] is None
        assert all(item["peer_rank"] != rank for item in records[1:])


def test_training_decorator_replaces_only_formed_noloco_algorithm(
    tmp_path: Path,
) -> None:
    data = manifest_data()
    data.update(
        {
            "algorithm_id": "noloco",
            "optimizer": "adam",
            "local_steps": 50,
            "round_count": 2,
            "learning_rate": 0.001,
            "algorithm_config": {
                "alpha": 0.5,
                "beta": 0.7,
                "gamma": 0.7,
                "inner_steps": 50,
                "adam": {
                    "learning_rate": 0.001,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "epsilon": 1e-8,
                    "gradient_clip_norm": 1.0,
                },
            },
            "artifact_codecs": [
                {"artifact_name": "outer_gradient", "codec_id": "identity-v1"},
                {"artifact_name": "slow_weights", "codec_id": "identity-v1"},
            ],
            "training": {
                "batch_size": 128,
                "momentum": 0.0,
                "weight_decay": 0.0,
                "learning_rate_milestones": [],
                "learning_rate_gamma": 0.1,
                "learning_rate_schedule": {
                    "schedule_id": "linear-warmup-cosine-v1",
                    "total_inner_steps": 100,
                    "warmup_inner_steps": 10,
                    "start_learning_rate": 0.0001,
                    "peak_learning_rate": 0.001,
                    "final_learning_rate": 0.0001,
                },
                "crop_padding": 4,
                "normalize": True,
                "final_consensus_rounds": 0,
            },
            "transport": {
                **data["transport"],
                "chunk_size_bytes": 1024,
                "window_size": 1,
            },
        }
    )
    draft_data = {
        key: value
        for key, value in data.items()
        if key
        not in {
            "draft_hash",
            "participants",
            "initial_checkpoint_hash",
            "tensor_schema",
        }
    }
    data["draft_hash"] = canonical_hash(DraftRunSpec.model_validate(draft_data))
    manifest = SealedManifest.model_validate(data)
    assert manifest.algorithm_config is not None
    algorithm = NoLoCoAlgorithm(
        trainer=_MatrixTrainer(),
        tensor_schema=manifest.tensor_schema,
        config=manifest.algorithm_config,
    )
    def load_checkpoint(path: Path) -> None:
        del path
    training = TrainingConfig(
        algorithm=algorithm,
        load_checkpoint=load_checkpoint,
        run_store=RunStore(tmp_path / "run-store"),
        artifact_root=tmp_path / "artifacts",
    )
    members = tuple(
        participant.public_key for participant in manifest.participants
    )
    context = DromeusTrajectoryContext(
        experiment_sha256="1" * 64,
        run_config_sha256="2" * 64,
        initial_checkpoint_sha256=manifest.initial_checkpoint_hash,
        tensor_schema_hash=canonical_hash(manifest.tensor_schema),
        world_size=4,
        benchmark_seed=17,
        rank=0,
        node_id=members[0],
        participants=members,
        scheduler_seed=manifest.peer_scheduler_seed,
        total_outer_steps=2,
        source_repository="gensyn-ai/noloco",
        source_commit="a1b4a425bdc4050a356cf9f4bae7c383419703ab",
        source_path="src/noloco/sparse_optimizer_c.py",
        outer_gradient_convention="phi-minus-theta-v1",
        pairing_convention="dromeus-peer-scheduler-v1",
    )
    decorate = create_trajectory_training_decorator(
        context=context,
        output_root=tmp_path / "trajectory",
    )

    decorated = decorate(
        FormationResult(
            manifest=manifest,
            manifest_hash=canonical_hash(manifest),
            checkpoint_path=tmp_path / "checkpoint.safetensors",
        ),
        members[0],
        training,
    )

    assert isinstance(decorated.algorithm, TrajectoryCapturingAlgorithm)
    assert decorated.load_checkpoint is load_checkpoint
    assert decorated.run_store is training.run_store
    assert decorated.artifact_root == training.artifact_root


def _dromeus_context(
    *,
    rank: int,
    node_id: str,
    members: tuple[str, ...],
) -> DromeusTrajectoryContext:
    return DromeusTrajectoryContext(
        experiment_sha256="1" * 64,
        run_config_sha256="2" * 64,
        initial_checkpoint_sha256="3" * 64,
        tensor_schema_hash="4" * 64,
        world_size=4,
        benchmark_seed=17,
        rank=rank,
        node_id=node_id,
        participants=members,
        scheduler_seed=17,
        total_outer_steps=2,
        source_repository="gensyn-ai/noloco",
        source_commit="a1b4a425bdc4050a356cf9f4bae7c383419703ab",
        source_path="src/noloco/sparse_optimizer_c.py",
        outer_gradient_convention="phi-minus-theta-v1",
        pairing_convention="dromeus-peer-scheduler-v1",
    )


def _reference_context(*, rank: int, node_id: str) -> TrajectoryContext:
    return TrajectoryContext(
        experiment_sha256="1" * 64,
        run_config_sha256="2" * 64,
        initial_checkpoint_sha256="3" * 64,
        tensor_schema_hash="4" * 64,
        world_size=4,
        benchmark_seed=17,
        rank=rank,
        node_id=node_id,
        interval=1,
        total_outer_steps=2,
        backend="nccl",
        acceptance_eligible=True,
        source_repository="gensyn-ai/noloco",
        source_commit="a1b4a425bdc4050a356cf9f4bae7c383419703ab",
        source_path="src/noloco/sparse_optimizer_c.py",
        outer_gradient_convention="phi-minus-theta-v1",
        pairing_convention="dromeus-peer-scheduler-v1",
    )
