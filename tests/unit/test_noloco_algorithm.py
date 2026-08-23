from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from support.sample_manifest import manifest_data
from torch import nn
from torch.utils.data import TensorDataset

from dromeus.algorithms.base import NamedValidatedUpdate, checksum_artifacts
from dromeus.algorithms.codec import NamedSafetensorsUpdateBundleCodec
from dromeus.algorithms.noloco import NoLoCoAlgorithm
from dromeus.manifests.models import (
    AdamSettings,
    NoLoCoConfig,
    SealedManifest,
    Tensor,
    TensorSchema,
)
from dromeus.persistence.archive import RunArchive
from dromeus.persistence.run_store import RunStore
from dromeus.training.trainer import PyTorchTrainer, TrainerSettings


class HandTraceTrainer:
    def __init__(
        self, initial: float | list[float], trained: float | list[float]
    ) -> None:
        self._weights = {"weight": np.atleast_1d(np.asarray(initial, dtype=np.float32))}
        self._trained = np.atleast_1d(np.asarray(trained, dtype=np.float32))
        self.train_calls: list[int] = []

    def train_local_steps(self, step_count: int) -> None:
        self.train_calls.append(step_count)
        self._weights["weight"] = self._trained.copy()

    def weights(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self._weights.items()}

    def load_weights(self, weights: dict[str, np.ndarray]) -> None:
        self._weights = {name: value.copy() for name, value in weights.items()}

    @property
    def local_loss(self) -> None:
        return None

    def evaluate(self) -> None:
        return None

    def checkpoint_tensors(self) -> dict[str, np.ndarray]:
        return {
            **self.weights(),
            "adam.first.weight": np.array([0.25], dtype=np.float32),
            "adam.second.weight": np.array([0.5], dtype=np.float32),
            "completed_steps": np.array([sum(self.train_calls)], dtype=np.int64),
            "loader_rng": np.array([1, 2, 3], dtype=np.uint8),
        }

    def load_checkpoint_tensors(self, state: dict[str, np.ndarray]) -> None:
        self.load_weights({"weight": state["weight"]})
        self.train_calls = [int(state["completed_steps"][0])]


def _config() -> NoLoCoConfig:
    return NoLoCoConfig(
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


def _bundle_codec(
    root: Path, *, sender: str, schema: TensorSchema
) -> NamedSafetensorsUpdateBundleCodec:
    return NamedSafetensorsUpdateBundleCodec(
        artifact_root=root,
        run_id="test-run",
        manifest_hash="0" * 64,
        sender_public_key=sender,
        algorithm_id="noloco",
        artifact_schemas={
            "outer_gradient": schema,
            "slow_weights": schema,
        },
    )


def _adam_trainer(*, initial: tuple[float, float], seed: int) -> PyTorchTrainer:
    model = nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor(initial, dtype=torch.float32).view(2, 1))
    data = TensorDataset(
        torch.tensor([[-2.0], [-1.0], [1.0], [2.0]], dtype=torch.float32),
        torch.tensor([0, 0, 1, 1], dtype=torch.long),
    )
    return PyTorchTrainer(
        model=model,
        model_definition="test-noloco-linear",
        train_data=data,  # pyright: ignore[reportArgumentType]
        settings=TrainerSettings(
            seed=seed,
            batch_size=2,
            learning_rate=0.001,
            optimizer="adam",
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            gradient_clip_norm=1.0,
            augment=False,
        ),
    )


def test_two_instances_match_hand_computed_noloco_trace(tmp_path: Path) -> None:
    schema = TensorSchema(tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),))
    first_trainer = HandTraceTrainer(initial=1.0, trained=0.5)
    second_trainer = HandTraceTrainer(initial=3.0, trained=1.5)
    first = NoLoCoAlgorithm(
        trainer=first_trainer,
        tensor_schema=schema,
        config=_config(),
        bundle_codec=_bundle_codec(tmp_path / "first", sender="first", schema=schema),
    )
    second = NoLoCoAlgorithm(
        trainer=second_trainer,
        tensor_schema=schema,
        config=_config(),
        bundle_codec=_bundle_codec(tmp_path / "second", sender="second", schema=schema),
    )

    first.pre_local(0)
    second.pre_local(0)
    first.local_training()
    second.local_training()
    first_bundle = first.post_local_bundle()
    second_bundle = second.post_local_bundle()
    try:
        first_update = first.validate_peer(second_bundle)
        second_update = second.validate_peer(first_bundle)
        first_snapshot = first.peer_apply(first_update)
        second_snapshot = second.peer_apply(second_update)
    finally:
        first.release_bundle(first_bundle)
        second.release_bundle(second_bundle)

    assert first_trainer.train_calls == [50]
    assert second_trainer.train_calls == [50]
    assert tuple(item.name for item in first_bundle.metadata.artifacts) == (
        "outer_gradient",
        "slow_weights",
    )
    assert np.array_equal(
        first_snapshot.weights["weight"], np.array([1.0], dtype=np.float32)
    )
    assert np.array_equal(
        second_snapshot.weights["weight"],
        np.array([1.6], dtype=np.float32),
    )
    assert np.array_equal(
        first.checkpoint_tensors()["noloco.v1.outer_momentum.weight"],
        np.array([0.0], dtype=np.float32),
    )
    assert np.array_equal(
        second.checkpoint_tensors()["noloco.v1.outer_momentum.weight"],
        np.array([-1.4], dtype=np.float32),
    )
    assert not any(tmp_path.rglob("*.safetensors"))


def test_identity_path_matches_pinned_upstream_outer_step_fixture(
    tmp_path: Path,
) -> None:
    fixture_path = (
        Path(__file__).parents[1] / "golden" / "noloco_upstream_outer_step_v1.json"
    )
    fixture = cast(dict[str, Any], json.loads(fixture_path.read_text(encoding="utf-8")))
    assert fixture["authority"]["commit"] == (
        "a1b4a425bdc4050a356cf9f4bae7c383419703ab"
    )
    assert fixture["authority"]["function"] == "outer_step"
    assert fixture["convention"] == {
        "alpha": 0.5,
        "beta": 0.7,
        "dromeus_outer_gradient": "slow_weights - fast_weights",
        "effective_gamma": 0.7,
    }
    nodes = {
        cast(int, node["rank"]): cast(dict[str, Any], node) for node in fixture["nodes"]
    }
    schema = TensorSchema(tensors=(Tensor(name="weight", dtype="float32", shape=(3,)),))
    algorithms: list[NoLoCoAlgorithm] = []
    for rank in range(2):
        node = nodes[rank]
        algorithm = NoLoCoAlgorithm(
            trainer=HandTraceTrainer(
                initial=cast(list[float], node["slow_weights"]),
                trained=cast(list[float], node["fast_weights"]),
            ),
            tensor_schema=schema,
            config=_config(),
            bundle_codec=_bundle_codec(
                tmp_path / f"rank-{rank}", sender=f"rank-{rank}", schema=schema
            ),
        )
        state = algorithm.checkpoint_tensors()
        state["noloco.v1.round_id"] = np.array([0], dtype=np.int64)
        state["noloco.v1.completed_outer_steps"] = np.array([1], dtype=np.int64)
        state["noloco.v1.phase"] = np.array([4], dtype=np.int64)
        state["noloco.v1.outer_momentum.weight"] = np.asarray(
            node["outer_momentum_before"], dtype=np.float32
        )
        algorithm.load_checkpoint_tensors(state)
        algorithms.append(algorithm)

    for algorithm in algorithms:
        algorithm.pre_local(1)
        algorithm.local_training()
    bundles = [algorithm.post_local_bundle() for algorithm in algorithms]
    tolerance = fixture["comparison_tolerance"]
    absolute = cast(float, tolerance["absolute"])
    relative = cast(float, tolerance["relative"])
    try:
        for rank, algorithm in enumerate(algorithms):
            local = algorithm.validate_peer(bundles[rank])
            np.testing.assert_allclose(
                local.artifacts["outer_gradient"]["weight"],
                np.asarray(nodes[rank]["outer_gradient"], dtype=np.float32),
                atol=absolute,
                rtol=relative,
            )
        snapshots = [
            algorithms[0].peer_apply(algorithms[0].validate_peer(bundles[1])),
            algorithms[1].peer_apply(algorithms[1].validate_peer(bundles[0])),
        ]
    finally:
        for algorithm, bundle in zip(algorithms, bundles, strict=True):
            algorithm.release_bundle(bundle)

    for rank, (algorithm, snapshot) in enumerate(
        zip(algorithms, snapshots, strict=True)
    ):
        np.testing.assert_allclose(
            snapshot.weights["weight"],
            np.asarray(nodes[rank]["slow_weights_after"], dtype=np.float32),
            atol=absolute,
            rtol=relative,
        )
        np.testing.assert_allclose(
            algorithm.checkpoint_tensors()["noloco.v1.outer_momentum.weight"],
            np.asarray(nodes[rank]["outer_momentum_after"], dtype=np.float32),
            atol=absolute,
            rtol=relative,
        )


def test_first_pre_local_adopts_checkpoint_loaded_after_construction(
    tmp_path: Path,
) -> None:
    schema = TensorSchema(tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),))
    trainer = HandTraceTrainer(initial=99.0, trained=0.5)
    algorithm = NoLoCoAlgorithm(
        trainer=trainer,
        tensor_schema=schema,
        config=_config(),
        bundle_codec=_bundle_codec(tmp_path, sender="local", schema=schema),
    )
    trainer.load_weights({"weight": np.array([2.0], dtype=np.float32)})

    algorithm.pre_local(0)

    assert np.array_equal(
        algorithm.snapshot().weights["weight"],
        np.array([2.0], dtype=np.float32),
    )


def test_checkpoint_round_trip_resumes_bit_identical_next_step(
    tmp_path: Path,
) -> None:
    schema = TensorSchema(tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),))
    original_trainer = HandTraceTrainer(initial=3.0, trained=1.5)
    peer_trainer = HandTraceTrainer(initial=1.0, trained=0.5)
    original = NoLoCoAlgorithm(
        trainer=original_trainer,
        tensor_schema=schema,
        config=_config(),
        bundle_codec=_bundle_codec(
            tmp_path / "original", sender="original", schema=schema
        ),
    )
    peer = NoLoCoAlgorithm(
        trainer=peer_trainer,
        tensor_schema=schema,
        config=_config(),
        bundle_codec=_bundle_codec(tmp_path / "peer", sender="peer", schema=schema),
    )
    original.pre_local(0)
    peer.pre_local(0)
    original.local_training()
    peer.local_training()
    original_bundle = original.post_local_bundle()
    peer_bundle = peer.post_local_bundle()
    try:
        original.peer_apply(original.validate_peer(peer_bundle))
        peer.peer_apply(peer.validate_peer(original_bundle))
    finally:
        original.release_bundle(original_bundle)
        peer.release_bundle(peer_bundle)

    checkpoint = original.checkpoint_tensors()
    assert checkpoint["noloco.v1.slow_weights.weight"].dtype == np.float32
    assert checkpoint["noloco.v1.outer_momentum.weight"].dtype == np.float32
    assert checkpoint["noloco.v1.error_feedback_residual.weight"].dtype == np.float32
    assert "noloco.v1.trainer.adam.first.weight" in checkpoint
    assert "noloco.v1.trainer.loader_rng" in checkpoint
    assert np.array_equal(
        checkpoint["noloco.v1.completed_outer_steps"],
        np.array([1], dtype=np.int64),
    )

    store = RunStore(tmp_path / "run-store")
    store.initialize(SealedManifest.model_validate(manifest_data()))
    store.persist_commit(
        committed_round=0,
        algorithm_state=checkpoint,
        state_checksum="a" * 64,
        schedule={"round_id": 0, "peer": "peer"},
    )
    archive = RunArchive.open(tmp_path / "run-store")
    assert archive.algorithm_state is not None

    restored = NoLoCoAlgorithm(
        trainer=HandTraceTrainer(initial=99.0, trained=1.5),
        tensor_schema=schema,
        config=_config(),
        bundle_codec=_bundle_codec(
            tmp_path / "restored", sender="restored", schema=schema
        ),
    )
    restored.load_checkpoint_tensors(archive.algorithm_state.load_tensors())
    assert np.array_equal(
        restored.snapshot().weights["weight"],
        original.snapshot().weights["weight"],
    )

    peer.pre_local(1)
    peer.local_training()
    next_peer_bundle = peer.post_local_bundle()
    original.pre_local(1)
    original.local_training()
    original_next_bundle = original.post_local_bundle()
    restored.pre_local(1)
    restored.local_training()
    restored_next_bundle = restored.post_local_bundle()
    try:
        original_next = original.peer_apply(original.validate_peer(next_peer_bundle))
        restored_next = restored.peer_apply(restored.validate_peer(next_peer_bundle))
    finally:
        peer.release_bundle(next_peer_bundle)
        original.release_bundle(original_next_bundle)
        restored.release_bundle(restored_next_bundle)

    assert np.array_equal(
        restored_next.weights["weight"], original_next.weights["weight"]
    )


def test_real_adam_state_restored_from_run_store_is_bit_identical(
    tmp_path: Path,
) -> None:
    schema = TensorSchema(
        tensors=(Tensor(name="weight", dtype="float32", shape=(2, 1)),)
    )
    original = NoLoCoAlgorithm(
        trainer=_adam_trainer(initial=(0.2, -0.1), seed=7),
        tensor_schema=schema,
        config=_config(),
        bundle_codec=_bundle_codec(
            tmp_path / "adam-original", sender="local", schema=schema
        ),
    )
    peer = NoLoCoAlgorithm(
        trainer=_adam_trainer(initial=(-0.3, 0.4), seed=11),
        tensor_schema=schema,
        config=_config(),
        bundle_codec=_bundle_codec(
            tmp_path / "adam-peer", sender="peer", schema=schema
        ),
    )
    original.pre_local(0)
    peer.pre_local(0)
    original.local_training()
    peer.local_training()
    original_bundle = original.post_local_bundle()
    peer_bundle = peer.post_local_bundle()
    try:
        original.peer_apply(original.validate_peer(peer_bundle))
        peer.peer_apply(peer.validate_peer(original_bundle))
    finally:
        original.release_bundle(original_bundle)
        peer.release_bundle(peer_bundle)

    store = RunStore(tmp_path / "adam-run-store")
    store.initialize(SealedManifest.model_validate(manifest_data()))
    store.persist_commit(
        committed_round=0,
        algorithm_state=original.checkpoint_tensors(),
        state_checksum="b" * 64,
        schedule={"round_id": 0, "peer": "peer"},
    )
    archive = RunArchive.open(tmp_path / "adam-run-store")
    assert archive.algorithm_state is not None
    restored = NoLoCoAlgorithm(
        trainer=_adam_trainer(initial=(9.0, -9.0), seed=999),
        tensor_schema=schema,
        config=_config(),
        bundle_codec=_bundle_codec(
            tmp_path / "adam-restored", sender="local", schema=schema
        ),
    )
    restored.load_checkpoint_tensors(archive.algorithm_state.load_tensors())

    peer.pre_local(1)
    peer.local_training()
    next_peer_bundle = peer.post_local_bundle()
    original.pre_local(1)
    original.local_training()
    original_bundle = original.post_local_bundle()
    restored.pre_local(1)
    restored.local_training()
    restored_bundle = restored.post_local_bundle()
    try:
        original_local = original.validate_peer(original_bundle)
        restored_local = restored.validate_peer(restored_bundle)
        assert all(
            np.array_equal(
                original_local.artifacts[artifact][name],
                restored_local.artifacts[artifact][name],
            )
            for artifact in ("outer_gradient", "slow_weights")
            for name in original_local.artifacts[artifact]
        )
        original_next = original.peer_apply(original.validate_peer(next_peer_bundle))
        restored_next = restored.peer_apply(restored.validate_peer(next_peer_bundle))
    finally:
        peer.release_bundle(next_peer_bundle)
        original.release_bundle(original_bundle)
        restored.release_bundle(restored_bundle)

    assert all(
        np.array_equal(value, restored_next.weights[name])
        for name, value in original_next.weights.items()
    )


@pytest.mark.parametrize("invalid_kind", ["missing", "non-finite"])
def test_outer_step_rejects_invalid_named_artifacts_before_mutation(
    tmp_path: Path,
    invalid_kind: str,
) -> None:
    schema = TensorSchema(tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),))
    algorithm = NoLoCoAlgorithm(
        trainer=HandTraceTrainer(initial=1.0, trained=0.5),
        tensor_schema=schema,
        config=_config(),
        bundle_codec=_bundle_codec(tmp_path, sender="local", schema=schema),
    )
    algorithm.pre_local(0)
    algorithm.local_training()
    bundle = algorithm.post_local_bundle()
    before = algorithm.snapshot()
    artifacts = {
        "outer_gradient": {"weight": np.array([0.5], dtype=np.float32)},
        "slow_weights": {"weight": np.array([1.0], dtype=np.float32)},
    }
    if invalid_kind == "missing":
        del artifacts["slow_weights"]
    else:
        artifacts["outer_gradient"]["weight"][0] = np.nan
    update = NamedValidatedUpdate(
        round_id=0,
        artifacts=artifacts,
        checksum=checksum_artifacts(artifacts),
    )

    try:
        with pytest.raises(ValueError, match="artifacts|non-finite"):
            algorithm.peer_apply(update)
    finally:
        algorithm.release_bundle(bundle)

    assert np.array_equal(
        algorithm.snapshot().weights["weight"], before.weights["weight"]
    )
