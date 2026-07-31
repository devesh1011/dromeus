from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dromeus.algorithms.codec import SafetensorsUpdateBundleCodec
from dromeus.algorithms.dpsgd import DPSGDAdapter
from dromeus.manifests.models import Tensor, TensorSchema


class CountingTrainer:
    def __init__(self, weights: np.ndarray) -> None:
        self.train_calls: list[int] = []
        self.optimizer_steps = 0
        self._weights = {
            "weight": weights.copy(),
        }

    def train_local_steps(self, step_count: int) -> None:
        self.train_calls.append(step_count)
        for _ in range(step_count):
            self._weights["weight"] += np.float32(1.0)
            self.optimizer_steps += 1

    def weights(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self._weights.items()}

    def load_weights(self, weights: dict[str, np.ndarray]) -> None:
        self._weights = {name: value.copy() for name, value in weights.items()}


def _bundle_codec(
    root: Path, *, sender: str, schema: TensorSchema
) -> SafetensorsUpdateBundleCodec:
    return SafetensorsUpdateBundleCodec(
        artifact_root=root,
        run_id="test-run",
        manifest_hash="0" * 64,
        sender_public_key=sender,
        algorithm_id="d-psgd",
        tensor_schema=schema,
    )


def test_two_nodes_run_production_local_sgd_lifecycle(tmp_path: Path) -> None:
    schema = TensorSchema(
        tensors=(Tensor(name="weight", dtype="float32", shape=(2,)),)
    )
    first_trainer = CountingTrainer(np.array([1.0, 3.0], dtype=np.float32))
    second_trainer = CountingTrainer(np.array([5.0, 9.0], dtype=np.float32))
    first = DPSGDAdapter(
        trainer=first_trainer,
        tensor_schema=schema,
        local_steps=3,
        bundle_codec=_bundle_codec(
            tmp_path / "first", sender="first", schema=schema
        ),
    )
    second = DPSGDAdapter(
        trainer=second_trainer,
        tensor_schema=schema,
        local_steps=3,
        bundle_codec=_bundle_codec(
            tmp_path / "second", sender="second", schema=schema
        ),
    )

    assert first.pre_local(round_id=0) is None
    assert second.pre_local(round_id=0) is None
    assert first.local_training() is None
    assert second.local_training() is None
    first_bundle = first.post_local_bundle()
    second_bundle = second.post_local_bundle()
    try:
        first_update = first.validate_peer(second_bundle)
        second_update = second.validate_peer(first_bundle)
        first_post_mix = first.peer_apply(first_update)
        second_post_mix = second.peer_apply(second_update)
    finally:
        first.release_bundle(first_bundle)
        second.release_bundle(second_bundle)

    expected = np.array([6.0, 9.0], dtype=np.float32)
    assert first_trainer.train_calls == [3]
    assert second_trainer.train_calls == [3]
    assert first_trainer.optimizer_steps == 3
    assert second_trainer.optimizer_steps == 3
    assert np.array_equal(first_post_mix.weights["weight"], expected)
    assert np.array_equal(second_post_mix.weights["weight"], expected)
    assert not any(tmp_path.rglob("*.safetensors"))


def test_dpsgd_rejects_invalid_peer_weights(tmp_path: Path) -> None:
    trainer = CountingTrainer(np.array([1.0, 3.0], dtype=np.float32))
    schema = TensorSchema(
        tensors=(Tensor(name="weight", dtype="float32", shape=(2,)),)
    )
    local_codec = _bundle_codec(tmp_path / "local", sender="local", schema=schema)
    peer_codec = _bundle_codec(tmp_path / "peer", sender="peer", schema=schema)
    algorithm = DPSGDAdapter(
        trainer=trainer,
        tensor_schema=schema,
        local_steps=1,
        bundle_codec=local_codec,
    )
    algorithm.pre_local(round_id=0)
    algorithm.local_training()
    local_bundle = algorithm.post_local_bundle()
    peer_weights = {"weight": np.array([np.nan, 1.0], dtype=np.float32)}
    with pytest.raises(ValueError, match="non-finite"):
        peer_codec.encode(round_id=0, tensors=peer_weights)
    local_codec.release(local_bundle)
