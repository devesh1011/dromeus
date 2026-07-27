from __future__ import annotations

import numpy as np
import pytest

from dromeus.algorithms.base import TrainedWeightsBundle
from dromeus.algorithms.dpsgd import DPSGDAdapter, checksum_tensors
from dromeus.manifests.models import Tensor, TensorSchema


class FakeTrainer:
    def __init__(self) -> None:
        self.local_steps: list[int] = []
        self._weights = {
            "weight": np.array([1.0, 3.0], dtype=np.float32),
        }

    def train_local_steps(self, step_count: int) -> None:
        self.local_steps.append(step_count)
        self._weights["weight"] = self._weights["weight"] + np.array(
            [1.0, 1.0], dtype=np.float32
        )

    def weights(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self._weights.items()}

    def load_weights(self, weights: dict[str, np.ndarray]) -> None:
        self._weights = {name: value.copy() for name, value in weights.items()}


class QuadraticTrainer:
    def __init__(self, value: float) -> None:
        self._weights = {"weight": np.array([value], dtype=np.float32)}

    def train_local_steps(self, step_count: int) -> None:
        for _ in range(step_count):
            gradient = self.stochastic_gradients()["weight"]
            self.load_weights(
                {"weight": self._weights["weight"] - 0.1 * gradient}
            )

    def stochastic_gradients(self) -> dict[str, np.ndarray]:
        return self.weights()

    def weights(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self._weights.items()}

    def load_weights(self, weights: dict[str, np.ndarray]) -> None:
        self._weights = {name: value.copy() for name, value in weights.items()}


class OptimizerOwningTrainer(QuadraticTrainer):
    def __init__(self) -> None:
        super().__init__(1.0)
        self.optimizer_calls: list[int] = []

    def train_local_steps(self, step_count: int) -> None:
        self.optimizer_calls.append(step_count)
        self._weights["weight"] -= np.float32(0.25)

    def stochastic_gradients(self) -> dict[str, np.ndarray]:
        raise AssertionError("adapter bypassed trainer-owned optimizer")


def test_dpsgd_step_matches_paper_weighted_average_then_gradient_update() -> None:
    trainer = QuadraticTrainer(1.0)
    schema = TensorSchema(
        tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),)
    )
    algorithm = DPSGDAdapter(
        trainer=trainer,
        tensor_schema=schema,
        local_steps=1,
        learning_rate=0.1,
    )
    peer_weights = {"weight": np.array([3.0], dtype=np.float32)}
    peer_bundle = TrainedWeightsBundle(
        round_id=0,
        tensors=peer_weights,
        checksum=checksum_tensors(peer_weights),
    )

    algorithm.pre_local(round_id=0)
    algorithm.step(peer_bundles=(peer_bundle,), peer_weights=(0.5,))

    assert np.allclose(algorithm.snapshot().weights["weight"], np.array([1.9]))


def test_dpsgd_local_training_runs_k_stochastic_gradient_steps() -> None:
    trainer = QuadraticTrainer(1.0)
    schema = TensorSchema(
        tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),)
    )
    algorithm = DPSGDAdapter(
        trainer=trainer,
        tensor_schema=schema,
        local_steps=2,
        learning_rate=0.1,
    )

    algorithm.pre_local(round_id=0)
    algorithm.local_training()

    assert np.allclose(algorithm.snapshot().weights["weight"], np.array([0.81]))


def test_dpsgd_local_training_never_bypasses_trainer_optimizer() -> None:
    trainer = OptimizerOwningTrainer()
    schema = TensorSchema(
        tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),)
    )
    algorithm = DPSGDAdapter(
        trainer=trainer,
        tensor_schema=schema,
        local_steps=7,
        learning_rate=0.1,
    )

    algorithm.pre_local(round_id=0)
    algorithm.local_training()

    assert trainer.optimizer_calls == [7]
    assert np.array_equal(
        algorithm.snapshot().weights["weight"],
        np.array([0.75], dtype=np.float32),
    )


def test_dpsgd_runs_local_steps_and_averages_peer_weights() -> None:
    trainer = FakeTrainer()
    schema = TensorSchema(
        tensors=(Tensor(name="weight", dtype="float32", shape=(2,)),)
    )
    algorithm = DPSGDAdapter(trainer=trainer, tensor_schema=schema, local_steps=5)

    algorithm.pre_local(round_id=0)
    algorithm.local_training()
    local_bundle = algorithm.post_local_bundle()
    peer_weights = {"weight": np.array([6.0, 10.0], dtype=np.float32)}
    peer_bundle = TrainedWeightsBundle(
        round_id=local_bundle.round_id,
        tensors=peer_weights,
        checksum=checksum_tensors(peer_weights),
    )
    algorithm.peer_apply(peer_bundle)

    assert trainer.local_steps == [5]
    assert np.array_equal(
        algorithm.snapshot().weights["weight"],
        np.array([4.0, 7.0], dtype=np.float32),
    )


def test_dpsgd_rejects_invalid_peer_weights() -> None:
    trainer = FakeTrainer()
    schema = TensorSchema(
        tensors=(Tensor(name="weight", dtype="float32", shape=(2,)),)
    )
    algorithm = DPSGDAdapter(trainer=trainer, tensor_schema=schema, local_steps=1)
    algorithm.pre_local(round_id=0)
    algorithm.local_training()
    local_bundle = algorithm.post_local_bundle()
    peer_weights = {"weight": np.array([np.nan, 1.0], dtype=np.float32)}
    bundle = TrainedWeightsBundle(
        round_id=local_bundle.round_id,
        tensors=peer_weights,
        checksum=checksum_tensors(peer_weights),
    )

    with pytest.raises(ValueError, match="non-finite"):
        algorithm.peer_apply(bundle)
