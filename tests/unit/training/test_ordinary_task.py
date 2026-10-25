"""Ordinary tasks share record semantics without requiring the privacy extra."""

import hashlib
import importlib
from dataclasses import replace
from typing import Any, Literal

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from dromeus.manifests.models import PrivateOptimizerSpec, TaskTrainingSpec
from dromeus.training.ordinary_task import OrdinaryTaskTrainer
from dromeus.training.private_model import describe_private_model
from dromeus.training.private_state import decode_state, encode_state
from dromeus.training.task import TensorBatchAdapter, TrainingTask


def model() -> nn.Module:
    linear = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        linear.weight.fill_(1)
    return linear


def losses(module: nn.Module, batch: Any) -> torch.Tensor:
    x, y = batch
    return (module(x) - y).square().flatten(1).mean(1)


def make_task() -> TrainingTask:
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.zeros_like(x)
    return TrainingTask(
        model_factory=model,
        dataset=TensorDataset(x, y),
        batch_adapter=TensorBatchAdapter((x[:2], y[:2]), "scalar regression"),
        loss_per_record=losses,
        model_definition="scalar linear without bias; initial weight one",
        task_definition="squared prediction error",
        loss_definition="one mean squared error per record",
        implementation_hash="a" * 64,
    )


def specification(
    task: TrainingTask,
    sampling: Literal["shuffle", "poisson"] = "shuffle",
    *,
    batch_size: int = 4,
    momentum: float = 0,
    optimizer_name: Literal["sgd", "adam", "adamw"] = "sgd",
    weight_decay: float = 0,
) -> TaskTrainingSpec:
    return TaskTrainingSpec(
        optimizer=PrivateOptimizerSpec(
            name=optimizer_name,
            torch_version=torch.__version__,
            parameter_names=("weight",),
            learning_rate=0.01,
            momentum=momentum,
            weight_decay=weight_decay,
        ),
        batch_size=batch_size,
        sampling=sampling,
        task_definition_hash=hashlib.sha256(task.task_definition.encode()).hexdigest(),
        implementation_hash=task.implementation_hash,
        batch_definition_hash=task.batch_adapter.definition_hash,
        loss_definition_hash=hashlib.sha256(task.loss_definition.encode()).hexdigest(),
        record_unit=task.record_unit,
        trainable_parameter_names=("weight",),
    )


def trainer(
    *,
    sampling: Literal["shuffle", "poisson"] = "shuffle",
    physical: int = 2,
    batch_size: int = 4,
    momentum: float = 0,
    seed: int = 3,
    decay: bool = False,
    optimizer_name: Literal["sgd", "adam", "adamw"] = "sgd",
    weight_decay: float = 0,
) -> OrdinaryTaskTrainer:
    task = make_task()
    spec = specification(
        task,
        sampling,
        batch_size=batch_size,
        momentum=momentum,
        optimizer_name=optimizer_name,
        weight_decay=weight_decay,
    )
    if decay:
        spec = spec.model_copy(update={"schedule": "linear_decay"})
    return OrdinaryTaskTrainer(
        task=task,
        specification=spec,
        local_data_identity="public-four-records-v1",
        max_logical_steps=8,
        research_seed=seed,
        max_physical_batch_size=physical,
    )


def test_ordinary_task_per_record_objective_matches_analytic_gradient() -> None:
    ordinary = trainer()
    ordinary.train_local_steps(1)
    # At w=1, mean gradient is 2*mean([1,4,9,16])=15; SGD gives .85.
    np.testing.assert_allclose(ordinary.weights()["weight"], [[0.85]], atol=1e-7)
    assert ordinary.completed_steps == 1
    assert ordinary.local_loss == pytest.approx(7.5)


def test_shuffle_execution_does_not_import_opacus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = importlib.import_module

    def import_without_privacy(name: str, package: str | None = None) -> Any:
        if name == "opacus" or name.startswith("opacus."):
            raise ModuleNotFoundError("privacy extra deliberately unavailable")
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", import_without_privacy)
    ordinary = trainer()
    ordinary.train_local_steps(1)
    np.testing.assert_allclose(ordinary.weights()["weight"], [[0.85]], atol=1e-7)


def test_poisson_uses_expected_batch_normalization() -> None:
    pytest.importorskip("opacus")
    ordinary = trainer(sampling="poisson", batch_size=1)
    ordinary.train_local_steps(1)
    # Seed 3 selects records x=2 and x=4. q=1/4 gives expected batch 1.
    # The sum gradient is 2*(4+16)=40, so w becomes 1-.01*40=.6.
    assert ordinary.sample_rate == 0.25
    np.testing.assert_allclose(ordinary.weights()["weight"], [[0.6]], atol=1e-7)


def test_empty_poisson_draw_is_one_zero_gradient_optimizer_update() -> None:
    pytest.importorskip("opacus")
    ordinary = trainer(sampling="poisson", batch_size=1, seed=2, momentum=0.9)
    ordinary.train_local_steps(1)
    np.testing.assert_array_equal(ordinary.weights()["weight"], [[1.0]])
    assert ordinary.completed_steps == ordinary.physical_steps == 1
    assert ordinary.local_loss == 0


@pytest.mark.parametrize("sampling", ["shuffle", "poisson"])
def test_physical_splitting_preserves_the_objective(
    sampling: Literal["shuffle", "poisson"],
) -> None:
    if sampling == "poisson":
        pytest.importorskip("opacus")
    split = trainer(sampling=sampling, physical=1, momentum=0.9)
    whole = trainer(sampling=sampling, physical=4, momentum=0.9)
    split.train_local_steps(4)
    whole.train_local_steps(4)
    np.testing.assert_allclose(
        split.weights()["weight"], whole.weights()["weight"], atol=1e-7
    )


@pytest.mark.parametrize("sampling", ["shuffle", "poisson"])
@pytest.mark.parametrize("optimizer_name", ["sgd", "adam", "adamw"])
@pytest.mark.parametrize("decay", [False, True])
def test_checkpoint_continuation_preserves_sampler_optimizer_and_schedule(
    sampling: Literal["shuffle", "poisson"],
    optimizer_name: Literal["sgd", "adam", "adamw"],
    decay: bool,
) -> None:
    if sampling == "poisson":
        pytest.importorskip("opacus")

    def make_trainer() -> OrdinaryTaskTrainer:
        return trainer(
            sampling=sampling,
            batch_size=3,
            momentum=0.9 if optimizer_name == "sgd" else 0,
            optimizer_name=optimizer_name,
            weight_decay=0.1 if optimizer_name == "adamw" else 0,
            decay=decay,
        )

    uninterrupted = make_trainer()
    uninterrupted.train_local_steps(3)
    checkpoint = uninterrupted.checkpoint_tensors()
    uninterrupted.train_local_steps(4)
    restored = make_trainer()
    restored.load_checkpoint_tensors(checkpoint)
    restored.train_local_steps(4)
    for key, value in uninterrupted.checkpoint_tensors().items():
        np.testing.assert_array_equal(restored.checkpoint_tensors()[key], value)


def test_malformed_checkpoint_rolls_back_without_changing_training() -> None:
    ordinary = trainer(momentum=0.9)
    ordinary.train_local_steps(1)
    before = ordinary.checkpoint_tensors()
    broken = {key: value.copy() for key, value in before.items()}
    broken["meta.steps"][:] = 3
    with pytest.raises(ValueError, match="counters"):
        ordinary.load_checkpoint_tensors(broken)
    for key, value in before.items():
        np.testing.assert_array_equal(ordinary.checkpoint_tensors()[key], value)


@pytest.mark.parametrize("optimizer_name", ["sgd", "adam", "adamw"])
@pytest.mark.parametrize(
    "corruption", ["learning_rate", "tensor_shape", "state_fields"]
)
def test_malformed_optimizer_restore_preserves_state_and_continuation(
    optimizer_name: Literal["sgd", "adam", "adamw"],
    corruption: str,
) -> None:
    def make_trainer() -> OrdinaryTaskTrainer:
        return trainer(
            optimizer_name=optimizer_name,
            momentum=0.9 if optimizer_name == "sgd" else 0,
            weight_decay=0.1 if optimizer_name == "adamw" else 0,
            decay=True,
        )

    ordinary = make_trainer()
    ordinary.train_local_steps(2)
    before = ordinary.checkpoint_tensors()
    payload = decode_state(
        {
            key.removeprefix("application."): value
            for key, value in before.items()
            if key.startswith("application.")
        }
    )
    optimizer_state = payload["optimizer"]
    state_field = "momentum_buffer" if optimizer_name == "sgd" else "exp_avg"
    if corruption == "learning_rate":
        optimizer_state["param_groups"][0]["lr"] = 0.5
    elif corruption == "tensor_shape":
        optimizer_state["state"][0][state_field] = torch.zeros(3)
    else:
        del optimizer_state["state"][0][state_field]
    broken = {
        key: value.copy()
        for key, value in before.items()
        if not key.startswith("application.")
    }
    broken.update(
        {f"application.{key}": value for key, value in encode_state(payload).items()}
    )
    with pytest.raises(ValueError):
        ordinary.load_checkpoint_tensors(broken)
    for key, value in before.items():
        np.testing.assert_array_equal(ordinary.checkpoint_tensors()[key], value)

    control = make_trainer()
    control.load_checkpoint_tensors(before)
    ordinary.train_local_steps(1)
    control.train_local_steps(1)
    for key, value in control.checkpoint_tensors().items():
        np.testing.assert_array_equal(ordinary.checkpoint_tensors()[key], value)


def test_undeclared_evaluation_rejects_before_preparation() -> None:
    def public_evaluate(_: nn.Module) -> dict[str, float]:
        return {"public_loss": 0.0}

    task = replace(make_task(), public_evaluate=public_evaluate)
    with pytest.raises(ValueError, match="evaluation provenance"):
        OrdinaryTaskTrainer(
            task=task,
            specification=specification(task),
            local_data_identity="public-four-records-v1",
            max_logical_steps=2,
            research_seed=3,
            max_physical_batch_size=2,
        )


def test_tied_frozen_and_public_buffer_state_survives_weight_exchange() -> None:
    class TiedModel(nn.Module):
        scale: torch.Tensor

        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, 1))
            self.alias = self.weight
            self.frozen = nn.Parameter(torch.ones(1), requires_grad=False)
            self.register_buffer("scale", torch.ones(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x @ self.weight + x @ self.alias + self.frozen + self.scale

    source = make_task()
    live = TiedModel()
    task = TrainingTask(
        model_factory=lambda: live,
        dataset=source.dataset,
        batch_adapter=source.batch_adapter,
        loss_per_record=losses,
        model_definition="tied two-use scalar with frozen bias and public scale",
        task_definition=source.task_definition,
        loss_definition=source.loss_definition,
        implementation_hash=source.implementation_hash,
    )
    description = describe_private_model(TiedModel())
    spec = specification(task).model_copy(
        update={
            "parameter_aliases": description.parameter_aliases,
            "public_buffer_hashes": dict(description.public_buffer_hashes),
            "frozen_parameter_hashes": dict(description.frozen_parameter_hashes),
        }
    )
    ordinary = OrdinaryTaskTrainer(
        task=task,
        specification=spec,
        local_data_identity="public-tied-task",
        max_logical_steps=3,
        research_seed=3,
        max_physical_batch_size=2,
    )
    assert set(ordinary.weights()) == {"weight"}
    ordinary.load_weights({"weight": np.array([[0.5]], dtype=np.float32)})
    ordinary.train_local_steps(1)
    checkpoint = ordinary.checkpoint_tensors()
    np.testing.assert_array_equal(checkpoint["model.weight"], checkpoint["model.alias"])
    np.testing.assert_array_equal(checkpoint["model.frozen"], [1])
    np.testing.assert_array_equal(checkpoint["model.scale"], [1])
    broken = {key: value.copy() for key, value in checkpoint.items()}
    broken["model.scale"][:] = 2
    with pytest.raises(ValueError, match="immutable"):
        ordinary.load_checkpoint_tensors(broken)
    live.scale.fill_(2)
    with pytest.raises(ValueError, match="public buffers"):
        ordinary.weights()
