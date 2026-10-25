"""Legacy tensor tasks retain their loss, sampling and local-state semantics."""

from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dromeus.manifests.canonical import load_safetensors
from dromeus.manifests.models import (
    PrivacyPolicy,
    PrivateOptimizerSpec,
    PrivateTrainingSpec,
)
from dromeus.persistence.privacy_ledger import PrivacyLedger
from dromeus.training.privacy import opacus_module
from dromeus.training.private_trainer import PrivateTrainer

pytest.importorskip("opacus")

type Loss = Literal["cross_entropy", "mse"]


def public_model(loss: Loss) -> nn.Module:
    model = nn.Linear(2, 3) if loss == "cross_entropy" else nn.Conv1d(2, 3, 1)
    assert model.bias is not None
    with torch.no_grad():
        model.weight.copy_(
            torch.arange(6, dtype=torch.float32).reshape_as(model.weight) / 10
        )
        model.bias.fill_(0.1)
    return model


def public_data(loss: Loss) -> tuple[torch.Tensor, torch.Tensor]:
    if loss == "cross_entropy":
        return (
            torch.arange(16, dtype=torch.float32).reshape(8, 2) / 10,
            torch.tensor([0, 2, 1, 0, 1, 2, 0, 1]),
        )
    return (
        torch.arange(48, dtype=torch.float32).reshape(8, 2, 3) / 10,
        torch.arange(72, dtype=torch.float32).reshape(8, 3, 3) / 20,
    )


def prepare(
    root: Path,
    loss: Loss,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    physical: int = 2,
) -> PrivateTrainer:
    return PrivateTrainer(
        model_factory=lambda: public_model(loss),
        inputs=inputs,
        targets=targets,
        policy=PrivacyPolicy(
            max_grad_norm=0.7,
            noise_multiplier=1,
            delta=1e-5,
            max_logical_steps=3,
            randomness_profile="reproducible_public_benchmark",
        ),
        specification=PrivateTrainingSpec(
            optimizer=PrivateOptimizerSpec(
                name="sgd",
                torch_version=torch.__version__,
                parameter_names=("weight", "bias"),
                learning_rate=0.03,
                momentum=0.7,
                weight_decay=0.03,
            ),
            loss=loss,
            batch_size=4,
        ),
        run_id="legacy-task-characterization",
        local_data_identity="public-legacy-task-v1",
        reservation_store=PrivacyLedger.create(root / "ledger.json", lineage="public"),
        max_physical_batch_size=physical,
        research_seed=31,
    )


@pytest.mark.parametrize("loss", ["cross_entropy", "mse"])
@pytest.mark.parametrize("physical", [1, 4])
def test_legacy_loss_trajectory_matches_direct_opacus(
    tmp_path: Path, loss: Loss, physical: int
) -> None:
    inputs, targets = public_data(loss)
    trainer = prepare(tmp_path, loss, inputs, targets, physical=physical)
    model = public_model(loss)
    base = torch.optim.SGD(
        model.parameters(),
        lr=0.03,
        momentum=0.7,
        weight_decay=0.03,
        foreach=False,
        fused=False,
    )
    loader = DataLoader(
        TensorDataset(inputs, targets),
        batch_size=4,
        generator=torch.Generator().manual_seed(31),
    )
    engine = opacus_module().PrivacyEngine(accountant="prv", secure_mode=False)
    wrapped, optimizer, private_loader = engine.make_private(
        module=model,
        optimizer=base,
        data_loader=loader,
        noise_multiplier=1,
        max_grad_norm=0.7,
        noise_generator=torch.Generator().manual_seed(32),
    )
    manager = opacus_module("opacus.utils.batch_memory_manager").BatchMemoryManager
    for logical_step in range(1, 4):
        with manager(
            data_loader=private_loader,
            max_physical_batch_size=physical,
            optimizer=optimizer,
        ) as batches:
            for batch_inputs, batch_targets in batches:
                optimizer.zero_grad()
                prediction = wrapped(batch_inputs)
                if len(batch_inputs) == 0:
                    objective = prediction.sum()
                elif loss == "cross_entropy":
                    objective = nn.functional.cross_entropy(prediction, batch_targets)
                else:
                    objective = nn.functional.mse_loss(prediction, batch_targets)
                cast(Any, objective).backward()
                optimizer.step()
                if sum(n for _, _, n in engine.accountant.history) == logical_step:
                    optimizer.zero_grad()
                    break
        trainer.train_local_steps(1)
        for name, expected in model.state_dict().items():
            np.testing.assert_allclose(
                trainer.weights()[name], expected.detach().numpy(), rtol=1e-6, atol=1e-7
            )
        assert trainer.accounting_history == tuple(engine.accountant.history)
        assert trainer.completed_steps == logical_step
        assert trainer.local_loss is None


@pytest.mark.parametrize("loss", ["cross_entropy", "mse"])
def test_legacy_data_copy_survives_continuation_and_model_export(
    tmp_path: Path, loss: Loss
) -> None:
    inputs, targets = public_data(loss)
    clean_inputs, clean_targets = inputs.clone(), targets.clone()
    trainer = prepare(tmp_path / "mutated", loss, inputs, targets)
    control = prepare(tmp_path / "control", loss, clean_inputs, clean_targets)
    inputs.fill_(1000)
    targets.fill_(999)
    trainer.train_local_steps(1)
    checkpoint = trainer.checkpoint_tensors()
    restored = prepare(tmp_path / "restored", loss, clean_inputs, clean_targets)
    restored.load_checkpoint_tensors(checkpoint)
    trainer.train_local_steps(2)
    restored.train_local_steps(2)
    control.train_local_steps(3)
    for key, expected in control.checkpoint_tensors().items():
        np.testing.assert_array_equal(trainer.checkpoint_tensors()[key], expected)
        np.testing.assert_array_equal(restored.checkpoint_tensors()[key], expected)
    export = tmp_path / "model.safetensors"
    restored.export_model(export)
    tensors = load_safetensors(str(export))
    assert set(tensors) == {"weight", "bias"}
    for key, expected in control.weights().items():
        np.testing.assert_array_equal(tensors[key], expected)
