"""Fail-closed private task composition before formation or reservation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from support.in_memory_transport import InMemoryNetwork, InMemoryTransport
from torch import nn
from torch.utils.data import TensorDataset

from dromeus.application import (
    LocalData,
    Privacy,
    draft_for_task,
    prepare_task,
)
from dromeus.manifests.models import DraftRunSpec, PrivateOptimizerSpec
from dromeus.persistence.privacy_ledger import PrivacyLedger
from dromeus.runtime import NodeRuntime
from dromeus.training import private_trainer as private_trainer_module
from dromeus.training.task import Optimizer, TensorBatchAdapter, TrainingTask

pytest.importorskip("opacus")


def linear() -> nn.Module:
    return nn.Linear(2, 1)


def squared_error(model: nn.Module, batch: Any) -> torch.Tensor:
    inputs, targets = batch
    return (model(inputs) - targets).square().flatten(1).mean(1)


def records() -> TensorDataset:
    return TensorDataset(torch.ones(8, 2), torch.zeros(8, 1))


def task_for(model_factory: Callable[[], nn.Module] = linear) -> TrainingTask:
    inputs = torch.ones(8, 2)
    targets = torch.zeros(8, 1)
    return TrainingTask(
        model_factory=model_factory,
        batch_adapter=TensorBatchAdapter(
            example=(inputs[:2], targets[:2]),
            definition="public-two-record-pairs-v1",
        ),
        loss_per_record=squared_error,
        model_definition="public-linear-v1",
        task_definition="public-synthetic-mse-v1",
        loss_definition="mean-squared-error-per-record-v1",
        implementation_hash="c" * 64,
        optimizer=Optimizer.adam(0.01),
        batch_size=4,
        record_unit="example",
    )


def draft_for(task: TrainingTask) -> DraftRunSpec:
    return draft_for_task(
        task,
        run_id="run-001",
        participants=4,
        rounds=3,
        local_steps=3,
        axl_version="1.0.0",
        privacy=Privacy(
            max_grad_norm=1,
            noise_multiplier=1,
            delta=1e-5,
            randomness="reproducible_public_benchmark",
        ),
        seed=8,
    )


def test_unsupported_private_model_rejected_before_reservation(tmp_path: Path) -> None:
    def with_batch_norm() -> nn.Module:
        return nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2), nn.Linear(2, 1))

    task = task_for(with_batch_norm)
    draft = draft_for(task)
    ledger = PrivacyLedger.create(tmp_path / "ledger.json", lineage="synthetic")
    with pytest.raises(ValueError, match="batch-dependent normalization"):
        prepare_task(
            draft=draft,
            task=task,
            local=LocalData(
                dataset=records(),
                data_identity="synthetic-fixed-v1",
                ledger=ledger,
                physical_batch_size=2,
                research_seed=10,
            ),
        )
    assert ledger.reservations() == ()


def test_retained_optimizer_cannot_change_frozen_learning_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    held: list[torch.optim.Optimizer] = []
    build = private_trainer_module.build_private_optimizer

    def retain(model: nn.Module, spec: PrivateOptimizerSpec) -> torch.optim.Optimizer:
        optimizer = build(model, spec)
        held.append(optimizer)
        return optimizer

    monkeypatch.setattr(private_trainer_module, "build_private_optimizer", retain)
    task = task_for()
    prepared = prepare_task(
        draft=draft_for(task),
        task=task,
        local=LocalData(
            dataset=records(),
            data_identity="synthetic-fixed-v1",
            ledger=PrivacyLedger.create(tmp_path / "ledger.json", lineage="synthetic"),
            physical_batch_size=2,
            research_seed=10,
        ),
    )
    held[0].param_groups[0]["lr"] = 0.2
    with pytest.raises(ValueError, match="learning rate"):
        prepared.create_initial_checkpoint(tmp_path / "initial.safetensors")


def test_private_runtime_refuses_plain_task_before_formation(tmp_path: Path) -> None:
    draft = draft_for(task_for())
    with pytest.raises(ValueError, match="before formation"):
        NodeRuntime(
            transport=InMemoryTransport(network=InMemoryNetwork(), public_key="node-a"),
            draft=draft,
            environment=draft.environment,
            dataset=draft.dataset,
            artifact_root=tmp_path,
        )
