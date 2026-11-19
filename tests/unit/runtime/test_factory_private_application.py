"""Factory optimizers through public task preparation and four NoLoCo runtimes."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import pytest
import torch
from support.factory_optimizer import optimizer_registration, public_model
from support.in_memory_transport import InMemoryNetwork, InMemoryTransport
from torch import nn
from torch.utils.data import TensorDataset

from dromeus.application import (
    LocalData,
    Privacy,
    draft_for_task,
    prepare_task,
)
from dromeus.errors import DraftError
from dromeus.manifests.models import DraftRunSpec
from dromeus.membership.formation import FormationResult, create_invitation
from dromeus.persistence.privacy_ledger import PrivacyLedger
from dromeus.runtime import (
    FailureConfig,
    InitiatorFormation,
    NodeRuntime,
    ParticipantFormation,
    TrainingConfig,
)
from dromeus.training.optimizer_factory import OptimizerRegistration
from dromeus.training.task import TensorBatchAdapter, TrainingTask

pytest.importorskip("opacus")


def squared_error(model: nn.Module, batch: Any) -> torch.Tensor:
    inputs, targets = batch
    return (model(inputs) - targets).square().flatten(1).mean(1)


def records(rank: int = 0) -> TensorDataset:
    return TensorDataset(torch.ones(8, 2) * (rank + 1), torch.zeros(8, 1))


def task_for(registration: OptimizerRegistration, rank: int = 0) -> TrainingTask:
    inputs = torch.ones(8, 2) * (rank + 1)
    targets = torch.zeros(8, 1)
    return TrainingTask(
        model_factory=public_model,
        batch_adapter=TensorBatchAdapter(
            example=(inputs[:2], targets[:2]),
            definition="public-two-record-pairs-v1",
        ),
        loss_per_record=squared_error,
        model_definition="public-linear-v1",
        task_definition="public-mse-v1",
        loss_definition="mean-squared-error-per-record-v1",
        implementation_hash="b" * 64,
        optimizer=registration,
        batch_size=4,
        schedule="linear_decay",
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


@pytest.mark.parametrize("kind", ["custom", "rmsprop"])
def test_factory_private_noloco_runs_four_nodes(
    tmp_path: Path, kind: Literal["custom", "rmsprop"]
) -> None:
    registration = optimizer_registration(kind)
    draft = draft_for(task_for(registration))
    prepared = [
        prepare_task(
            draft=draft,
            task=task_for(registration, rank),
            local=LocalData(
                dataset=records(rank),
                data_identity=f"public-node-{rank}",
                ledger=PrivacyLedger.create(
                    tmp_path / f"ledger-{rank}.json", lineage=f"public-{rank}"
                ),
                physical_batch_size=2,
                research_seed=17 + rank,
            ),
        )
        for rank in range(4)
    ]
    initial = prepared[0].create_initial_checkpoint(tmp_path / "initial.safetensors")
    network = InMemoryNetwork()
    nodes = [
        NodeRuntime(
            transport=InMemoryTransport(network=network, public_key=f"peer-{rank}"),
            draft=draft,
            environment=draft.environment,
            dataset=draft.dataset,
            artifact_root=tmp_path / f"formation-{rank}",
            failure=FailureConfig.for_run_root(tmp_path / f"node-{rank}"),
            private_trainer=prepared[rank].private_trainer,
        )
        for rank in range(4)
    ]
    invitation = create_invitation(
        draft=draft, initiator_public_key="peer-0", bootstrap_uri="axl://factory"
    )

    async def run() -> None:
        async def node(rank: int) -> None:
            def training(result: FormationResult) -> TrainingConfig:
                return prepared[rank].build_config(
                    result=result,
                    local_public_key=f"peer-{rank}",
                    run_root=tmp_path / f"node-{rank}",
                )

            outcome = await nodes[rank].run_to_completion(
                formation=InitiatorFormation(
                    bootstrap_uri="axl://factory",
                    checkpoint_path=initial.path,
                    tensor_schema=initial.tensor_schema,
                )
                if rank == 0
                else ParticipantFormation(invitation=invitation),
                training_factory=training,
            )
            assert len(outcome.commits) == 3
            trainer = prepared[rank].private_trainer
            assert trainer is not None and trainer.completed_steps == 9
            assert trainer.accounting_history == ((1.0, 0.5, 9),)
            assert trainer.local_loss is None
            prepared[rank].export_model(tmp_path / f"export-{rank}.safetensors")

        await asyncio.wait_for(asyncio.gather(*(node(i) for i in range(4))), timeout=20)

    asyncio.run(run())


def test_factory_identity_mismatch_fails_before_reservation(tmp_path: Path) -> None:
    registration = optimizer_registration()
    draft = draft_for(task_for(registration))
    ledger = PrivacyLedger.create(tmp_path / "ledger.json", lineage="public")
    altered = replace(registration, implementation_hash="f" * 64)
    with pytest.raises(DraftError, match="optimizer"):
        prepare_task(
            draft=draft,
            task=task_for(altered),
            local=LocalData(
                dataset=records(),
                data_identity="public",
                ledger=ledger,
                physical_batch_size=2,
                research_seed=17,
            ),
        )
    assert ledger.reservations() == ()
