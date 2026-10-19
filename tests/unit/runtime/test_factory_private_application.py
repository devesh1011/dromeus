"""Factory optimizers through public preparation and four NoLoCo runtimes."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest
import torch
from support.application_fixture import application_draft
from support.factory_optimizer import optimizer_registration, public_model
from support.in_memory_transport import InMemoryNetwork, InMemoryTransport

from dromeus.application import definition_hash, prepare_private_application
from dromeus.manifests.models import DraftRunSpec, PrivacyPolicy, PrivateTrainingSpec
from dromeus.membership.formation import FormationResult, create_invitation
from dromeus.persistence.privacy_ledger import PrivacyLedger
from dromeus.runtime import (
    FailureConfig,
    InitiatorFormation,
    NodeRuntime,
    ParticipantFormation,
    TrainingConfig,
)
from dromeus.training.private_trainer import private_training_definition

pytest.importorskip("opacus")


def draft_for(kind: Literal["custom", "rmsprop"] = "custom") -> DraftRunSpec:
    values = application_draft().model_dump(mode="python")
    spec = PrivateTrainingSpec(
        optimizer=optimizer_registration(kind).describe(public_model()),
        loss="mse",
        batch_size=4,
        schedule="linear_decay",
    )
    values.update(
        manifest_version=5,
        model_definition_hash=definition_hash("public-linear-v1"),
        private_training=spec,
        privacy=PrivacyPolicy(
            max_grad_norm=1,
            noise_multiplier=1,
            delta=1e-5,
            max_logical_steps=9,
            randomness_profile="reproducible_public_benchmark",
        ),
    )
    values["environment"]["model_definition_hash"] = values["model_definition_hash"]
    values["application_training"]["definition_hash"] = definition_hash(
        private_training_definition(spec)
    )
    values["dataset"]["definition_hash"] = definition_hash("public-mse-v1")
    return DraftRunSpec.model_validate(values)


@pytest.mark.parametrize("kind", ["custom", "rmsprop"])
def test_factory_private_noloco_runs_four_nodes(
    tmp_path: Path, kind: Literal["custom", "rmsprop"]
) -> None:
    draft = draft_for(kind)
    prepared = [
        prepare_private_application(
            draft=draft,
            model_factory=public_model,
            inputs=torch.ones(8, 2) * (rank + 1),
            targets=torch.zeros(8, 1),
            model_definition="public-linear-v1",
            task_definition="public-mse-v1",
            local_data_identity=f"public-node-{rank}",
            reservation_store=PrivacyLedger.create(
                tmp_path / f"ledger-{rank}.json", lineage=f"public-{rank}"
            ),
            max_physical_batch_size=2,
            research_seed=17 + rank,
            optimizer_registration=optimizer_registration(kind),
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
    ledger = PrivacyLedger.create(tmp_path / "ledger.json", lineage="public")
    with pytest.raises(ValueError, match="implementation"):
        prepare_private_application(
            draft=draft_for(),
            model_factory=public_model,
            inputs=torch.ones(8, 2),
            targets=torch.zeros(8, 1),
            model_definition="public-linear-v1",
            task_definition="public-mse-v1",
            local_data_identity="public",
            reservation_store=ledger,
            max_physical_batch_size=2,
            research_seed=17,
            optimizer_registration=replace(
                optimizer_registration(), implementation_hash="f" * 64
            ),
        )
    assert ledger.reservations() == ()
