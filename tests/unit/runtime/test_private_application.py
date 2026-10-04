"""Fail-closed private application composition before formation."""

from pathlib import Path

import pytest
import torch
from support.application_fixture import application_draft
from torch import nn

from dromeus.application import definition_hash, prepare_private_application
from dromeus.manifests.models import (
    DraftRunSpec,
    PrivacyPolicy,
    PrivateOptimizerSpec,
    PrivateTrainingSpec,
)
from dromeus.persistence.privacy_ledger import PrivacyLedger
from dromeus.training.private_trainer import private_training_definition

pytest.importorskip("opacus")


def private_draft() -> DraftRunSpec:
    draft = application_draft()
    spec = PrivateTrainingSpec(
        optimizer=PrivateOptimizerSpec(
            name="adam",
            torch_version=torch.__version__,
            parameter_names=("weight", "bias"),
            learning_rate=0.01,
        ),
        loss="mse",
        batch_size=4,
    )
    values = draft.model_dump(mode="python")
    values.update(
        manifest_version=5,
        model_definition_hash=definition_hash("linear-2-1-v1"),
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
    values["dataset"]["definition_hash"] = definition_hash("public-synthetic-mse-v1")
    return DraftRunSpec.model_validate(values)


def test_private_application_prepares_before_any_network_output(tmp_path: Path) -> None:
    prepared = prepare_private_application(
        draft=private_draft(),
        model_factory=lambda: nn.Linear(2, 1),
        inputs=torch.ones(8, 2),
        targets=torch.zeros(8, 1),
        model_definition="linear-2-1-v1",
        task_definition="public-synthetic-mse-v1",
        local_data_identity="synthetic-fixed-v1",
        reservation_store=PrivacyLedger.create(
            tmp_path / "ledger.json", lineage="synthetic"
        ),
        max_physical_batch_size=2,
        research_seed=10,
    )
    prepared.validate_draft(private_draft())
    prepared.create_initial_checkpoint(tmp_path / "initial.safetensors")
    with pytest.raises(ValueError, match="draft"):
        prepared.validate_draft(private_draft().model_copy(update={"run_id": "other"}))
    assert (
        len(PrivacyLedger(tmp_path / "ledger.json", lineage="synthetic").reservations())
        == 1
    )


@pytest.mark.parametrize(
    "bad", ["batchnorm", "rmsprop", "foreach", "omitted", "private_eval"]
)
def test_unsupported_private_profiles_rejected_before_reservation(
    tmp_path: Path, bad: str
) -> None:
    draft = private_draft()

    def model() -> nn.Module:
        return nn.BatchNorm1d(2) if bad == "batchnorm" else nn.Linear(2, 1)

    def optimizer(model: nn.Module) -> torch.optim.Optimizer:
        if bad == "rmsprop":
            return torch.optim.RMSprop(model.parameters(), lr=0.01)
        return torch.optim.Adam(
            list(model.parameters())[:1] if bad == "omitted" else model.parameters(),
            lr=0.01,
            foreach=bad == "foreach",
            fused=False,
            differentiable=False,
            maximize=False,
            capturable=False,
        )

    ledger = PrivacyLedger.create(tmp_path / "ledger.json", lineage="synthetic")
    with pytest.raises(ValueError):
        prepare_private_application(
            draft=draft,
            model_factory=model,
            inputs=torch.ones(8, 2),
            targets=torch.zeros(8, 1),
            model_definition="linear-2-1-v1",
            task_definition="public-synthetic-mse-v1",
            local_data_identity="synthetic-fixed-v1",
            reservation_store=ledger,
            max_physical_batch_size=2,
            research_seed=10,
            optimizer_factory=optimizer,
            public_evaluate=(lambda _: {"loss": 1.0})
            if bad == "private_eval"
            else None,
        )
    assert ledger.reservations() == ()


def test_retained_optimizer_cannot_change_frozen_learning_rate(tmp_path: Path) -> None:
    from dromeus.training.private_trainer import build_private_optimizer

    held: list[torch.optim.Optimizer] = []
    draft = private_draft()
    assert draft.private_training is not None
    optimizer_spec = draft.private_training.optimizer

    def factory(model: nn.Module) -> torch.optim.Optimizer:
        optimizer = build_private_optimizer(model, optimizer_spec)
        held.append(optimizer)
        return optimizer

    prepared = prepare_private_application(
        draft=draft,
        model_factory=lambda: nn.Linear(2, 1),
        inputs=torch.ones(8, 2),
        targets=torch.zeros(8, 1),
        model_definition="linear-2-1-v1",
        task_definition="public-synthetic-mse-v1",
        local_data_identity="synthetic-fixed-v1",
        reservation_store=PrivacyLedger.create(
            tmp_path / "ledger.json", lineage="synthetic"
        ),
        max_physical_batch_size=2,
        research_seed=10,
        optimizer_factory=factory,
    )
    held[0].param_groups[0]["lr"] = 0.2
    with pytest.raises(ValueError, match="learning rate"):
        prepared.create_initial_checkpoint(tmp_path / "initial.safetensors")


def test_private_runtime_refuses_plain_factory_before_formation(tmp_path: Path) -> None:
    from support.in_memory_transport import InMemoryNetwork, InMemoryTransport

    from dromeus.runtime import NodeRuntime

    draft = private_draft()
    with pytest.raises(ValueError, match="before formation"):
        NodeRuntime(
            transport=InMemoryTransport(network=InMemoryNetwork(), public_key="node-a"),
            draft=draft,
            environment=draft.environment,
            dataset=draft.dataset,
            artifact_root=tmp_path,
        )


def test_legacy_v4_restore_identity_matches_accepted_baseline() -> None:
    from dromeus.application import prepare_application
    from dromeus.training.trainer import PyTorchTrainer
    from examples.custom_training import (
        MODEL_DEFINITION,
        TASK_DEFINITION,
        TRAINING_DEFINITION,
    )

    trainer = PyTorchTrainer(
        model=nn.Linear(3, 1),
        train_step=lambda _: None,
        save_state=lambda: {},
        load_state=lambda _: None,
        state_identity="legacy-state-v1",
    )
    prepare_application(
        draft=application_draft(),
        trainer=trainer,
        model_definition=MODEL_DEFINITION,
        task_definition=TASK_DEFINITION,
        training_definition=TRAINING_DEFINITION,
    )
    # Captured with application.py at accepted M2 SHA d949fff40e58.
    assert (
        trainer.checkpoint_tensors()["meta.identity"].tobytes().hex()
        == "b1d7ecd627b36200667f73d1d1f86146b6f0a56233c5b54348fa3918adc16b33"
    )
