"""Accounted model-only initialization fails before getters or reservations."""

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from dromeus.application import describe_task
from dromeus.manifests.canonical import file_sha256, save_safetensors
from dromeus.manifests.models import (
    AccountedInitializationAuthority,
    AccountedInitializationProvenance,
    PrivacyPolicy,
    PrivacyReservation,
    PrivateOptimizerSpec,
)
from dromeus.training.private_trainer import PrivateTrainer
from dromeus.training.task import (
    AccountedInitializationEvidence,
    TensorBatchAdapter,
    TrainingTask,
)


class UnreadDataset(Dataset[torch.Tensor]):
    reads = 0

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> torch.Tensor:
        self.reads += 1
        raise AssertionError("protected getter must not run during preparation")


def model_factory() -> nn.Module:
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.25)
    return model


def fixture(tmp_path: Path) -> tuple[TrainingTask, AccountedInitializationProvenance]:
    path = tmp_path / "parent.safetensors"
    save_safetensors(
        {k: v.detach().numpy() for k, v in model_factory().state_dict().items()},
        str(path),
    )
    provenance = AccountedInitializationProvenance(
        parent_run_id="parent-run",
        parent_model_sha256=file_sha256(path),
        parent_audit_sha256="a" * 64,
        accounting_authorities=(
            AccountedInitializationAuthority(
                lineage_id="protected-lineage",
                ledger_sha256="b" * 64,
                reserved_steps=4096,
                reservation_count=1,
            ),
        ),
    )
    task = TrainingTask(
        model_factory=model_factory,
        dataset=UnreadDataset(),
        batch_adapter=TensorBatchAdapter(
            example=torch.tensor([[1.0, 2.0], [2.0, 3.0]]),
            definition="public-two-feature-v1",
        ),
        loss_per_record=lambda model, batch: model(batch).square().mean(1),
        model_definition="tiny-accounted-model-v1",
        task_definition="tiny-fixed-record-task-v1",
        loss_definition="mean-square-per-record-v1",
        implementation_hash="c" * 64,
        initialization_evidence=AccountedInitializationEvidence(
            provenance=provenance, model_path=path, verify_parent=lambda _: True
        ),
    )
    return task, provenance


def describe(task: TrainingTask, provenance: AccountedInitializationProvenance):
    return describe_task(
        task,
        batch_size=2,
        optimizer=PrivateOptimizerSpec(
            name="adamw",
            torch_version=torch.__version__,
            parameter_names=("weight",),
            learning_rate=1e-4,
        ),
        initialization="accounted_private_pretrained",
        initialization_provenance=provenance,
    )


def test_matching_evidence_constructs_exact_parent_model(tmp_path: Path) -> None:
    task, provenance = fixture(tmp_path)
    specification = describe(task, provenance)
    task.check_definition(specification)
    torch.testing.assert_close(task.model_factory().weight, torch.full((1, 2), 0.25))
    assert specification.initialization_provenance == provenance
    assert cast(UnreadDataset, task.dataset).reads == 0


@pytest.mark.parametrize("public_mode", ["public_untrained", "public_pretrained"])
def test_accounted_evidence_cannot_be_declared_public(
    tmp_path: Path, public_mode: Any
) -> None:
    task, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="cannot be declared public"):
        describe_task(task, batch_size=2, initialization=public_mode)


def test_missing_or_different_evidence_fails_before_factory(tmp_path: Path) -> None:
    task, provenance = fixture(tmp_path)
    missing = replace(task, initialization_evidence=None)
    with pytest.raises(ValueError, match="matching locally verified"):
        describe(missing, provenance)
    other = provenance.model_copy(update={"parent_audit_sha256": "d" * 64})
    with pytest.raises(ValueError, match="matching locally verified"):
        describe(task, other)


def test_every_factory_call_rechecks_actual_values(tmp_path: Path) -> None:
    task, provenance = fixture(tmp_path)
    describe(task, provenance)

    def changed_factory() -> nn.Module:
        model = model_factory()
        with torch.no_grad():
            cast(nn.Linear, model).weight.add_(1)
        return model

    changed = replace(task, model_factory=changed_factory)
    with pytest.raises(ValueError, match="model tensors differ"):
        changed.model_factory()
    assert cast(UnreadDataset, task.dataset).reads == 0


@pytest.mark.parametrize("kind", ["values", "inventory", "dtype"])
def test_factory_cannot_substitute_same_schema_or_inventory(
    tmp_path: Path, kind: str
) -> None:
    task, _ = fixture(tmp_path)

    def changed_factory() -> nn.Module:
        model = model_factory()
        if kind == "values":
            with torch.no_grad():
                cast(nn.Linear, model).weight.fill_(float("inf"))
        elif kind == "inventory":
            model.register_buffer("extra", torch.ones(1))
        else:
            model.double()
        return model

    with pytest.raises(
        ValueError, match="initialization (model tensors|tensor inventory)"
    ):
        replace(task, model_factory=changed_factory).model_factory()


def test_changed_parent_file_rejected(tmp_path: Path) -> None:
    task, _ = fixture(tmp_path)
    assert task.initialization_evidence is not None
    task.initialization_evidence.model_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="model source differs"):
        task.model_factory()


def test_source_mutation_inside_factory_rejected(tmp_path: Path) -> None:
    task, _ = fixture(tmp_path)
    assert task.initialization_evidence is not None
    path = task.initialization_evidence.model_path

    def mutating_factory() -> nn.Module:
        model = model_factory()
        path.write_bytes(b"changed by factory")
        return model

    with pytest.raises(ValueError, match="source changed during preparation"):
        replace(task, model_factory=mutating_factory).model_factory()


def test_unverified_ancestry_prevents_construction_and_reservation(
    tmp_path: Path,
) -> None:
    task, provenance = fixture(tmp_path)
    specification = describe(task, provenance)
    assert task.initialization_evidence is not None

    def reject_parent(_: AccountedInitializationProvenance) -> bool:
        return False

    rejected = replace(
        task,
        initialization_evidence=replace(
            task.initialization_evidence, verify_parent=reject_parent
        ),
    )

    class Store:
        calls = 0

        def reserve(self, *args: Any, **kwargs: Any) -> None:
            self.calls += 1
            raise AssertionError("reservation must follow initialization verification")

        def check(self, reservation: PrivacyReservation) -> None:
            raise AssertionError("not prepared")

    store = Store()
    with pytest.raises(ValueError, match="ancestry was not verified"):
        PrivateTrainer(
            task=rejected,
            specification=specification,
            policy=PrivacyPolicy(
                max_grad_norm=1,
                noise_multiplier=0.5,
                delta=1e-6,
                max_logical_steps=4,
                randomness_profile="reproducible_public_benchmark",
            ),
            run_id="new-run",
            local_data_identity="protected-lineage",
            reservation_store=store,
            max_physical_batch_size=1,
            research_seed=17,
        )
    assert store.calls == cast(UnreadDataset, task.dataset).reads == 0


def test_verified_parent_prepares_fresh_optimizer_without_getters(
    tmp_path: Path,
) -> None:
    task, provenance = fixture(tmp_path)
    specification = describe(task, provenance)

    class Store:
        reserved: list[PrivacyReservation] = []

        def reserve(self, reservation: PrivacyReservation, **kwargs: Any) -> None:
            self.reserved.append(reservation)

        def check(self, reservation: PrivacyReservation) -> None:
            assert reservation in self.reserved

    store = Store()
    trainer = PrivateTrainer(
        task=task,
        specification=specification,
        policy=PrivacyPolicy(
            max_grad_norm=1,
            noise_multiplier=0.5,
            delta=1e-6,
            max_logical_steps=4,
            randomness_profile="reproducible_public_benchmark",
        ),
        run_id="new-linked-run",
        local_data_identity="protected-lineage",
        reservation_store=store,
        max_physical_batch_size=1,
        research_seed=19,
    )
    assert trainer.completed_steps == 0
    assert trainer.accounting_history == ()
    state = trainer.checkpoint_tensors()
    assert int(state["meta.steps"][0]) == 0
    assert store.reserved[0].logical_steps == 4
    assert cast(UnreadDataset, task.dataset).reads == 0


def test_registered_alias_frozen_and_buffer_values_are_all_checked(
    tmp_path: Path,
) -> None:
    task, provenance = fixture(tmp_path)

    class Shared(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.left = nn.Linear(2, 2, bias=False)
            self.right = nn.Linear(2, 2, bias=False)
            self.right.weight = self.left.weight
            self.frozen = nn.Parameter(torch.ones(2), requires_grad=False)
            self.register_buffer("constant", torch.ones(2))
            with torch.no_grad():
                self.left.weight.fill_(0.25)

    path = tmp_path / "shared.safetensors"
    save_safetensors(
        {k: v.detach().numpy().copy() for k, v in Shared().state_dict().items()},
        str(path),
    )
    provenance = provenance.model_copy(
        update={"parent_model_sha256": file_sha256(path)}
    )
    evidence = AccountedInitializationEvidence(provenance, path, lambda _: True)
    shared = replace(task, model_factory=Shared, initialization_evidence=evidence)
    model = cast(Shared, shared.model_factory())
    assert model.left.weight is model.right.weight
    for name in ["constant", "frozen"]:

        def changed() -> nn.Module:
            model = Shared()
            with torch.no_grad():
                cast(torch.Tensor, getattr(model, name)).add_(1)
            return model

        with pytest.raises(ValueError, match="model tensors differ"):
            replace(shared, model_factory=changed).model_factory()
