"""Opt-in B1 backend admission, real private updates and restore isolation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from dromeus.application import describe_task
from dromeus.manifests.canonical import canonical_hash
from dromeus.manifests.models import PrivacyPolicy, PrivateOptimizerSpec
from dromeus.persistence.privacy_ledger import PrivacyLedger
from dromeus.training.private_trainer import PrivateTrainer
from dromeus.training.task import TensorBatchAdapter, TrainingTask


class PublicCancellationRecords(Dataset[torch.Tensor]):
    def __init__(self) -> None:
        self.reads = 0

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> torch.Tensor:
        if not 0 <= index < 2:
            raise IndexError(index)
        self.reads += 1
        return torch.tensor([1000.0, 0.001, -1000.0, 0.001] * 256).reshape(1024, 1)


def public_task() -> tuple[TrainingTask, PublicCancellationRecords]:
    records = PublicCancellationRecords()

    def model() -> nn.Module:
        with cast(Any, torch.random).fork_rng(devices=[]):
            value = nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                value.weight.fill_(0.125)
            return value

    def loss(model: nn.Module, batch: Any) -> torch.Tensor:
        return cast(torch.Tensor, model(batch)).mean(dim=(1, 2))

    return TrainingTask(
        model_factory=model,
        dataset=records,
        batch_adapter=TensorBatchAdapter(
            example=torch.ones(2, 1024, 1), definition="public-sequence-fixture-v1"
        ),
        loss_per_record=loss,
        model_definition="public-one-linear-weight-0125",
        task_definition="public-token-reduction",
        loss_definition="mean-output-per-record",
        implementation_hash="1" * 64,
    ), records


def optimizer(name: str = "sgd") -> PrivateOptimizerSpec:
    if name in ("adam", "adamw"):
        return PrivateOptimizerSpec(
            name=cast(Any, name),
            torch_version=torch.__version__,
            parameter_names=("weight",),
            learning_rate=0.01,
            betas=(0.9, 0.999),
            numerical_epsilon=1e-8,
            weight_decay=0,
        )
    return PrivateOptimizerSpec(
        name="sgd",
        torch_version=torch.__version__,
        parameter_names=("weight",),
        learning_rate=0.01,
        momentum=0,
        weight_decay=0,
    )


def policy() -> PrivacyPolicy:
    return PrivacyPolicy(
        max_grad_norm=1,
        noise_multiplier=1,
        delta=1e-6,
        max_logical_steps=2,
        randomness_profile="reproducible_public_benchmark",
    )


def test_default_and_native_specs_are_explicitly_distinct() -> None:
    task, records = public_task()
    original = describe_task(task, batch_size=2, optimizer=optimizer())
    native = describe_task(
        task,
        batch_size=2,
        optimizer=optimizer(),
        gradient_backend="opacus_native_b1",
    )
    assert original.gradient_backend == "opacus_hooks"
    assert native.gradient_backend == "opacus_native_b1"
    assert canonical_hash(original) != canonical_hash(native)
    assert native.model_dump(exclude={"gradient_backend"}) == original.model_dump(
        exclude={"gradient_backend"}
    )
    assert records.reads == 0


@pytest.mark.parametrize("name", ["sgd", "adam", "adamw"])
def test_owned_private_update_handles_public_token_cancellation(
    tmp_path: Path, name: str
) -> None:
    task, records = public_task()
    spec = describe_task(
        task,
        batch_size=2,
        optimizer=optimizer(name),
        gradient_backend="opacus_native_b1",
    )
    ledger = PrivacyLedger.create(tmp_path / "owner.json", lineage="public-native-b1")
    trainer = PrivateTrainer(
        task=task,
        policy=policy(),
        specification=spec,
        run_id="public-native-b1",
        local_data_identity="public-native-b1",
        reservation_store=ledger,
        max_physical_batch_size=1,
        research_seed=17,
    )
    assert records.reads == 0 and len(ledger.reservations()) == 1
    trainer.train_local_steps(1)
    assert trainer.completed_steps == 1 and records.reads == 2
    assert all(np.isfinite(v).all() for v in trainer.weights().values())
    # Independent full-model autograd, explicit clipped-sum noise and an ordinary
    # optimizer; no native hook sampler implementation is reused by this oracle.
    reference = task.model_factory()
    record = torch.tensor([1000.0, 0.001, -1000.0, 0.001] * 256).reshape(1, 1024, 1)
    reference(record).mean().backward()
    parameter = next(reference.parameters())
    assert parameter.grad is not None
    gradient = parameter.grad.detach().clone()
    norm = cast(torch.Tensor, cast(Any, torch.linalg).vector_norm(gradient))
    clipped = gradient * torch.clamp(torch.reciprocal(norm + 1e-6), max=1)
    noise = torch.normal(
        0.0, 1.0, size=parameter.shape, generator=torch.Generator().manual_seed(18)
    )
    parameter.grad = (2 * clipped + noise) / 2
    optimizers = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    reference_optimizer = optimizers[name](
        reference.parameters(), lr=0.01, weight_decay=0, foreach=False
    )
    cast(Any, reference_optimizer).step()
    torch.testing.assert_close(
        torch.tensor(trainer.weights()["weight"]), parameter.detach(), rtol=0, atol=1e-7
    )
    assert trainer.accounting_history == ((1.0, 1.0, 1),)


def test_native_empty_draw_keeps_noise_accounting_and_no_record_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task, records = public_task()
    ledger = PrivacyLedger.create(tmp_path / "owner.json", lineage="public-empty")
    trainer = PrivateTrainer(
        task=task,
        policy=policy(),
        specification=describe_task(
            task,
            batch_size=1,
            optimizer=optimizer(),
            gradient_backend="opacus_native_b1",
        ),
        run_id="public-empty",
        local_data_identity="public-empty",
        reservation_store=ledger,
        max_physical_batch_size=1,
        research_seed=17,
    )
    before = trainer.weights()["weight"].copy()

    def empty_draw(*shape: Any, **kwargs: Any) -> torch.Tensor:
        size = cast(
            tuple[int, ...],
            shape[0]
            if len(shape) == 1 and isinstance(shape[0], (tuple, list))
            else shape,
        )
        return torch.full(size, 0.75, device=kwargs.get("device", "cpu"))

    monkeypatch.setattr(torch, "rand", empty_draw)
    trainer.train_local_steps(1)
    assert records.reads == 0 and trainer.completed_steps == 1
    assert trainer.accounting_history == ((1.0, 0.5, 1),)
    assert not np.array_equal(before, trainer.weights()["weight"])


def test_default_backend_retains_its_strict_coverage_guard(tmp_path: Path) -> None:
    task, records = public_task()
    ledger = PrivacyLedger.create(tmp_path / "owner.json", lineage="public-default")
    trainer = PrivateTrainer(
        task=task,
        policy=policy(),
        specification=describe_task(task, batch_size=2, optimizer=optimizer()),
        run_id="public-default",
        local_data_identity="public-default",
        reservation_store=ledger,
        max_physical_batch_size=1,
        research_seed=17,
    )
    with pytest.raises(ValueError, match="private local training failed"):
        trainer.train_local_steps(1)
    assert records.reads == 1 and trainer.completed_steps == 0
    assert ledger.reservations()[0].logical_steps == 2


def test_native_rejects_larger_physical_cap_before_reservation(tmp_path: Path) -> None:
    task, records = public_task()
    spec = describe_task(
        task,
        batch_size=2,
        optimizer=optimizer(),
        gradient_backend="opacus_native_b1",
    )
    ledger = PrivacyLedger.create(tmp_path / "owner.json", lineage="public-cap")
    with pytest.raises(ValueError, match="physical"):
        PrivateTrainer(
            task=task,
            policy=policy(),
            specification=spec,
            run_id="public-cap",
            local_data_identity="public-cap",
            reservation_store=ledger,
            max_physical_batch_size=2,
            research_seed=17,
        )
    assert records.reads == 0 and ledger.reservations() == ()


def test_sampler_replacement_in_loss_stops_before_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task, records = public_task()
    original_loss = task.loss_per_record
    mutation = {"enabled": False}

    def loss(model: nn.Module, batch: Any) -> torch.Tensor:
        result = original_loss(model, batch)
        if mutation["enabled"]:
            wrapped = cast(Any, model)
            wrapped.GRAD_SAMPLERS = dict(wrapped.GRAD_SAMPLERS)
        return result

    task = replace(task, loss_per_record=loss)
    spec = describe_task(
        task,
        batch_size=2,
        optimizer=optimizer(),
        gradient_backend="opacus_native_b1",
    )
    ledger = PrivacyLedger.create(tmp_path / "owner.json", lineage="public-mutation")
    trainer = PrivateTrainer(
        task=task,
        policy=policy(),
        specification=spec,
        run_id="public-mutation",
        local_data_identity="public-mutation",
        reservation_store=ledger,
        max_physical_batch_size=1,
        research_seed=17,
    )
    owned = cast(Any, trainer)._optimizer
    original_step = owned.step
    calls: list[bool] = []

    def step(*args: Any, **kwargs: Any) -> Any:
        calls.append(True)
        return original_step(*args, **kwargs)

    monkeypatch.setattr(owned, "step", step)
    mutation["enabled"] = True
    with pytest.raises(ValueError, match="private local training failed"):
        trainer.train_local_steps(1)
    assert (
        calls == []
        and trainer.completed_steps == 0
        and trainer.accounting_history == ()
    )
    assert records.reads == 1 and ledger.reservations()[0].logical_steps == 2


def test_research_restore_cannot_cross_gradient_backends(tmp_path: Path) -> None:
    task, records = public_task()
    # Independent hypothetical authorities, with identical bytes. No training
    # occurs: every identity input except the backend is held equal.
    paths = [tmp_path / f"owner-{index}.json" for index in range(2)]
    ledgers = [PrivacyLedger.create(p, lineage="public-restore") for p in paths]
    assert paths[0].read_bytes() == paths[1].read_bytes()
    original = describe_task(task, batch_size=2, optimizer=optimizer())
    native = describe_task(
        task,
        batch_size=2,
        optimizer=optimizer(),
        gradient_backend="opacus_native_b1",
    )
    trainers = [
        PrivateTrainer(
            task=task,
            policy=policy(),
            specification=spec,
            run_id="public-restore",
            local_data_identity="public-restore",
            reservation_store=ledgers[index],
            max_physical_batch_size=1,
            research_seed=17,
        )
        for index, spec in enumerate((original, native))
    ]
    assert paths[0].read_bytes() == paths[1].read_bytes()
    before = [t.checkpoint_tensors() for t in trainers]
    assert not np.array_equal(before[0]["meta.identity"], before[1]["meta.identity"])
    for index in range(2):
        with pytest.raises(ValueError):
            trainers[index].load_checkpoint_tensors(before[1 - index])
        after = trainers[index].checkpoint_tensors()
        assert before[index].keys() == after.keys()
        assert all(np.array_equal(before[index][k], after[k]) for k in after)
    assert records.reads == 0
