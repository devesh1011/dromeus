"""Group-selected private optimizers preserve update rules and safe state."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
import torch
from support.factory_optimizer import (
    CustomOptimizer,
    optimizer_registration,
    public_model,
)
from torch.utils.data import DataLoader, TensorDataset

from dromeus.manifests.models import PrivacyPolicy, PrivateTrainingSpec
from dromeus.persistence.privacy_ledger import PrivacyLedger
from dromeus.training.optimizer_factory import PrivateOptimizerFactory
from dromeus.training.privacy import opacus_module
from dromeus.training.private_state import decode_state, encode_state
from dromeus.training.private_trainer import PrivateTrainer

pytest.importorskip("opacus")
Kind = Literal["custom", "rmsprop", "adagrad"]


def make_trainer(
    path: Path,
    registration: PrivateOptimizerFactory,
    *,
    physical: int = 2,
    specification: PrivateTrainingSpec | None = None,
) -> PrivateTrainer:
    return PrivateTrainer(
        model_factory=public_model,
        inputs=torch.arange(16, dtype=torch.float32).reshape(8, 2) / 8,
        targets=torch.ones(8, 1),
        policy=PrivacyPolicy(
            max_grad_norm=1,
            noise_multiplier=1,
            delta=1e-5,
            max_logical_steps=4,
            randomness_profile="reproducible_public_benchmark",
        ),
        specification=specification
        or PrivateTrainingSpec(
            optimizer=registration.describe(public_model()),
            loss="mse",
            batch_size=4,
            schedule="linear_decay",
        ),
        run_id="factory-conformance",
        local_data_identity="public-synthetic",
        reservation_store=PrivacyLedger.create(path / "ledger.json", lineage="public"),
        max_physical_batch_size=physical,
        research_seed=31,
        optimizer_registration=registration,
    )


def application_state(checkpoint: dict[str, np.ndarray]) -> Any:
    return decode_state(
        {
            key.removeprefix("application."): value
            for key, value in checkpoint.items()
            if key.startswith("application.")
        }
    )


@pytest.mark.parametrize("kind", ["custom", "rmsprop", "adagrad"])
def test_registered_optimizer_matches_direct_opacus(tmp_path: Path, kind: Kind) -> None:
    registration = optimizer_registration(kind)
    trainer = make_trainer(tmp_path, registration)
    model = public_model()
    base = registration.create(model)
    if isinstance(base, CustomOptimizer):
        # Direct Opacus exposes internal attributes; Dromeus deliberately hides them.
        base.check_sanitized = False
    initial_rates = [group["lr"] for group in base.param_groups]
    loader = DataLoader(
        TensorDataset(
            torch.arange(16, dtype=torch.float32).reshape(8, 2) / 8, torch.ones(8, 1)
        ),
        batch_size=4,
        generator=torch.Generator().manual_seed(31),
    )
    engine = opacus_module().PrivacyEngine(accountant="prv", secure_mode=False)
    wrapped, optimizer, private_loader = engine.make_private(
        module=model,
        optimizer=base,
        data_loader=loader,
        noise_multiplier=1,
        max_grad_norm=1,
        noise_generator=torch.Generator().manual_seed(32),
    )
    manager = opacus_module("opacus.utils.batch_memory_manager").BatchMemoryManager
    for step in range(4):
        with manager(
            data_loader=private_loader, max_physical_batch_size=2, optimizer=optimizer
        ) as physical_loader:
            for inputs, targets in physical_loader:
                optimizer.zero_grad()
                prediction = wrapped(inputs)
                loss = (
                    (prediction - targets).square().mean()
                    if len(inputs)
                    else prediction.sum()
                )
                loss.backward()
                optimizer.step()
                if sum(count for _, _, count in engine.accountant.history) == step + 1:
                    for group, initial_rate in zip(
                        base.param_groups, initial_rates, strict=True
                    ):
                        group["lr"] = initial_rate * (1 - (step + 1) / 4)
                    optimizer.zero_grad()
                    break
        trainer.train_local_steps(1)
        for name, value in model.state_dict().items():
            np.testing.assert_array_equal(trainer.weights()[name], value.numpy())
    expected = encode_state(base.state_dict())
    actual = encode_state(application_state(trainer.checkpoint_tensors())["optimizer"])
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])
    assert trainer.accounting_history == tuple(engine.accountant.history)


@pytest.mark.parametrize("kind", ["custom", "rmsprop", "adagrad"])
def test_registered_state_continues_after_restore(tmp_path: Path, kind: Kind) -> None:
    registration = optimizer_registration(kind)
    original = make_trainer(tmp_path / "original", registration)
    original.train_local_steps(2)
    snapshot = original.checkpoint_tensors()
    original.train_local_steps(2)
    restored = make_trainer(tmp_path / "restored", registration)
    restored.load_checkpoint_tensors(snapshot)
    restored.train_local_steps(2)
    for key, value in restored.checkpoint_tensors().items():
        if key != "meta.identity":
            np.testing.assert_array_equal(value, original.checkpoint_tensors()[key])


def test_weight_copy_keeps_parameter_objects_and_group_history(tmp_path: Path) -> None:
    registration = optimizer_registration()
    created: list[torch.optim.Optimizer] = []

    def capture(model: torch.nn.Module) -> torch.optim.Optimizer:
        optimizer = registration.create(model)
        created.append(optimizer)
        return optimizer

    captured = replace(registration, create=capture)
    specification = PrivateTrainingSpec(
        optimizer=registration.describe(public_model()),
        loss="mse",
        batch_size=4,
        schedule="linear_decay",
    )
    trainer = make_trainer(tmp_path, captured, specification=specification)
    optimizer = created[-1]
    identities = tuple(
        id(p) for group in optimizer.param_groups for p in group["params"]
    )
    trainer.train_local_steps(1)
    before = encode_state(optimizer.state_dict())
    trainer.load_weights(
        {name: value + 0.25 for name, value in trainer.weights().items()}
    )
    assert identities == tuple(
        id(p) for group in optimizer.param_groups for p in group["params"]
    )
    for name, value in before.items():
        np.testing.assert_array_equal(encode_state(optimizer.state_dict())[name], value)
    trainer.train_local_steps(1)
    np.testing.assert_allclose(
        [group["lr"] for group in optimizer.param_groups], [0.015, 0.005]
    )
    assert all(state["count"] == 2 for state in optimizer.state.values())


def test_custom_empty_draw_is_one_sanitized_optimizer_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sampler = opacus_module(
        "opacus.utils.uniform_sampler"
    ).UniformWithReplacementSampler

    def empty_draw(self: object) -> Iterator[list[int]]:
        yield []

    monkeypatch.setattr(sampler, "__iter__", empty_draw)
    trainer = make_trainer(tmp_path, optimizer_registration(), physical=1)
    before = trainer.weights()
    trainer.train_local_steps(1)
    assert trainer.completed_steps == trainer.physical_steps == 1
    assert trainer.accounting_history == ((1.0, 0.5, 1),)
    state = application_state(trainer.checkpoint_tensors())["optimizer"]["state"]
    assert all(fields["count"] == 1 for fields in state.values())
    assert any(
        not np.array_equal(before[key], value)
        for key, value in trainer.weights().items()
    )


def test_custom_physical_splitting_preserves_update(tmp_path: Path) -> None:
    registration = optimizer_registration()
    split = make_trainer(tmp_path / "split", registration, physical=1)
    unsplit = make_trainer(tmp_path / "unsplit", registration, physical=32)
    split.train_local_steps(4)
    unsplit.train_local_steps(4)
    for key, value in unsplit.weights().items():
        np.testing.assert_allclose(split.weights()[key], value, atol=1e-7)
    assert split.accounting_history == unsplit.accounting_history
    assert split.physical_steps > unsplit.physical_steps


@pytest.mark.parametrize("corruption", ["shape", "counter", "group", "state_id"])
def test_custom_malformed_restore_is_atomic(tmp_path: Path, corruption: str) -> None:
    trainer = make_trainer(tmp_path, optimizer_registration())
    trainer.train_local_steps(1)
    before = trainer.checkpoint_tensors()
    payload = application_state(before)
    optimizer = payload["optimizer"]
    if corruption == "shape":
        optimizer["state"][0]["energy"] = torch.zeros(3)
    elif corruption == "counter":
        optimizer["state"][0]["count"] = 9
    elif corruption == "group":
        optimizer["param_groups"][0]["decay"] = 0.9
    else:
        optimizer["state"][7] = optimizer["state"].pop(0)
    candidate = {
        key: value
        for key, value in before.items()
        if not key.startswith("application.")
    }
    candidate.update(
        {f"application.{key}": value for key, value in encode_state(payload).items()}
    )
    with pytest.raises(ValueError):
        trainer.load_checkpoint_tensors(candidate)
    for key, value in trainer.checkpoint_tensors().items():
        np.testing.assert_array_equal(value, before[key])


@pytest.mark.parametrize("mismatch", ["hash", "options", "groups"])
def test_factory_mismatch_rejects_before_reservation(
    tmp_path: Path, mismatch: str
) -> None:
    registration = optimizer_registration()
    specification = PrivateTrainingSpec(
        optimizer=registration.describe(public_model()),
        loss="mse",
        batch_size=4,
    )

    def changed(model: torch.nn.Module) -> torch.optim.Optimizer:
        optimizer = registration.create(model)
        if mismatch == "options":
            optimizer.param_groups[0]["decay"] = 0.7
        else:
            optimizer.param_groups.reverse()
        return optimizer

    actual = (
        replace(registration, implementation_hash="f" * 64)
        if mismatch == "hash"
        else replace(registration, create=changed)
    )
    with pytest.raises(ValueError):
        make_trainer(tmp_path, actual, specification=specification)
    assert (
        PrivacyLedger(tmp_path / "ledger.json", lineage="public").reservations() == ()
    )


def test_required_closure_rejected_before_reservation(tmp_path: Path) -> None:
    registration = optimizer_registration()
    specification = PrivateTrainingSpec(
        optimizer=registration.describe(public_model()),
        loss="mse",
        batch_size=4,
    )

    class ClosureOptimizer(CustomOptimizer):
        def step(self, closure: Any) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
            return closure()

    actual = replace(
        registration, optimizer_type=ClosureOptimizer, create=ClosureOptimizer
    )
    # Supply the matching declared class so this exercises closure validation.
    implementation_id = f"{ClosureOptimizer.__module__}.{ClosureOptimizer.__qualname__}"
    spec = specification.model_copy(
        update={
            "optimizer": specification.optimizer.model_copy(
                update={"implementation_id": implementation_id}
            )
        }
    )
    with pytest.raises(ValueError, match="closure"):
        make_trainer(tmp_path, actual, specification=spec)
    assert (
        PrivacyLedger(tmp_path / "ledger.json", lineage="public").reservations() == ()
    )


def test_mutating_shared_options_and_optimizer_cannot_change_contract(
    tmp_path: Path,
) -> None:
    from dromeus.manifests.models import FactoryOptimizerSpec

    held: list[torch.optim.Optimizer] = []
    registration = optimizer_registration("custom")
    original = registration.create

    def create(model: torch.nn.Module) -> torch.optim.Optimizer:
        optimizer = original(model)
        held.append(optimizer)
        return optimizer

    registration = replace(registration, create=create)
    trainer = make_trainer(tmp_path, registration)
    spec = trainer.specification.optimizer
    assert isinstance(spec, FactoryOptimizerSpec)
    spec.groups[0].options["decay"] = 0.5
    held[-1].param_groups[0]["decay"] = 0.5
    with pytest.raises(ValueError, match="specification changed"):
        trainer.train_local_steps(1)
    with pytest.raises(ValueError, match="specification changed"):
        trainer.weights()
    assert trainer.completed_steps == 0


def test_integer_learning_rate_survives_linear_schedule(tmp_path: Path) -> None:
    registration = optimizer_registration("custom")
    original = registration.create

    def create(model: torch.nn.Module) -> torch.optim.Optimizer:
        optimizer = original(model)
        for group in optimizer.param_groups:
            group["lr"] = 1
        return optimizer

    trainer = make_trainer(tmp_path, replace(registration, create=create))
    trainer.train_local_steps(2)
    checkpoint = trainer.checkpoint_tensors()
    trainer.load_checkpoint_tensors(checkpoint)
    trainer.train_local_steps(2)
    assert trainer.completed_steps == 4


def test_secure_factory_uses_same_mechanism_and_refuses_resume(tmp_path: Path) -> None:
    pytest.importorskip("torchcsprng")
    registration = optimizer_registration()
    spec = PrivateTrainingSpec(
        optimizer=registration.describe(public_model()), loss="mse", batch_size=4
    )
    trainer = PrivateTrainer(
        model_factory=public_model,
        inputs=torch.ones(8, 2),
        targets=torch.zeros(8, 1),
        specification=spec,
        policy=PrivacyPolicy(
            max_grad_norm=1,
            noise_multiplier=1,
            delta=1e-5,
            max_logical_steps=2,
            randomness_profile="secure",
        ),
        run_id="secure-factory",
        local_data_identity="public-secure-synthetic",
        reservation_store=PrivacyLedger.create(
            tmp_path / "ledger.json", lineage="public"
        ),
        max_physical_batch_size=1,
        optimizer_registration=registration,
    )
    trainer.train_local_steps(2)
    state = trainer.checkpoint_tensors()
    payload = application_state(state)
    assert payload["sampling_rng"] is None and payload["noise_rng"] is None
    assert trainer.accounting_history == ((1.0, 0.5, 2),)
    with pytest.raises(ValueError, match="fresh-start"):
        trainer.load_checkpoint_tensors(state)


def test_optimizer_failure_does_not_refund_reservation(tmp_path: Path) -> None:

    class FailingOptimizer(CustomOptimizer):
        def step(self, closure: Any = None) -> None:
            super().step(closure)
            raise RuntimeError("synthetic update failure")

    registration = optimizer_registration()
    registration = replace(
        registration,
        optimizer_type=FailingOptimizer,
        create=FailingOptimizer,
    )
    trainer = make_trainer(tmp_path, registration)
    with pytest.raises(ValueError, match="private local training failed"):
        trainer.train_local_steps(1)
    assert trainer.accounting_history == ((1.0, 0.5, 1),)
    assert (
        PrivacyLedger(tmp_path / "ledger.json", lineage="public")
        .reservations()[0]
        .logical_steps
        == 4
    )
    with pytest.raises(ValueError, match="private trainer failed"):
        trainer.weights()


def test_documented_registration_trains_and_restores(tmp_path: Path) -> None:
    from examples.private_optimizer import registration

    trainer = make_trainer(tmp_path, registration())
    trainer.train_local_steps(2)
    checkpoint = trainer.checkpoint_tensors()
    trainer.train_local_steps(2)
    expected = trainer.weights()
    restored = make_trainer(tmp_path / "restored", registration())
    restored.load_checkpoint_tensors(checkpoint)
    restored.train_local_steps(2)
    for name, value in expected.items():
        np.testing.assert_array_equal(restored.weights()[name], value)
