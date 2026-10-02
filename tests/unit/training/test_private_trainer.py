"""Public private-training adapter behavior on fixed synthetic data."""

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch
from torch import nn

from dromeus.manifests.models import (
    PrivacyPolicy,
    PrivateOptimizerSpec,
    PrivateTrainingSpec,
)
from dromeus.persistence.privacy_ledger import PrivacyLedger
from dromeus.training.private_trainer import PrivateTrainer

pytest.importorskip("opacus")


def make_trainer(
    tmp_path: Path,
    name: Literal["sgd", "adam", "adamw"] = "adam",
    physical: int = 2,
    secure: bool = False,
    nesterov: bool = False,
    sample_count: int = 8,
) -> PrivateTrainer:
    policy = PrivacyPolicy(
        max_grad_norm=1,
        noise_multiplier=1,
        delta=1e-5,
        max_logical_steps=4,
        randomness_profile="secure" if secure else "reproducible_public_benchmark",
    )
    spec = PrivateTrainingSpec(
        optimizer=PrivateOptimizerSpec(
            name=name,
            torch_version=torch.__version__,
            parameter_names=("weight", "bias"),
            learning_rate=0.01,
            momentum=0.9 if nesterov else 0,
            nesterov=nesterov,
        ),
        loss="mse",
        batch_size=4,
    )
    ledger = PrivacyLedger.create(tmp_path / "ledger.json", lineage="public-synthetic")
    return PrivateTrainer(
        model_factory=lambda: nn.Linear(2, 1),
        inputs=torch.ones(sample_count, 2),
        targets=torch.zeros(sample_count, 1),
        policy=policy,
        specification=spec,
        run_id="synthetic-run",
        local_data_identity="synthetic-v1",
        reservation_store=ledger,
        max_physical_batch_size=physical,
        research_seed=None if secure else 31,
    )


@pytest.mark.parametrize("name", ["sgd", "adam", "adamw"])
def test_owned_trainer_counts_logical_steps_and_preserves_weights(
    tmp_path: Path, name: Literal["sgd", "adam", "adamw"]
) -> None:
    trainer = make_trainer(tmp_path, name)
    assert set(trainer.weights()) == {"weight", "bias"}
    trainer.train_local_steps(2)
    trainer.load_weights(trainer.weights())
    trainer.train_local_steps(2)
    assert trainer.completed_steps == 4
    assert trainer.accounting_history == ((1.0, 0.5, 4),)
    assert trainer.local_loss is None
    assert trainer.physical_steps >= 4
    with pytest.raises(ValueError, match="horizon"):
        trainer.train_local_steps(1)
    state = trainer.checkpoint_tensors()
    assert not state["meta.loss"].size
    assert all(np.isfinite(v).all() for v in state.values())


@pytest.mark.parametrize("name", ["sgd", "adam", "adamw"])
def test_research_checkpoint_continuation_matches_uninterrupted(
    tmp_path: Path, name: Literal["sgd", "adam", "adamw"]
) -> None:
    trainer = make_trainer(tmp_path / "first", name)
    trainer.train_local_steps(2)
    checkpoint = trainer.checkpoint_tensors()
    trainer.train_local_steps(2)
    restored = make_trainer(tmp_path / "restored", name)
    restored.load_checkpoint_tensors(checkpoint)
    restored.train_local_steps(2)
    for key, actual in restored.checkpoint_tensors().items():
        if key != "meta.identity":
            np.testing.assert_array_equal(actual, trainer.checkpoint_tensors()[key])


def test_bad_checkpoint_does_not_change_state(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path)
    trainer.train_local_steps(1)
    old = trainer.checkpoint_tensors()
    bad = {k: v.copy() for k, v in old.items()}
    bad["meta.steps"][:] = 3
    with pytest.raises(ValueError, match="counters"):
        trainer.load_checkpoint_tensors(bad)
    for key, value in trainer.checkpoint_tensors().items():
        np.testing.assert_array_equal(value, old[key])


def test_missing_ledger_blocks_training_and_model_export(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path)
    (tmp_path / "ledger.json").unlink()
    with pytest.raises(ValueError, match="ledger"):
        trainer.train_local_steps(1)
    with pytest.raises(ValueError, match="ledger"):
        trainer.export_model(tmp_path / "model.safetensors")
    assert not (tmp_path / "model.safetensors").exists()


@pytest.mark.parametrize("name", ["sgd", "adam", "adamw"])
@pytest.mark.parametrize("nesterov", [False, True])
def test_direct_opacus_trajectory_parity(
    tmp_path: Path, name: Literal["sgd", "adam", "adamw"], nesterov: bool
) -> None:
    if nesterov and name != "sgd":
        nesterov = False
    from torch.utils.data import DataLoader, TensorDataset

    from dromeus.training.privacy import opacus_module
    from dromeus.training.private_trainer import build_private_optimizer

    trainer = make_trainer(tmp_path, name, nesterov=nesterov)
    model = nn.Linear(2, 1)
    with torch.no_grad():
        for key, target in model.state_dict().items():
            target.copy_(torch.tensor(trainer.weights()[key]))
    specification = trainer.specification
    base = build_private_optimizer(model, specification.optimizer)
    sampling = torch.Generator().manual_seed(31)
    noise = torch.Generator().manual_seed(32)
    loader = DataLoader(
        TensorDataset(torch.ones(8, 2), torch.zeros(8, 1)),
        batch_size=4,
        generator=sampling,
    )
    engine = opacus_module().PrivacyEngine(accountant="prv", secure_mode=False)
    wrapped, optimizer, private_loader = engine.make_private(
        module=model,
        optimizer=base,
        data_loader=loader,
        noise_multiplier=1,
        max_grad_norm=1,
        noise_generator=noise,
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
                if sum(n for _, _, n in engine.accountant.history) == step + 1:
                    optimizer.zero_grad()
                    break
        trainer.train_local_steps(1)
        for key, expected in model.state_dict().items():
            np.testing.assert_array_equal(
                trainer.weights()[key], expected.detach().numpy()
            )
    from dromeus.training.private_state import decode_state

    checkpoint = trainer.checkpoint_tensors()
    owned = decode_state(
        {
            k.removeprefix("application."): v
            for k, v in checkpoint.items()
            if k.startswith("application.")
        }
    )["optimizer"]
    reference = base.state_dict()
    assert owned["param_groups"] == reference["param_groups"]
    assert owned["state"].keys() == reference["state"].keys()
    for index, fields in reference["state"].items():
        assert owned["state"][index].keys() == fields.keys()
        for field, value in fields.items():
            np.testing.assert_array_equal(
                owned["state"][index][field].numpy(), value.detach().numpy()
            )
    assert trainer.accounting_history == tuple(engine.accountant.history)
    assert float(trainer.accounting()["privacy_epsilon"]) > 0


@pytest.mark.parametrize("name", ["sgd", "adam", "adamw"])
@pytest.mark.parametrize(("noise", "expected_gradient"), [(0.0, -0.5), (0.2, -0.4)])
def test_analytic_clipped_gradient_preserves_optimizer_rule(
    tmp_path: Path,
    name: Literal["sgd", "adam", "adamw"],
    monkeypatch: pytest.MonkeyPatch,
    noise: float,
    expected_gradient: float,
) -> None:
    from dromeus.training.privacy import opacus_module
    from dromeus.training.private_trainer import build_private_optimizer

    # Independent analytic result: two gradients (-2,0), (0,-2), each
    # clipped to unit length, plus fixed sum noise (0.2,0.2),
    # then averaged, give (-0.4,-0.4); Opacus uses a 1e-6 clip stabilizer.
    def controlled_noise(**kwargs: object) -> torch.Tensor:
        return torch.full_like(kwargs["reference"], noise)  # type: ignore[arg-type]

    monkeypatch.setattr(
        opacus_module("opacus.optimizers.optimizer"),
        "_generate_noise",
        controlled_noise,
    )
    spec = PrivateTrainingSpec(
        optimizer=PrivateOptimizerSpec(
            name=name,
            torch_version=torch.__version__,
            parameter_names=("weight",),
            learning_rate=0.1,
            weight_decay=0.05,
        ),
        loss="mse",
        batch_size=2,
    )

    def model_factory() -> nn.Module:
        model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        return model

    seen: list[torch.Tensor] = []

    def capture_optimizer(model: nn.Module) -> torch.optim.Optimizer:
        optimizer = build_private_optimizer(model, spec.optimizer)
        original_step = optimizer.step

        def capture_step(*args: object, **kwargs: object) -> object:
            for parameter in model.parameters():
                assert parameter.grad is not None
                seen.append(parameter.grad.detach().clone())
            return original_step()  # pyright: ignore[reportUnknownMemberType]

        optimizer.step = capture_step  # type: ignore[method-assign]
        return optimizer

    trainer = PrivateTrainer(
        optimizer_factory=capture_optimizer,
        model_factory=model_factory,
        inputs=torch.eye(2),
        targets=torch.ones(2, 1),
        policy=PrivacyPolicy(
            max_grad_norm=1,
            noise_multiplier=1,
            delta=1e-5,
            max_logical_steps=1,
            randomness_profile="reproducible_public_benchmark",
        ),
        specification=spec,
        run_id="analytic",
        local_data_identity="public",
        reservation_store=PrivacyLedger.create(
            tmp_path / "ledger.json", lineage="analytic"
        ),
        max_physical_batch_size=1,
        research_seed=2,
    )
    reference = model_factory()
    optimizer = build_private_optimizer(reference, spec.optimizer)
    for parameter in reference.parameters():
        parameter.grad = torch.tensor([[expected_gradient, expected_gradient]])
    optimizer.step()
    trainer.train_local_steps(1)
    assert len(seen) == 1
    np.testing.assert_allclose(
        seen[0].numpy(), [[expected_gradient, expected_gradient]], atol=5e-7
    )
    np.testing.assert_allclose(
        trainer.weights()["weight"], reference.state_dict()["weight"].numpy(), atol=1e-7
    )


def test_empty_draw_consumes_noise_only_logical_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dromeus.training.privacy import opacus_module

    sampler = opacus_module(
        "opacus.utils.uniform_sampler"
    ).UniformWithReplacementSampler

    def empty_draw(_: object) -> Iterator[list[int]]:
        draws: list[list[int]] = [[]]
        return iter(draws)

    monkeypatch.setattr(sampler, "__iter__", empty_draw)
    trainer = make_trainer(tmp_path)
    before = trainer.weights()
    trainer.train_local_steps(1)
    assert trainer.accounting_history == ((1.0, 0.5, 1),)
    assert trainer.completed_steps == trainer.physical_steps == 1
    assert trainer.local_loss is None
    assert any(not np.array_equal(before[k], v) for k, v in trainer.weights().items())


@pytest.mark.parametrize("name", ["sgd", "adam", "adamw"])
def test_secure_mode_steps_export_and_refuses_resume(
    tmp_path: Path, name: Literal["sgd", "adam", "adamw"]
) -> None:
    pytest.importorskip("torchcsprng")
    from dromeus.manifests.canonical import load_safetensors

    trainer = make_trainer(tmp_path, name, secure=True)
    trainer.train_local_steps(1)
    state = trainer.checkpoint_tensors()
    with pytest.raises(ValueError, match="fresh-start"):
        trainer.load_checkpoint_tensors(state)
    trainer.export_model(tmp_path / "model.safetensors")
    exported = load_safetensors(str(tmp_path / "model.safetensors"))
    assert set(exported) == {"weight", "bias"}
    assert trainer.accounting_history == ((1.0, 0.5, 1),)


def test_optimizer_exception_does_not_refund_reserved_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = make_trainer(tmp_path)

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("private-example-id-must-not-escape")

    monkeypatch.setattr(torch.optim.Adam, "step", fail)
    with pytest.raises(ValueError, match="^private local training failed$"):
        trainer.train_local_steps(1)
    assert trainer.accounting_history == ((1.0, 0.5, 1),)
    assert (
        PrivacyLedger(tmp_path / "ledger.json", lineage="public-synthetic")
        .reservations()[0]
        .logical_steps
        == 4
    )
    with pytest.raises(ValueError):
        trainer.export_model(tmp_path / "model.safetensors")


def test_split_and_unsplit_mechanisms_match(tmp_path: Path) -> None:
    split = make_trainer(tmp_path / "split", physical=1)
    unsplit = make_trainer(tmp_path / "unsplit", physical=32)
    unsplit.load_weights(split.weights())
    split.train_local_steps(4)
    unsplit.train_local_steps(4)
    assert split.accounting_history == unsplit.accounting_history
    for key, expected in unsplit.weights().items():
        np.testing.assert_allclose(split.weights()[key], expected, atol=1e-7)
    assert split.physical_steps > unsplit.physical_steps


def test_custom_forward_mutation_rejected_before_reservation(tmp_path: Path) -> None:
    class RawMutation(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                self.linear.weight.add_(inputs.sum())
            return self.linear(inputs)

    spec = PrivateTrainingSpec(
        optimizer=PrivateOptimizerSpec(
            name="sgd",
            torch_version=torch.__version__,
            parameter_names=("linear.weight", "linear.bias"),
            learning_rate=0.01,
        ),
        loss="mse",
        batch_size=2,
    )
    ledger = PrivacyLedger.create(tmp_path / "ledger.json", lineage="synthetic")
    with pytest.raises(ValueError, match="unsupported private model"):
        PrivateTrainer(
            model_factory=RawMutation,
            inputs=torch.ones(8, 2),
            targets=torch.zeros(8, 1),
            policy=PrivacyPolicy(
                max_grad_norm=1,
                noise_multiplier=1,
                delta=1e-5,
                max_logical_steps=4,
                randomness_profile="reproducible_public_benchmark",
            ),
            specification=spec,
            run_id="mutation",
            local_data_identity="public",
            reservation_store=ledger,
            max_physical_batch_size=2,
            research_seed=1,
        )
    assert ledger.reservations() == ()


def test_secure_stream_wiring_and_one_noise_per_logical_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torchcsprng")
    from typing import Any

    from dromeus.training.privacy import opacus_module

    engines: list[Any] = []
    original = opacus_module().PrivacyEngine

    def engine_factory(**kwargs: Any) -> Any:
        engine = original(**kwargs)
        make_private = engine.make_private

        def prepare(**options: Any) -> Any:
            result = make_private(**options)
            assert result[1].generator is engine.secure_rng
            assert result[2].generator is engine.secure_rng
            return result

        engine.make_private = prepare
        engines.append(engine)
        return engine

    monkeypatch.setattr(opacus_module(), "PrivacyEngine", engine_factory)
    noise_module = opacus_module("opacus.optimizers.optimizer")
    noise = noise_module._generate_noise
    noise_events: list[object] = []

    def capture_noise(**kwargs: Any) -> Any:
        noise_events.append(kwargs["generator"])
        return noise(**kwargs)

    monkeypatch.setattr(noise_module, "_generate_noise", capture_noise)
    first = make_trainer(tmp_path / "first", secure=True, physical=1)
    second = make_trainer(tmp_path / "second", secure=True, physical=1)
    second.load_weights(first.weights())
    assert engines[0].secure_rng is not engines[1].secure_rng
    first.train_local_steps(2)
    second.train_local_steps(2)
    assert len(noise_events) == 8  # two parameters * four logical steps
    assert any(
        not np.array_equal(first.weights()[k], second.weights()[k])
        for k in first.weights()
    )
    before = len(noise_events)
    from dromeus.algorithms.noloco import NoLoCoAlgorithm
    from dromeus.manifests.models import NoLoCoConfig

    algorithm = NoLoCoAlgorithm(
        trainer=first,
        tensor_schema=first.tensor_schema,
        config=NoLoCoConfig(alpha=0.4, beta=0.3, gamma=0.2, inner_steps=1),
    )
    algorithm.configure_bundle_codec(
        run_id="retry",
        manifest_hash="a" * 64,
        sender_public_key="node-a",
        algorithm_id="noloco",
        artifact_root=tmp_path / "bundles",
    )
    algorithm.pre_local(0)
    algorithm.local_training()
    bundle = algorithm.post_local_bundle()
    payloads = [artifact.path.read_bytes() for artifact in bundle.artifacts]
    for _ in range(2):
        assert [artifact.path.read_bytes() for artifact in bundle.artifacts] == payloads
    assert len(noise_events) == before + 2
    algorithm.release_bundle(bundle)


@pytest.mark.parametrize(
    "field",
    [
        "policy",
        "sample_rate",
        "accountant",
        "optimizer",
        "sampling_rng",
        "specification",
    ],
)
def test_malformed_research_restore_is_atomic(tmp_path: Path, field: str) -> None:
    from dromeus.training.private_state import decode_state, encode_state

    trainer = make_trainer(tmp_path)
    trainer.train_local_steps(1)
    before = trainer.checkpoint_tensors()
    payload = decode_state(
        {
            k.removeprefix("application."): v
            for k, v in before.items()
            if k.startswith("application.")
        }
    )
    if field in {"policy", "specification"}:
        payload[field] = "f" * 64
    elif field == "sample_rate":
        payload[field] = 0.25
    elif field == "accountant":
        payload[field] = [(1, 0.5, 2)]
    elif field == "optimizer":
        payload[field]["state"][0]["exp_avg"] = torch.zeros(3)
    else:
        payload[field] = torch.zeros(3, dtype=torch.uint8)
    candidate = {k: v for k, v in before.items() if not k.startswith("application.")}
    candidate.update({f"application.{k}": v for k, v in encode_state(payload).items()})
    with pytest.raises(ValueError):
        trainer.load_checkpoint_tensors(candidate)
    for key, value in trainer.checkpoint_tensors().items():
        np.testing.assert_array_equal(value, before[key])


def test_equal_accounting_across_optimizers(tmp_path: Path) -> None:
    histories: list[object] = []
    epsilons: list[float | int | str] = []
    for name in ("sgd", "adam", "adamw"):
        trainer = make_trainer(tmp_path / name, name)
        trainer.train_local_steps(4)
        histories.append(trainer.accounting_history)
        epsilons.append(trainer.accounting()["privacy_epsilon"])
    assert histories[0] == histories[1] == histories[2]
    assert epsilons[0] == epsilons[1] == epsilons[2]


def test_seeded_gaussian_noise_variance_and_expected_batch_normalization(
    tmp_path: Path,
) -> None:
    # Zero inputs/targets and zero initialization give exactly zero raw gradients.
    # SGD with lr=1 exposes only the normalized Gaussian draw in the model delta.
    spec = PrivateTrainingSpec(
        optimizer=PrivateOptimizerSpec(
            name="sgd",
            torch_version=torch.__version__,
            parameter_names=("weight",),
            learning_rate=1,
        ),
        loss="mse",
        batch_size=32,
    )

    def model() -> nn.Module:
        module = nn.Linear(1, 4096, bias=False)
        with torch.no_grad():
            module.weight.zero_()
        return module

    trainer = PrivateTrainer(
        model_factory=model,
        inputs=torch.zeros(32, 1),
        targets=torch.zeros(32, 4096),
        specification=spec,
        policy=PrivacyPolicy(
            max_grad_norm=3,
            noise_multiplier=2,
            delta=1e-5,
            max_logical_steps=1,
            randomness_profile="reproducible_public_benchmark",
        ),
        run_id="variance",
        local_data_identity="public-zero-data",
        reservation_store=PrivacyLedger.create(
            tmp_path / "ledger.json", lineage="variance"
        ),
        max_physical_batch_size=4,
        research_seed=83,
    )
    trainer.train_local_steps(1)
    values = trainer.weights()["weight"].ravel()
    assert abs(float(values.mean())) < 0.01
    # (sigma*C/expected_batch_size)^2 = (2*3/32)^2 = 0.03515625.
    assert 0.032 < float(values.var()) < 0.039
    assert trainer.accounting_history == ((2.0, 1.0, 1),)


def test_actual_sampler_rate_accounts_for_loader_rounding(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path, sample_count=10)
    assert trainer.sample_rate == 1 / 3
    assert trainer.sample_rate != 4 / 10
    trainer.train_local_steps(2)
    assert trainer.accounting_history == ((1.0, 1 / 3, 2),)


def test_secure_rng_unavailable_never_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dromeus.training.privacy import opacus_module

    def unavailable(**kwargs: object) -> object:
        assert kwargs["secure_mode"] is True
        raise ImportError("torchcsprng absent")

    monkeypatch.setattr(opacus_module(), "PrivacyEngine", unavailable)
    with pytest.raises(ValueError, match="private preparation failed"):
        make_trainer(tmp_path, secure=True)
    assert (
        PrivacyLedger(
            tmp_path / "ledger.json", lineage="public-synthetic"
        ).reservations()
        == ()
    )
