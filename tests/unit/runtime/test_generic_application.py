"""Model-independent application behavior through public task preparation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pytest
import torch
from support.application_fixture import application_draft
from torch import nn
from torch.utils.data import Dataset

from dromeus.application import (
    LocalTrainingOptions,
    PreparedApplication,
    definition_hash,
    describe_task,
    prepare_task,
)
from dromeus.manifests.canonical import load_safetensors
from dromeus.manifests.models import DraftRunSpec, PrivacyPolicy, PrivateOptimizerSpec
from dromeus.persistence.privacy_ledger import PrivacyLedger
from dromeus.training.private_data import make_task_loader
from dromeus.training.private_trainer import PrivateTrainer, private_training_definition
from dromeus.training.task import TensorBatchAdapter, TrainingTask

pytest.importorskip("opacus")


class LocalTokens(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, count: int = 8) -> None:
        self.count = count
        self.reads = 0

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        self.reads += 1
        tokens = torch.tensor([(index + j) % 7 for j in range(3)], dtype=torch.long)
        return {
            "tokens": tokens,
            "targets": (tokens + 1) % 7,
            "mask": torch.tensor([index % 3 != 0, True, index % 2 == 0]),
        }


class LocalLanguageModel(nn.Module):
    scale: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.tokens = nn.Embedding(7, 4)
        self.positions = nn.Embedding(3, 4)
        self.positions.requires_grad_(False)
        self.normalization = nn.LayerNorm(4)
        self.head = nn.Linear(4, 7, bias=False)
        self.head.weight = self.tokens.weight
        self.register_buffer("scale", torch.tensor([0.75]))

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        positions = positions.unsqueeze(0).expand_as(tokens)
        hidden = self.tokens(tokens) + self.positions(positions)
        return {"logits": self.head(self.normalization(hidden)) * self.scale}


def model_factory() -> nn.Module:
    with cast(Any, torch.random).fork_rng():
        cast(Any, torch).manual_seed(77)
        return LocalLanguageModel()


def sequence_loss(model: nn.Module, batch: Any) -> torch.Tensor:
    logits = model(batch["tokens"])["logits"]
    token_loss = nn.functional.cross_entropy(
        logits.transpose(1, 2), batch["targets"], reduction="none"
    )
    mask = batch["mask"].to(token_loss.dtype)
    return (token_loss * mask).sum(1) / mask.sum(1).clamp(min=1)


def task_for(dataset: LocalTokens | None = None) -> TrainingTask:
    return TrainingTask(
        model_factory=model_factory,
        dataset=dataset if dataset is not None else LocalTokens(),
        batch_adapter=TensorBatchAdapter(
            example={
                "tokens": torch.tensor([[0, 1, 2], [3, 4, 5]]),
                "targets": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                "mask": torch.tensor([[True, True, False], [False, False, False]]),
            },
            definition="public-fixed-length-token-tree-v1",
        ),
        loss_per_record=sequence_loss,
        model_definition="custom-tied-language-model-with-frozen-positions-v1",
        task_definition="masked-next-token-seven-token-public-synthetic-v1",
        loss_definition="mean-valid-token-loss-per-sequence-zero-all-masked-v1",
        implementation_hash="a" * 64,
        record_unit="fixed-three-token-sequence",
    )


def draft_for(
    task: TrainingTask,
    *,
    private: bool = True,
    batch_size: int = 4,
    optimizer_name: Literal["sgd", "adam"] = "sgd",
) -> DraftRunSpec:
    model = task.model_factory()
    specification = describe_task(
        task,
        batch_size=batch_size,
        optimizer=PrivateOptimizerSpec(
            name=optimizer_name,
            torch_version=torch.__version__,
            parameter_names=tuple(
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ),
            learning_rate=0.01,
        ),
    )
    values = application_draft().model_dump(mode="python")
    values.update(
        manifest_version=6,
        model_id="custom-language-model",
        model_definition_hash=definition_hash(task.model_definition),
        private_training=specification if private else None,
        privacy=PrivacyPolicy(
            max_grad_norm=0.5,
            noise_multiplier=1,
            delta=1e-5,
            max_logical_steps=9,
            randomness_profile="reproducible_public_benchmark",
        )
        if private
        else None,
    )
    values["environment"]["model_definition_hash"] = values["model_definition_hash"]
    values["dataset"]["definition_hash"] = definition_hash(task.task_definition)
    values["application_training"]["definition_hash"] = definition_hash(
        private_training_definition(specification)
    )
    return DraftRunSpec.model_validate(values)


def prepared_task(
    tmp_path: Path,
    task: TrainingTask,
    *,
    physical: int = 2,
    seed: int = 31,
    batch_size: int = 4,
    optimizer_name: Literal["sgd", "adam"] = "sgd",
) -> tuple[PreparedApplication, PrivateTrainer, PrivacyLedger]:
    draft = draft_for(task, batch_size=batch_size, optimizer_name=optimizer_name)
    ledger = PrivacyLedger.create(tmp_path / "ledger.json", lineage="local-text")
    prepared = prepare_task(
        draft=draft,
        task=task,
        local=LocalTrainingOptions(
            data_identity="fixed-local-token-records-v1",
            reservation_store=ledger,
            max_physical_batch_size=physical,
            research_seed=seed,
        ),
    )
    trainer = prepared.private_trainer
    assert trainer is not None
    return prepared, trainer, ledger


def test_custom_text_task_preserves_ties_frozen_state_and_private_exports(
    tmp_path: Path,
) -> None:
    records = LocalTokens()
    task = task_for(records)
    prepared, trainer, ledger = prepared_task(tmp_path, task)
    assert records.reads == 0
    prepared.validate_draft(draft_for(task))
    initial = trainer.weights()
    trainer.train_local_steps(2)
    assert records.reads > 0
    assert trainer.completed_steps == 2
    assert trainer.accounting_history == ((1.0, 0.5, 2),)
    assert trainer.local_loss is None
    assert len(ledger.reservations()) == 1
    assert len(prepared.tensor_schema.tensors) == len(initial)
    assert "head.weight" not in initial
    assert "tokens.weight" in initial
    assert "positions.weight" not in initial
    assert "scale" not in initial
    path = tmp_path / "model.safetensors"
    prepared.export_model(path)
    exported = load_safetensors(str(path))
    np.testing.assert_array_equal(exported["head.weight"], exported["tokens.weight"])
    public_model = task.model_factory()
    assert set(exported) == set(public_model.state_dict())
    np.testing.assert_array_equal(
        exported["positions.weight"],
        public_model.state_dict()["positions.weight"].numpy(),
    )
    assert all("rng" not in name and "loss" not in name for name in exported)
    assert trainer.checkpoint_tensors()["meta.loss"].size == 0


def test_same_application_factory_selects_ordinary_or_private_from_policy(
    tmp_path: Path,
) -> None:
    task = task_for()
    private_draft = draft_for(task)
    specification = private_draft.private_training
    from dromeus.manifests.models import TaskTrainingSpec

    assert isinstance(specification, TaskTrainingSpec)
    ordinary_draft = draft_for(task, private=False)
    ordinary = prepare_task(
        draft=ordinary_draft,
        task=task,
        specification=specification,
        local=LocalTrainingOptions(
            data_identity="ordinary-local-records", research_seed=31
        ),
    )
    private, _, _ = prepared_task(tmp_path / "private", task)
    assert ordinary.private_trainer is None
    assert private.private_trainer is not None
    assert ordinary.tensor_schema == private.tensor_schema
    ordinary.validate_draft(ordinary_draft)
    ordinary.create_initial_checkpoint(tmp_path / "ordinary-init.safetensors")
    private.create_initial_checkpoint(tmp_path / "private-init.safetensors")
    left = load_safetensors(str(tmp_path / "ordinary-init.safetensors"))
    right = load_safetensors(str(tmp_path / "private-init.safetensors"))
    assert set(left) == set(right)
    for name in left:
        np.testing.assert_array_equal(left[name], right[name])
    with pytest.raises(ValueError, match="draft"):
        ordinary.validate_draft(private_draft)


@pytest.mark.parametrize(
    "field", ["task_definition", "loss_definition", "model_definition", "record_unit"]
)
def test_changed_task_semantics_fail_before_any_record_read_or_reservation(
    tmp_path: Path,
    field: str,
) -> None:
    records = LocalTokens()
    task = task_for(records)
    draft = draft_for(task)
    ledger = PrivacyLedger.create(tmp_path / "ledger.json", lineage="local-text")
    altered = replace(task, **{field: "different-semantics"})
    with pytest.raises(ValueError, match="definition|contract|manifest"):
        prepare_task(
            draft=draft,
            task=altered,
            local=LocalTrainingOptions(
                data_identity="local-text", reservation_store=ledger, research_seed=31
            ),
        )
    assert records.reads == 0
    assert ledger.reservations() == ()


def test_split_and_unsplit_private_token_updates_match(tmp_path: Path) -> None:
    _, split, _ = prepared_task(
        tmp_path / "split", task_for(), physical=1, batch_size=8
    )
    _, whole, _ = prepared_task(
        tmp_path / "whole", task_for(), physical=8, batch_size=8
    )
    split.train_local_steps(3)
    whole.train_local_steps(3)
    assert split.accounting_history == whole.accounting_history == ((1.0, 1.0, 3),)
    assert split.physical_steps == 24
    assert whole.physical_steps == 3
    for name, value in whole.weights().items():
        np.testing.assert_allclose(split.weights()[name], value, rtol=2e-5, atol=2e-7)


def test_empty_draw_performs_noise_only_step_without_application_callback(
    tmp_path: Path,
) -> None:
    # Find a known empty public Poisson draw through the same public data interface.
    seed = None
    for candidate in range(20):
        fixture = task_for(LocalTokens(64))
        loader = make_task_loader(
            fixture.dataset,
            fixture.batch_adapter,
            batch_size=1,
            generator=torch.Generator().manual_seed(candidate),
        )
        if fixture.batch_adapter.size(next(iter(loader))) == 0:
            seed = candidate
            break
    assert seed is not None
    calls: list[int] = []
    records = LocalTokens(64)
    task = task_for(records)

    def observed_loss(model: nn.Module, batch: Any) -> torch.Tensor:
        calls.append(len(batch["tokens"]))
        return sequence_loss(model, batch)

    task = replace(task, loss_per_record=observed_loss)
    _, trainer, ledger = prepared_task(tmp_path, task, seed=seed, batch_size=1)
    # Public compatibility probes ran during preparation; actual draws start now.
    calls.clear()
    before = trainer.weights()
    assert records.reads == 0
    trainer.train_local_steps(1)
    assert records.reads == 0
    assert calls == []
    assert trainer.completed_steps == 1
    assert trainer.accounting_history == ((1.0, 1 / 64, 1),)
    assert len(ledger.reservations()) == 1
    assert any(
        not np.array_equal(before[name], value)
        for name, value in trainer.weights().items()
    )


def test_research_checkpoint_continues_token_sampler_and_noise(tmp_path: Path) -> None:
    _, original, _ = prepared_task(tmp_path / "original", task_for())
    original.train_local_steps(2)
    checkpoint = original.checkpoint_tensors()
    original.train_local_steps(2)
    _, resumed, _ = prepared_task(tmp_path / "resumed", task_for())
    resumed.load_checkpoint_tensors(checkpoint)
    resumed.train_local_steps(2)
    assert resumed.accounting_history == original.accounting_history
    for name, value in original.checkpoint_tensors().items():
        if name != "meta.identity":
            np.testing.assert_array_equal(resumed.checkpoint_tensors()[name], value)


@pytest.mark.parametrize(
    "name", ["model.positions.weight", "model.scale", "model.head.weight"]
)
def test_bad_immutable_or_tied_checkpoint_is_rejected_atomically(
    tmp_path: Path,
    name: str,
) -> None:
    _, trainer, _ = prepared_task(tmp_path, task_for())
    trainer.train_local_steps(1)
    before = trainer.checkpoint_tensors()
    damaged = {key: value.copy() for key, value in before.items()}
    damaged[name].flat[0] += 1
    with pytest.raises(ValueError, match="immutable|tied"):
        trainer.load_checkpoint_tensors(damaged)
    after = trainer.checkpoint_tensors()
    for key in before:
        np.testing.assert_array_equal(after[key], before[key])


def test_failed_private_callback_retains_reservation_and_blocks_export(
    tmp_path: Path,
) -> None:
    enabled = False

    def failing_loss(model: nn.Module, batch: Any) -> torch.Tensor:
        if enabled:
            raise RuntimeError("private-text-must-not-appear-in-error")
        return sequence_loss(model, batch)

    task = replace(task_for(), loss_per_record=failing_loss)
    prepared, trainer, ledger = prepared_task(tmp_path, task, batch_size=8)
    enabled = True
    with pytest.raises(ValueError, match="private local training failed") as error:
        trainer.train_local_steps(1)
    assert "private-text" not in str(error.value)
    assert len(ledger.reservations()) == 1
    assert ledger.reservations()[0].logical_steps == 9
    assert trainer.local_loss is None
    with pytest.raises(ValueError, match="failed"):
        prepared.export_model(tmp_path / "forbidden.safetensors")
    assert not (tmp_path / "forbidden.safetensors").exists()


def test_undeclared_evaluation_callback_is_rejected_before_reservation(
    tmp_path: Path,
) -> None:
    task = task_for()
    draft = draft_for(task)

    def undeclared_evaluation(_: nn.Module) -> dict[str, float]:
        return {"private_value": 123.0}

    altered = replace(task, public_evaluate=undeclared_evaluation)
    ledger = PrivacyLedger.create(tmp_path / "ledger.json", lineage="local-text")
    with pytest.raises(ValueError, match="evaluation"):
        prepare_task(
            draft=draft,
            task=altered,
            local=LocalTrainingOptions(
                data_identity="local-text", reservation_store=ledger, research_seed=31
            ),
        )
    assert ledger.reservations() == ()


def test_outer_weight_copy_preserves_adam_accountant_and_sampler_state(
    tmp_path: Path,
) -> None:
    _, trainer, _ = prepared_task(tmp_path, task_for(), optimizer_name="adam")
    trainer.train_local_steps(2)
    before = trainer.checkpoint_tensors()
    exchanged = {
        name: value + np.float32(0.01) for name, value in trainer.weights().items()
    }
    trainer.load_weights(exchanged)
    after = trainer.checkpoint_tensors()
    for key in before:
        if not key.startswith("model."):
            np.testing.assert_array_equal(after[key], before[key])
    np.testing.assert_array_equal(
        after["model.tokens.weight"], after["model.head.weight"]
    )
    trainer.train_local_steps(1)
    assert trainer.accounting_history == ((1.0, 0.5, 3),)


def test_runtime_buffer_mutation_cannot_be_released(tmp_path: Path) -> None:
    enabled = False

    def mutating_loss(model: nn.Module, batch: Any) -> torch.Tensor:
        if enabled:
            next(model.buffers()).add_(1)
        return sequence_loss(model, batch)

    task = replace(task_for(), loss_per_record=mutating_loss)
    prepared, trainer, ledger = prepared_task(tmp_path, task, batch_size=8)
    enabled = True
    with pytest.raises(ValueError, match="private local training failed"):
        trainer.train_local_steps(1)
    with pytest.raises(ValueError, match="failed"):
        prepared.export_model(tmp_path / "unsafe-buffer.safetensors")
    assert ledger.reservations()[0].logical_steps == 9
    assert not (tmp_path / "unsafe-buffer.safetensors").exists()


@pytest.mark.parametrize("where", ["record_loss", "collation"])
def test_undeclared_runtime_randomness_fails_without_releasing_model(
    tmp_path: Path,
    where: str,
) -> None:
    task = task_for()
    if where == "record_loss":

        def stochastic_loss(model: nn.Module, batch: Any) -> torch.Tensor:
            values = sequence_loss(model, batch)
            # The declared public probe has no token 6; actual local records do.
            if bool((batch["tokens"] == 6).any()):
                values = values * torch.rand_like(values)
            return values

        task = replace(task, loss_per_record=stochastic_loss)
    else:
        original_adapter = task.batch_adapter

        def stochastic_collation(records: Any) -> Any:
            torch.rand(())
            return original_adapter.collate(records)

        task = replace(
            task,
            batch_adapter=TensorBatchAdapter(
                example=original_adapter.example,
                definition=original_adapter.definition,
                collate_fn=stochastic_collation,
            ),
        )
    prepared, trainer, ledger = prepared_task(tmp_path, task, batch_size=8)
    with pytest.raises(ValueError, match="private local training failed"):
        trainer.train_local_steps(1)
    assert len(ledger.reservations()) == 1
    assert ledger.reservations()[0].logical_steps == 9
    with pytest.raises(ValueError, match="failed"):
        prepared.export_model(tmp_path / "randomness-failure.safetensors")
    assert not (tmp_path / "randomness-failure.safetensors").exists()


def test_nonpersistent_public_buffer_remains_local_across_restore_and_export(
    tmp_path: Path,
) -> None:
    def with_positions() -> nn.Module:
        model = model_factory()
        model.register_buffer("position_ids", torch.arange(3), persistent=False)
        return model

    task = replace(task_for(), model_factory=with_positions)
    prepared, trainer, _ = prepared_task(tmp_path, task)
    trainer.train_local_steps(1)
    checkpoint = trainer.checkpoint_tensors()
    assert "model.position_ids" not in checkpoint
    trainer.load_checkpoint_tensors(checkpoint)
    trainer.train_local_steps(1)
    prepared.export_model(tmp_path / "model.safetensors")
    exported = load_safetensors(str(tmp_path / "model.safetensors"))
    assert "position_ids" not in exported
    assert trainer.completed_steps == 2


def test_extra_serialized_application_state_fails_before_data_or_spending(
    tmp_path: Path,
) -> None:
    class ExtraStateModel(LocalLanguageModel):
        def get_extra_state(self) -> Any:
            return torch.tensor([42.0])

    records = LocalTokens()
    task = task_for(records)
    draft = draft_for(task)
    altered = replace(task, model_factory=ExtraStateModel)
    ledger = PrivacyLedger.create(tmp_path / "ledger.json", lineage="local-text")
    with pytest.raises(ValueError, match="registered parameters and buffers"):
        prepare_task(
            draft=draft,
            task=altered,
            local=LocalTrainingOptions(
                data_identity="local-text", reservation_store=ledger, research_seed=31
            ),
        )
    assert records.reads == 0
    assert ledger.reservations() == ()
