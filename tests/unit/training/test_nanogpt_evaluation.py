"""Public metrics retain training normalization and isolated generation state."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from benchmarks.workloads.nanogpt.evaluation import (
    evaluate_public_domains,
    generate_public_prompts,
)
from benchmarks.workloads.nanogpt.training import GPT, GPTConfig, build_task
from dromeus.application import describe_task
from dromeus.manifests.models import PrivacyPolicy, PrivateOptimizerSpec
from dromeus.persistence.privacy_ledger import PrivacyLedger
from dromeus.training.private_trainer import PrivateTrainer


class Records(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, targets: list[list[int]], masks: list[list[bool]]) -> None:
        self.targets, self.masks = targets, masks

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "tokens": torch.zeros(len(self.targets[index]), dtype=torch.int64),
            "targets": torch.tensor(self.targets[index]),
            "mask": torch.tensor(self.masks[index]),
        }


class FixedLogits(nn.Module):
    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, None]:
        # Index3 is a padded vocabulary entry. It must affect NLL but must never
        # be generated when the tokenizer has only3 valid entries.
        logits = torch.tensor([0.0, 1.0, 2.0, 9.0])
        return logits.expand(*tokens.shape, 4), None


def test_token_weighting_full_vocab_and_domain_aggregation() -> None:
    model = FixedLogits()
    domains: dict[str, Dataset[dict[str, torch.Tensor]]] = {
        "one": Records([[0, 1], [2, 999]], [[True, True], [True, False]]),
        "two": Records([[2, 999]], [[True, False]]),
    }
    result = evaluate_public_domains(model, domains, device="cpu")
    ce = F.cross_entropy(
        torch.tensor([[0.0, 1.0, 2.0, 9.0]]).expand(3, 4),
        torch.tensor([0, 1, 2]),
        reduction="none",
    )
    one, two = float(ce.double().mean()), float(ce[2])
    assert result["one_nll"] == pytest.approx(one)
    assert result["two_nll"] == pytest.approx(two)
    assert result["loss"] == pytest.approx((3 * one + two) / 4)
    assert result["macro_domain_nll"] == pytest.approx((one + two) / 2)
    assert result["worst_domain_nll"] == one
    assert result["target_tokens"] == 4
    assert result["perplexity"] == pytest.approx(math.exp(result["loss"]))
    assert result["loss"] > 7  # Would fail if padded logits were masked for NLL.


def test_generation_masks_padding_and_does_not_consume_global_rng() -> None:
    model = FixedLogits()
    rng = torch.get_rng_state().clone()
    result = generate_public_prompts(
        model,
        [[0]],
        device="cpu",
        context_length=2,
        valid_vocabulary=3,
        max_new_tokens=4,
        temperature=1.0,
        top_k=1,
        seed=17,
    )
    assert result == [[0, 2, 2, 2, 2]]
    assert torch.equal(rng, torch.get_rng_state())
    assert model.training


def test_actual_gpt_evaluation_preserves_weights_gradients_and_mixed_modes() -> None:
    model = GPT(GPTConfig(block_size=2, vocab_size=5, n_layer=1, n_head=1, n_embd=4))
    model.transformer.eval()
    modes = [m.training for m in model.modules()]
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    rng = torch.get_rng_state().clone()
    evaluate_public_domains(
        model, {"test": Records([[1, 2]], [[True, True]])}, device="cpu"
    )
    assert modes == [m.training for m in model.modules()]
    assert all(torch.equal(v, weights[k]) for k, v in model.state_dict().items())
    assert all(
        p.grad is not None and torch.equal(p.grad, torch.ones_like(p))
        for p in model.parameters()
    )
    assert torch.equal(rng, torch.get_rng_state())


def test_evaluation_failure_restores_modes_and_rejects_empty_targets() -> None:
    model = FixedLogits()
    with pytest.raises(ValueError, match="valid targets"):
        evaluate_public_domains(
            model, {"test": Records([[999]], [[False]])}, device="cpu"
        )
    assert model.training


def test_private_trainer_evaluation_preserves_complete_training_state(
    tmp_path: Path,
) -> None:
    records = Records([[1, 2], [2, 3], [3, 4], [4, 0]], [[True, True]] * 4)
    config = GPTConfig(block_size=2, vocab_size=5, n_layer=1, n_head=1, n_embd=4)
    task = build_task(records, config, tokenizer_definition="public-tiny-fixture")
    def evaluate(model: nn.Module) -> dict[str, float]:
        return evaluate_public_domains(model, {"public": records}, device="cpu")

    task = replace(task, public_evaluate=evaluate)
    spec = describe_task(
        task,
        batch_size=2,
        optimizer=PrivateOptimizerSpec(
            name="adamw",
            torch_version=torch.__version__,
            parameter_names=tuple(
                n for n, _ in task.model_factory().named_parameters()
            ),
            learning_rate=1e-4,
            betas=(0.9, 0.999),
            numerical_epsilon=1e-8,
            weight_decay=0,
        ),
    )
    trainer = PrivateTrainer(
        task=task,
        policy=PrivacyPolicy(
            max_grad_norm=1,
            noise_multiplier=1,
            delta=1e-6,
            max_logical_steps=2,
            randomness_profile="reproducible_public_benchmark",
        ),
        specification=spec,
        run_id="evaluation-side-effects",
        local_data_identity="public-evaluation-fixture",
        reservation_store=PrivacyLedger.create(
            tmp_path / "ledger.json", lineage="test"
        ),
        max_physical_batch_size=1,
        research_seed=19,
    )
    trainer.train_local_steps(1)
    before = trainer.checkpoint_tensors()
    assert trainer.evaluate() is not None
    after = trainer.checkpoint_tensors()
    assert before.keys() == after.keys()
    assert all(np.array_equal(before[key], after[key]) for key in before)
