"""The pinned nanoGPT architecture trains through a generic record-loss task."""

import copy
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from dromeus.training.private_model import validate_private_model
from examples.nanogpt_training import (
    GPT,
    GPTConfig,
    MMapTokenDataset,
    build_task,
    per_record_cross_entropy,
)


def tiny_config() -> GPTConfig:
    return GPTConfig(block_size=4, vocab_size=11, n_layer=1, n_head=2, n_embd=8)


def upstream_logits(model: GPT, tokens: torch.Tensor) -> torch.Tensor:
    """Upstream's unbatched position lookup, independent of adapted GPT.forward."""
    position = torch.arange(tokens.shape[1], device=tokens.device)
    hidden = model.transformer["wte"](tokens) + model.transformer["wpe"](position)
    for block in cast(nn.ModuleList, model.transformer["h"]):
        hidden = block(hidden)
    return model.lm_head(model.transformer["ln_f"](hidden))


def test_batched_positions_preserve_upstream_logits_and_tied_gradients() -> None:
    torch.set_rng_state(torch.Generator().manual_seed(5).get_state())
    model = GPT(tiny_config())
    reference = copy.deepcopy(model)
    tokens = torch.tensor([[1, 3, 5, 7], [2, 4, 6, 8]])
    targets = torch.tensor([[3, 5, 7, 9], [4, 6, 8, 10]])
    logits, loss = model(tokens, targets)
    expected = upstream_logits(reference, tokens)
    torch.testing.assert_close(logits, expected, rtol=1e-6, atol=1e-7)
    assert loss is not None
    expected_loss = F.cross_entropy(expected.flatten(0, 1), targets.flatten())
    torch.testing.assert_close(loss, expected_loss)
    actual_grad = torch.autograd.grad(loss, tuple(model.parameters()))
    expected_grad = torch.autograd.grad(expected_loss, tuple(reference.parameters()))
    for actual, wanted in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=1e-5, atol=1e-7)
    embedding = cast(nn.Embedding, model.transformer["wte"])
    assert embedding.weight is model.lm_head.weight


def test_sequence_loss_uses_each_records_mask_and_all_masked_is_zero() -> None:
    model = GPT(tiny_config())
    batch = {
        "tokens": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]),
        "targets": torch.tensor([[2, 3, 4, 5], [3, 2, 1, 0]]),
        "mask": torch.tensor([[True, False, True, False], [False] * 4]),
    }
    losses = per_record_cross_entropy(model, batch)
    logits, _ = model(batch["tokens"])
    expected = (
        F.cross_entropy(logits[0, 0].unsqueeze(0), torch.tensor([2]))
        + F.cross_entropy(logits[0, 2].unsqueeze(0), torch.tensor([4]))
    ) / 2
    torch.testing.assert_close(losses[0], expected)
    assert losses[1].item() == 0
    assert losses.requires_grad
    gradients = torch.autograd.grad(losses[1], tuple(model.parameters()))
    assert all(torch.count_nonzero(value).item() == 0 for value in gradients)


def test_mmap_records_are_fixed_sequences_and_task_passes_hook_parity(
    tmp_path: Path,
) -> None:
    pytest.importorskip("opacus")
    path = tmp_path / "records.npy"
    np.save(path, np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], dtype=np.uint16))
    records = MMapTokenDataset(path, sequence_length=4, vocab_size=11)
    assert len(records) == 2
    assert torch.equal(records[0]["tokens"], torch.tensor([1, 2, 3, 4]))
    assert torch.equal(records[0]["targets"], torch.tensor([2, 3, 4, 5]))
    task = build_task(
        records, tiny_config(), tokenizer_definition="public-test-tokenizer"
    )
    model = task.model_factory()
    report = validate_private_model(
        model, task.loss_per_record, task.batch_adapter, task.batch_adapter.example
    )
    assert report.parameter_aliases == (("lm_head.weight", "transformer.wte.weight"),)
