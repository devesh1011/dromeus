"""Clipping-only diagnostics are ordinary, reproducible public-data experiments."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_matched_baseline import digit_reference
from torch import nn

from benchmarks.m3.clipped_digits import (
    ClippedDiagnostic,
    ClippedRecipe,
    diagnostic_draft,
    execute,
    recipe_for,
)
from benchmarks.m3.matched_baseline import BaselineRecipe, MatchedBaseline
from benchmarks.m3.matched_digits import SOURCES

pytest.importorskip("opacus")


def model() -> nn.Linear:
    result = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        result.weight.fill_(1)
    return result


def diagnostic(
    *,
    clipping: float = 1,
    batch: int = 2,
    physical: int = 1,
    seed: int = 17,
    steps: int = 12,
    adam: bool = False,
) -> ClippedDiagnostic:
    return ClippedDiagnostic(
        model_factory=model,
        optimizer_factory=lambda m: (
            torch.optim.Adam(m.parameters(), lr=0.1)
            if adam
            else torch.optim.SGD(m.parameters(), lr=0.1)
        ),
        inputs=torch.tensor([[1.0], [2.0]]),
        targets=torch.zeros(2, 1),
        recipe=ClippedRecipe(
            batch_size=batch,
            logical_steps=steps,
            loss="mse",
            optimizer_definition="Adam:.1" if adam else "SGD:.1",
            schedule="linear_decay" if adam else "constant",
            clipping=clipping,
        ),
        sampling_seed=seed,
        local_data_identity="public-pair",
        max_physical_batch_size=physical,
    )


def test_per_example_clipping_changes_analytic_update() -> None:
    clipped = diagnostic()
    clipped.trainer.train_local_steps(1)
    # Gradients 2 and 8 each clip to ~1; divide sum by expected batch 2.
    expected = 1 - 0.1 * ((2 / (2 + 1e-6)) + (8 / (8 + 1e-6))) / 2
    np.testing.assert_allclose(
        clipped.trainer.weights()["weight"], [[expected]], atol=1e-7
    )
    assert clipped.statistics["logical_steps"] == 1
    assert clipped.statistics["physical_steps"] == 2
    assert clipped.statistics["epsilon"] is None
    assert clipped.statistics["privacy_claim"] is False


def test_large_clip_matches_unclipped_poisson_trajectory() -> None:
    clipped = diagnostic(clipping=1e6, batch=1)
    baseline = MatchedBaseline(
        model_factory=model,
        optimizer_factory=lambda m: torch.optim.SGD(m.parameters(), lr=0.1),
        inputs=torch.tensor([[1.0], [2.0]]),
        targets=torch.zeros(2, 1),
        recipe=BaselineRecipe(
            batch_size=1, logical_steps=12, loss="mse", optimizer_definition="SGD:.1"
        ),
        sampling_seed=17,
        local_data_identity="public-pair",
        max_physical_batch_size=1,
    )
    for _ in range(12):
        clipped.trainer.train_local_steps(1)
        baseline.trainer.train_local_steps(1)
        np.testing.assert_allclose(
            clipped.trainer.weights()["weight"],
            baseline.trainer.weights()["weight"],
            atol=1e-7,
        )
    for key in ("empty_draws", "physical_steps", "logical_steps", "sampled_examples"):
        assert clipped.statistics[key] == baseline.statistics[key]
    empty = clipped.statistics["empty_draws"]
    assert isinstance(empty, int) and empty > 0


def test_checkpoint_restores_sampler_optimizer_and_split_counts() -> None:
    continuous = diagnostic(batch=1, adam=True)
    continuous.trainer.train_local_steps(5)
    snapshot = continuous.trainer.checkpoint_tensors()
    continuous.trainer.train_local_steps(7)
    restored = diagnostic(batch=1, adam=True)
    restored.trainer.load_checkpoint_tensors(snapshot)
    restored.trainer.train_local_steps(7)
    assert restored.statistics == continuous.statistics
    for key, value in continuous.trainer.checkpoint_tensors().items():
        np.testing.assert_array_equal(restored.trainer.checkpoint_tensors()[key], value)
    with pytest.raises(ValueError, match="identity"):
        diagnostic(clipping=2, batch=1, adam=True).trainer.load_checkpoint_tensors(
            snapshot
        )


def test_digit_diagnostic_contract_and_export(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    reference = digit_reference()
    recipe = recipe_for(reference, evaluation_interval=1)
    draft = diagnostic_draft(reference, recipe, run_id="clipped-test")
    assert draft.privacy is None and draft.private_training is None
    assert draft.application_training is not None
    assert draft.application_training.policy_id == "public-clipping-diagnostic-v1"
    assert (
        recipe.training_definition
        != recipe.model_copy(update={"clipping": 2}).training_definition
    )
    data = tmp_path / "data"
    data.mkdir()
    x = np.full((4, 1, 32, 32), 0.5, dtype=np.float32)
    y = np.array([0, 1, 2, 3], dtype=np.int64)
    # The rank only owns MNIST training data, plus the public test datasets.
    np.savez(data / "mnist-train.npz", inputs=x, labels=y)
    for source in SOURCES:
        np.savez(data / f"{source}-test.npz", inputs=x, labels=y)
    root = tmp_path / "run"
    execute(
        mode="local",
        reference=reference,
        data=data,
        root=root,
        rank=0,
        device="cpu",
        seed=17,
        physical=2,
        run_id="clipped-test",
        evaluation_interval=1,
    )
    summary = json.loads((root / "clipped.json").read_text())
    assert summary["arm"] == "non_private_clipping_diagnostic"
    assert summary["epsilon"] is None
    assert summary["logical_steps"] == 2
    assert summary["physical_steps"] == 4
    assert summary["noise_multiplier"] == 0
    assert len(summary["history"]) == 3
    assert (root / "clipped-model.safetensors").is_file()
    assert not (root / "privacy-ledger").exists()
