"""Actual owned private digit execution and paired reference semantics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.m3.dp_digits import DigitPrivateWorkload, execute, make_draft
from benchmarks.m3.matched_digits import SOURCES, make_baseline, model_definition
from dromeus.manifests.canonical import file_sha256


def data_at(root: Path) -> Path:
    root.mkdir()
    rng = np.random.default_rng(3)
    for source in SOURCES:
        for split, count in (("train", 8), ("test", 2)):
            np.savez(
                root / f"{source}-{split}.npz",
                inputs=rng.random((count, 1, 32, 32), dtype=np.float32),
                labels=np.arange(count, dtype=np.int64) % 10,
            )
    return root


def test_private_digit_execution_reconciles_and_composes(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    data = data_at(tmp_path / "data")
    ledger = tmp_path / "ledger.json"
    reference = make_draft("a" * 40, run_id="dp-test-one", steps=2, local_steps=1)
    baseline = make_baseline(
        reference, data=data, rank=0, device="cpu", seed=17, physical=4
    )
    initial = baseline.trainer.create_initial_checkpoint(
        tmp_path / "baseline-init.safetensors", model_definition=model_definition(17)
    )
    for index in (1, 2):
        draft = make_draft("a" * 40, run_id=f"dp-test-{index}", steps=2, local_steps=1)
        root = tmp_path / f"run-{index}"
        execute(
            mode="local",
            reference=draft,
            data=data,
            root=root,
            rank=0,
            device="cpu",
            seed=17,
            physical=4,
            ledger_path=ledger,
            create_ledger=index == 1,
            evaluation_interval=1,
            train_count=8,
            test_count=2,
        )
        result = json.loads((root / "private.json").read_text())
        assert result["accounting"]["logical_steps"] == 2
        assert result["physical_steps"] == 4
        assert result["sample_rate"] == baseline.statistics["sample_rate"] == 1
        assert result["initial_checkpoint_sha256"] == initial.sha256
        assert result["model_sha256"] == file_sha256(root / "private-model.safetensors")
        assert len(result["reservations"]) == index
        assert len(result["prior_reservations"]) == index - 1
        assert len(result["history"]) == 3
        assert result["history"][-1]["accuracy_macro_source"] >= 0
        if index == 2:
            assert (
                result["cumulative_reserved_prv_epsilon"]
                > result["accounting"]["privacy_epsilon"]
            )
    with pytest.raises(ValueError, match="fresh output"):
        execute(
            mode="local",
            reference=reference,
            data=data,
            root=tmp_path / "run-1",
            rank=0,
            device="cpu",
            seed=17,
            physical=4,
            ledger_path=ledger,
        )


def test_private_digit_missing_ledger_and_counts_fail_closed(tmp_path: Path) -> None:
    data = data_at(tmp_path / "data")
    reference = make_draft("a" * 40, run_id="dp-test", steps=1, local_steps=1)
    ledger = tmp_path / "ledger.json"
    with pytest.raises(ValueError, match="sample counts"):
        DigitPrivateWorkload(
            reference,
            data=data,
            rank=0,
            device="cpu",
            seed=17,
            physical=4,
            ledger_path=ledger,
            create_ledger=True,
        )
    assert not ledger.exists()
    with pytest.raises(ValueError, match="ledger missing"):
        DigitPrivateWorkload(
            reference,
            data=data,
            rank=0,
            device="cpu",
            seed=17,
            physical=4,
            ledger_path=ledger,
            train_count=8,
            test_count=2,
        )
    assert not ledger.exists()


def test_private_reference_matches_pilot_settings() -> None:
    draft = make_draft("a" * 40, run_id="bounded-public-pilot")
    assert draft.round_count == 75 and draft.local_steps == 10
    assert draft.private_training is not None and draft.privacy is not None
    assert draft.private_training.schedule == "linear_decay"
    assert draft.private_training.batch_size == 64
    assert draft.privacy.max_logical_steps == 750
    assert draft.privacy.randomness_profile == "reproducible_public_benchmark"
