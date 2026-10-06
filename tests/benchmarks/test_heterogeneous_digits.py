"""Pilot data semantics and matched local workload, without network access."""

import bz2
from pathlib import Path

import numpy as np
import pytest

from benchmarks.m3.heterogeneous_data import SOURCES, read_usps, select_indices
from benchmarks.m3.heterogeneous_training import DigitTraining, make_draft


def test_source_label_mapping_and_balanced_disjoint_selection(tmp_path: Path) -> None:
    path = tmp_path / "usps.bz2"
    path.write_bytes(bz2.compress(b"1 1:-1 2:1\n10 1:1 2:-1\n"))
    x, y = read_usps(path)
    np.testing.assert_array_equal(y, [0, 9])
    assert x.shape == (2, 1, 16, 16)
    assert x[0, 0, 0, 0] == 0 and x[0, 0, 0, 1] == 1
    labels = np.repeat(np.arange(10), 5)
    indices = select_indices(labels, per_class=3, seed=17)
    assert len(set(indices)) == 30
    np.testing.assert_array_equal(np.bincount(labels[indices]), [3] * 10)
    with pytest.raises(ValueError):
        select_indices(labels, per_class=6, seed=17)


def test_node_uses_only_own_training_source_and_baseline_state_matches(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(17)
    x = rng.random((80, 1, 32, 32), dtype=np.float32)
    y = np.arange(80, dtype=np.int64) % 10
    np.savez(tmp_path / "mnist-train.npz", inputs=x, labels=y)
    for name in SOURCES:
        np.savez(tmp_path / f"{name}-test.npz", inputs=x[:10], labels=y[:10])
    first = DigitTraining(tmp_path, 0, "cpu")
    second = DigitTraining(tmp_path, 0, "cpu")
    for key, value in first.trainer.weights().items():
        np.testing.assert_array_equal(value, second.trainer.weights()[key])
    first.trainer.train_local_steps(2)
    second.trainer.train_local_steps(2)
    for key, value in first.trainer.weights().items():
        np.testing.assert_array_equal(value, second.trainer.weights()[key])
    checkpoint = first.trainer.checkpoint_tensors()
    first.trainer.train_local_steps(1)
    second.trainer.load_checkpoint_tensors(checkpoint)
    second.trainer.train_local_steps(1)
    for key, value in first.trainer.weights().items():
        np.testing.assert_array_equal(value, second.trainer.weights()[key])
    assert first.trainer.evaluate() is not None
    assert make_draft("abcdef0").privacy is None
