from __future__ import annotations

from pathlib import Path

import numpy as np

from dromeus.training.pytorch import (
    CIFAR10Data,
    CIFAR10Trainer,
    create_initial_checkpoint,
)


def test_iid_partitions_are_reproducible_and_disjoint() -> None:
    data = CIFAR10Data.synthetic(sample_count=40, seed=4)

    first = data.split_iid(participant_count=4, seed=11)
    second = data.split_iid(participant_count=4, seed=11)

    assert [part.labels.tolist() for part in first] == [
        part.labels.tolist() for part in second
    ]
    assert all(part.images.shape == (10, 3, 32, 32) for part in first)
    assert len({int(label) for part in first for label in part.labels}) > 1


def test_checkpoint_is_deterministic_and_matches_trainer_schema(tmp_path: Path) -> None:
    first_path = tmp_path / "first.safetensors"
    second_path = tmp_path / "second.safetensors"

    create_initial_checkpoint(first_path, seed=17)
    create_initial_checkpoint(second_path, seed=17)

    assert first_path.read_bytes() == second_path.read_bytes()

    trainer = CIFAR10Trainer(
        train_data=CIFAR10Data.synthetic(sample_count=8, seed=2),
        seed=17,
        batch_size=4,
    )
    trainer.load_checkpoint(first_path)
    assert trainer.tensor_schema == trainer.tensor_schema_for_model()
    assert trainer.checkpoint_hash(first_path) == trainer.checkpoint_hash(second_path)


def test_trainer_runs_sgd_and_evaluates() -> None:
    data = CIFAR10Data.synthetic(sample_count=16, seed=8)
    trainer = CIFAR10Trainer(train_data=data, seed=3, batch_size=4, learning_rate=0.05)
    before = trainer.weights()

    trainer.train_local_steps(2)
    loss, accuracy = trainer.evaluate(data)

    assert any(
        not np.array_equal(before[name], value)
        for name, value in trainer.weights().items()
    )
    assert np.isfinite(loss)
    assert 0.0 <= accuracy <= 1.0
