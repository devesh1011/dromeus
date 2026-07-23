from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from dromeus.training.pytorch import (
    CIFAR10Data,
    CIFAR10Trainer,
    CIFARDataError,
    create_initial_checkpoint,
)


@pytest.fixture(scope="session")
def cifar10_data() -> CIFAR10Data:
    cache_dir = Path(
        os.environ.get(
            "DROMEUS_CIFAR_CACHE",
            Path.home() / ".cache" / "dromeus" / "cifar10",
        )
    )
    if not cache_dir.exists():
        pytest.skip("real CIFAR-10 data unavailable in DROMEUS_CIFAR_CACHE")
    try:
        return CIFAR10Data.from_huggingface(
            cache_dir=cache_dir,
            train=False,
        )
    except CIFARDataError:
        pytest.skip(
            "real CIFAR-10 data unavailable; set DROMEUS_CIFAR_CACHE to writable cache"
        )


def test_huggingface_loader_returns_real_cifar10(cifar10_data: CIFAR10Data) -> None:
    image, label = cifar10_data[0]

    assert len(cifar10_data) == 10_000
    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert 0 <= label < 10


def test_iid_partitions_are_reproducible_and_disjoint(
    cifar10_data: CIFAR10Data,
) -> None:
    first = cifar10_data.split_iid(participant_count=4, seed=11)
    second = cifar10_data.split_iid(participant_count=4, seed=11)

    assert [[part[index][1] for index in range(len(part))] for part in first] == [
        [part[index][1] for index in range(len(part))] for part in second
    ]
    assert all(part[0][0].shape == (3, 32, 32) for part in first)
    assert len({part[index][1] for part in first for index in range(len(part))}) > 1


def test_checkpoint_is_deterministic_and_matches_trainer_schema(
    tmp_path: Path,
    cifar10_data: CIFAR10Data,
) -> None:
    first_path = tmp_path / "first.safetensors"
    second_path = tmp_path / "second.safetensors"

    first = create_initial_checkpoint(first_path, seed=17)
    second = create_initial_checkpoint(second_path, seed=17)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first.path == first_path
    assert first.tensor_schema == second.tensor_schema
    assert first.sha256 == second.sha256

    trainer = CIFAR10Trainer(
        train_data=cifar10_data,
        seed=17,
        batch_size=4,
    )
    trainer.load_checkpoint(first_path)
    assert trainer.tensor_schema == trainer.tensor_schema_for_model()
    assert trainer.checkpoint_hash(first_path) == trainer.checkpoint_hash(second_path)


def test_initial_checkpoint_returns_formation_handoff(tmp_path: Path) -> None:
    prepared = create_initial_checkpoint(tmp_path / "checkpoint.safetensors", seed=17)

    assert prepared.path.is_file()
    assert prepared.tensor_schema.tensors
    assert prepared.sha256 == create_initial_checkpoint(
        tmp_path / "checkpoint-2.safetensors", seed=17
    ).sha256


def test_trainer_runs_sgd_and_evaluates(cifar10_data: CIFAR10Data) -> None:
    trainer = CIFAR10Trainer(
        train_data=cifar10_data,
        seed=3,
        batch_size=4,
        learning_rate=0.05,
    )
    before = trainer.weights()

    trainer.train_local_steps(2)
    loss, accuracy = trainer.evaluate(cifar10_data)

    assert any(
        not np.array_equal(before[name], value)
        for name, value in trainer.weights().items()
    )
    assert np.isfinite(loss)
    assert 0.0 <= accuracy <= 1.0
