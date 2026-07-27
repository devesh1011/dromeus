from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import dromeus.training.cifar10 as cifar10_recipe
from dromeus.training.cifar10 import (
    DATASET_REPOSITORY,
    DATASET_REVISION,
    DATASET_VERSION,
    PREPROCESSING_DEFINITION,
    PREPROCESSING_HASH,
    CIFAR10DataError,
    create_initial_checkpoint,
    create_trainer,
    load_cifar10,
)
from dromeus.training.data import ClassificationData, IIDPartitionProvenance
from dromeus.training.trainer import derive_benchmark_seed


def test_cifar_contract_constants_are_canonical() -> None:
    assert DATASET_VERSION == f"huggingface-{DATASET_REVISION}"
    assert PREPROCESSING_HASH == hashlib.sha256(
        PREPROCESSING_DEFINITION.encode()
    ).hexdigest()


def test_benchmark_seed_concerns_are_stable_and_separate() -> None:
    values = {
        derive_benchmark_seed(17, purpose)
        for purpose in (
            "model-initialization",
            "local-training",
            "consensus-sketch",
        )
    }

    assert len(values) == 3
    assert derive_benchmark_seed(17, "local-training") == derive_benchmark_seed(
        17, "local-training"
    )


@pytest.fixture(scope="session")
def cifar10_data() -> ClassificationData:
    cache_dir = Path(
        os.environ.get(
            "DROMEUS_CIFAR_CACHE",
            Path.home() / ".cache" / "dromeus" / "cifar10",
        )
    )
    if not cache_dir.exists():
        pytest.skip("real CIFAR-10 data unavailable in DROMEUS_CIFAR_CACHE")
    try:
        return load_cifar10(
            cache_dir=cache_dir,
            train=False,
        )
    except CIFAR10DataError:
        pytest.skip(
            "real CIFAR-10 data unavailable; set DROMEUS_CIFAR_CACHE to writable cache"
        )


def test_huggingface_loader_returns_real_cifar10(
    cifar10_data: ClassificationData,
) -> None:
    image, label = cifar10_data[0]

    assert len(cifar10_data) == 10_000
    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert 0 <= label < 10


def test_huggingface_loader_pins_source_revision_and_decodes_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeDataset:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, object]:
            assert index == 0
            return {"img": Image.new("RGB", (32, 32), "red"), "label": 3}

    def fake_load_dataset(repository: str, **kwargs: object) -> FakeDataset:
        calls.append({"repository": repository, **kwargs})
        return FakeDataset()

    monkeypatch.setattr(cifar10_recipe, "load_dataset", fake_load_dataset)

    data = load_cifar10(cache_dir=tmp_path, train=False)
    image, label = data[0]

    assert calls == [
        {
            "repository": DATASET_REPOSITORY,
            "split": "test",
            "revision": DATASET_REVISION,
            "cache_dir": str(tmp_path),
        }
    ]
    assert data.matches_source(
        source="huggingface-uoft-cs-cifar10",
        split="test",
    )
    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert label == 3


def test_iid_partitions_are_reproducible_and_disjoint(
    cifar10_data: ClassificationData,
) -> None:
    first = cifar10_data.split_iid(participant_count=4, seed=11)
    second = cifar10_data.split_iid(participant_count=4, seed=11)

    assert [[part[index][1] for index in range(len(part))] for part in first] == [
        [part[index][1] for index in range(len(part))] for part in second
    ]
    assert all(part[0][0].shape == (3, 32, 32) for part in first)
    assert len({part[index][1] for part in first for index in range(len(part))}) > 1
    provenance = [part.partition_provenance for part in first]
    assert all(isinstance(value, IIDPartitionProvenance) for value in provenance)
    assert [
        (value.seed, value.participant_count, value.partition_index)
        for value in provenance
        if value
    ] == [(11, 4, index) for index in range(4)]
    assert all(
        value and value.source_sample_count == len(cifar10_data) for value in provenance
    )
    assert len({value.indices_sha256 for value in provenance if value}) == 4


def test_checkpoint_is_deterministic_and_matches_trainer_schema(
    tmp_path: Path,
    cifar10_data: ClassificationData,
) -> None:
    first_path = tmp_path / "first.safetensors"
    second_path = tmp_path / "second.safetensors"

    first = create_initial_checkpoint(first_path, seed=17)
    second = create_initial_checkpoint(second_path, seed=17)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first.path == first_path
    assert first.tensor_schema == second.tensor_schema
    assert first.sha256 == second.sha256

    trainer = create_trainer(
        train_data=cifar10_data,
        seed=17,
        batch_size=4,
    )
    trainer.load_checkpoint(first_path)
    assert trainer.tensor_schema == first.tensor_schema
    assert trainer.checkpoint_hash(first_path) == trainer.checkpoint_hash(second_path)


def test_initial_checkpoint_returns_formation_handoff(tmp_path: Path) -> None:
    prepared = create_initial_checkpoint(tmp_path / "checkpoint.safetensors", seed=17)

    assert prepared.path.is_file()
    assert prepared.tensor_schema.tensors
    assert (
        prepared.sha256
        == create_initial_checkpoint(
            tmp_path / "checkpoint-2.safetensors", seed=17
        ).sha256
    )


def test_trainer_runs_sgd_and_evaluates(
    cifar10_data: ClassificationData,
) -> None:
    trainer = create_trainer(
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


def test_resnet_trainer_uses_momentum_schedule_and_full_float_state(
    cifar10_data: ClassificationData,
) -> None:
    trainer = create_trainer(
        train_data=cifar10_data,
        seed=3,
        batch_size=4,
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        learning_rate_milestones=(1,),
        learning_rate_gamma=0.1,
        crop_padding=4,
        normalize=True,
    )
    before = trainer.weights()

    trainer.train_local_steps(2)
    checkpoint_state = trainer.checkpoint_tensors()

    assert trainer.learning_rate == pytest.approx(0.01)
    assert int(checkpoint_state["__dromeus_training__.completed_steps"][0]) == 2
    assert any(
        name.startswith("__dromeus_training__.momentum.")
        for name in checkpoint_state
    )
    assert any(name.endswith("running_mean") for name in before)
    assert any(
        name.endswith("running_mean")
        and not np.array_equal(before[name], trainer.weights()[name])
        for name in before
    )

    restored = create_trainer(
        train_data=cifar10_data,
        seed=99,
        batch_size=4,
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        learning_rate_milestones=(1,),
        learning_rate_gamma=0.1,
        crop_padding=4,
        normalize=True,
    )
    restored.load_checkpoint_tensors(checkpoint_state)

    assert restored.learning_rate == pytest.approx(0.01)
    assert all(
        np.array_equal(value, restored.weights()[name])
        for name, value in trainer.weights().items()
    )

    trainer.train_local_steps(1)
    restored.train_local_steps(1)

    assert all(
        np.array_equal(value, restored.weights()[name])
        for name, value in trainer.weights().items()
    )
