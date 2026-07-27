from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from dromeus.training.pytorch import (
    CIFAR10_ARCHIVE_MD5,
    CIFAR10_DATASET_VERSION,
    CIFAR_RESNET32_MODEL_ID,
    CIFAR_RESNET32_PREPROCESSING_DEFINITION,
    CIFAR_RESNET32_PREPROCESSING_HASH,
    PREPROCESSING_DEFINITION,
    PREPROCESSING_HASH,
    CIFAR10Data,
    CIFAR10Trainer,
    CIFARDataError,
    IIDPartitionProvenance,
    build_model,
    create_initial_checkpoint,
    derive_benchmark_seed,
    tensor_schema_for_model,
)


def test_cifar_contract_constants_are_canonical() -> None:
    assert CIFAR10_DATASET_VERSION == f"torchvision-python-{CIFAR10_ARCHIVE_MD5}"
    assert PREPROCESSING_HASH == hashlib.sha256(
        PREPROCESSING_DEFINITION.encode()
    ).hexdigest()
    assert CIFAR_RESNET32_PREPROCESSING_HASH == hashlib.sha256(
        CIFAR_RESNET32_PREPROCESSING_DEFINITION.encode()
    ).hexdigest()


def test_resnet32_v2_has_capacity_and_exchanges_batchnorm_state() -> None:
    model = build_model(seed=17, model_id=CIFAR_RESNET32_MODEL_ID)
    state_names = {tensor.name for tensor in tensor_schema_for_model(model).tensors}

    assert sum(parameter.numel() for parameter in model.parameters()) >= 450_000
    assert "stages.0.0.bn1.running_mean" in state_names
    assert "stages.0.0.bn1.num_batches_tracked" not in state_names


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
    assert (
        prepared.sha256
        == create_initial_checkpoint(
            tmp_path / "checkpoint-2.safetensors", seed=17
        ).sha256
    )


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


def test_resnet_trainer_uses_momentum_schedule_and_full_float_state(
    cifar10_data: CIFAR10Data,
) -> None:
    trainer = CIFAR10Trainer(
        train_data=cifar10_data,
        seed=3,
        model_id=CIFAR_RESNET32_MODEL_ID,
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

    restored = CIFAR10Trainer(
        train_data=cifar10_data,
        seed=99,
        model_id=CIFAR_RESNET32_MODEL_ID,
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
