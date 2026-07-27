from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest
import torch
from support.sample_manifest import manifest_data
from torch.utils.data import Dataset, TensorDataset

from benchmarks.cifar10.fedavg_reference import (
    FedAvgConfig,
    FedAvgSeedInput,
    average_weights,
    run_fedavg,
    run_fedavg_seeds,
)
from dromeus.manifests.models import SealedManifest
from dromeus.training.pytorch import (
    CIFAR10Data,
    CIFARDataProvenance,
    create_initial_checkpoint,
)


def _config(
    *,
    seed: int = 17,
    partition_seed: int = 7,
    round_count: int = 1,
) -> FedAvgConfig:
    data = manifest_data()
    dataset = data["dataset"]
    dataset["iid_partition_seed"] = partition_seed
    dataset["sample_count"] = 8
    dataset["partition_sample_counts"] = [2, 2, 2, 2]
    manifest = SealedManifest.model_validate(data)
    base = FedAvgConfig.from_manifest(
        manifest,
        trainer_seed=seed,
        batch_size=2,
        augment=False,
    )
    return replace(
        base,
        local_steps=1,
        round_count=round_count,
        learning_rate=0.01,
        data_source="test-fixture",
        test_sample_count=8,
    )


def _data(
    *,
    split: Literal["train", "test"],
    source: str = "test-fixture",
) -> CIFAR10Data:
    images = torch.zeros((8, 3, 32, 32), dtype=torch.float32)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    return CIFAR10Data(
        cast(Dataset[tuple[torch.Tensor, int]], TensorDataset(images, labels)),
        _source_provenance=CIFARDataProvenance(
            source=source,
            split=split,
        ),
    )


def test_average_weights_returns_equal_central_model() -> None:
    averaged = average_weights(
        [
            {"weight": np.array([1.0, 3.0], dtype=np.float32)},
            {"weight": np.array([3.0, 5.0], dtype=np.float32)},
            {"weight": np.array([5.0, 7.0], dtype=np.float32)},
            {"weight": np.array([7.0, 9.0], dtype=np.float32)},
        ]
    )

    assert np.array_equal(averaged["weight"], np.array([4.0, 6.0], dtype=np.float32))


def test_run_fedavg_reuses_four_partitions_and_common_test_set(tmp_path: Path) -> None:
    data = _data(split="train")
    test_data = _data(split="test")
    partitions = data.split_iid(participant_count=4, seed=7)
    checkpoint = create_initial_checkpoint(tmp_path / "initial.safetensors", seed=17)

    result = run_fedavg(
        partitions=partitions,
        test_data=test_data,
        initial_checkpoint=checkpoint.path,
        config=_config(round_count=2),
    )

    assert [round_result.round_id for round_result in result.rounds] == [0, 1]
    assert all(len(round_result.local_losses) == 4 for round_result in result.rounds)
    assert all(0.0 <= round_result.accuracy <= 1.0 for round_result in result.rounds)
    assert result.final_accuracy == result.rounds[-1].accuracy


def test_run_fedavg_seeds_requires_three_matching_frozen_configs(
    tmp_path: Path,
) -> None:
    data = _data(split="train")
    test_data = _data(split="test")
    partitions = tuple(data.split_iid(participant_count=4, seed=7))
    inputs = tuple(
        FedAvgSeedInput(
            seed=seed,
            partitions=partitions,
            test_data=test_data,
            initial_checkpoint=create_initial_checkpoint(
                tmp_path / f"initial-{seed}.safetensors", seed=17
            ).path,
            config=_config(seed=seed),
        )
        for seed in (17, 23, 29)
    )

    results = run_fedavg_seeds(inputs)

    assert [seed for seed, _ in results] == [17, 23, 29]
    assert all(result.config is not None for _, result in results)


def test_run_fedavg_rejects_partitions_from_different_seed(tmp_path: Path) -> None:
    data = _data(split="train")
    test_data = _data(split="test")
    partitions = data.split_iid(participant_count=4, seed=7)
    checkpoint = create_initial_checkpoint(tmp_path / "initial.safetensors", seed=17)

    with pytest.raises(ValueError, match="partition provenance"):
        run_fedavg(
            partitions=partitions,
            test_data=test_data,
            initial_checkpoint=checkpoint.path,
            config=_config(partition_seed=8),
        )


def test_run_fedavg_rejects_fixture_claiming_official_torchvision_source(
    tmp_path: Path,
) -> None:
    data = _data(split="train", source="torchvision-cifar10")
    test_data = _data(split="test", source="torchvision-cifar10")
    checkpoint = create_initial_checkpoint(tmp_path / "initial.safetensors", seed=17)

    with pytest.raises(ValueError, match="test data source"):
        run_fedavg(
            partitions=data.split_iid(participant_count=4, seed=7),
            test_data=test_data,
            initial_checkpoint=checkpoint.path,
            config=replace(_config(), data_source="torchvision-cifar10"),
        )
