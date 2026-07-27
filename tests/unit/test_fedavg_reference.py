from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

from benchmarks.cifar10.fedavg_reference import (
    FedAvgConfig,
    FedAvgSeedInput,
    average_weights,
    run_fedavg,
    run_fedavg_seeds,
)
from dromeus.training.pytorch import CIFAR10Data, create_initial_checkpoint


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
    images = torch.zeros((8, 3, 32, 32), dtype=torch.float32)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    data = CIFAR10Data(
        cast(Dataset[tuple[torch.Tensor, int]], TensorDataset(images, labels))
    )
    partitions = data.split_iid(participant_count=4, seed=7)
    checkpoint = create_initial_checkpoint(tmp_path / "initial.safetensors", seed=17)

    result = run_fedavg(
        partitions=partitions,
        test_data=data,
        initial_checkpoint=checkpoint.path,
        config=FedAvgConfig(
            local_steps=1,
            round_count=2,
            learning_rate=0.01,
            batch_size=2,
            augment=False,
        ),
    )

    assert [round_result.round_id for round_result in result.rounds] == [0, 1]
    assert all(len(round_result.local_losses) == 4 for round_result in result.rounds)
    assert all(0.0 <= round_result.accuracy <= 1.0 for round_result in result.rounds)
    assert result.final_accuracy == result.rounds[-1].accuracy


def test_run_fedavg_seeds_requires_three_matching_frozen_configs(
    tmp_path: Path,
) -> None:
    images = torch.zeros((8, 3, 32, 32), dtype=torch.float32)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    data = CIFAR10Data(
        cast(Dataset[tuple[torch.Tensor, int]], TensorDataset(images, labels))
    )
    partitions = tuple(data.split_iid(participant_count=4, seed=7))
    inputs = tuple(
        FedAvgSeedInput(
            seed=seed,
            partitions=partitions,
            test_data=data,
            initial_checkpoint=create_initial_checkpoint(
                tmp_path / f"initial-{seed}.safetensors", seed=17
            ).path,
            config=FedAvgConfig(
                local_steps=1,
                round_count=1,
                learning_rate=0.01,
                trainer_seed=seed,
                augment=False,
            ),
        )
        for seed in (17, 23, 29)
    )

    results = run_fedavg_seeds(inputs)

    assert [seed for seed, _ in results] == [17, 23, 29]
    assert all(result.config is not None for _, result in results)
