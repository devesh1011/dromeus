"""Owned Poisson sampling for finite application datasets without eager reads."""

from __future__ import annotations

import math
from collections.abc import Sized
from typing import Any, cast

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler

from dromeus.training.privacy import opacus_module
from dromeus.training.task import TensorBatchAdapter


def dataset_size(dataset: Dataset[Any]) -> int:
    """Only metadata is consulted; never read a record during preparation."""
    if isinstance(dataset, IterableDataset) or not isinstance(dataset, Sized):
        raise ValueError("private task requires a finite indexed dataset")
    count = len(dataset)
    if count <= 0:
        raise ValueError("private task requires a nonempty fixed dataset")
    return count


class _FixedPopulation(Dataset[Any]):
    def __init__(self, dataset: Dataset[Any], count: int) -> None:
        self.source = dataset
        self.count = count

    def __len__(self) -> int:
        if dataset_size(self.source) != self.count:
            raise ValueError("private dataset population changed")
        return self.count

    def __getitem__(self, index: int) -> Any:
        if not 0 <= index < len(self):
            raise IndexError("private record index out of range")
        return self.source[index]


class TaskDataLoader(DataLoader[Any]):
    """An owned sampler and collator; no user-provided sampling distribution."""

    sample_rate: float
    population_size: int

    def __init__(
        self,
        dataset: Dataset[Any],
        adapter: TensorBatchAdapter,
        *,
        batch_size: int,
        generator: torch.Generator | None,
    ) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("logical batch size must be a positive integer")
        self.population_size = dataset_size(dataset)
        steps = math.ceil(self.population_size / batch_size)
        self.sample_rate = 1 / steps
        sampler_type = opacus_module(
            "opacus.utils.uniform_sampler"
        ).UniformWithReplacementSampler
        sampler = sampler_type(
            num_samples=self.population_size,
            sample_rate=self.sample_rate,
            generator=generator,
            steps=steps,
        )
        super().__init__(  # pyright: ignore[reportUnknownMemberType]
            _FixedPopulation(dataset, self.population_size),
            batch_sampler=cast(Sampler[list[int]], sampler),
            collate_fn=adapter.collate,
            generator=generator,
            num_workers=0,
        )

    def check_population(self) -> None:
        if len(cast(Sized, self.dataset)) != self.population_size:
            raise ValueError("private dataset population changed")

    def __iter__(self) -> Any:
        self.check_population()
        return super().__iter__()


def make_task_loader(
    dataset: Dataset[Any],
    adapter: TensorBatchAdapter,
    *,
    batch_size: int,
    generator: torch.Generator | None,
) -> TaskDataLoader:
    """Use with PrivacyEngine(poisson_sampling=False): sampling is already owned."""
    return TaskDataLoader(dataset, adapter, batch_size=batch_size, generator=generator)
