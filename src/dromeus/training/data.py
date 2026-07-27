"""Dataset views and deterministic IID partitioning."""

from __future__ import annotations

import hashlib
from collections.abc import Sized
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class IIDPartitionProvenance:
    """Identity of one deterministic IID split member."""

    seed: int
    participant_count: int
    partition_index: int
    source_sample_count: int
    indices_sha256: str


@dataclass(frozen=True, slots=True)
class DataProvenance:
    """Trusted dataset source and split identity."""

    source: str
    split: Literal["train", "test"]


@dataclass(frozen=True, slots=True)
class ClassificationData(Dataset[tuple[Tensor, int]]):
    """Indexed view over a tensor classification dataset."""

    _dataset: Dataset[tuple[Tensor, int]]
    _indices: tuple[int, ...] | None = None
    _partition_provenance: IIDPartitionProvenance | None = None
    _source_provenance: DataProvenance | None = None

    def split_iid(
        self,
        *,
        participant_count: int,
        seed: int,
    ) -> tuple[ClassificationData, ...]:
        """Return equal, disjoint, deterministic IID partitions."""
        index_groups = iid_partition_indices(
            source_sample_count=len(self),
            participant_count=participant_count,
            seed=seed,
        )
        return tuple(
            ClassificationData(
                self._dataset,
                indices,
                IIDPartitionProvenance(
                    seed=seed,
                    participant_count=participant_count,
                    partition_index=partition_index,
                    source_sample_count=len(self),
                    indices_sha256=_indices_hash(indices),
                ),
                self._source_provenance,
            )
            for partition_index, indices in enumerate(index_groups)
        )

    @property
    def partition_provenance(self) -> IIDPartitionProvenance | None:
        return self._partition_provenance

    @property
    def source_provenance(self) -> DataProvenance | None:
        return self._source_provenance

    def matches_source(
        self,
        *,
        source: str,
        split: Literal["train", "test"],
    ) -> bool:
        """Return whether this view came from the declared trusted loader."""
        return self._source_provenance == DataProvenance(source=source, split=split)

    def __len__(self) -> int:
        return (
            len(self._indices)
            if self._indices is not None
            else len(cast(Sized, self._dataset))
        )

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        dataset_index = self._indices[index] if self._indices is not None else index
        image, label = self._dataset[dataset_index]
        return image, int(label)


def iid_partition_indices(
    *,
    source_sample_count: int,
    participant_count: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    """Return the canonical deterministic IID index groups."""
    if source_sample_count <= 0 or participant_count <= 0:
        raise ValueError("sample and participant counts must be positive")
    if source_sample_count % participant_count:
        raise ValueError("sample count must divide participant count exactly")
    order = np.random.default_rng(seed).permutation(source_sample_count)
    return tuple(
        tuple(int(index) for index in indices)
        for indices in np.array_split(order, participant_count)
    )


def iid_partition_index_hashes(
    *,
    source_sample_count: int,
    participant_count: int,
    seed: int,
) -> tuple[str, ...]:
    """Return stable identities for the canonical IID index groups."""
    return tuple(
        _indices_hash(indices)
        for indices in iid_partition_indices(
            source_sample_count=source_sample_count,
            participant_count=participant_count,
            seed=seed,
        )
    )


def _indices_hash(indices: tuple[int, ...]) -> str:
    values = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


__all__ = [
    "ClassificationData",
    "DataProvenance",
    "IIDPartitionProvenance",
    "iid_partition_index_hashes",
    "iid_partition_indices",
]
