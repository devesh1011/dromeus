"""Node-local classification data, validated without sharing source records.

The supported developer interface is an immutable, deterministic, map-style
PyTorch dataset. Each item is an input/target pair. Inputs are FP32 tensors or
NumPy arrays; targets are finite integer-valued scalars. Random augmentation
belongs in the trainer after this boundary, not in dataset access.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sized
from dataclasses import dataclass
from numbers import Integral, Real
from operator import index as integer_index
from pathlib import Path
from typing import Annotated, Literal, cast

import numpy as np
import torch
from numpy.lib.npyio import NpzFile
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset, Subset

from dromeus.manifests.canonical import canonical_hash
from dromeus.manifests.models import ClassificationTaskContract


@dataclass(frozen=True, slots=True)
class LocalClassificationData:
    """Private node binding; its paths and source records never enter a manifest."""

    train_data: Dataset[object]
    evaluation_data: Dataset[object] | None
    label_names: tuple[str, ...]
    preprocessing_hash: str
    source_id: str = "local"


@dataclass(frozen=True, slots=True)
class ValidatedLocalData:
    """Normalized datasets and a content identity for local checkpoint binding."""

    train_data: Dataset[tuple[Tensor, int]]
    evaluation_data: Dataset[tuple[Tensor, int]] | None
    train_sample_count: int
    evaluation_sample_count: int | None
    training_label_counts: tuple[int, ...]
    evaluation_label_counts: tuple[int, ...] | None
    fingerprint: str
    source_id: str


def _dataset_length(data: object) -> int:
    if (
        not isinstance(data, Dataset)
        or isinstance(data, IterableDataset)
        or not isinstance(data, Sized)
    ):
        raise TypeError("local data requires a sized map-style PyTorch dataset")
    count = len(data)
    if count <= 0:
        raise ValueError("local datasets must contain at least one sample")
    return count


def _target_value(value: object, class_count: int) -> int:
    if isinstance(value, Tensor):
        if value.ndim != 0 or value.layout != torch.strided:
            raise ValueError("classification targets must be scalar class indices")
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("classification targets cannot be boolean")
    if isinstance(value, Integral):
        target = int(value)
    elif isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("classification targets must be finite integral values")
        target = int(numeric)
    else:
        raise ValueError("classification targets must be numeric scalar class indices")
    if not 0 <= target < class_count:
        raise ValueError("classification target is outside the declared label range")
    return target


def _normalize_sample(
    sample: object, contract: ClassificationTaskContract
) -> tuple[Tensor, int]:
    if not isinstance(sample, (tuple, list)):
        raise ValueError("local samples must be input/target pairs")
    values = cast(tuple[object, ...] | list[object], sample)
    if len(values) != 2:
        raise ValueError("local samples must be input/target pairs")
    inputs, target = values
    if isinstance(inputs, np.ndarray):
        array = cast(np.ndarray, inputs)
        if array.dtype != np.float32:
            raise ValueError("local inputs must have float32 dtype")
        inputs = torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
            array.copy()
        )
    if not isinstance(inputs, Tensor) or inputs.dtype != torch.float32:
        raise ValueError("local inputs must be float32 tensors or NumPy arrays")
    if inputs.layout != torch.strided or inputs.device.type == "meta":
        raise ValueError("local inputs require materialized dense tensors")
    if tuple(inputs.shape) != contract.input_shape:
        raise ValueError("local input shape does not match the shared task contract")
    if not bool(torch.isfinite(inputs).all().item()):
        raise ValueError("local inputs must contain only finite values")
    # A trainer may transform a sample in place; never hand out dataset storage.
    return (
        inputs.detach().cpu().contiguous().clone(),
        _target_value(target, contract.class_count),
    )


def _sample_digest(sample: tuple[Tensor, int]) -> bytes:
    inputs, target = sample
    digest = hashlib.sha256(b"dromeus.local-sample.v1\x00")
    digest.update(inputs.numpy().astype("<f4", copy=False).tobytes(order="C"))
    digest.update(struct.pack(">q", target))
    return digest.digest()


class _ValidatedDataset(Dataset[tuple[Tensor, int]]):
    def __init__(
        self,
        data: Dataset[object],
        contract: ClassificationTaskContract,
        sample_hashes: tuple[bytes, ...],
    ) -> None:
        self._data = data
        self._contract = contract
        self._sample_hashes = sample_hashes

    def __len__(self) -> int:
        if _dataset_length(self._data) != len(self._sample_hashes):
            raise ValueError("local dataset length changed after validation")
        return len(self._sample_hashes)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        count = len(self)
        if index < 0:
            index += count
        if not 0 <= index < count:
            raise IndexError(index)
        sample = _normalize_sample(self._data[index], self._contract)
        if _sample_digest(sample) != self._sample_hashes[index]:
            raise ValueError(
                "local dataset records or ordering changed after validation"
            )
        return sample


def _validate_dataset(
    data: Dataset[object], contract: ClassificationTaskContract
) -> tuple[_ValidatedDataset, tuple[int, ...], tuple[bytes, ...]]:
    count = _dataset_length(data)
    label_counts = [0] * contract.class_count
    sample_hashes: list[bytes] = []
    for index in range(count):
        sample = _normalize_sample(data[index], contract)
        label_counts[sample[1]] += 1
        sample_hashes.append(_sample_digest(sample))
    hashes = tuple(sample_hashes)
    normalized = _ValidatedDataset(data, contract, hashes)
    # Read twice before READY: reject stochastic transforms and changes observed
    # during preflight, then keep checking content and schema during training.
    for index in range(count):
        normalized[index]
    return normalized, tuple(label_counts), hashes


def _ordered_content_hash(hashes: tuple[bytes, ...]) -> bytes:
    digest = hashlib.sha256(b"dromeus.local-dataset.v1\x00")
    digest.update(struct.pack(">Q", len(hashes)))
    for sample_hash in hashes:
        digest.update(sample_hash)
    return digest.digest()


def _source_rows(
    data: Dataset[object],
) -> tuple[Dataset[object], frozenset[int] | None]:
    """Resolve ordinary and nested Subset views without reading source records.

    None denotes the complete source dataset. Custom dataset wrappers are opaque;
    their callers remain responsible for assigning disjoint held-out records.
    """
    source = data
    indices: tuple[int, ...] | None = None
    seen: set[int] = set()
    while isinstance(source, Subset):
        if id(source) in seen:
            raise ValueError("local subset dataset references must be acyclic")
        seen.add(id(source))
        subset = source
        indices = (
            tuple(integer_index(value) for value in subset.indices)
            if indices is None
            else tuple(integer_index(subset.indices[value]) for value in indices)
        )
        source = subset.dataset
    if indices is None:
        return source, None
    count = _dataset_length(source)
    return source, frozenset(value + count if value < 0 else value for value in indices)


def _shared_source_rows(train: Dataset[object], evaluation: Dataset[object]) -> bool:
    train_source, train_rows = _source_rows(train)
    evaluation_source, evaluation_rows = _source_rows(evaluation)
    if train_source is not evaluation_source:
        return False
    if train_rows is None or evaluation_rows is None:
        return True
    return bool(train_rows & evaluation_rows)


def validate_local_data(
    data: LocalClassificationData, contract: ClassificationTaskContract
) -> ValidatedLocalData:
    """Validate private data and semantics before formation readiness.

    Different nodes may omit classes and have unequal, tiny datasets. Evaluation
    must be explicitly absent or caller-designated, disjoint held-out data; no
    training-data fallback is supplied. Known shared Dataset/Subset source rows
    and identical complete collections reject. Arbitrary custom dataset ownership
    is opaque, so this validation does not certify statistical independence or
    reject individual matching observations across independently owned datasets.
    The fingerprint binds source identity, shared semantics, and ordered
    training/evaluation content, independent of local filesystem paths.
    """
    if data.label_names != contract.label_names:
        raise ValueError("local label meanings do not match the shared task contract")
    if data.preprocessing_hash != contract.preprocessing_hash:
        raise ValueError("local preprocessing does not match the shared task contract")
    if not data.source_id.strip():
        raise ValueError("local source_id must not be blank")
    if data.evaluation_data is data.train_data:
        raise ValueError("evaluation data must not be identical to training data")
    train, training_counts, train_hashes = _validate_dataset(data.train_data, contract)
    evaluation: _ValidatedDataset | None = None
    evaluation_counts: tuple[int, ...] | None = None
    evaluation_hashes: tuple[bytes, ...] | None = None
    if data.evaluation_data is not None:
        evaluation, evaluation_counts, evaluation_hashes = _validate_dataset(
            data.evaluation_data, contract
        )
        if _shared_source_rows(data.train_data, data.evaluation_data):
            raise ValueError("evaluation data overlaps training source records")
        if sorted(train_hashes) == sorted(evaluation_hashes):
            raise ValueError("evaluation data must not be identical to training data")
    digest = hashlib.sha256(b"dromeus.local-classification-binding.v1\x00")
    digest.update(canonical_hash(contract).encode("ascii"))
    source_id = data.source_id.encode("utf-8")
    digest.update(struct.pack(">Q", len(source_id)))
    digest.update(source_id)
    digest.update(_ordered_content_hash(train_hashes))
    digest.update(b"no-evaluation" if evaluation_hashes is None else b"held-out")
    if evaluation_hashes is not None:
        digest.update(_ordered_content_hash(evaluation_hashes))
    return ValidatedLocalData(
        train_data=train,
        evaluation_data=evaluation,
        train_sample_count=len(train_hashes),
        evaluation_sample_count=(
            None if evaluation_hashes is None else len(evaluation_hashes)
        ),
        training_label_counts=training_counts,
        evaluation_label_counts=evaluation_counts,
        fingerprint=digest.hexdigest(),
        source_id=data.source_id,
    )


class _NpzDataset(Dataset[object]):
    def __init__(self, inputs: np.ndarray, labels: np.ndarray) -> None:
        self._inputs = inputs.copy()
        self._labels = labels.copy()

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> object:
        return (
            torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
                self._inputs[index].copy()
            ),
            int(self._labels[index]),
        )


def load_npz_dataset(path: Path) -> Dataset[object]:
    """Read only inputs float32[N,*shape] and labels int64[N], without pickle."""
    loaded = np.load(path, allow_pickle=False)
    if not isinstance(loaded, NpzFile):
        raise ValueError("local dataset must be an NPZ archive")
    with loaded:
        if len(loaded.files) != 2 or set(loaded.files) != {"inputs", "labels"}:
            raise ValueError("local NPZ must contain exactly inputs and labels")
        inputs = cast(np.ndarray, loaded["inputs"])
        labels = cast(np.ndarray, loaded["labels"])
        if inputs.dtype != np.float32 or inputs.ndim < 2:
            raise ValueError("NPZ inputs must be float32 with shape [N, *input_shape]")
        if labels.dtype != np.int64 or labels.ndim != 1:
            raise ValueError("NPZ labels must be int64 with shape [N]")
        if inputs.shape[0] != labels.shape[0]:
            raise ValueError("NPZ input and label sample counts must match")
        return _NpzDataset(inputs, labels)


class _LocalBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    source_id: Annotated[str, StringConstraints(min_length=1)]
    label_names: tuple[Annotated[str, StringConstraints(min_length=1)], ...] = Field(
        min_length=2
    )
    preprocessing_hash: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    train_path: Annotated[str, StringConstraints(min_length=1)]
    evaluation_path: Annotated[str, StringConstraints(min_length=1)] | None

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_schema_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("local binding schema_version must be integer 1")
        return value


def _contained_path(root: Path, path: str) -> Path:
    relative = Path(path)
    if relative.is_absolute():
        raise ValueError(
            "local dataset paths must be relative to the binding directory"
        )
    resolved = (root / relative).resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(
            "local dataset paths must be files inside the binding directory"
        )
    return resolved


def load_local_data(path: Path) -> LocalClassificationData:
    """Load a closed, private JSON binding with explicit held-out evaluation policy."""
    binding = _LocalBinding.model_validate_json(path.read_bytes())
    if (
        not binding.source_id.strip()
        or any(not label.strip() for label in binding.label_names)
        or len(set(binding.label_names)) != len(binding.label_names)
    ):
        raise ValueError("local source and unique label names must not be blank")
    root = path.resolve(strict=True).parent
    train_path = _contained_path(root, binding.train_path)
    evaluation_path = (
        None
        if binding.evaluation_path is None
        else _contained_path(root, binding.evaluation_path)
    )
    if evaluation_path == train_path:
        raise ValueError("evaluation data must not be identical to training data")
    return LocalClassificationData(
        train_data=load_npz_dataset(train_path),
        evaluation_data=(
            None if evaluation_path is None else load_npz_dataset(evaluation_path)
        ),
        label_names=binding.label_names,
        preprocessing_hash=binding.preprocessing_hash,
        source_id=binding.source_id,
    )


__all__ = [
    "LocalClassificationData",
    "ValidatedLocalData",
    "load_local_data",
    "load_npz_dataset",
    "validate_local_data",
]
