"""Application-owned task semantics shared by ordinary and private training.

Callbacks are trusted application code. They may construct the forward/loss
computation, but private execution owns sampling, backward and optimizer steps.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from torch import nn
from torch.utils.data import Dataset

from dromeus.manifests.models import TaskTrainingSpec
from dromeus.training.optimizer_factory import PrivateOptimizerFactory

Batch = Any
CollateRecords = Callable[[Sequence[Any]], Batch]
RecordLoss = Callable[[nn.Module, Batch], torch.Tensor]


def _collate_tensors(records: Sequence[Any]) -> Batch:
    """Stack tensor records without changing tuple/list semantics."""
    first = records[0]
    if isinstance(first, torch.Tensor):
        if not all(isinstance(item, torch.Tensor) for item in records):
            raise ValueError("record tensor structure changed")
        return torch.stack(cast(list[torch.Tensor], list(records)))
    if type(first) is dict:
        keys = tuple(cast(dict[str, Any], first))
        if not all(type(item) is dict and tuple(item) == keys for item in records):
            raise ValueError("record mapping structure changed")
        return {key: _collate_tensors([item[key] for item in records]) for key in keys}
    if type(first) in (tuple, list):
        count = len(first)
        if not all(
            type(item) is type(first) and len(item) == count for item in records
        ):
            raise ValueError("record sequence structure changed")
        values = [_collate_tensors([item[i] for item in records]) for i in range(count)]
        return tuple(values) if type(first) is tuple else values
    raise ValueError("records must be tensor trees; constants need a task adapter")


def _map_tensors(
    value: Batch, operation: Callable[[torch.Tensor], torch.Tensor]
) -> Batch:
    if isinstance(value, torch.Tensor):
        return operation(value)
    if type(value) is dict:
        return {
            k: _map_tensors(v, operation)
            for k, v in cast(dict[str, Any], value).items()
        }
    mapped = [_map_tensors(item, operation) for item in value]
    return tuple(mapped) if type(value) is tuple else mapped


def _batch_tensors(value: Batch, *, path: str = "batch") -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            raise ValueError(f"{path}: tensor requires record axis 0")
        if value.layout != torch.strided or value.is_complex():
            raise ValueError(f"{path}: dense real tensors required")
        if not torch.isfinite(value).all():
            raise ValueError(f"{path}: nonfinite batch tensor")
        return [value]
    if type(value) is dict:
        mapping = cast(dict[Any, Any], value)
        if not mapping or not all(type(k) is str for k in mapping):
            raise ValueError(f"{path}: nonempty string-keyed mapping required")
        leaves: list[torch.Tensor] = []
        for key, item in mapping.items():
            leaves.extend(_batch_tensors(item, path=f"{path}.{key}"))
        return leaves
    if type(value) in (tuple, list) and len(value):
        leaves = []
        for index, item in enumerate(value):
            leaves.extend(_batch_tensors(item, path=f"{path}[{index}]"))
        return leaves
    raise ValueError(
        f"{path}: only tensor trees supported; declare constants in task code"
    )


def _schema(value: Batch) -> Any:
    if isinstance(value, torch.Tensor):
        return ("tensor", str(value.dtype), tuple(value.shape[1:]))
    if type(value) is dict:
        return (
            "dict",
            tuple((k, _schema(v)) for k, v in cast(dict[str, Any], value).items()),
        )
    return (type(value).__name__, tuple(_schema(item) for item in value))


@dataclass(frozen=True)
class TensorBatchAdapter:
    """Fixed trailing shapes and record axis 0 for nested tensor batches.

    ``example`` is an explicitly public/synthetic batch of at least two records.
    It describes empty batches without touching the protected dataset. Collation
    must preserve each selected record and may not use cross-record statistics.
    Scalars and non-batched constants belong in trusted task code for this profile.
    """

    example: Batch
    definition: str
    collate_fn: CollateRecords = _collate_tensors
    _schema: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.definition.strip():
            raise ValueError("batch definition is required")
        tensors = _batch_tensors(self.example)
        count = len(tensors[0])
        if count < 2 or any(len(tensor) != count for tensor in tensors):
            raise ValueError(
                "public batch fixture requires at least two aligned records"
            )
        copied = _map_tensors(self.example, lambda t: t.detach().clone())
        object.__setattr__(self, "example", copied)
        object.__setattr__(self, "_schema", _schema(copied))

    @property
    def definition_hash(self) -> str:
        """Bind declared semantics and tensor schema, never public probe values."""
        payload = json.dumps(
            {"definition": self.definition, "schema": self._schema},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def size(self, batch: Batch) -> int:
        tensors = _batch_tensors(batch)
        if _schema(batch) != self._schema:
            raise ValueError("batch structure, dtype or trailing shape changed")
        count = len(tensors[0])
        if any(len(tensor) != count for tensor in tensors):
            raise ValueError("batch tensors disagree on record axis")
        return count

    def collate(self, records: Sequence[Any]) -> Batch:
        batch = self.collate_fn(records) if records else self.empty()
        if self.size(batch) != len(records):
            raise ValueError("collation changed the sampled record count")
        return batch

    def to_device(self, batch: Batch, device: torch.device | str) -> Batch:
        self.size(batch)
        return _map_tensors(batch, lambda t: t.to(device))

    def slice(self, batch: Batch, start: int, end: int) -> Batch:
        count = self.size(batch)
        if not 0 <= start <= end <= count:
            raise ValueError("invalid physical batch record range")
        return _map_tensors(batch, lambda t: t[start:end])

    def empty(self) -> Batch:
        return _map_tensors(self.example, lambda t: t[:0].detach().clone())


def validate_record_losses(losses: object, count: int) -> torch.Tensor:
    """Check the application returns one finite differentiable loss per record."""
    if (
        not isinstance(losses, torch.Tensor)
        or losses.shape != (count,)
        or not losses.is_floating_point()
        or not torch.isfinite(losses).all()
        or (count > 0 and not losses.requires_grad)
    ):
        raise ValueError(
            "task loss requires finite differentiable per-record values [B]"
        )
    return losses


@dataclass(frozen=True)
class TrainingTask:
    """Small application description; no private gradient mechanism lives here."""

    model_factory: Callable[[], nn.Module]
    dataset: Dataset[Any]
    batch_adapter: TensorBatchAdapter
    loss_per_record: RecordLoss
    model_definition: str
    task_definition: str
    loss_definition: str
    implementation_hash: str
    optimizer_registration: PrivateOptimizerFactory | None = None
    record_unit: str = "example"
    public_evaluate: Callable[[nn.Module], Mapping[str, float]] | None = None

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (
                self.model_definition,
                self.task_definition,
                self.loss_definition,
                self.record_unit,
            )
        ):
            raise ValueError(
                "model, task, loss and protected record definitions required"
            )
        if re.fullmatch(r"[0-9a-f]{64}", self.implementation_hash) is None:
            raise ValueError("task implementation requires a SHA-256 identity")

    def check_definition(self, specification: TaskTrainingSpec) -> None:
        """Bind trusted stateless task callbacks to the group's declared semantics.

        Stateful task transforms need a separately versioned state contract.
        Hash agreement does not attest the behavior of arbitrary Python code.
        """
        expected = {
            "task": (
                hashlib.sha256(self.task_definition.encode()).hexdigest(),
                specification.task_definition_hash,
            ),
            "implementation": (
                self.implementation_hash,
                specification.implementation_hash,
            ),
            "batch": (
                self.batch_adapter.definition_hash,
                specification.batch_definition_hash,
            ),
            "loss": (
                hashlib.sha256(self.loss_definition.encode()).hexdigest(),
                specification.loss_definition_hash,
            ),
            "record unit": (self.record_unit, specification.record_unit),
        }
        for name, (actual, declared) in expected.items():
            if actual != declared:
                raise ValueError(f"task {name} definition differs from shared contract")
