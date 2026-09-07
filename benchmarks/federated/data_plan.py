"""Deterministic local-data plans, separate from the frozen M2 CIFAR matrix.

The first four features carry class signal; the last four are nuisance features.
Every profile uses the same class meanings, centers, and noise distribution.
Feature skew changes only the nuisance features. Each node's evaluation data is an
independent held-out draw from its local distribution; the explicitly declared
common evaluation set is balanced and has no node-specific domain transformation.

Version 1 uses 128 training and 64 held-out examples per node. Quantity skew repeats
the counts 32, 64, 128, and 256 across successive four-node groups. Moderate skew
draws the node's dominant class with probability 0.7 and each other class with
probability 0.1. Severe skew supplies only the node's dominant class. Random
streams are derived independently by seed, role, rank, and purpose. In particular,
the common evaluation set is identical across profiles and world sizes at a seed.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

PROFILES = (
    "iid",
    "label-skew-moderate",
    "label-skew-severe",
    "quantity-skew",
    "feature-skew",
)
LABEL_NAMES = ("class-0", "class-1", "class-2", "class-3")
INPUT_SHAPE = (8,)
_TRAIN_COUNT = 128
_EVALUATION_COUNT = 64
_COMMON_EVALUATION_COUNT = 512
_PLAN_FILENAME = "data-plan.json"


class DataPlanError(ValueError):
    """A data plan or one of its bound local datasets is invalid."""


@dataclass(frozen=True, slots=True)
class DatasetBinding:
    """A verified local NPZ file and its retained data statistics."""

    path: Path
    sha256: str
    sample_count: int
    label_histogram: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NodeDataPlan:
    rank: int
    train: DatasetBinding
    evaluation: DatasetBinding


@dataclass(frozen=True, slots=True)
class DataPlan:
    """Immutable selection; hashes and paths must be revalidated before a run."""

    path: Path
    sha256: str
    version: int
    profile: str
    seed: int
    world_size: int
    label_names: tuple[str, ...]
    input_shape: tuple[int, ...]
    nodes: tuple[NodeDataPlan, ...]
    common_evaluation: DatasetBinding


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class _DatasetSpec(_ClosedModel):
    path: str
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    sample_count: Annotated[int, Field(gt=0)]
    label_histogram: tuple[Annotated[int, Field(ge=0)], ...]


class _NodeSpec(_ClosedModel):
    rank: Annotated[int, Field(ge=0)]
    train: _DatasetSpec
    evaluation: _DatasetSpec


class _PlanSpec(_ClosedModel):
    version: Literal[1]
    profile: Literal[
        "iid",
        "label-skew-moderate",
        "label-skew-severe",
        "quantity-skew",
        "feature-skew",
    ]
    seed: Annotated[int, Field(ge=0, lt=2**64)]
    world_size: Literal[4, 8, 16]
    label_names: tuple[str, ...]
    input_shape: tuple[int, ...]
    nodes: tuple[_NodeSpec, ...]
    common_evaluation: _DatasetSpec

    @field_validator("version", "world_size", mode="before")
    @classmethod
    def integer_selections(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("version and world_size must be integers")
        return value


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _rng(seed: int, role: str, rank: int, purpose: str) -> np.random.Generator:
    identity = json.dumps(
        ["federated-data-v1", seed, role, rank, purpose], separators=(",", ":")
    ).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(identity).digest()))


def _train_count(profile: str, rank: int) -> int:
    return 32 * 2 ** (rank % 4) if profile == "quantity-skew" else _TRAIN_COUNT


def _dataset(
    *,
    profile: str,
    seed: int,
    world_size: int,
    rank: int,
    role: str,
    sample_count: int,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    label_rng = _rng(seed, role, rank, "labels")
    if profile == "label-skew-severe":
        labels = np.full(sample_count, rank % 4, dtype=np.int64)
    elif profile == "label-skew-moderate":
        probabilities = np.full(4, 0.1)
        probabilities[rank % 4] = 0.7
        labels = label_rng.choice(4, size=sample_count, p=probabilities).astype(
            np.int64
        )
    else:
        labels = np.tile(np.arange(4, dtype=np.int64), sample_count // 4)
    label_rng.shuffle(labels)
    inputs = (
        _rng(seed, role, rank, "features")
        .normal(0.0, 0.7, size=(sample_count, 8))
        .astype(np.float32)
    )
    inputs[:, :4] -= np.float32(1.0)
    inputs[np.arange(sample_count), labels] += np.float32(3.5)
    if profile == "feature-skew":
        shift = np.float32(-3.0 + 6.0 * rank / (world_size - 1))
        direction = np.array([1.0, -0.5, 0.75, -1.0], dtype=np.float32)
        inputs[:, 4:] *= np.float32(0.6 + 0.15 * (rank % 4))
        inputs[:, 4:] += shift * direction
    return inputs, labels


def _npz_bytes(inputs: NDArray[np.float32], labels: NDArray[np.int64]) -> bytes:
    """Write stable ZIP headers instead of wall-clock-dependent archive metadata."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in (("inputs", inputs), ("labels", labels)):
            payload = io.BytesIO()
            np.save(payload, array, allow_pickle=False)
            entry = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_STORED
            entry.create_system = 3
            entry.external_attr = 0o600 << 16
            archive.writestr(entry, payload.getvalue())
    return output.getvalue()


def _materialize(
    root: Path,
    relative_path: str,
    *,
    inputs: NDArray[np.float32],
    labels: NDArray[np.int64],
) -> _DatasetSpec:
    payload = _npz_bytes(inputs, labels)
    destination = root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(payload)
    return _DatasetSpec(
        path=relative_path,
        sha256=_sha256(payload),
        sample_count=len(labels),
        label_histogram=tuple(int(count) for count in np.bincount(labels, minlength=4)),
    )


def create_data_plan(
    root: Path, *, profile: str, seed: int, world_size: int = 4
) -> DataPlan:
    """Create an isolated, reproducible plan without overwriting retained datasets.

    Repeating the exact selection at an existing complete plan revalidates it and
    returns it. An incomplete or different selection must use a fresh directory.
    """
    if profile not in PROFILES:
        raise DataPlanError(f"unsupported data profile: {profile!r}")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise DataPlanError("seed must be an integer in [0, 2**64)")
    if type(world_size) is not int or world_size not in (4, 8, 16):
        raise DataPlanError("world_size must be 4, 8, or 16")
    root = root.resolve()
    path = root / _PLAN_FILENAME
    if path.is_symlink():
        raise DataPlanError("data plan must not be a symbolic link")
    if path.exists():
        retained = load_data_plan(path)
        if (retained.profile, retained.seed, retained.world_size) != (
            profile,
            seed,
            world_size,
        ):
            raise DataPlanError("existing plan has a different selection")
        return retained
    destinations = [root / "common-evaluation.npz", path]
    destinations.extend(
        root / f"node-{rank}" / filename
        for rank in range(world_size)
        for filename in ("train.npz", "evaluation.npz")
    )
    if any(
        destination.exists() or destination.is_symlink() for destination in destinations
    ):
        raise DataPlanError("incomplete data plan exists; use a fresh directory")
    if any((root / f"node-{rank}").is_symlink() for rank in range(world_size)):
        raise DataPlanError("node data directories must not be symbolic links")
    root.mkdir(parents=True, exist_ok=True)
    nodes: list[_NodeSpec] = []
    for rank in range(world_size):
        bindings: dict[str, _DatasetSpec] = {}
        for role, sample_count in (
            ("train", _train_count(profile, rank)),
            ("evaluation", _EVALUATION_COUNT),
        ):
            inputs, labels = _dataset(
                profile=profile,
                seed=seed,
                world_size=world_size,
                rank=rank,
                role=role,
                sample_count=sample_count,
            )
            bindings[role] = _materialize(
                root, f"node-{rank}/{role}.npz", inputs=inputs, labels=labels
            )
        nodes.append(
            _NodeSpec(
                rank=rank, train=bindings["train"], evaluation=bindings["evaluation"]
            )
        )
    inputs, labels = _dataset(
        profile="iid",
        seed=seed,
        world_size=world_size,
        rank=-1,
        role="common-evaluation",
        sample_count=_COMMON_EVALUATION_COUNT,
    )
    common = _materialize(root, "common-evaluation.npz", inputs=inputs, labels=labels)
    spec = _PlanSpec(
        version=1,
        profile=profile,
        seed=seed,
        world_size=world_size,
        label_names=LABEL_NAMES,
        input_shape=INPUT_SHAPE,
        nodes=tuple(nodes),
        common_evaluation=common,
    )
    payload = json.dumps(spec.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"
    with path.open("x") as stream:
        stream.write(payload)
    return load_data_plan(path)


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise DataPlanError(f"duplicate JSON field: {key}")
        value[key] = item
    return value


def _validate_binding(
    root: Path, spec: _DatasetSpec, *, expected_path: str, expected_count: int
) -> DatasetBinding:
    if spec.path != expected_path:
        raise DataPlanError(f"dataset path must be {expected_path}")
    path = root / spec.path
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError) as error:
        raise DataPlanError("dataset path cannot be safely resolved") from error
    if not resolved.is_relative_to(root) or resolved != path:
        raise DataPlanError("dataset path escapes its root or uses a symbolic link")
    if spec.sample_count != expected_count:
        raise DataPlanError("dataset sample count does not match the profile")
    if len(spec.label_histogram) != 4 or sum(spec.label_histogram) != expected_count:
        raise DataPlanError("invalid declared label histogram")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise DataPlanError(f"cannot read dataset: {spec.path}") from error
    if _sha256(payload) != spec.sha256:
        raise DataPlanError(f"dataset hash mismatch: {spec.path}")
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if sorted(archive.files) != ["inputs", "labels"]:
                raise DataPlanError("NPZ fields must be exactly inputs and labels")
            inputs = archive["inputs"]
            labels = archive["labels"]
            if inputs.dtype != np.dtype(np.float32) or inputs.shape != (
                expected_count,
                8,
            ):
                raise DataPlanError("inputs must be float32 with declared shape [N, 8]")
            if labels.dtype != np.dtype(np.int64) or labels.shape != (expected_count,):
                raise DataPlanError("labels must be int64 with declared shape [N]")
            if not np.isfinite(inputs).all():
                raise DataPlanError("inputs contain non-finite values")
            if np.any(labels < 0) or np.any(labels >= 4):
                raise DataPlanError("labels must be in the shared range [0, 4)")
            histogram = tuple(int(count) for count in np.bincount(labels, minlength=4))
            if histogram != spec.label_histogram:
                raise DataPlanError(
                    "dataset label histogram does not match its binding"
                )
    except (
        OSError,
        EOFError,
        ValueError,
        TypeError,
        AttributeError,
        zipfile.BadZipFile,
    ) as error:
        raise DataPlanError(f"invalid dataset {spec.path}: {error}") from error
    return DatasetBinding(
        path=resolved,
        sha256=spec.sha256,
        sample_count=spec.sample_count,
        label_histogram=spec.label_histogram,
    )


def load_data_plan(path: Path) -> DataPlan:
    """Validate closed metadata, safe paths, file hashes, and actual array schemas."""
    if path.is_symlink():
        raise DataPlanError("data plan must not be a symbolic link")
    path = path.resolve()
    try:
        payload = path.read_bytes()
        json.loads(payload, object_pairs_hook=_unique_fields)
        spec = _PlanSpec.model_validate_json(payload)
    except (OSError, ValueError, ValidationError) as error:
        raise DataPlanError(f"invalid data plan: {error}") from error
    if spec.label_names != LABEL_NAMES or spec.input_shape != INPUT_SHAPE:
        raise DataPlanError(
            "plan does not match the shared version-1 classification task"
        )
    if tuple(node.rank for node in spec.nodes) != tuple(range(spec.world_size)):
        raise DataPlanError("nodes must contain every rank exactly once in rank order")
    nodes = tuple(
        NodeDataPlan(
            rank=node.rank,
            train=_validate_binding(
                path.parent,
                node.train,
                expected_path=f"node-{node.rank}/train.npz",
                expected_count=_train_count(spec.profile, node.rank),
            ),
            evaluation=_validate_binding(
                path.parent,
                node.evaluation,
                expected_path=f"node-{node.rank}/evaluation.npz",
                expected_count=_EVALUATION_COUNT,
            ),
        )
        for node in spec.nodes
    )
    common = _validate_binding(
        path.parent,
        spec.common_evaluation,
        expected_path="common-evaluation.npz",
        expected_count=_COMMON_EVALUATION_COUNT,
    )
    if common.label_histogram != (128, 128, 128, 128):
        raise DataPlanError("common evaluation must contain 128 examples of each class")
    if spec.profile == "label-skew-severe":
        for node in nodes:
            for binding in (node.train, node.evaluation):
                if binding.label_histogram[node.rank % 4] != binding.sample_count:
                    raise DataPlanError(
                        "severe label skew must contain only the local class"
                    )
    return DataPlan(
        path=path,
        sha256=_sha256(payload),
        version=spec.version,
        profile=spec.profile,
        seed=spec.seed,
        world_size=spec.world_size,
        label_names=spec.label_names,
        input_shape=spec.input_shape,
        nodes=nodes,
        common_evaluation=common,
    )
