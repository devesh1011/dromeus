from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset, Subset, TensorDataset

from dromeus.adapters.classification.data import (
    LocalClassificationData,
    load_local_data,
    load_npz_dataset,
    validate_local_data,
)
from dromeus.manifests.models import ClassificationTaskContract


class _Records(Dataset[object]):
    def __init__(self, values: list[object]) -> None:
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> object:
        return self.values[index]


def _contract() -> ClassificationTaskContract:
    return ClassificationTaskContract(
        dataset_id="local-classification-v1",
        input_shape=(2,),
        input_dtype="float32",
        label_names=("red", "green", "blue"),
        preprocessing_hash="a" * 64,
    )


def _records() -> _Records:
    return _Records(
        [
            (torch.tensor([1.0, 2.0]), 0),
            (torch.tensor([3.0, 4.0]), 0),
            (torch.tensor([5.0, 6.0]), 1),
        ]
    )


def _data(
    train: Dataset[object] | None = None, evaluation: Dataset[object] | None = None
) -> LocalClassificationData:
    task = _contract()
    return LocalClassificationData(
        train_data=_records() if train is None else train,
        evaluation_data=evaluation,
        label_names=task.label_names,
        preprocessing_hash=task.preprocessing_hash,
    )


def test_normalizes_real_pytorch_datasets_and_missing_classes() -> None:
    train = TensorDataset(
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        torch.tensor([0, 0, 1]),
    )
    evaluation = TensorDataset(torch.tensor([[7.0, 8.0]]), torch.tensor([2]))
    data = validate_local_data(_data(train, evaluation), _contract())

    assert data.train_sample_count == 3
    assert data.evaluation_sample_count == 1
    assert data.training_label_counts == (2, 1, 0)
    assert data.evaluation_label_counts == (0, 0, 1)
    inputs, target = data.train_data[0]
    assert isinstance(target, int)
    assert inputs.dtype == torch.float32
    assert inputs.shape == (2,)
    inputs.fill_(99)
    assert torch.equal(data.train_data[0][0], torch.tensor([1.0, 2.0]))
    assert len(data.fingerprint) == 64


def test_tiny_data_has_explicitly_unavailable_evaluation() -> None:
    data = validate_local_data(
        _data(_Records([(np.array([1.0, 2.0], dtype=np.float32), np.int64(1))])),
        _contract(),
    )
    assert data.train_sample_count == 1
    assert data.training_label_counts == (0, 1, 0)
    assert data.evaluation_data is None
    assert data.evaluation_sample_count is None
    assert data.evaluation_label_counts is None


@pytest.mark.parametrize("target", (0, np.int64(0), 0.0, torch.tensor(0.0)))
def test_integral_scalar_targets_normalize_to_python_int(target: object) -> None:
    data = validate_local_data(
        _data(_Records([(torch.tensor([1.0, 2.0]), target)])), _contract()
    )
    assert type(data.train_data[0][1]) is int
    assert data.train_data[0][1] == 0


@pytest.mark.parametrize(
    ("sample", "message"),
    (
        ({"input": [1.0, 2.0], "target": 0}, "input/target pairs"),
        ((torch.ones(2), 0, "extra"), "input/target pairs"),
        ((torch.ones(2, dtype=torch.float64), 0), "float32"),
        ((np.ones(2, dtype=np.float64), 0), "float32"),
        ((torch.ones(3), 0), "input shape"),
        ((torch.tensor([float("nan"), 1.0]), 0), "finite"),
        ((torch.tensor([float("inf"), 1.0]), 0), "finite"),
        ((torch.ones(2), True), "boolean"),
        ((torch.ones(2), 0.25), "integral"),
        ((torch.ones(2), float("nan")), "integral"),
        ((torch.ones(2), "1"), "numeric scalar"),
        ((torch.ones(2), torch.tensor([1])), "scalar class"),
        ((torch.ones(2), torch.tensor(1.25)), "integral"),
        ((torch.ones(2), -1), "label range"),
        ((torch.ones(2), 3), "label range"),
    ),
)
def test_actual_invalid_samples_reject_before_preparation(
    sample: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_local_data(_data(_Records([sample])), _contract())


@pytest.mark.parametrize("empty_split", ("train", "evaluation"))
def test_empty_provided_datasets_reject(empty_split: str) -> None:
    data = _data(
        train=_Records([]) if empty_split == "train" else _records(),
        evaluation=_Records([]) if empty_split == "evaluation" else None,
    )
    with pytest.raises(ValueError, match="at least one sample"):
        validate_local_data(data, _contract())


class _Stream(IterableDataset[object]):
    def __iter__(self) -> Iterator[object]:
        yield torch.ones(2), 0


def test_iterable_data_rejects() -> None:
    with pytest.raises(TypeError, match="map-style"):
        validate_local_data(_data(_Stream()), _contract())


@pytest.mark.parametrize("mismatch", ("labels", "preprocessing"))
def test_local_semantics_must_match_shared_contract(mismatch: str) -> None:
    data = _data()
    changed = (
        replace(data, label_names=("green", "red", "blue"))
        if mismatch == "labels"
        else replace(data, preprocessing_hash="b" * 64)
    )
    with pytest.raises(ValueError, match="shared task contract"):
        validate_local_data(changed, _contract())


@pytest.mark.parametrize("shared", ("object", "copy", "reordered"))
def test_training_records_cannot_be_reported_as_held_out(shared: str) -> None:
    train = _records()
    evaluation = train if shared == "object" else _records()
    if shared == "reordered":
        evaluation.values.reverse()
    with pytest.raises(ValueError, match="identical to training"):
        validate_local_data(_data(train, evaluation), _contract())


@pytest.mark.parametrize(
    "view", ("base-train", "base-evaluation", "siblings", "nested", "negative")
)
def test_evaluation_rejects_identifiable_subset_source_overlap(view: str) -> None:
    source = _records()
    train: Dataset[object] = source
    evaluation: Dataset[object] = Subset(source, [0, 1])
    if view == "base-evaluation":
        train, evaluation = evaluation, train
    elif view == "siblings":
        train = Subset(source, [0, 1])
        evaluation = Subset(source, [1, 2])
    elif view == "nested":
        train = Subset(Subset(source, [2, 0, 1]), [1, 2])
        evaluation = Subset(Subset(source, [1, 0, 2]), [0, 2])
    elif view == "negative":
        train = Subset(source, [0, -1])
        evaluation = Subset(source, [1, 2])
    with pytest.raises(ValueError, match="overlaps training source records"):
        validate_local_data(_data(train, evaluation), _contract())


def test_nested_disjoint_subsets_remain_valid_held_out_data() -> None:
    source = _records()
    train = Subset(Subset(source, [2, 0, 1]), [1, 2])
    evaluation = Subset(Subset(source, [1, 2, 0]), [1])
    validated = validate_local_data(_data(train, evaluation), _contract())
    assert validated.train_sample_count == 2
    assert validated.evaluation_sample_count == 1


def test_matching_observations_do_not_imply_shared_record_identity() -> None:
    train = _records()
    independent = _records()
    independent.values[1] = torch.tensor([8.0, 9.0]), 2
    # One observation has matching values, but the underlying sources are distinct.
    validated = validate_local_data(
        _data(Subset(train, [0, 1]), Subset(independent, [0, 1])), _contract()
    )
    assert validated.evaluation_sample_count == 2


def test_fingerprint_binds_order_content_semantics_source_and_evaluation() -> None:
    original = _data()
    task = _contract()
    fingerprint = validate_local_data(original, task).fingerprint
    assert fingerprint == validate_local_data(_data(), task).fingerprint
    reordered = _records()
    reordered.values.reverse()
    changed = _records()
    changed.values[0] = torch.tensor([1.0, 2.25]), 0
    relabeled = _records()
    relabeled.values[0] = torch.tensor([1.0, 2.0]), 1
    evaluation = _Records([(torch.tensor([8.0, 9.0]), 2)])
    for data in (
        _data(reordered),
        _data(changed),
        _data(relabeled),
        _data(evaluation=evaluation),
        replace(original, source_id="another-source"),
    ):
        assert validate_local_data(data, task).fingerprint != fingerprint
    new_task = task.model_copy(update={"preprocessing_hash": "b" * 64})
    assert (
        validate_local_data(
            replace(original, preprocessing_hash="b" * 64), new_task
        ).fingerprint
        != fingerprint
    )


@pytest.mark.parametrize("mutation", ("record", "length", "schema"))
def test_validated_dataset_rejects_changes_during_training(mutation: str) -> None:
    records = _records()
    data = validate_local_data(_data(records), _contract())
    if mutation == "record":
        records.values[0] = torch.tensor([2.0, 3.0]), 0
    elif mutation == "length":
        records.values.pop()
    else:
        records.values[0] = torch.ones(3), 0
    with pytest.raises(ValueError, match="changed|input shape"):
        data.train_data[0]


class _Nondeterministic(_Records):
    def __getitem__(self, index: int) -> object:
        self.values[index] = torch.rand(2), 0
        return self.values[index]


def test_nondeterministic_datasets_reject_during_preflight() -> None:
    with pytest.raises(ValueError, match="changed after validation"):
        validate_local_data(_data(_Nondeterministic([(torch.ones(2), 0)])), _contract())


def _write_npz(path: Path, offset: float = 0.0) -> None:
    np.savez(
        path,
        inputs=np.array([[1.0 + offset, 2.0], [3.0, 4.0]], dtype=np.float32),
        labels=np.array([0, 1], dtype=np.int64),
    )


def _binding(path: Path, **changes: object) -> Path:
    values: dict[str, object] = {
        "schema_version": 1,
        "source_id": "node-a",
        "label_names": list(_contract().label_names),
        "preprocessing_hash": "a" * 64,
        "train_path": "train.npz",
        "evaluation_path": None,
    }
    values.update(changes)
    path.write_text(json.dumps(values))
    return path


def test_npz_and_closed_json_binding_supply_local_data(tmp_path: Path) -> None:
    _write_npz(tmp_path / "train.npz")
    _write_npz(tmp_path / "held-out.npz", 0.25)
    binding_path = _binding(tmp_path / "binding.json", evaluation_path="held-out.npz")
    validated = validate_local_data(load_local_data(binding_path), _contract())
    assert validated.train_sample_count == 2
    assert validated.evaluation_sample_count == 2
    assert validated.source_id == "node-a"
    inputs, target = cast(
        tuple[Tensor, int], load_npz_dataset(tmp_path / "train.npz")[0]
    )
    assert torch.equal(inputs, torch.tensor([1.0, 2.0]))
    assert target == 0


@pytest.mark.parametrize(
    "changes",
    (
        {"schema_version": 2},
        {"schema_version": True},
        {"schema_version": 1.0},
        {"unknown": "field"},
        {"label_names": ["red", "red"]},
        {"source_id": " "},
        {"preprocessing_hash": "invalid"},
    ),
)
def test_closed_json_binding_rejects_invalid_configuration(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    _write_npz(tmp_path / "train.npz")
    with pytest.raises(ValueError):
        load_local_data(_binding(tmp_path / "binding.json", **changes))


def test_json_binding_requires_explicit_evaluation_policy(tmp_path: Path) -> None:
    path = _binding(tmp_path / "binding.json")
    contents = cast(dict[str, object], json.loads(path.read_text()))
    del contents["evaluation_path"]
    path.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="evaluation_path"):
        load_local_data(path)


@pytest.mark.parametrize("escape", ("absolute", "parent", "symlink"))
def test_binding_paths_cannot_escape_the_binding_directory(
    tmp_path: Path, escape: str
) -> None:
    outside = tmp_path / "outside.npz"
    _write_npz(outside)
    local = tmp_path / "node"
    local.mkdir()
    if escape == "absolute":
        path = str(outside)
    elif escape == "parent":
        path = "../outside.npz"
    else:
        (local / "linked.npz").symlink_to(outside)
        path = "linked.npz"
    with pytest.raises(ValueError, match="binding directory"):
        load_local_data(_binding(local / "binding.json", train_path=path))


@pytest.mark.parametrize(
    "invalid",
    ("extra-field", "input-dtype", "label-dtype", "label-shape", "count", "object"),
)
def test_npz_requires_exact_fields_shapes_and_safe_dtypes(
    tmp_path: Path, invalid: str
) -> None:
    arrays: dict[str, np.ndarray] = {
        "inputs": np.ones((2, 2), dtype=np.float32),
        "labels": np.array([0, 1], dtype=np.int64),
    }
    if invalid == "extra-field":
        arrays["extra"] = np.array([1])
    elif invalid == "input-dtype":
        arrays["inputs"] = np.ones((2, 2), dtype=np.float64)
    elif invalid == "label-dtype":
        arrays["labels"] = np.array([0, 1], dtype=np.float32)
    elif invalid == "label-shape":
        arrays["labels"] = np.array([[0], [1]], dtype=np.int64)
    elif invalid == "count":
        arrays["labels"] = np.array([0], dtype=np.int64)
    else:
        arrays["inputs"] = np.array([[object()]], dtype=object)
    path = tmp_path / "invalid.npz"
    if invalid == "extra-field":
        np.savez(
            path,
            inputs=arrays["inputs"],
            labels=arrays["labels"],
            extra=arrays["extra"],
        )
    else:
        np.savez(path, inputs=arrays["inputs"], labels=arrays["labels"])
    with pytest.raises(ValueError):
        load_npz_dataset(path)
