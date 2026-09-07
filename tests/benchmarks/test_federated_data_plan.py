from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from benchmarks.federated.data_plan import (
    LABEL_NAMES,
    PROFILES,
    create_data_plan,
    load_data_plan,
)


@pytest.mark.parametrize("profile", PROFILES)
def test_data_plan_replays_identical_bytes_with_independent_local_data(
    tmp_path: Path, profile: str
) -> None:
    first = create_data_plan(tmp_path / "a", profile=profile, seed=17)
    second = create_data_plan(tmp_path / "b", profile=profile, seed=17)
    assert first.sha256 == second.sha256
    assert first.path.read_bytes() == second.path.read_bytes()
    assert first.label_names == LABEL_NAMES
    assert first.input_shape == (8,)
    assert len({node.train.sha256 for node in first.nodes}) == 4
    assert first.common_evaluation.label_histogram == (128, 128, 128, 128)
    assert first.common_evaluation.sample_count == 512
    for node, replayed in zip(first.nodes, second.nodes, strict=True):
        assert node.train.path.read_bytes() == replayed.train.path.read_bytes()
        assert (
            node.evaluation.path.read_bytes() == replayed.evaluation.path.read_bytes()
        )
        assert node.train.sha256 != node.evaluation.sha256
        assert node.train.path.parent.name == f"node-{node.rank}"
        assert node.evaluation.sample_count == 64
        with np.load(node.train.path, allow_pickle=False) as arrays:
            assert set(arrays.files) == {"inputs", "labels"}
            assert arrays["inputs"].dtype == np.float32
            assert arrays["inputs"].shape == (node.train.sample_count, 8)
            assert arrays["labels"].dtype == np.int64
            assert arrays["labels"].shape == (node.train.sample_count,)
    assert create_data_plan(tmp_path / "a", profile=profile, seed=17) == first
    assert load_data_plan(first.path) == first
    with pytest.raises(FrozenInstanceError):
        setattr(first.nodes[0].train, "sample_count", 7)


@pytest.mark.parametrize("world_size", (4, 8, 16))
def test_data_plan_declares_all_ranks_and_unequal_positive_counts(
    tmp_path: Path, world_size: int
) -> None:
    plan = create_data_plan(
        tmp_path, profile="quantity-skew", seed=41, world_size=world_size
    )
    assert tuple(node.rank for node in plan.nodes) == tuple(range(world_size))
    assert [node.train.sample_count for node in plan.nodes] == [32, 64, 128, 256] * (
        world_size // 4
    )
    for node in plan.nodes:
        assert len(set(node.train.label_histogram)) == 1


def test_profiles_isolate_skew_and_share_the_common_evaluation(tmp_path: Path) -> None:
    plans = {
        profile: create_data_plan(tmp_path / profile, profile=profile, seed=29)
        for profile in PROFILES
    }
    assert len({plan.sha256 for plan in plans.values()}) == len(PROFILES)
    assert len({plan.common_evaluation.sha256 for plan in plans.values()}) == 1
    for node in plans["iid"].nodes:
        assert node.train.label_histogram == (32, 32, 32, 32)
    for node in plans["label-skew-moderate"].nodes:
        assert 0.55 <= node.train.label_histogram[node.rank % 4] / 128 <= 0.85
        assert min(node.train.label_histogram) > 0
    for node in plans["label-skew-severe"].nodes:
        for binding in (node.train, node.evaluation):
            assert binding.label_histogram.count(0) == 3
            assert binding.label_histogram[node.rank % 4] == binding.sample_count
    for iid, skewed in zip(
        plans["iid"].nodes, plans["feature-skew"].nodes, strict=True
    ):
        with (
            np.load(iid.train.path, allow_pickle=False) as control,
            np.load(skewed.train.path, allow_pickle=False) as domain,
        ):
            np.testing.assert_array_equal(control["labels"], domain["labels"])
            np.testing.assert_array_equal(
                control["inputs"][:, :4], domain["inputs"][:, :4]
            )
            assert not np.array_equal(control["inputs"][:, 4:], domain["inputs"][:, 4:])


def test_seed_changes_datasets_without_overwriting_an_existing_selection(
    tmp_path: Path,
) -> None:
    first = create_data_plan(tmp_path / "first", profile="iid", seed=17)
    other = create_data_plan(tmp_path / "other", profile="iid", seed=29)
    assert first.sha256 != other.sha256
    assert first.nodes[0].train.sha256 != other.nodes[0].train.sha256
    before = first.path.read_bytes()
    with pytest.raises(ValueError, match="different selection"):
        create_data_plan(tmp_path / "first", profile="iid", seed=29)
    assert first.path.read_bytes() == before


@pytest.mark.parametrize("world_size", (0, 1, 3, 5, 32, True))
def test_data_plan_rejects_invalid_membership(tmp_path: Path, world_size: int) -> None:
    with pytest.raises(ValueError, match="world_size"):
        create_data_plan(tmp_path, profile="iid", seed=17, world_size=world_size)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("seed", (-1, 2**64, True))
def test_data_plan_rejects_invalid_seed(tmp_path: Path, seed: int) -> None:
    with pytest.raises(ValueError, match="seed"):
        create_data_plan(tmp_path, profile="iid", seed=seed)


def test_data_plan_rejects_unknown_profile_and_partial_existing_data(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="profile"):
        create_data_plan(tmp_path, profile="unknown", seed=17)
    (tmp_path / "node-0").mkdir()
    existing = tmp_path / "node-0" / "train.npz"
    existing.write_bytes(b"keep these bytes")
    with pytest.raises(ValueError, match="incomplete"):
        create_data_plan(tmp_path, profile="iid", seed=17)
    assert existing.read_bytes() == b"keep these bytes"


def test_data_plan_rejects_changed_or_missing_bound_file(tmp_path: Path) -> None:
    plan = create_data_plan(tmp_path, profile="iid", seed=17)
    binding = plan.nodes[0].train
    original = binding.path.read_bytes()
    binding.path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_data_plan(plan.path)
    binding.path.unlink()
    with pytest.raises(ValueError, match="cannot read"):
        load_data_plan(plan.path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("unknown", True),
        ("version", 2),
        ("version", True),
        ("version", 1.0),
        ("seed", "17"),
        ("world_size", 6),
        ("world_size", 4.0),
        ("input_shape", [9]),
        ("label_names", ["dog", "cat", "bird", "fish"]),
        ("nodes", []),
    ),
)
def test_data_plan_rejects_invalid_closed_metadata(
    tmp_path: Path, field: str, value: object
) -> None:
    plan = create_data_plan(tmp_path, profile="iid", seed=17)
    payload = json.loads(plan.path.read_bytes())
    payload[field] = value
    plan.path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_data_plan(plan.path)


@pytest.mark.parametrize("changed", ("path", "count", "histogram", "unknown", "rank"))
def test_data_plan_rejects_invalid_node_binding(tmp_path: Path, changed: str) -> None:
    plan = create_data_plan(tmp_path, profile="iid", seed=17)
    payload = json.loads(plan.path.read_bytes())
    node = payload["nodes"][0]
    if changed == "path":
        node["train"]["path"] = "../outside.npz"
    elif changed == "count":
        node["train"]["sample_count"] = 256
    elif changed == "histogram":
        node["train"]["label_histogram"] = [31, 33, 32, 32]
    elif changed == "unknown":
        node["train"]["unexpected"] = "value"
    else:
        node["rank"] = 1
    plan.path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_data_plan(plan.path)


@pytest.mark.parametrize(
    "invalid", ("float64", "shape", "nan", "labels", "label-dtype", "extra", "pickle")
)
def test_data_plan_validates_arrays_even_when_modified_bytes_have_a_matching_hash(
    tmp_path: Path, invalid: str
) -> None:
    plan = create_data_plan(tmp_path, profile="iid", seed=17)
    binding = plan.nodes[0].train
    with np.load(binding.path, allow_pickle=False) as archive:
        inputs = archive["inputs"]
        labels = archive["labels"]
    if invalid == "float64":
        inputs = inputs.astype(np.float64)
    elif invalid == "shape":
        inputs = inputs[:, :7]
    elif invalid == "nan":
        inputs[0, 0] = np.nan
    elif invalid == "labels":
        labels[0] = 4
    elif invalid == "label-dtype":
        labels = labels.astype(np.int32)
    elif invalid == "pickle":
        inputs = inputs.astype(object)
    if invalid == "extra":
        np.savez(binding.path, inputs=inputs, labels=labels, extra=labels)
    else:
        np.savez(binding.path, inputs=inputs, labels=labels)
    payload = json.loads(plan.path.read_bytes())
    payload["nodes"][0]["train"]["sha256"] = hashlib.sha256(
        binding.path.read_bytes()
    ).hexdigest()
    plan.path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="invalid dataset"):
        load_data_plan(plan.path)


def test_data_plan_rejects_symlink_and_duplicate_json_fields(tmp_path: Path) -> None:
    plan = create_data_plan(tmp_path / "plan", profile="iid", seed=17)
    binding = plan.nodes[0].train
    external = tmp_path / "external.npz"
    binding.path.rename(external)
    binding.path.symlink_to(external)
    with pytest.raises(ValueError, match="symbolic link"):
        load_data_plan(plan.path)
    binding.path.unlink()
    external.rename(binding.path)
    source = plan.path.read_text()
    plan.path.write_text(source.replace('"version": 1', '"version": 1, "version": 1'))
    with pytest.raises(ValueError, match="duplicate JSON field"):
        load_data_plan(plan.path)
