"""Frozen public evaluation membership excludes every previously used image."""

import hashlib
from pathlib import Path

import numpy as np

from benchmarks.m3.freeze_data import select_unused


def test_selection_is_balanced_deterministic_and_excludes_indices_and_images() -> None:
    images = np.arange(40, dtype=np.float32).reshape(40, 1, 1, 1)
    labels = np.repeat(np.arange(10), 4).astype(np.int64)
    images[2] = images[1]  # A duplicate must not get another slot.
    blocked = {hashlib.sha256(images[4].tobytes()).hexdigest()}
    selected = select_unused(
        images,
        labels,
        excluded_indices={0},
        excluded_hashes=blocked,
        per_class=2,
        seed=19,
    )
    again = select_unused(
        images,
        labels,
        excluded_indices={0},
        excluded_hashes=blocked,
        per_class=2,
        seed=19,
    )
    assert selected == again
    assert len(selected.indices) == len(set(selected.image_hashes)) == 20
    assert 0 not in selected.indices and 4 not in selected.indices
    assert not (set(selected.image_hashes) & blocked)
    np.testing.assert_array_equal(np.bincount(labels[list(selected.indices)]), [2] * 10)


def test_insufficient_unique_class_fails_instead_of_silently_shortening() -> None:
    import pytest

    images = np.arange(20, dtype=np.float32).reshape(20, 1, 1, 1)
    labels = np.repeat(np.arange(10), 2).astype(np.int64)
    images[1] = images[0]
    with pytest.raises(ValueError, match="insufficient unique unused"):
        select_unused(
            images,
            labels,
            excluded_indices=set(),
            excluded_hashes=set(),
            per_class=2,
            seed=1,
        )


def test_invalid_candidate_and_out_of_range_exclusion_fail() -> None:
    import pytest

    images = np.arange(20, dtype=np.float32).reshape(20, 1, 1, 1)
    labels = np.repeat(np.arange(10), 2).astype(np.int64)
    for exclusions in ({-1}, {20}):
        with pytest.raises(ValueError, match="invalid candidate"):
            select_unused(
                images,
                labels,
                excluded_indices=exclusions,
                excluded_hashes=set(),
                per_class=1,
                seed=1,
            )
    images[0] = np.nan
    with pytest.raises(ValueError, match="invalid candidate"):
        select_unused(
            images,
            labels,
            excluded_indices=set(),
            excluded_hashes=set(),
            per_class=1,
            seed=1,
        )


def test_freeze_refuses_existing_output_before_modifying_it(tmp_path: Path) -> None:
    import pytest

    from benchmarks.m3.freeze_data import freeze_data

    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("retain")
    with pytest.raises(FileExistsError):
        freeze_data(
            cache=tmp_path,
            previous_data=tmp_path,
            previous_provenance=tmp_path / "absent.json",
            output=output,
            evidence=tmp_path / "evidence",
        )
    assert sentinel.read_text() == "retain"


def test_original_checksum_change_rejects_freeze_before_output(tmp_path: Path) -> None:
    import json

    import pytest

    from benchmarks.m3.freeze_data import freeze_data
    from benchmarks.m3.heterogeneous_data import FILES

    files: list[dict[str, str]] = []
    for name in FILES:
        (tmp_path / name).write_bytes(b"original")
        files.append({"file": name, "sha256": hashlib.sha256(b"original").hexdigest()})
    (tmp_path / next(iter(FILES))).write_bytes(b"changed")
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"files": files}))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="original source checksum mismatch"):
        freeze_data(
            cache=tmp_path,
            previous_data=tmp_path,
            previous_provenance=prior,
            output=output,
            evidence=tmp_path / "evidence",
        )
    assert not output.exists()


def test_other_source_selected_images_are_excluded_from_next_selection() -> None:
    images = np.arange(40, dtype=np.float32).reshape(40, 1, 1, 1)
    labels = np.repeat(np.arange(10), 4).astype(np.int64)
    first = select_unused(
        images,
        labels,
        excluded_indices=set(),
        excluded_hashes=set(),
        per_class=2,
        seed=1,
    )
    second = select_unused(
        images,
        labels,
        excluded_indices=set(),
        excluded_hashes=set(first.image_hashes),
        per_class=2,
        seed=2,
    )
    assert not (set(first.image_hashes) & set(second.image_hashes))
    assert not (set(first.indices) & set(second.indices))
    assert len(second.indices) == 20
