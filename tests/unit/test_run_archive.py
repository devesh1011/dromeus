from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from support.sample_manifest import manifest_data

from dromeus.manifests.models import SealedManifest
from dromeus.persistence.archive import (
    ARCHIVE_VERSION,
    RunArchive,
    RunArchiveError,
)
from dromeus.persistence.run_store import RunStore, RunStoreError


def _write_archive(root: Path, *, complete: bool = True) -> RunStore:
    store = RunStore(root)
    store.initialize(SealedManifest.model_validate(manifest_data()))
    store.persist_commit(
        committed_round=0,
        algorithm_state={"weight": np.array([2.0], dtype=np.float32)},
        pre_mix_state={"weight": np.array([1.0], dtype=np.float32)},
        post_mix_state={"weight": np.array([2.0], dtype=np.float32)},
        state_checksum="a" * 64,
        schedule={"round_id": 0, "peer": "peer-1"},
    )
    if complete:
        store.record_terminal("complete", {"committed_rounds": 1})
    return store


def _read_state(root: Path) -> dict[str, object]:
    return cast(
        dict[str, object],
        json.loads((root / "state.json").read_text(encoding="utf-8")),
    )


def _write_state(root: Path, state: dict[str, object]) -> None:
    (root / "state.json").write_text(
        json.dumps(state, sort_keys=True), encoding="utf-8"
    )


def test_open_returns_verified_immutable_archive(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_archive(root)

    archive = RunArchive.open(root).require_complete(1)

    assert archive.archive_version == ARCHIVE_VERSION
    assert archive.state.integrity_recorded
    assert archive.algorithm_state is not None
    assert np.array_equal(
        archive.algorithm_state.load_tensors()["weight"],
        np.array([2.0], dtype=np.float32),
    )
    archive.verify_checkpoint_integrity()
    checkpoints = cast(dict[int, object], archive.state.pre_mix_checkpoints)
    with pytest.raises(TypeError):
        checkpoints[1] = checkpoints[0]


def test_open_allows_incomplete_archive_but_completion_check_rejects_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _write_archive(root, complete=False)

    archive = RunArchive.open(root)

    assert archive.terminal is None
    with pytest.raises(RunArchiveError, match="not complete"):
        archive.require_complete(1)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("archive_version", 2, "unsupported archive version"),
        ("unexpected", True, "fields are invalid"),
    ],
)
def test_open_rejects_unsupported_versions_and_unknown_fields(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    root = tmp_path / "run"
    _write_archive(root)
    state = _read_state(root)
    state[field] = value
    _write_state(root, state)

    with pytest.raises(RunArchiveError, match=message):
        RunArchive.open(root)


def test_checkpoint_hash_detects_tampering(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_archive(root)
    archive = RunArchive.open(root)
    assert archive.algorithm_state is not None
    checkpoint = root / archive.algorithm_state.relative_path
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")

    with pytest.raises(RunArchiveError, match="hash mismatch"):
        archive.algorithm_state.load_tensors()
    with pytest.raises(RunArchiveError, match="hash mismatch"):
        RunArchive.open(root).verify_checkpoint_integrity()


@pytest.mark.parametrize(
    "unsafe_path",
    ["/tmp/checkpoint.safetensors", "../checkpoint.safetensors"],
)
def test_open_rejects_unsafe_checkpoint_paths(tmp_path: Path, unsafe_path: str) -> None:
    root = tmp_path / "run"
    _write_archive(root)
    state = _read_state(root)
    algorithm_state = cast(dict[str, object], state["algorithm_state"])
    algorithm_state["path"] = unsafe_path
    _write_state(root, state)

    with pytest.raises(RunArchiveError, match="path is unsafe"):
        RunArchive.open(root)


def test_open_rejects_checkpoint_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_archive(root)
    target = root / "checkpoints" / "committed-round-000000.safetensors"
    link = root / "checkpoints" / "linked.safetensors"
    link.symlink_to(target)
    state = _read_state(root)
    algorithm_state = cast(dict[str, object], state["algorithm_state"])
    algorithm_state["path"] = "checkpoints/linked.safetensors"
    _write_state(root, state)

    with pytest.raises(RunArchiveError, match="symlink"):
        RunArchive.open(root)


def test_legacy_v0_is_read_only_and_has_no_integrity_claim(tmp_path: Path) -> None:
    root = tmp_path / "run"
    store = _write_archive(root)
    state = _read_state(root)
    del state["archive_version"]
    del state["prepared_commit"]
    for key in ("algorithm_state",):
        checkpoint = cast(dict[str, str], state[key])
        state[key] = checkpoint["path"]
    for key in ("pre_mix_checkpoints", "post_mix_checkpoints"):
        checkpoints = cast(dict[str, dict[str, str]], state[key])
        state[key] = {
            round_id: checkpoint["path"] for round_id, checkpoint in checkpoints.items()
        }
    _write_state(root, state)

    archive = RunArchive.open(root)

    assert archive.archive_version == 0
    assert not archive.state.integrity_recorded
    assert archive.algorithm_state is not None
    assert "weight" in archive.algorithm_state.load_tensors()
    with pytest.raises(RunArchiveError, match="legacy archive"):
        archive.verify_checkpoint_integrity()
    with pytest.raises(RunStoreError, match="read-only"):
        store.record_terminal("failed")


def test_open_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_archive(root)
    state = _read_state(root)
    state["manifest_hash"] = "f" * 64
    _write_state(root, state)

    with pytest.raises(RunArchiveError, match="manifest hash mismatch"):
        RunArchive.open(root)


def test_open_rejects_symlinked_state_file(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_archive(root)
    state_path = root / "state.json"
    target = tmp_path / "state-target.json"
    state_path.replace(target)
    state_path.symlink_to(target)

    with pytest.raises(RunArchiveError, match="symlink"):
        RunArchive.open(root)
