from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from dromeus.persistence import files
from dromeus.persistence.files import atomic_write_bytes


def test_atomic_write_replaces_contents_and_leaves_no_temp(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    target.write_bytes(b"old")

    atomic_write_bytes(target, b"new")

    assert target.read_bytes() == b"new"
    assert [path.name for path in tmp_path.iterdir()] == ["state.json"]


def test_atomic_write_applies_private_mode(tmp_path: Path) -> None:
    target = tmp_path / "ledger.json"

    atomic_write_bytes(target, b"{}", mode=0o600)

    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_failed_write_keeps_old_file_and_removes_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "invitation.json"
    target.write_bytes(b"old")

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(files.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk full"):
        atomic_write_bytes(target, b"new")

    assert target.read_bytes() == b"old"
    assert os.listdir(tmp_path) == ["invitation.json"]
