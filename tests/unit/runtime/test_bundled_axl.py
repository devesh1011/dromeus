"""Drafts pin the AXL build bundled with Dromeus unless told otherwise."""

from __future__ import annotations

from pathlib import Path

import pytest

from dromeus import DraftError, make_draft
from dromeus import application as application_module
from dromeus.application import bundled_axl_version, load_draft
from examples.custom_training import task


def _draft(path: Path, axl_version: str | None = None) -> str:
    make_draft(
        task(),
        path,
        run_id="bundled-axl",
        participants=4,
        rounds=2,
        local_steps=1,
        axl_version=axl_version,
    )
    return load_draft(path).environment.axl_version


def test_bundled_version_is_the_installed_dromeus_axl_commit() -> None:
    dromeus_axl = pytest.importorskip("dromeus_axl")

    assert bundled_axl_version() == dromeus_axl.AXL_COMMIT
    assert dromeus_axl.binary().is_file()


def test_draft_defaults_to_the_bundled_axl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(application_module, "bundled_axl_version", lambda: "abc123")

    assert _draft(tmp_path / "draft.yaml") == "abc123"


def test_named_axl_build_overrides_the_bundled_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(application_module, "bundled_axl_version", lambda: "abc123")

    assert _draft(tmp_path / "draft.yaml", axl_version="own-build") == "own-build"


def test_without_a_bundled_axl_the_version_must_be_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(application_module, "bundled_axl_version", lambda: None)

    with pytest.raises(DraftError, match="no AXL is bundled"):
        _draft(tmp_path / "draft.yaml")
