"""Nodes on different PyTorch builds of one release can train together.

`torch.__version__` carries a build label (`2.13.0+cpu` on Linux CPU wheels,
`2.13.0+cu130` on CUDA, none on macOS). Drafts record the release only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from examples.custom_training import local_data, task

from dromeus import DraftError, check_draft, make_draft
from dromeus.application import load_draft, prepare_task


def _release(version: str) -> str:
    return version.split("+", 1)[0]


RELEASE = _release(torch.__version__)


def _write_draft(path: Path) -> None:
    make_draft(task(), path, run_id="builds", participants=4, rounds=2,
               local_steps=1, axl_version="test-axl")


def test_draft_records_the_release_without_build_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch, "__version__", f"{RELEASE}+cpu")
    draft = tmp_path / "draft.yaml"

    _write_draft(draft)

    loaded = load_draft(draft)
    assert loaded.environment.pytorch_version == RELEASE
    assert loaded.training.optimizer.torch_version == RELEASE


def test_other_build_of_the_same_release_checks_and_prepares(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    draft = tmp_path / "draft.yaml"
    monkeypatch.setattr(torch, "__version__", f"{RELEASE}+cpu")
    _write_draft(draft)
    records = tmp_path / "local.npz"
    np.savez(records, inputs=np.zeros((8, 3), dtype=np.float32),
             targets=np.zeros((8, 1), dtype=np.float32))

    monkeypatch.setattr(torch, "__version__", f"{RELEASE}+cu130")
    check_draft(draft, task())
    prepare_task(draft=load_draft(draft), task=task(), local=local_data(records))


def test_another_release_is_still_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    draft = tmp_path / "draft.yaml"
    _write_draft(draft)

    monkeypatch.setattr(torch, "__version__", "1.0.0+cpu")
    with pytest.raises(DraftError, match="PyTorch"):
        check_draft(draft, task())
