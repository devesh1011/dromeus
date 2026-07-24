from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from safetensors.numpy import load_file  # pyright: ignore[reportUnknownVariableType]
from support.sample_manifest import manifest_data

from dromeus.manifests.models import SealedManifest
from dromeus.persistence.run_store import RunStore, RunStoreError

_load_file = cast(
    Callable[[str], dict[str, np.ndarray]],
    load_file,
)


def test_run_store_persists_round_artifacts_and_terminal_result(tmp_path: Path) -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    store = RunStore(tmp_path / "run")

    manifest_hash = store.initialize(manifest)
    state = store.persist_commit(
        committed_round=0,
        algorithm_state={"layer.weight": np.array([[2.0]], dtype=np.float32)},
        pre_mix_state={"layer.weight": np.array([[1.0]], dtype=np.float32)},
        post_mix_state={"layer.weight": np.array([[2.0]], dtype=np.float32)},
        state_checksum="a" * 64,
        schedule={"round_id": 0, "peer": "peer-1"},
        metrics={"loss": 0.5},
        transfer_diagnostics={"retries": 1},
    )

    assert state["manifest_hash"] == manifest_hash
    assert state["committed_round"] == 0
    committed = tmp_path / "run" / state["algorithm_state"]
    loaded = _load_file(str(committed))
    assert np.array_equal(loaded["layer.weight"], np.array([[2.0]], dtype=np.float32))
    assert (tmp_path / "run" / state["pre_mix_checkpoints"]["0"]).is_file()
    assert (tmp_path / "run" / state["post_mix_checkpoints"]["0"]).is_file()
    assert not list((tmp_path / "run" / "checkpoints").glob("*.tmp"))

    store.record_terminal("complete", {"round": 0})
    store.record_terminal("complete", {"round": 0})
    assert store.load_state()["terminal"] == {
        "result": "complete",
        "diagnostics": {"round": 0},
    }
    with pytest.raises(RunStoreError, match="terminal result already recorded"):
        store.record_terminal("failed", {"round": 0})


def test_run_store_persists_consensus_result_once(tmp_path: Path) -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    store = RunStore(tmp_path / "run")
    store.initialize(manifest)

    store.record_consensus(
        round_id=0,
        normalized_rms=0.125,
        sketch_count=4,
    )
    store.record_consensus(
        round_id=0,
        normalized_rms=0.125,
        sketch_count=4,
    )

    assert store.load_state()["consensus"] == [
        {"round_id": 0, "normalized_rms": 0.125, "sketch_count": 4}
    ]
    with pytest.raises(RunStoreError, match="consensus result already recorded"):
        store.record_consensus(
            round_id=0,
            normalized_rms=0.25,
            sketch_count=4,
        )
