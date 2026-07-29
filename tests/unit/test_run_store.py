from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from support.sample_manifest import manifest_data

from dromeus.manifests.models import SealedManifest
from dromeus.persistence.archive import ARCHIVE_VERSION, RunArchive
from dromeus.persistence.run_store import RunStore, RunStoreError


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
        schedule={
            "round_id": 0,
            "peer": "peer-1",
            "nested": {"items": [1]},
        },
        metrics={"loss": 0.5},
        transfer_diagnostics={"retries": 1},
    )

    assert state.archive_version == ARCHIVE_VERSION
    assert state.manifest_hash == manifest_hash
    assert state.committed_round == 0
    archive = RunArchive.open(tmp_path / "run")
    assert archive.algorithm_state is not None
    loaded = archive.algorithm_state.load_tensors()
    assert np.array_equal(loaded["layer.weight"], np.array([[2.0]], dtype=np.float32))
    assert state.pre_mix_checkpoints[0].sha256 is not None
    assert state.post_mix_checkpoints[0].sha256 is not None
    assert state.schedule_history[0]["nested"] == {"items": (1,)}
    archive.verify_checkpoint_integrity()
    assert not list((tmp_path / "run" / "checkpoints").glob("*.tmp"))

    store.record_terminal("complete", {"round": 0})
    store.record_terminal("complete", {"round": 0})
    terminal = store.load_state().terminal
    assert terminal is not None
    assert terminal.result == "complete"
    assert terminal.diagnostics == {"round": 0}
    with pytest.raises(RunStoreError, match="terminal result already recorded"):
        store.record_terminal("failed", {"round": 0})


def test_run_store_exposes_round_only_after_peer_confirmation(tmp_path: Path) -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    store = RunStore(tmp_path / "run")
    store.initialize(manifest)

    prepared = store.persist_prepared_commit(
        committed_round=0,
        algorithm_state={"layer.weight": np.array([[2.0]], dtype=np.float32)},
        pre_mix_state={"layer.weight": np.array([[1.0]], dtype=np.float32)},
        post_mix_state={"layer.weight": np.array([[2.0]], dtype=np.float32)},
        state_checksum="a" * 64,
        schedule={"round_id": 0, "peer": "peer-1"},
    )

    assert prepared.committed_round == -1
    assert prepared.algorithm_state is None
    assert prepared.prepared_commit is not None
    assert prepared.prepared_commit.round_id == 0
    confirmed = store.confirm_prepared_commit(
        committed_round=0,
        state_checksum="a" * 64,
    )
    assert confirmed.committed_round == 0
    assert confirmed.prepared_commit is None
    assert confirmed.algorithm_state is not None
    assert confirmed.algorithm_state.relative_path.endswith(
        "committed-round-000000.safetensors"
    )


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

    consensus = store.load_state().consensus
    assert len(consensus) == 1
    assert consensus[0].round_id == 0
    assert consensus[0].normalized_rms == 0.125
    assert consensus[0].sketch_count == 4
    with pytest.raises(RunStoreError, match="consensus result already recorded"):
        store.record_consensus(
            round_id=0,
            normalized_rms=0.25,
            sketch_count=4,
        )
