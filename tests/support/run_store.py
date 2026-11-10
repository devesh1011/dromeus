"""Test helper: persist and confirm one round in a single call."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from dromeus.persistence.archive import ArchiveState
from dromeus.persistence.run_store import JsonRecord, RunStore


def persist_commit(
    store: RunStore,
    *,
    committed_round: int,
    algorithm_state: Mapping[str, np.ndarray],
    state_checksum: str,
    schedule: JsonRecord,
    metrics: JsonRecord | None = None,
    transfer_diagnostics: JsonRecord | None = None,
) -> ArchiveState:
    """Run the runtime's two-phase commit (prepare, then confirm) in one step."""
    store.persist_prepared_commit(
        committed_round=committed_round,
        algorithm_state=algorithm_state,
        state_checksum=state_checksum,
        schedule=schedule,
        metrics=metrics,
        transfer_diagnostics=transfer_diagnostics,
    )
    return store.confirm_prepared_commit(
        committed_round=committed_round,
        state_checksum=state_checksum,
    )
