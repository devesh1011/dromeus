"""Atomic run-state storage."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from threading import RLock
from typing import cast

import numpy as np
from safetensors.numpy import save_file  # pyright: ignore[reportUnknownVariableType]

from dromeus.manifests.canonical import (
    canonical_hash,
    canonical_json,
    file_sha256,
)
from dromeus.manifests.models import SealedManifest, Sha256
from dromeus.persistence.archive import (
    ARCHIVE_VERSION,
    ArchiveState,
    RunArchiveError,
    load_archive_state,
)


class RunStoreError(RuntimeError):
    """A durable run-state operation could not be completed safely."""


JsonRecord = Mapping[str, object]


class RunStore:
    """Persist audit state with one crash-safe committed checkpoint."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._checkpoint_root = root / "checkpoints"
        self._root.mkdir(parents=True, exist_ok=True)
        self._checkpoint_root.mkdir(parents=True, exist_ok=True)
        self._manifest_path = root / "manifest.json"
        self._state_path = root / "state.json"
        self._lock = RLock()

    def initialize(self, manifest: SealedManifest) -> Sha256:
        """Write the sealed manifest once and initialize current archive state."""
        with self._lock:
            manifest_bytes = canonical_json(manifest)
            manifest_hash = canonical_hash(manifest)
            if self._manifest_path.exists():
                if self._manifest_path.read_bytes() != manifest_bytes:
                    raise RunStoreError("run store manifest does not match")
            else:
                _atomic_write(self._manifest_path, manifest_bytes)

            if self._state_path.exists():
                state = self.load_state()
                if state.manifest_hash != manifest_hash:
                    raise RunStoreError("run store state manifest does not match")
                if state.archive_version == ARCHIVE_VERSION:
                    self._remove_unreferenced_checkpoints(state)
            else:
                _atomic_write_json(self._state_path, _initial_state(manifest_hash))
            return manifest_hash

    def load_state(self) -> ArchiveState:
        """Return validated immutable archive state."""
        with self._lock:
            if not self._state_path.is_file():
                raise RunStoreError("run store has not been initialized")
            try:
                return load_archive_state(self._state_path)
            except RunArchiveError as error:
                raise RunStoreError(str(error)) from error

    def persist_commit(
        self,
        *,
        committed_round: int,
        algorithm_state: Mapping[str, np.ndarray],
        state_checksum: str,
        schedule: JsonRecord,
        metrics: JsonRecord | None = None,
        transfer_diagnostics: JsonRecord | None = None,
    ) -> ArchiveState:
        """Persist and immediately confirm one committed round."""
        self.persist_prepared_commit(
            committed_round=committed_round,
            algorithm_state=algorithm_state,
            state_checksum=state_checksum,
            schedule=schedule,
            metrics=metrics,
            transfer_diagnostics=transfer_diagnostics,
        )
        return self.confirm_prepared_commit(
            committed_round=committed_round,
            state_checksum=state_checksum,
        )

    def persist_prepared_commit(
        self,
        *,
        committed_round: int,
        algorithm_state: Mapping[str, np.ndarray],
        state_checksum: str,
        schedule: JsonRecord,
        metrics: JsonRecord | None = None,
        transfer_diagnostics: JsonRecord | None = None,
    ) -> ArchiveState:
        """Persist round artifacts without exposing the round as committed."""
        with self._lock:
            if committed_round < 0:
                raise ValueError("committed_round must be non-negative")
            state = self._writable_state()
            if committed_round <= state.committed_round:
                if (
                    committed_round == state.committed_round
                    and state.state_checksum == state_checksum
                ):
                    return state
                raise RunStoreError("committed round must advance monotonically")
            prepared = state.prepared_commit
            if prepared is not None:
                if (
                    prepared.round_id == committed_round
                    and prepared.state_checksum == state_checksum
                ):
                    return state
                raise RunStoreError("another prepared commit already exists")

            committed_name = f"committed-round-{committed_round:06d}.safetensors"
            committed_hash = _atomic_save_tensors(
                self._checkpoint_root / committed_name, algorithm_state
            )

            next_state = state.as_json()
            next_state["prepared_commit"] = {
                "round_id": committed_round,
                "state_checksum": state_checksum,
                "algorithm_state": _checkpoint_json(committed_name, committed_hash),
                "schedule": dict(schedule),
                "metrics": dict(metrics) if metrics is not None else None,
                "transfer_diagnostics": (
                    dict(transfer_diagnostics)
                    if transfer_diagnostics is not None
                    else None
                ),
            }
            try:
                _atomic_write_json(self._state_path, next_state)
            except Exception:
                _remove_file(self._checkpoint_root / committed_name)
                raise
            return self.load_state()

    def confirm_prepared_commit(
        self,
        *,
        committed_round: int,
        state_checksum: str,
    ) -> ArchiveState:
        """Expose a prepared round after the peer confirms its commit."""
        with self._lock:
            if committed_round < 0:
                raise ValueError("committed_round must be non-negative")
            state = self._writable_state()
            if committed_round <= state.committed_round:
                if (
                    committed_round == state.committed_round
                    and state.state_checksum == state_checksum
                ):
                    return state
                raise RunStoreError("committed round must advance monotonically")
            prepared = state.prepared_commit
            if (
                prepared is None
                or prepared.round_id != committed_round
                or prepared.state_checksum != state_checksum
            ):
                raise RunStoreError("prepared commit does not match confirmation")

            previous_checkpoint = state.algorithm_state
            next_state = state.as_json()
            prepared_json = cast(
                dict[str, object], next_state["prepared_commit"]
            )
            next_state.update(
                {
                    "committed_round": committed_round,
                    "state_checksum": state_checksum,
                    "algorithm_state": prepared.algorithm_state.as_json(),
                    "prepared_commit": None,
                }
            )
            cast(list[object], next_state["schedule_history"]).append(
                prepared_json["schedule"]
            )
            if prepared.metrics is not None:
                cast(list[object], next_state["metrics"]).append(
                    prepared_json["metrics"]
                )
            if prepared.transfer_diagnostics is not None:
                cast(list[object], next_state["transfer_diagnostics"]).append(
                    prepared_json["transfer_diagnostics"]
                )
            _atomic_write_json(self._state_path, next_state)
            if previous_checkpoint is not None:
                _remove_file(self._root / previous_checkpoint.relative_path)
            return self.load_state()

    def record_consensus(
        self,
        *,
        round_id: int,
        normalized_rms: float,
        sketch_count: int,
    ) -> ArchiveState:
        """Append one completed live consensus result atomically and idempotently."""
        if round_id < 0:
            raise ValueError("round_id must be non-negative")
        if not np.isfinite(normalized_rms) or normalized_rms < 0:
            raise ValueError("normalized_rms must be finite and non-negative")
        if sketch_count <= 0:
            raise ValueError("sketch_count must be positive")
        with self._lock:
            state = self._writable_state()
            for existing in state.consensus:
                if existing.round_id != round_id:
                    continue
                if (
                    existing.normalized_rms != normalized_rms
                    or existing.sketch_count != sketch_count
                ):
                    raise RunStoreError("consensus result already recorded")
                return state
            next_state = state.as_json()
            cast(list[object], next_state["consensus"]).append(
                {
                    "round_id": round_id,
                    "normalized_rms": float(normalized_rms),
                    "sketch_count": sketch_count,
                }
            )
            _atomic_write_json(self._state_path, next_state)
            return self.load_state()

    def record_terminal(
        self, result: str, diagnostics: JsonRecord | None = None
    ) -> None:
        """Record one terminal result; identical duplicate calls are idempotent."""
        with self._lock:
            if not result:
                raise ValueError("terminal result must not be empty")
            state = self._writable_state()
            terminal = {"result": result, "diagnostics": dict(diagnostics or {})}
            if state.terminal is not None:
                existing = {
                    "result": state.terminal.result,
                    "diagnostics": dict(state.terminal.diagnostics),
                }
                if existing == terminal:
                    return
                raise RunStoreError("terminal result already recorded")
            next_state = state.as_json()
            discarded = (
                state.prepared_commit.algorithm_state
                if result != "complete" and state.prepared_commit is not None
                else None
            )
            if discarded is not None:
                next_state["prepared_commit"] = None
            next_state["terminal"] = terminal
            _atomic_write_json(self._state_path, next_state)
            if discarded is not None:
                _remove_file(self._root / discarded.relative_path)

    def _writable_state(self) -> ArchiveState:
        state = self.load_state()
        if state.archive_version != ARCHIVE_VERSION:
            raise RunStoreError("legacy run archive is read-only")
        return state

    def _remove_unreferenced_checkpoints(self, state: ArchiveState) -> None:
        referenced = {
            record.relative_path for record in state.checkpoint_records()
        }
        for path in self._checkpoint_root.glob("*.safetensors"):
            relative_path = path.relative_to(self._root).as_posix()
            if relative_path not in referenced:
                _remove_file(path)


def _initial_state(manifest_hash: Sha256) -> dict[str, object]:
    return {
        "archive_version": ARCHIVE_VERSION,
        "manifest_hash": manifest_hash,
        "committed_round": -1,
        "state_checksum": None,
        "algorithm_state": None,
        "prepared_commit": None,
        "schedule_history": [],
        "metrics": [],
        "transfer_diagnostics": [],
        "consensus": [],
        "terminal": None,
    }


def _checkpoint_json(name: str, digest: Sha256) -> dict[str, str]:
    return {"path": f"checkpoints/{name}", "sha256": digest}


def _atomic_save_tensors(path: Path, tensors: Mapping[str, np.ndarray]) -> Sha256:
    if not tensors:
        raise ValueError("checkpoint must contain at least one tensor")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        save_file(
            {name: np.ascontiguousarray(value) for name, value in tensors.items()},
            str(temporary),
        )
        _fsync_file(temporary)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        return file_sha256(path)
    except (OSError, ValueError, TypeError) as error:
        temporary.unlink(missing_ok=True)
        raise RunStoreError(f"could not persist checkpoint {path.name}") from error


def _atomic_write_json(path: Path, value: object) -> None:
    try:
        data = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError) as error:
        raise RunStoreError("run state is not JSON serializable") from error
    _atomic_write(path, data)


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise RunStoreError(f"could not persist {path.name}") from error


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
        _fsync_directory(path.parent)
    except OSError:
        pass


__all__ = ["RunStore", "RunStoreError"]
