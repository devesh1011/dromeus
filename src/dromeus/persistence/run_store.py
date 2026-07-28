"""Atomic run-state storage."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from threading import RLock
from typing import Any, cast

import numpy as np
from safetensors.numpy import save_file  # pyright: ignore[reportUnknownVariableType]

from dromeus.manifests.canonical import canonical_hash, canonical_json
from dromeus.manifests.models import SealedManifest, Sha256


class RunStoreError(RuntimeError):
    """A durable run-state operation could not be completed safely."""


JsonRecord = Mapping[str, object]


class RunStore:
    """Persist audit state with immutable checkpoints and atomic JSON commits."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._checkpoint_root = root / "checkpoints"
        self._root.mkdir(parents=True, exist_ok=True)
        self._checkpoint_root.mkdir(parents=True, exist_ok=True)
        self._manifest_path = root / "manifest.json"
        self._state_path = root / "state.json"
        self._lock = RLock()

    def initialize(self, manifest: SealedManifest) -> Sha256:
        """Write the sealed manifest once and initialize an empty run state."""
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
                if state.get("manifest_hash") != manifest_hash:
                    raise RunStoreError("run store state manifest does not match")
            else:
                _atomic_write_json(self._state_path, _initial_state(manifest_hash))
            return manifest_hash

    def load_state(self) -> dict[str, Any]:
        with self._lock:
            if not self._state_path.is_file():
                raise RunStoreError("run store has not been initialized")
            try:
                value = cast(object, json.loads(self._state_path.read_text()))
            except (OSError, ValueError) as error:
                raise RunStoreError("run store state is unreadable") from error
            if not isinstance(value, dict):
                raise RunStoreError("run store state must be a JSON object")
            return cast(dict[str, Any], value)

    def persist_commit(
        self,
        *,
        committed_round: int,
        algorithm_state: Mapping[str, np.ndarray],
        pre_mix_state: Mapping[str, np.ndarray],
        post_mix_state: Mapping[str, np.ndarray],
        state_checksum: str,
        schedule: JsonRecord,
        metrics: JsonRecord | None = None,
        transfer_diagnostics: JsonRecord | None = None,
    ) -> dict[str, Any]:
        """Persist and immediately confirm one committed round."""
        self.persist_prepared_commit(
            committed_round=committed_round,
            algorithm_state=algorithm_state,
            pre_mix_state=pre_mix_state,
            post_mix_state=post_mix_state,
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
        pre_mix_state: Mapping[str, np.ndarray],
        post_mix_state: Mapping[str, np.ndarray],
        state_checksum: str,
        schedule: JsonRecord,
        metrics: JsonRecord | None = None,
        transfer_diagnostics: JsonRecord | None = None,
    ) -> dict[str, Any]:
        """Persist round artifacts without exposing the round as committed."""
        with self._lock:
            if committed_round < 0:
                raise ValueError("committed_round must be non-negative")
            state = self.load_state()
            previous_round = int(state["committed_round"])
            if committed_round <= previous_round:
                if (
                    committed_round == previous_round
                    and state.get("state_checksum") == state_checksum
                ):
                    return state
                raise RunStoreError("committed round must advance monotonically")
            prepared_value = state.get("prepared_commit")
            if prepared_value is not None:
                prepared = (
                    cast(dict[str, object], prepared_value)
                    if isinstance(prepared_value, dict)
                    else None
                )
                if (
                    prepared is not None
                    and prepared.get("round_id") == committed_round
                    and prepared.get("state_checksum") == state_checksum
                ):
                    return state
                raise RunStoreError("another prepared commit already exists")

            pre_name = f"pre-mix-round-{committed_round:06d}.safetensors"
            post_name = f"post-mix-round-{committed_round:06d}.safetensors"
            committed_name = f"committed-round-{committed_round:06d}.safetensors"
            pre_path = self._checkpoint_root / pre_name
            post_path = self._checkpoint_root / post_name
            committed_path = self._checkpoint_root / committed_name
            _atomic_save_tensors(pre_path, pre_mix_state)
            _atomic_save_tensors(post_path, post_mix_state)
            _atomic_save_tensors(committed_path, algorithm_state)

            next_state = dict(state)
            next_state["prepared_commit"] = {
                "round_id": committed_round,
                "state_checksum": state_checksum,
                "algorithm_state": f"checkpoints/{committed_name}",
                "pre_mix_checkpoint": f"checkpoints/{pre_name}",
                "post_mix_checkpoint": f"checkpoints/{post_name}",
                "schedule": dict(schedule),
                "metrics": dict(metrics) if metrics is not None else None,
                "transfer_diagnostics": (
                    dict(transfer_diagnostics)
                    if transfer_diagnostics is not None
                    else None
                ),
            }
            _atomic_write_json(self._state_path, next_state)
            return next_state

    def confirm_prepared_commit(
        self,
        *,
        committed_round: int,
        state_checksum: str,
    ) -> dict[str, Any]:
        """Expose a prepared round after the peer confirms its commit."""
        with self._lock:
            if committed_round < 0:
                raise ValueError("committed_round must be non-negative")
            state = self.load_state()
            previous_round = int(state["committed_round"])
            if committed_round <= previous_round:
                if (
                    committed_round == previous_round
                    and state.get("state_checksum") == state_checksum
                ):
                    return state
                raise RunStoreError("committed round must advance monotonically")
            prepared_value = state.get("prepared_commit")
            if not isinstance(prepared_value, dict):
                raise RunStoreError("prepared commit does not match confirmation")
            prepared = cast(dict[str, object], prepared_value)
            if (
                prepared.get("round_id") != committed_round
                or prepared.get("state_checksum") != state_checksum
            ):
                raise RunStoreError("prepared commit does not match confirmation")

            round_key = str(committed_round)
            next_state = dict(state)
            next_state.update(
                {
                    "committed_round": committed_round,
                    "state_checksum": state_checksum,
                    "algorithm_state": prepared["algorithm_state"],
                    "prepared_commit": None,
                }
            )
            next_state["pre_mix_checkpoints"] = {
                **cast(dict[str, str], state["pre_mix_checkpoints"]),
                round_key: prepared["pre_mix_checkpoint"],
            }
            next_state["post_mix_checkpoints"] = {
                **cast(dict[str, str], state["post_mix_checkpoints"]),
                round_key: prepared["post_mix_checkpoint"],
            }
            next_state["schedule_history"] = [
                *cast(list[object], state["schedule_history"]),
                prepared["schedule"],
            ]
            metrics = prepared.get("metrics")
            if metrics is not None:
                next_state["metrics"] = [
                    *cast(list[object], state["metrics"]),
                    metrics,
                ]
            transfer_diagnostics = prepared.get("transfer_diagnostics")
            if transfer_diagnostics is not None:
                next_state["transfer_diagnostics"] = [
                    *cast(list[object], state["transfer_diagnostics"]),
                    transfer_diagnostics,
                ]
            _atomic_write_json(self._state_path, next_state)
            return next_state

    def record_consensus(
        self,
        *,
        round_id: int,
        normalized_rms: float,
        sketch_count: int,
    ) -> dict[str, Any]:
        """Append one completed live consensus result atomically and idempotently."""
        if round_id < 0:
            raise ValueError("round_id must be non-negative")
        if not np.isfinite(normalized_rms) or normalized_rms < 0:
            raise ValueError("normalized_rms must be finite and non-negative")
        if sketch_count <= 0:
            raise ValueError("sketch_count must be positive")
        with self._lock:
            state = self.load_state()
            record = {
                "round_id": round_id,
                "normalized_rms": float(normalized_rms),
                "sketch_count": sketch_count,
            }
            records = cast(list[dict[str, object]], state.get("consensus", []))
            for existing in records:
                if existing.get("round_id") != round_id:
                    continue
                if existing != record:
                    raise RunStoreError("consensus result already recorded")
                return state
            next_state = dict(state)
            next_state["consensus"] = [*records, record]
            _atomic_write_json(self._state_path, next_state)
            return next_state

    def record_terminal(
        self, result: str, diagnostics: JsonRecord | None = None
    ) -> None:
        """Record one terminal result; identical duplicate calls are idempotent."""
        with self._lock:
            if not result:
                raise ValueError("terminal result must not be empty")
            state = self.load_state()
            terminal = {"result": result, "diagnostics": dict(diagnostics or {})}
            existing = state.get("terminal")
            if existing is not None:
                if existing == terminal:
                    return
                raise RunStoreError("terminal result already recorded")
            state["terminal"] = terminal
            _atomic_write_json(self._state_path, state)


def _initial_state(manifest_hash: Sha256) -> dict[str, object]:
    return {
        "manifest_hash": manifest_hash,
        "committed_round": -1,
        "state_checksum": None,
        "algorithm_state": None,
        "prepared_commit": None,
        "pre_mix_checkpoints": {},
        "post_mix_checkpoints": {},
        "schedule_history": [],
        "metrics": [],
        "transfer_diagnostics": [],
        "consensus": [],
        "terminal": None,
    }


def _atomic_save_tensors(path: Path, tensors: Mapping[str, np.ndarray]) -> None:
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


__all__ = ["RunStore", "RunStoreError"]
