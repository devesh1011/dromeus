"""Atomic run-state storage."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
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

    def initialize(self, manifest: SealedManifest) -> Sha256:
        """Write the sealed manifest once and initialize an empty run state."""
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
        """Persist one committed round; state JSON becomes visible last."""
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
        next_state.update(
            {
                "committed_round": committed_round,
                "state_checksum": state_checksum,
                "algorithm_state": f"checkpoints/{committed_name}",
            }
        )
        next_state["pre_mix_checkpoints"] = {
            **cast(dict[str, str], state["pre_mix_checkpoints"]),
            str(committed_round): f"checkpoints/{pre_name}",
        }
        next_state["post_mix_checkpoints"] = {
            **cast(dict[str, str], state["post_mix_checkpoints"]),
            str(committed_round): f"checkpoints/{post_name}",
        }
        next_state["schedule_history"] = [
            *cast(list[object], state["schedule_history"]),
            dict(schedule),
        ]
        if metrics is not None:
            next_state["metrics"] = [
                *cast(list[object], state["metrics"]),
                dict(metrics),
            ]
        if transfer_diagnostics is not None:
            next_state["transfer_diagnostics"] = [
                *cast(list[object], state["transfer_diagnostics"]),
                dict(transfer_diagnostics),
            ]
        _atomic_write_json(self._state_path, next_state)
        return next_state

    def record_terminal(
        self, result: str, diagnostics: JsonRecord | None = None
    ) -> None:
        """Record one terminal result; identical duplicate calls are idempotent."""
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
        "pre_mix_checkpoints": {},
        "post_mix_checkpoints": {},
        "schedule_history": [],
        "metrics": [],
        "transfer_diagnostics": [],
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
