"""Validated immutable read model for one persisted run archive."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import cast

import numpy as np
from safetensors import SafetensorError
from safetensors.numpy import (
    load as _load,  # pyright: ignore[reportUnknownVariableType]
)

from dromeus.manifests.canonical import canonical_hash
from dromeus.manifests.models import SealedManifest, Sha256

ARCHIVE_VERSION = 2
_CURRENT_STATE_FIELDS = frozenset(
    {
        "archive_version",
        "manifest_hash",
        "committed_round",
        "state_checksum",
        "algorithm_state",
        "prepared_commit",
        "schedule_history",
        "metrics",
        "transfer_diagnostics",
        "consensus",
        "terminal",
    }
)
_LEGACY_STATE_FIELDS = _CURRENT_STATE_FIELDS | {
    "pre_mix_checkpoints",
    "post_mix_checkpoints",
}
_LOAD_SAFETENSORS = cast(
    Callable[[bytes], dict[str, np.ndarray]],
    _load,
)


class RunArchiveError(ValueError):
    """A persisted run archive is unsafe, inconsistent, or corrupt."""


JsonRecord = Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    relative_path: str
    sha256: Sha256 | None

    def as_json(self) -> dict[str, str]:
        if self.sha256 is None:
            raise RunArchiveError("legacy checkpoint has no recorded hash")
        return {"path": self.relative_path, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class PreparedCommitState:
    round_id: int
    state_checksum: Sha256
    algorithm_state: CheckpointRecord
    schedule: JsonRecord
    metrics: JsonRecord | None
    transfer_diagnostics: JsonRecord | None
    legacy_checkpoint_records: tuple[CheckpointRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class ConsensusState:
    round_id: int
    normalized_rms: float
    sketch_count: int


@dataclass(frozen=True, slots=True)
class TerminalState:
    result: str
    diagnostics: JsonRecord


@dataclass(frozen=True, slots=True)
class ArchiveState:
    archive_version: int
    manifest_hash: Sha256
    committed_round: int
    state_checksum: Sha256 | None
    algorithm_state: CheckpointRecord | None
    prepared_commit: PreparedCommitState | None
    schedule_history: tuple[JsonRecord, ...]
    metrics: tuple[JsonRecord, ...]
    transfer_diagnostics: tuple[JsonRecord, ...]
    consensus: tuple[ConsensusState, ...]
    terminal: TerminalState | None
    legacy_checkpoint_records: tuple[CheckpointRecord, ...] = ()

    @property
    def integrity_recorded(self) -> bool:
        return self.archive_version >= 1 and all(
            record.sha256 is not None for record in self.checkpoint_records()
        )

    def checkpoint_records(self) -> tuple[CheckpointRecord, ...]:
        records = list(self.legacy_checkpoint_records)
        if self.algorithm_state is not None:
            records.append(self.algorithm_state)
        if self.prepared_commit is not None:
            records.append(self.prepared_commit.algorithm_state)
            records.extend(self.prepared_commit.legacy_checkpoint_records)
        return tuple(records)

    def as_json(self) -> dict[str, object]:
        if self.archive_version != ARCHIVE_VERSION:
            raise RunArchiveError("legacy archive state is read-only")
        return {
            "archive_version": ARCHIVE_VERSION,
            "manifest_hash": self.manifest_hash,
            "committed_round": self.committed_round,
            "state_checksum": self.state_checksum,
            "algorithm_state": (
                self.algorithm_state.as_json()
                if self.algorithm_state is not None
                else None
            ),
            "prepared_commit": (
                _prepared_as_json(self.prepared_commit)
                if self.prepared_commit is not None
                else None
            ),
            "schedule_history": [
                _thaw_json(record) for record in self.schedule_history
            ],
            "metrics": [_thaw_json(record) for record in self.metrics],
            "transfer_diagnostics": [
                _thaw_json(record) for record in self.transfer_diagnostics
            ],
            "consensus": [
                {
                    "round_id": record.round_id,
                    "normalized_rms": record.normalized_rms,
                    "sketch_count": record.sketch_count,
                }
                for record in self.consensus
            ],
            "terminal": (
                {
                    "result": self.terminal.result,
                    "diagnostics": _thaw_json(self.terminal.diagnostics),
                }
                if self.terminal is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class CheckpointRef:
    """Archive-bound checkpoint reference that cannot bypass validation."""

    _root: Path
    relative_path: str
    recorded_sha256: Sha256 | None

    def verify(self) -> None:
        if self.recorded_sha256 is None:
            raise RunArchiveError(
                f"checkpoint hash is not recorded: {self.relative_path}"
            )
        data = _read_safe_file(self._root, self.relative_path)
        if hashlib.sha256(data).hexdigest() != self.recorded_sha256:
            raise RunArchiveError(f"checkpoint hash mismatch: {self.relative_path}")

    def load_tensors(self) -> dict[str, np.ndarray]:
        data = _read_safe_file(self._root, self.relative_path)
        if (
            self.recorded_sha256 is not None
            and hashlib.sha256(data).hexdigest() != self.recorded_sha256
        ):
            raise RunArchiveError(f"checkpoint hash mismatch: {self.relative_path}")
        try:
            return {
                name: np.ascontiguousarray(value)
                for name, value in _LOAD_SAFETENSORS(data).items()
            }
        except (SafetensorError, ValueError, TypeError) as error:
            raise RunArchiveError(
                f"invalid checkpoint: {self.relative_path}"
            ) from error


@dataclass(frozen=True, slots=True)
class RunArchive:
    """Validated manifest, state, and safe checkpoint references."""

    root: Path
    manifest: SealedManifest
    manifest_hash: Sha256
    state: ArchiveState
    algorithm_state: CheckpointRef | None

    @classmethod
    def open(cls, root: Path) -> RunArchive:
        try:
            resolved_root = root.resolve(strict=True)
        except OSError as error:
            raise RunArchiveError(f"run archive does not exist: {root}") from error
        if not resolved_root.is_dir():
            raise RunArchiveError(f"run archive is not a directory: {root}")
        try:
            manifest = SealedManifest.model_validate_json(
                _read_safe_file(resolved_root, "manifest.json")
            )
        except (ValueError, TypeError) as error:
            raise RunArchiveError(f"invalid run manifest: {root}") from error
        state = load_archive_state(resolved_root / "state.json")
        manifest_hash = canonical_hash(manifest)
        if state.manifest_hash != manifest_hash:
            raise RunArchiveError(f"manifest hash mismatch in {root}")
        _validate_state_consistency(state)
        records = state.checkpoint_records()
        for record in records:
            _safe_file_path(resolved_root, record.relative_path)
        return cls(
            root=resolved_root,
            manifest=manifest,
            manifest_hash=manifest_hash,
            state=state,
            algorithm_state=(
                _bind_checkpoint(resolved_root, state.algorithm_state)
                if state.algorithm_state is not None
                else None
            ),
        )

    @property
    def archive_version(self) -> int:
        return self.state.archive_version

    @property
    def terminal(self) -> TerminalState | None:
        return self.state.terminal

    @property
    def prepared_commit(self) -> PreparedCommitState | None:
        return self.state.prepared_commit

    def require_complete(self, expected_rounds: int) -> RunArchive:
        """Require exactly ``expected_rounds`` committed rounds."""
        if expected_rounds <= 0:
            raise ValueError("expected_rounds must be positive")
        if (
            self.terminal is None
            or self.terminal.result != "complete"
            or self.state.committed_round != expected_rounds - 1
            or self.prepared_commit is not None
        ):
            raise RunArchiveError(
                f"run archive is not complete through {expected_rounds} rounds"
            )
        return self

    def verify_checkpoint_integrity(self) -> None:
        if not self.state.integrity_recorded:
            raise RunArchiveError("legacy archive has no recorded checkpoint integrity")
        seen: set[tuple[str, str]] = set()
        for record in self.state.checkpoint_records():
            assert record.sha256 is not None
            key = (record.relative_path, record.sha256)
            if key in seen:
                continue
            seen.add(key)
            _bind_checkpoint(self.root, record).verify()


def load_archive_state(path: Path) -> ArchiveState:
    try:
        root = path.parent.resolve(strict=True)
        value = cast(
            object,
            json.loads(
                _read_safe_file(root, path.name),
                parse_constant=_reject_json_constant,
            ),
        )
    except (OSError, ValueError) as error:
        raise RunArchiveError(f"run state is unreadable: {path}") from error
    record = _mapping(value, "run state")
    version = _integer(record.get("archive_version", 0), "archive_version")
    if version not in (0, 1, ARCHIVE_VERSION):
        raise RunArchiveError(f"unsupported archive version: {version}")
    fields = (
        _CURRENT_STATE_FIELDS
        if version == ARCHIVE_VERSION
        else _LEGACY_STATE_FIELDS
    )
    unknown = set(record) - fields
    optional_fields: set[str] = (
        {"archive_version", "prepared_commit"}
        if version == 0
        else set()
    )
    missing = fields - optional_fields - set(record)
    if unknown or missing:
        raise RunArchiveError("run state fields are invalid")
    empty_checkpoint_map: Mapping[int, CheckpointRecord] = MappingProxyType({})
    legacy_pre_mix: Mapping[int, CheckpointRecord] = (
        _checkpoint_map(
            record.get("pre_mix_checkpoints"), version, "pre_mix_checkpoints"
        )
        if version < ARCHIVE_VERSION
        else empty_checkpoint_map
    )
    legacy_post_mix: Mapping[int, CheckpointRecord] = (
        _checkpoint_map(
            record.get("post_mix_checkpoints"), version, "post_mix_checkpoints"
        )
        if version < ARCHIVE_VERSION
        else empty_checkpoint_map
    )
    if set(legacy_pre_mix) != set(legacy_post_mix):
        raise RunArchiveError("pre/post checkpoint rounds do not match")
    legacy_checkpoints = (
        *legacy_pre_mix.values(),
        *legacy_post_mix.values(),
    )
    state = ArchiveState(
        archive_version=version,
        manifest_hash=_sha256(record.get("manifest_hash"), "manifest_hash"),
        committed_round=_integer(
            record.get("committed_round"), "committed_round", minimum=-1
        ),
        state_checksum=_optional_sha256(record.get("state_checksum"), "state_checksum"),
        algorithm_state=_optional_checkpoint(
            record.get("algorithm_state"), version, "algorithm_state"
        ),
        prepared_commit=_optional_prepared(record.get("prepared_commit"), version),
        schedule_history=_record_list(
            record.get("schedule_history"), "schedule_history"
        ),
        metrics=_record_list(record.get("metrics"), "metrics"),
        transfer_diagnostics=_record_list(
            record.get("transfer_diagnostics"), "transfer_diagnostics"
        ),
        consensus=_consensus_list(record.get("consensus")),
        terminal=_optional_terminal(record.get("terminal")),
        legacy_checkpoint_records=legacy_checkpoints,
    )
    _validate_state_consistency(state)
    return state


def _prepared_as_json(value: PreparedCommitState) -> dict[str, object]:
    return {
        "round_id": value.round_id,
        "state_checksum": value.state_checksum,
        "algorithm_state": value.algorithm_state.as_json(),
        "schedule": _thaw_json(value.schedule),
        "metrics": _thaw_json(value.metrics) if value.metrics is not None else None,
        "transfer_diagnostics": (
            _thaw_json(value.transfer_diagnostics)
            if value.transfer_diagnostics is not None
            else None
        ),
    }


def _optional_prepared(value: object, version: int) -> PreparedCommitState | None:
    if value is None:
        return None
    record = _mapping(value, "prepared_commit")
    expected = {
        "round_id",
        "state_checksum",
        "algorithm_state",
        "schedule",
        "metrics",
        "transfer_diagnostics",
    }
    if version < ARCHIVE_VERSION:
        expected |= {"pre_mix_checkpoint", "post_mix_checkpoint"}
    if set(record) != expected:
        raise RunArchiveError("prepared_commit fields are invalid")
    legacy_checkpoints = (
        (
            _checkpoint(
                record["pre_mix_checkpoint"], version, "prepared pre_mix_checkpoint"
            ),
            _checkpoint(
                record["post_mix_checkpoint"], version, "prepared post_mix_checkpoint"
            ),
        )
        if version < ARCHIVE_VERSION
        else ()
    )
    return PreparedCommitState(
        round_id=_integer(record["round_id"], "prepared round", minimum=0),
        state_checksum=_sha256(record["state_checksum"], "prepared state_checksum"),
        algorithm_state=_checkpoint(
            record["algorithm_state"], version, "prepared algorithm_state"
        ),
        schedule=_immutable_record(record["schedule"], "prepared schedule"),
        metrics=_optional_record(record["metrics"], "prepared metrics"),
        transfer_diagnostics=_optional_record(
            record["transfer_diagnostics"], "prepared transfer_diagnostics"
        ),
        legacy_checkpoint_records=legacy_checkpoints,
    )


def _checkpoint_map(
    value: object, version: int, label: str
) -> Mapping[int, CheckpointRecord]:
    entries = _mapping(value, label)
    result: dict[int, CheckpointRecord] = {}
    for round_value, checkpoint_value in entries.items():
        if not round_value.isdecimal():
            raise RunArchiveError(f"{label} contains an invalid round")
        round_id = int(round_value)
        if round_id in result:
            raise RunArchiveError(f"{label} contains a duplicate round")
        result[round_id] = _checkpoint(
            checkpoint_value, version, f"{label}[{round_value}]"
        )
    return MappingProxyType(result)


def _optional_checkpoint(
    value: object, version: int, label: str
) -> CheckpointRecord | None:
    return None if value is None else _checkpoint(value, version, label)


def _checkpoint(value: object, version: int, label: str) -> CheckpointRecord:
    if version == 0:
        if not isinstance(value, str):
            raise RunArchiveError(f"{label} must be a path")
        relative_path = value
        digest = None
    else:
        record = _mapping(value, label)
        if set(record) != {"path", "sha256"}:
            raise RunArchiveError(f"{label} fields are invalid")
        path_value = record["path"]
        if not isinstance(path_value, str):
            raise RunArchiveError(f"{label} path is invalid")
        relative_path = path_value
        digest = _sha256(record["sha256"], f"{label} sha256")
    _validate_relative_path(relative_path)
    return CheckpointRecord(relative_path=relative_path, sha256=digest)


def _record_list(value: object, label: str) -> tuple[JsonRecord, ...]:
    if not isinstance(value, list):
        raise RunArchiveError(f"{label} must be a list")
    return tuple(
        _immutable_record(item, f"{label} entry") for item in cast(list[object], value)
    )


def _consensus_list(value: object) -> tuple[ConsensusState, ...]:
    if not isinstance(value, list):
        raise RunArchiveError("consensus must be a list")
    result: list[ConsensusState] = []
    seen: set[int] = set()
    for item in cast(list[object], value):
        record = _mapping(item, "consensus entry")
        if set(record) != {"round_id", "normalized_rms", "sketch_count"}:
            raise RunArchiveError("consensus entry fields are invalid")
        round_id = _integer(record["round_id"], "consensus round", minimum=0)
        distance = record["normalized_rms"]
        if (
            isinstance(distance, bool)
            or not isinstance(distance, (int, float))
            or not math.isfinite(distance)
            or distance < 0
        ):
            raise RunArchiveError("consensus distance is invalid")
        sketch_count = _integer(
            record["sketch_count"], "consensus sketch_count", minimum=1
        )
        if round_id in seen:
            raise RunArchiveError("consensus rounds must be unique")
        seen.add(round_id)
        result.append(
            ConsensusState(
                round_id=round_id,
                normalized_rms=float(distance),
                sketch_count=sketch_count,
            )
        )
    return tuple(result)


def _optional_terminal(value: object) -> TerminalState | None:
    if value is None:
        return None
    record = _mapping(value, "terminal")
    if set(record) != {"result", "diagnostics"}:
        raise RunArchiveError("terminal fields are invalid")
    result = record["result"]
    if not isinstance(result, str) or not result:
        raise RunArchiveError("terminal result is invalid")
    return TerminalState(
        result=result,
        diagnostics=_immutable_record(record["diagnostics"], "terminal diagnostics"),
    )


def _validate_state_consistency(state: ArchiveState) -> None:
    if state.committed_round == -1:
        if (
            state.state_checksum is not None
            or state.algorithm_state is not None
            or state.schedule_history
        ):
            raise RunArchiveError("empty archive contains committed state")
    else:
        if state.archive_version < ARCHIVE_VERSION and len(
            state.legacy_checkpoint_records
        ) != 2 * (state.committed_round + 1):
            raise RunArchiveError("committed archive state is incomplete")
        if (
            state.state_checksum is None
            or state.algorithm_state is None
            or len(state.schedule_history) != state.committed_round + 1
        ):
            raise RunArchiveError("committed archive state is incomplete")
    if (
        state.prepared_commit is not None
        and state.prepared_commit.round_id <= state.committed_round
    ):
        raise RunArchiveError("prepared commit does not advance the archive")
    if state.archive_version == ARCHIVE_VERSION and not state.integrity_recorded:
        raise RunArchiveError("archive v2 checkpoint hashes are incomplete")


def _bind_checkpoint(root: Path, record: CheckpointRecord) -> CheckpointRef:
    _safe_file_path(root, record.relative_path)
    return CheckpointRef(
        _root=root,
        relative_path=record.relative_path,
        recorded_sha256=record.sha256,
    )


def _read_safe_file(root: Path, relative_path: str) -> bytes:
    path = _safe_file_path(root, relative_path)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RunArchiveError(f"cannot open archive file: {relative_path}") from error
    try:
        with os.fdopen(descriptor, "rb") as handle:
            return handle.read()
    except OSError as error:
        raise RunArchiveError(f"cannot read archive file: {relative_path}") from error


def _safe_file_path(root: Path, relative_path: str) -> Path:
    _validate_relative_path(relative_path)
    path = root
    try:
        for part in PurePosixPath(relative_path).parts:
            path = path / part
            if path.is_symlink():
                raise RunArchiveError(
                    f"archive path contains a symlink: {relative_path}"
                )
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise RunArchiveError(f"archive file is missing: {relative_path}") from error
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise RunArchiveError(f"archive path is unsafe: {relative_path}")
    return resolved


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in value
    ):
        raise RunArchiveError(f"archive path is unsafe: {value}")


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RunArchiveError(f"{label} must be a JSON object")
    entries = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in entries):
        raise RunArchiveError(f"{label} must be a JSON object")
    return cast(dict[str, object], entries)


def _immutable_record(value: object, label: str) -> JsonRecord:
    record = _mapping(value, label)
    return cast(JsonRecord, _freeze_json(record))


def _optional_record(value: object, label: str) -> JsonRecord | None:
    return None if value is None else _immutable_record(value, label)


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        entries = cast(dict[object, object], value)
        if not all(isinstance(key, str) for key in entries):
            raise RunArchiveError("JSON object keys must be strings")
        return MappingProxyType(
            {cast(str, key): _freeze_json(item) for key, item in entries.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in cast(list[object], value))
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise RunArchiveError("archive contains an invalid JSON value")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        entries = cast(Mapping[object, object], value)
        return {cast(str, key): _thaw_json(item) for key, item in entries.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in cast(tuple[object, ...], value)]
    return value


def _integer(value: object, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RunArchiveError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise RunArchiveError(f"{label} is below its minimum")
    return value


def _sha256(value: object, label: str) -> Sha256:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RunArchiveError(f"{label} must be a SHA-256 digest")
    return value


def _optional_sha256(value: object, label: str) -> Sha256 | None:
    return None if value is None else _sha256(value, label)


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"invalid JSON constant: {value}")


__all__ = [
    "ARCHIVE_VERSION",
    "ArchiveState",
    "CheckpointRecord",
    "CheckpointRef",
    "ConsensusState",
    "PreparedCommitState",
    "RunArchive",
    "RunArchiveError",
    "TerminalState",
    "load_archive_state",
]
