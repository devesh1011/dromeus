"""Progress items read from one node's live JSONL event log.

`read_progress` follows the log a node writes under its run root and turns the
few events a live terminal view needs (start, readiness, committed rounds,
consensus distance, completion or failure) into typed items. Every other event
is ignored. The CLI only formats these items; the parsing rules live here.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from io import BufferedReader
from pathlib import Path
from typing import cast

from dromeus.errors import ConfigError

_CHUNK_BYTES = 64 * 1024
_ROUND_SECONDS_FIELDS = (
    "local_compute_seconds",
    "peer_wait_seconds",
    "transfer_seconds",
    "mixing_seconds",
    "evaluation_seconds",
)


@dataclass(frozen=True, slots=True)
class NodeStarted:
    run_id: str
    role: str


@dataclass(frozen=True, slots=True)
class NodeReady:
    run_id: str
    manifest_hash: str


@dataclass(frozen=True, slots=True)
class RoundCommitted:
    round_id: int
    peer_id: str
    local_loss: float | None
    metrics: dict[str, float]
    round_seconds: float
    encoded_bytes: int | None


@dataclass(frozen=True, slots=True)
class ConsensusMeasured:
    round_id: int
    normalized_rms: float


@dataclass(frozen=True, slots=True)
class RunCompleted:
    committed_rounds: int


@dataclass(frozen=True, slots=True)
class RunFailed:
    error: str
    round_id: int | None


type Progress = (
    NodeStarted
    | NodeReady
    | RoundCommitted
    | ConsensusMeasured
    | RunCompleted
    | RunFailed
)


class _Malformed(Exception):
    """A record is missing or mistypes a field its progress item needs."""


def parse_progress(record: Mapping[str, object]) -> Progress | None:
    """Map one decoded log record to a progress item, or None if it is not one."""
    event = record.get("event")
    try:
        if event == "node_start":
            return NodeStarted(
                run_id=_str(record, "run_id"),
                role=_str(record, "role"),
            )
        if event == "node_ready":
            return NodeReady(
                run_id=_str(record, "run_id"),
                manifest_hash=_str(record, "manifest_hash"),
            )
        if event == "round_metrics":
            return _round_committed(record)
        if event == "consensus_observation":
            return ConsensusMeasured(
                round_id=_int(record, "round_id"),
                normalized_rms=_float(record, "normalized_rms"),
            )
        if event == "node_complete":
            return RunCompleted(committed_rounds=_int(record, "committed_rounds"))
        if event in ("run_failed", "formation_failed"):
            return RunFailed(
                error=_str(record, "error"),
                round_id=_optional_int(record, "round_id"),
            )
    except _Malformed:
        return None
    return None


def read_progress(
    path: Path,
    *,
    follow: bool,
    poll_seconds: float = 0.5,
) -> Iterator[Progress]:
    """Yield progress items from a node's JSONL log, in log order.

    With `follow=False` the existing file is read to its end. With `follow=True`
    the file may appear later and is polled like `tail -f`. The iterator stops
    right after a terminal item (`RunCompleted` or `RunFailed`). A missing file
    raises `ConfigError` immediately when `follow=False`.
    """
    if not follow and not path.is_file():
        raise _missing_log(path)
    return _stream(path, follow=follow, poll_seconds=poll_seconds)


def _stream(path: Path, *, follow: bool, poll_seconds: float) -> Iterator[Progress]:
    with _open_log(path, follow=follow, poll_seconds=poll_seconds) as handle:
        pending = b""
        while True:
            chunk = handle.read(_CHUNK_BYTES)
            if not chunk:
                if not follow:
                    return
                time.sleep(poll_seconds)
                continue
            # Newline bytes never occur inside a UTF-8 sequence, so each complete
            # line decodes on its own even when a chunk ends mid-character.
            *lines, pending = (pending + chunk).split(b"\n")
            for line in lines:
                item = _parse_line(line)
                if item is None:
                    continue
                yield item
                if isinstance(item, RunCompleted | RunFailed):
                    return


def _open_log(path: Path, *, follow: bool, poll_seconds: float) -> BufferedReader:
    while True:
        try:
            return path.open("rb")
        except FileNotFoundError:
            if not follow:
                raise _missing_log(path) from None
            time.sleep(poll_seconds)


def _parse_line(line: bytes) -> Progress | None:
    try:
        value = cast(object, json.loads(line.decode("utf-8")))
    except (ValueError, RecursionError):
        return None
    if not isinstance(value, dict):
        return None
    return parse_progress(cast(dict[str, object], value))


def _missing_log(path: Path) -> ConfigError:
    return ConfigError(
        f"node event log not found: {path}; check that the run root is right "
        "and that the node has started"
    )


def _round_committed(record: Mapping[str, object]) -> RoundCommitted:
    round_seconds = 0.0
    for key in _ROUND_SECONDS_FIELDS:
        round_seconds += _float(record, key)
    return RoundCommitted(
        round_id=_int(record, "round_id"),
        peer_id=_str(record, "peer_id"),
        local_loss=_optional_float(record, "local_loss"),
        metrics=_metrics(record, "evaluation_metrics"),
        round_seconds=round_seconds,
        encoded_bytes=_optional_int(record, "encoded_artifact_bytes"),
    )


def _str(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise _Malformed(key)
    return value


def _int(record: Mapping[str, object], key: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _Malformed(key)
    return value


def _float(record: Mapping[str, object], key: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _Malformed(key)
    return float(value)


def _optional_int(record: Mapping[str, object], key: str) -> int | None:
    if record.get(key) is None:
        return None
    return _int(record, key)


def _optional_float(record: Mapping[str, object], key: str) -> float | None:
    if record.get(key) is None:
        return None
    return _float(record, key)


def _metrics(record: Mapping[str, object], key: str) -> dict[str, float]:
    value = record.get(key)
    if not isinstance(value, dict):
        raise _Malformed(key)
    metrics: dict[str, float] = {}
    for name, number in cast(dict[object, object], value).items():
        if not isinstance(name, str) or isinstance(number, bool):
            raise _Malformed(key)
        if not isinstance(number, int | float):
            raise _Malformed(key)
        metrics[name] = float(number)
    return metrics


__all__ = [
    "ConsensusMeasured",
    "NodeReady",
    "NodeStarted",
    "Progress",
    "RoundCommitted",
    "RunCompleted",
    "RunFailed",
    "parse_progress",
    "read_progress",
]
