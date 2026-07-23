"""Structured event output shared by Dromeus runtime modules."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Protocol


class EventSink(Protocol):
    def append(self, record: Mapping[str, object]) -> None: ...


@dataclass(slots=True)
class JsonlEventSink:
    """Thread-safe append-only JSONL sink for one node's telemetry."""

    path: Path
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Mapping[str, object]) -> None:
        line = json.dumps(
            dict(record),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")
            handle.flush()


def emit_event(
    event: str,
    *,
    run_id: str | None = None,
    manifest_hash: str | None = None,
    node_id: str | None = None,
    message_id: str | None = None,
    transfer_id: str | None = None,
    peer_id: str | None = None,
    round_id: int | None = None,
    sink: EventSink | None = None,
    **fields: object,
) -> None:
    """Write one JSON event, retaining correlation IDs when supplied."""
    record = event_record(
        event,
        run_id=run_id,
        manifest_hash=manifest_hash,
        node_id=node_id,
        message_id=message_id,
        transfer_id=transfer_id,
        peer_id=peer_id,
        round_id=round_id,
        **fields,
    )
    if sink is not None:
        sink.append(record)
        return
    print(json.dumps(record, separators=(",", ":"), sort_keys=True), file=sys.stdout)


def event_record(
    event: str,
    *,
    run_id: str | None = None,
    manifest_hash: str | None = None,
    node_id: str | None = None,
    message_id: str | None = None,
    transfer_id: str | None = None,
    peer_id: str | None = None,
    round_id: int | None = None,
    **fields: object,
) -> dict[str, object]:
    """Build one deterministic structured event without writing it."""
    record: dict[str, object] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": event,
    }
    identifiers: Mapping[str, str | int | None] = {
        "run_id": run_id,
        "manifest_hash": manifest_hash,
        "node_id": node_id,
        "message_id": message_id,
        "transfer_id": transfer_id,
        "peer_id": peer_id,
        "round_id": round_id,
    }
    record.update(
        {key: value for key, value in identifiers.items() if value is not None}
    )
    record.update(fields)
    return record


__all__ = ["EventSink", "JsonlEventSink", "emit_event", "event_record"]
