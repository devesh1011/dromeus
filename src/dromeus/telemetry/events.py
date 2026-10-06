"""Structured event output shared by Dromeus runtime modules."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
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
        try:
            sink.append(record)
        except Exception:
            pass
        return
    try:
        print(
            json.dumps(record, separators=(",", ":"), sort_keys=True),
            file=sys.stdout,
        )
    except Exception:
        pass


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


class BoundedEventSink:
    """Nonblocking facade over a possibly blocking sink, with bounded shutdown.

    A timed-out in-flight write may complete later; no further writes are started.
    All users sharing the target sink must share this facade to avoid lock contention.
    """

    def __init__(self, sink: EventSink | None, *, capacity: int = 64) -> None:
        if capacity <= 0:
            raise ValueError("event capacity must be positive")
        self._sink = sink
        self._queue: Queue[Mapping[str, object]] = Queue(maxsize=capacity)
        self._count_lock = Lock()
        self._stop = Event()
        self._abort = Event()
        self._thread: Thread | None = None
        self._in_flight = False
        self._dropped = 0

    @property
    def dropped(self) -> int:
        with self._count_lock:
            return self._dropped

    def start(self) -> None:
        with self._count_lock:
            if self._thread is not None or self._stop.is_set():
                return
            self._thread = Thread(
                target=self._run, name="dromeus-event-delivery", daemon=True
            )
            self._thread.start()

    def append(self, record: Mapping[str, object]) -> None:
        with self._count_lock:
            if self._stop.is_set():
                self._dropped += 1
                return
            try:
                self._queue.put_nowait(dict(record))
            except Full:
                self._dropped += 1

    def _run(self) -> None:
        while not self._abort.is_set():
            if self._stop.is_set() and self._queue.empty():
                return
            try:
                record = self._queue.get(timeout=0.05)
            except Empty:
                continue
            with self._count_lock:
                if self._abort.is_set():
                    self._dropped += 1
                    self._queue.task_done()
                    return
                self._in_flight = True
            success = False
            try:
                if self._sink is not None:
                    self._sink.append(record)
                    success = True
            except Exception:
                pass
            with self._count_lock:
                if not success and not self._abort.is_set():
                    self._dropped += 1
                self._in_flight = False
            self._queue.task_done()

    def stop(self, *, timeout_seconds: float = 0.2) -> bool:
        if timeout_seconds <= 0:
            raise ValueError("event stop timeout must be positive")
        with self._count_lock:
            self._stop.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout_seconds)
            if not thread.is_alive():
                return True
        with self._count_lock:
            if not self._abort.is_set():
                self._abort.set()
                self._dropped += int(self._in_flight)
            while True:
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                    self._dropped += 1
                except Empty:
                    break
        return thread is None or not thread.is_alive()
