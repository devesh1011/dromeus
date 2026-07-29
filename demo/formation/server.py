"""Serve and drive the real-AXL formation dashboard."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import threading
import time
import webbrowser
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from safetensors.numpy import save_file  # pyright: ignore[reportUnknownVariableType]

from dromeus.manifests.models import DraftRunSpec, Tensor, TensorSchema
from dromeus.membership.protocol import create_invitation
from dromeus.runtime import NodeRuntime, NodeState
from dromeus.training.cifar10 import DATASET_VERSION, PREPROCESSING_HASH
from dromeus.training.models import MODEL_DEFINITION_HASH
from dromeus.transport.axl import AXLBridgeConfig, AXLTransport
from dromeus.transport.base import ReceivedBytes
from dromeus.transport.envelope import Envelope, MessageType, decode_envelope
from dromeus.transport.transfer import ArtifactStore

ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = Path(__file__).with_name("static")
CACHE_ROOT = ROOT / ".demo-cache"
AXL_SOURCE = "https://github.com/gensyn-ai/axl.git"
AXL_COMMIT = "628e28ace077f26dfe8d0259009b357216a9d8d4"
AXL_PORTS = (9302, 9303, 9304, 9305)
AXL_LISTEN_PORT = 9300
MAX_ENVELOPE_BYTES = 4 * 1024 * 1024


class TransportObserver(Protocol):
    """Observation seam shared by local and containerized demo transports."""

    def record_send(self, envelope: Envelope, destination: str) -> int: ...

    def send_accepted(self, sequence: int) -> None: ...

    def send_failed(self, sequence: int, error: str) -> None: ...

    def record_receive(self, envelope: Envelope, destination: str) -> None: ...


class DashboardState:
    """Thread-safe state projected into the browser."""

    def __init__(self, *, containerized: bool = False) -> None:
        self._lock = threading.Lock()
        self._containerized = containerized
        self._generation = 0
        self._started_at: float | None = None
        self._completed_elapsed: float | None = None
        self._key_to_node: dict[str, str] = {}
        self._nodes: list[dict[str, object]] = []
        self._remote_event_keys: set[tuple[str, int]] = set()
        self._remote_event_sequences: dict[tuple[str, int], int] = {}
        self._event_wire_keys: dict[int, tuple[str, str]] = {}
        self._remote_deliveries: set[tuple[str, str, str]] = set()
        self._pending_remote_deliveries: set[tuple[str, str, str]] = set()
        self._reset_locked(round_count=2)

    def reset(self, *, round_count: int = 2) -> None:
        with self._lock:
            self._generation += 1
            self._reset_locked(round_count=round_count)

    def _reset_locked(self, *, round_count: int) -> None:
        self._status = "idle"
        self._status_detail = "Ready to launch four local AXL nodes"
        self._run_id = "run-axl-demo"
        self._round_count = round_count
        self._manifest_hash: str | None = None
        self._error: str | None = None
        self._events: list[dict[str, object]] = []
        self._next_sequence = 1
        self._started_at = None
        self._completed_elapsed = None
        self._key_to_node = {}
        self._remote_event_keys = set()
        self._remote_event_sequences = {}
        self._event_wire_keys = {}
        self._remote_deliveries = set()
        self._pending_remote_deliveries = set()
        self._training_log_keys: set[tuple[str, str, str]] = set()
        self._training_logs: list[dict[str, object]] = []
        self._nodes = [
            {
                "id": f"node-{index}",
                "name": "Initiator" if index == 0 else f"Participant {index}",
                "role": "initiator" if index == 0 else "participant",
                "port": port,
                "container": f"dromeus-node-{index}",
                "key": None,
                "short_key": "identity pending",
                "state": "offline",
                "progress": 0,
            }
            for index, port in enumerate(AXL_PORTS)
        ]

    def preparing(self, detail: str) -> None:
        with self._lock:
            self._status = "preparing"
            self._status_detail = detail
            for node in self._nodes:
                node["state"] = "booting"

    def register_nodes(self, keys: list[str]) -> None:
        with self._lock:
            self._key_to_node = {
                key: cast(str, self._nodes[index]["id"])
                for index, key in enumerate(keys)
            }
            for index, key in enumerate(keys):
                self._nodes[index]["key"] = key
                self._nodes[index]["short_key"] = _short_key(key)
                self._nodes[index]["state"] = "connected"
            self._status = "running"
            self._status_detail = "Formation envelopes are crossing real AXL bridges"
            self._started_at = time.monotonic()

    def register_node(self, index: int, key: str) -> None:
        """Register one node reported by a container worker."""
        with self._lock:
            if not 0 <= index < len(self._nodes):
                return
            self._key_to_node[key] = cast(str, self._nodes[index]["id"])
            self._nodes[index]["key"] = key
            self._nodes[index]["short_key"] = _short_key(key)
            self._nodes[index]["state"] = "connected"
            self._relabel_remote_events_locked()
            if all(node["key"] for node in self._nodes):
                if self._status == "preparing":
                    self._status = "running"
                    self._status_detail = (
                        "Four containerized nodes are crossing real AXL bridges"
                    )
                    self._started_at = self._started_at or time.monotonic()

    def ingest_remote_snapshot(
        self, index: int, snapshot: dict[str, object]
    ) -> None:
        """Merge one worker's event stream into the dashboard ledger."""
        key = snapshot.get("key")
        if isinstance(key, str):
            self.register_node(index, key)
        with self._lock:
            self._ingest_training_locked(index, snapshot)
        events = snapshot.get("events")
        if not isinstance(events, list):
            return
        with self._lock:
            for raw in cast(list[object], events):
                if not isinstance(raw, dict):
                    continue
                event = cast(dict[object, object], raw)
                sequence_value = event.get("sequence")
                if not isinstance(sequence_value, int):
                    continue
                remote_id = (str(index), sequence_value)
                kind = event.get("kind")
                if remote_id in self._remote_event_keys:
                    dashboard_sequence = self._remote_event_sequences.get(remote_id)
                    if dashboard_sequence is not None:
                        self._update_remote_event_locked(dashboard_sequence, event)
                    continue
                self._remote_event_keys.add(remote_id)
                if kind == "receive":
                    self._record_remote_delivery_locked(event)
                    continue
                if kind == "invitation":
                    source = event.get("source")
                    detail = event.get("detail")
                    if isinstance(source, str):
                        self._append_event_locked(
                            event_type="INVITATION",
                            source=self._node_id_locked(source),
                            target="all-participants",
                            transport="OUT_OF_BAND",
                            message_id="invitation",
                            payload_bytes=0,
                            detail=str(detail or ""),
                            delivered=True,
                        )
                    continue
                if kind != "send":
                    continue
                source = event.get("source")
                target = event.get("target")
                event_type = event.get("type")
                message_id = event.get("message_id")
                if not all(
                    isinstance(item, str)
                    for item in (source, target, event_type, message_id)
                ):
                    continue
                payload_bytes = event.get("payload_bytes", 0)
                dashboard_sequence = self._append_event_locked(
                    event_type=cast(str, event_type),
                    source=self._node_id_locked(cast(str, source)),
                    target=self._node_id_locked(cast(str, target)),
                    transport="AXL",
                    message_id=cast(str, message_id),
                    payload_bytes=(
                        payload_bytes if isinstance(payload_bytes, int) else 0
                    ),
                    detail=str(event.get("detail", "")),
                    delivered=bool(event.get("delivered", False)),
                )
                self._remote_event_sequences[remote_id] = dashboard_sequence
                self._event_wire_keys[dashboard_sequence] = (
                    cast(str, source),
                    cast(str, target),
                )
                self._update_remote_event_locked(dashboard_sequence, event)
            self._relabel_remote_events_locked()

    def _ingest_training_locked(
        self, index: int, snapshot: dict[str, object]
    ) -> None:
        if not 0 <= index < len(self._nodes):
            return
        training = snapshot.get("training")
        if not isinstance(training, dict):
            return
        node = self._nodes[index]
        completed_rounds = training.get("completed_rounds")
        if isinstance(completed_rounds, int):
            node["training_round"] = completed_rounds
            node["progress"] = min(
                100,
                round(100 * completed_rounds / max(self._round_count, 1)),
            )
        node["training_round_count"] = self._round_count
        for field in ("local_loss", "evaluation_loss", "evaluation_accuracy"):
            value = training.get(field)
            if isinstance(value, (int, float)):
                node[field] = value
        status = snapshot.get("status")
        if status == "formed":
            node["state"] = "ready"
            if self._status in {"running", "formed"}:
                self._status = "formed"
                self._status_detail = (
                    "Four nodes formed; ready to start training"
                )
        elif status == "training":
            node["state"] = "training"
            self._status = "training"
            self._status_detail = (
                f"Training round {completed_rounds or 0} / {self._round_count} "
                "over real AXL"
            )
        elif status == "complete":
            node["state"] = "ready"
            node["progress"] = 100
        logs = training.get("logs")
        if not isinstance(logs, list):
            return
        for raw in logs:
            if not isinstance(raw, dict):
                continue
            log = cast(dict[object, object], raw)
            timestamp = log.get("timestamp")
            message = log.get("message")
            if not isinstance(timestamp, str) or not isinstance(message, str):
                continue
            key = (str(index), timestamp, message)
            if key in self._training_log_keys:
                continue
            self._training_log_keys.add(key)
            self._training_logs.append(
                {
                    "timestamp": timestamp,
                    "node": f"N{index}",
                    "message": message,
                    "round_id": log.get("round_id"),
                    "local_loss": log.get("local_loss"),
                    "evaluation_loss": log.get("evaluation_loss"),
                    "evaluation_accuracy": log.get("evaluation_accuracy"),
                }
            )
        self._training_logs.sort(key=lambda item: str(item["timestamp"]))
        del self._training_logs[:-300]

    def _update_remote_event_locked(
        self, dashboard_sequence: int, event: dict[object, object]
    ) -> None:
        current = self._event_locked(dashboard_sequence)
        if current is None:
            return
        current["bridge_accepted"] = bool(
            event.get("bridge_accepted", current["bridge_accepted"])
        )
        current["delivered"] = bool(event.get("delivered", current["delivered"]))
        if event.get("failed"):
            current["failed"] = True
            current["error"] = str(event.get("error", ""))

    def _record_remote_delivery_locked(self, event: dict[object, object]) -> None:
        source = event.get("source")
        target = event.get("target")
        message_id = event.get("message_id")
        if not all(isinstance(item, str) for item in (source, target, message_id)):
            return
        delivery = (cast(str, source), cast(str, target), cast(str, message_id))
        self._remote_deliveries.add(delivery)
        for sequence, wire_keys in self._event_wire_keys.items():
            event_record = self._event_locked(sequence)
            if (
                event_record is not None
                and wire_keys == delivery[:2]
                and event_record["message_id"] == delivery[2]
            ):
                event_record["delivered"] = True
                event_record["received_at"] = _utc_now()
                return
        self._pending_remote_deliveries.add(delivery)

    def _relabel_remote_events_locked(self) -> None:
        for sequence, (source, target) in self._event_wire_keys.items():
            event = self._event_locked(sequence)
            if event is None:
                continue
            event["source"] = self._node_id_locked(source)
            event["target"] = self._node_id_locked(target)
        pending = self._pending_remote_deliveries
        for sequence, wire_keys in self._event_wire_keys.items():
            event = self._event_locked(sequence)
            if event is None:
                continue
            delivery = (*wire_keys, event["message_id"])
            if delivery in self._remote_deliveries or delivery in pending:
                event["delivered"] = True
                pending.discard(delivery)

    def invitation(self, *, source_key: str, draft_hash: str) -> None:
        with self._lock:
            self._append_event_locked(
                event_type="INVITATION",
                source=self._node_id_locked(source_key),
                target="all-participants",
                transport="OUT_OF_BAND",
                message_id="invitation",
                payload_bytes=0,
                detail=f"draft {draft_hash[:12]}",
                delivered=True,
            )
            for node in self._nodes[1:]:
                node["state"] = "invited"

    def record_send(self, envelope: Envelope, destination: str) -> int:
        with self._lock:
            sequence = self._append_event_locked(
                event_type=envelope.message_type,
                source=self._node_id_locked(envelope.sender_public_key),
                target=self._node_id_locked(destination),
                transport="AXL",
                message_id=envelope.message_id,
                payload_bytes=envelope.payload_length,
                detail=_event_detail(envelope),
                delivered=False,
            )
            self._apply_send_locked(envelope.message_type, destination)
            return sequence

    def send_accepted(self, sequence: int) -> None:
        with self._lock:
            event = self._event_locked(sequence)
            if event is not None:
                event["bridge_accepted"] = True

    def send_failed(self, sequence: int, error: str) -> None:
        with self._lock:
            event = self._event_locked(sequence)
            if event is not None:
                event["failed"] = True
                event["error"] = error

    def record_receive(self, envelope: Envelope, destination: str) -> None:
        with self._lock:
            source = self._node_id_locked(envelope.sender_public_key)
            target = self._node_id_locked(destination)
            for event in reversed(self._events):
                if (
                    event["message_id"] == envelope.message_id
                    and event["source"] == source
                    and event["target"] == target
                    and not event["delivered"]
                ):
                    event["delivered"] = True
                    event["received_at"] = _utc_now()
                    break
            self._apply_receive_locked(
                envelope.message_type, envelope.sender_public_key
            )

    def complete(self, manifest_hash: str, *, detail: str | None = None) -> None:
        with self._lock:
            if self._started_at is not None:
                self._completed_elapsed = time.monotonic() - self._started_at
            self._status = "complete"
            self._status_detail = detail or (
                "Four nodes entered round zero with one manifest"
            )
            self._manifest_hash = manifest_hash
            for node in self._nodes:
                node["state"] = "ready"
                node["progress"] = 100

    def formation_ready(self, manifest_hash: str) -> None:
        with self._lock:
            self._status = "formed"
            self._status_detail = "Four nodes formed; ready to start training"
            self._manifest_hash = manifest_hash
            for node in self._nodes:
                node["state"] = "ready"
                node["progress"] = 100

    def training_started(self) -> None:
        with self._lock:
            self._status = "training"
            self._status_detail = (
                f"Training round 0 / {self._round_count} over real AXL"
            )

    def fail(self, error: str) -> None:
        with self._lock:
            was_training = self._status == "training"
            self._status = "failed"
            self._status_detail = (
                "Training stopped" if was_training else "Formation stopped"
            )
            self._error = error
            for node in self._nodes:
                if node["state"] != "ready":
                    node["state"] = "failed"

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            elapsed = self._completed_elapsed
            if elapsed is None and self._started_at is not None:
                elapsed = time.monotonic() - self._started_at
            return {
                "generation": self._generation,
                "status": self._status,
                "status_detail": self._status_detail,
                "run_id": self._run_id,
                "round_count": self._round_count,
                "manifest_hash": self._manifest_hash,
                "error": self._error,
                "elapsed_seconds": round(elapsed or 0.0, 2),
                "transport": "AXLTransport",
                "in_memory": False,
                "deployment": "docker" if self._containerized else "local",
                "nodes": [dict(node) for node in self._nodes],
                "events": [dict(event) for event in self._events],
                "training": {
                    "completed_rounds": max(
                        (
                            int(node.get("training_round", 0))
                            for node in self._nodes
                        ),
                        default=0,
                    ),
                    "round_count": self._round_count,
                    "logs": [dict(log) for log in self._training_logs],
                },
                "phases": self._phases_locked(),
            }

    def _append_event_locked(
        self,
        *,
        event_type: MessageType | str,
        source: str,
        target: str,
        transport: str,
        message_id: str,
        payload_bytes: int,
        detail: str,
        delivered: bool,
    ) -> int:
        sequence = self._next_sequence
        self._next_sequence += 1
        self._events.append(
            {
                "sequence": sequence,
                "timestamp": _utc_now(),
                "type": str(event_type),
                "source": source,
                "target": target,
                "transport": transport,
                "message_id": message_id,
                "payload_bytes": payload_bytes,
                "detail": detail,
                "bridge_accepted": transport != "AXL",
                "delivered": delivered,
                "failed": False,
            }
        )
        return sequence

    def _event_locked(self, sequence: int) -> dict[str, object] | None:
        return next(
            (event for event in self._events if event["sequence"] == sequence),
            None,
        )

    def _node_id_locked(self, public_key: str) -> str:
        return self._key_to_node.get(public_key, "unknown")

    def _node_locked(self, public_key: str) -> dict[str, object] | None:
        node_id = self._node_id_locked(public_key)
        return next((node for node in self._nodes if node["id"] == node_id), None)

    def _apply_send_locked(
        self, message_type: MessageType, destination: str
    ) -> None:
        destination_node = self._node_locked(destination)
        if destination_node is None:
            return
        if message_type is MessageType.JOIN_REQUEST:
            return
        if message_type is MessageType.JOIN_ACCEPTED:
            destination_node["state"] = "accepted"
        elif message_type is MessageType.MANIFEST_SEALED:
            destination_node["state"] = "manifest"
            destination_node["progress"] = 20
        elif message_type is MessageType.TRANSFER_BEGIN:
            destination_node["state"] = "syncing"
            destination_node["progress"] = 35
        elif message_type is MessageType.CHUNK:
            destination_node["state"] = "syncing"
            destination_node["progress"] = 75
        elif message_type is MessageType.TRANSFER_COMPLETE:
            destination_node["state"] = "verifying"
            destination_node["progress"] = 100
        elif message_type is MessageType.START:
            destination_node["state"] = "starting"

    def _apply_receive_locked(
        self, message_type: MessageType, source_public_key: str
    ) -> None:
        source_node = self._node_locked(source_public_key)
        if source_node is None:
            return
        if message_type is MessageType.JOIN_REQUEST:
            source_node["state"] = "joining"
        elif message_type is MessageType.READY:
            source_node["state"] = "ready"
        elif message_type is MessageType.START_ACK:
            source_node["state"] = "ready"

    def _phases_locked(self) -> list[dict[str, object]]:
        delivered = [event for event in self._events if event["delivered"]]

        def count(message_type: str) -> int:
            return sum(event["type"] == message_type for event in delivered)

        phases = [
            ("Invitation shared", count("INVITATION"), 1),
            ("Members accepted", count(str(MessageType.JOIN_ACCEPTED)), 3),
            ("Manifest agreed", count(str(MessageType.MANIFEST_SEALED)), 3),
            ("Checkpoint verified", count(str(MessageType.TRANSFER_COMPLETE)), 3),
            ("Readiness confirmed", count(str(MessageType.READY)), 3),
            ("Formation acknowledged", count(str(MessageType.START_ACK)), 3),
        ]
        return [
            {
                "name": name,
                "count": min(current, required),
                "required": required,
                "complete": current >= required,
            }
            for name, current, required in phases
        ]


class ObservedAXLTransport(AXLTransport):
    """Real AXL adapter with read-only dashboard observation."""

    def __init__(self, config: AXLBridgeConfig, state: TransportObserver) -> None:
        super().__init__(config)
        self._state = state

    async def send(self, destination: str, payload: bytes) -> None:
        local_key = await self.local_public_key()
        envelope = decode_envelope(
            payload,
            authenticated_sender=local_key,
            participant_keys=None,
            max_payload_bytes=MAX_ENVELOPE_BYTES,
        )
        sequence = self._state.record_send(envelope, destination)
        try:
            await super().send(destination, payload)
        except BaseException as error:
            self._state.send_failed(sequence, str(error))
            raise
        self._state.send_accepted(sequence)

    async def recv(self, timeout_seconds: float) -> ReceivedBytes | None:
        received = await super().recv(timeout_seconds)
        if received is None:
            return None
        envelope = decode_envelope(
            received.payload,
            authenticated_sender=received.sender_public_key,
            participant_keys=None,
            max_payload_bytes=MAX_ENVELOPE_BYTES,
        )
        self._state.record_receive(envelope, await self.local_public_key())
        return received


class ProcessRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: list[subprocess.Popen[str]] = []

    def set(self, processes: list[subprocess.Popen[str]]) -> None:
        with self._lock:
            self._processes = processes

    def terminate(self) -> None:
        with self._lock:
            processes, self._processes = self._processes, []
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()


class DemoController:
    def __init__(self, worker_urls: tuple[str, ...] = ()) -> None:
        self.state = DashboardState(containerized=bool(worker_urls))
        self._registry = ProcessRegistry()
        self._worker_urls = worker_urls
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._round_count = 2
        self._formation_ready = False
        self._training_requested = threading.Event()
        self._stopping = threading.Event()

    def start(self, *, round_count: int = 2) -> bool:
        return self.begin_formation(round_count=round_count)

    def begin_formation(self, *, round_count: int = 2) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                if self.state.snapshot()["status"] != "formed":
                    return False
                self._stopping.set()
                self._training_requested.set()
                self._thread.join(timeout=5.0)
                if self._thread.is_alive():
                    return False
            self._round_count = round_count
            self._formation_ready = False
            self._training_requested.clear()
            self._stopping.clear()
            self.state.reset(round_count=round_count)
            self.state.preparing("Checking the pinned AXL runtime")
            self._thread = threading.Thread(
                target=self._run_formation_stage,
                name="dromeus-formation-demo",
                daemon=True,
            )
            self._thread.start()
            return True

    def start_training(self) -> bool:
        with self._lock:
            if (
                not self._worker_urls
                or not self._formation_ready
                or self._stopping.is_set()
                or self.state.snapshot()["status"] != "formed"
            ):
                return False
            if self._thread is None or not self._thread.is_alive():
                return False
            self.state.training_started()
            self._training_requested.set()
            return True

    def stop(self) -> None:
        self._stopping.set()
        self._training_requested.set()
        self._registry.terminate()
        for url in self._worker_urls:
            try:
                _post_json(f"{url}/stop", {})
            except (HTTPError, URLError, TimeoutError):
                pass

    def _run_formation_stage(self) -> None:
        try:
            if self._worker_urls:
                manifest_hash = asyncio.run(
                    _run_remote_formation(
                        self.state, self._worker_urls, self._round_count
                    )
                )
                with self._lock:
                    self._formation_ready = True
                self.state.formation_ready(manifest_hash)
                self._training_requested.wait()
                if self._stopping.is_set():
                    return
                asyncio.run(
                    _run_remote_training(
                        self.state, self._worker_urls, self._round_count
                    )
                )
            else:
                asyncio.run(_run_formation(self.state, self._registry))
        except BaseException as error:
            self.state.fail(str(error))
            if self._worker_urls:
                self.stop()
        finally:
            self._registry.terminate()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            snapshot = self.state.snapshot()
            thread_running = self._thread is not None and self._thread.is_alive()
            snapshot["can_begin_formation"] = (
                not thread_running or snapshot["status"] == "formed"
            )
            snapshot["can_start_training"] = (
                bool(self._worker_urls)
                and thread_running
                and self._formation_ready
                and snapshot["status"] == "formed"
            )
            return snapshot


class DashboardHandler(BaseHTTPRequestHandler):
    controller: DemoController

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/state":
            self._json(HTTPStatus.OK, self.controller.snapshot())
            return
        static_files = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            "/favicon.svg": ("favicon.svg", "image/svg+xml"),
        }
        static = static_files.get(path)
        if static is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        filename, content_type = static
        body = (STATIC_ROOT / filename).read_bytes()
        self.send_response(HTTPStatus.OK)
        self._security_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path not in {"/api/start", "/api/train"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if path == "/api/train":
            started = self.controller.start_training()
            payload: dict[str, object] = {"started": started}
            if not started:
                payload["error"] = (
                    "formation is not ready or another run is still stopping"
                )
            self._json(
                HTTPStatus.ACCEPTED if started else HTTPStatus.CONFLICT,
                payload,
            )
            return
        try:
            payload = self._read_json()
            round_count = payload.get("round_count", 2)
            if not isinstance(round_count, int) or isinstance(round_count, bool):
                raise ValueError("round_count must be an integer")
            if not 1 <= round_count <= 100:
                raise ValueError("round_count must be between 1 and 100")
        except (ValueError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        started = self.controller.start(round_count=round_count)
        response: dict[str, object] = {"started": started}
        if not started:
            response["error"] = "another run is still stopping; retry shortly"
        self._json(
            HTTPStatus.ACCEPTED if started else HTTPStatus.CONFLICT,
            response,
        )

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, status: HTTPStatus, payload: object) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 4096:
            raise ValueError("request body is too large")
        if length == 0:
            return {}
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return cast(dict[str, object], value)

    def _security_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self'; "
            "script-src 'self'; connect-src 'self'",
        )


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True


async def _run_formation(
    state: DashboardState, registry: ProcessRegistry
) -> None:
    binary = _ensure_axl_binary(state)
    run_root = CACHE_ROOT / "runs" / datetime.now(UTC).strftime(
        "%Y%m%d-%H%M%S-%f"
    )
    run_root.mkdir(parents=True)
    state.preparing("Starting four pinned AXL nodes")
    processes = _start_axl_nodes(binary, run_root)
    registry.set(processes)
    await asyncio.gather(*(_wait_for_topology(port) for port in AXL_PORTS))
    await asyncio.gather(
        _wait_for_peer_count(AXL_PORTS[0], 3),
        *(_wait_for_peer_count(port, 1) for port in AXL_PORTS[1:]),
    )

    transports = [
        ObservedAXLTransport(
            AXLBridgeConfig(base_url=f"http://127.0.0.1:{port}"), state
        )
        for port in AXL_PORTS
    ]
    keys = list(await asyncio.gather(*(item.local_public_key() for item in transports)))
    state.register_nodes(keys)

    draft = _draft()
    checkpoint = run_root / "checkpoint.safetensors"
    tensor_schema = _write_checkpoint(checkpoint)
    nodes = [
        NodeRuntime(
            transport=transport,
            draft=draft,
            environment=draft.environment,
            dataset=draft.dataset,
            artifact_store=ArtifactStore(run_root / f"artifacts-{index}"),
        )
        for index, transport in enumerate(transports)
    ]
    invitation = create_invitation(
        draft=draft,
        initiator_public_key=keys[0],
        bootstrap_uri=f"tls://127.0.0.1:{AXL_LISTEN_PORT}",
    )
    state.invitation(source_key=keys[0], draft_hash=invitation.draft_hash)

    try:
        tasks = [
            asyncio.create_task(
                nodes[0].initiate(
                    bootstrap_uri=invitation.bootstrap_uri,
                    checkpoint_path=checkpoint,
                    tensor_schema=tensor_schema,
                )
            ),
            *(
                asyncio.create_task(node.join(invitation=invitation))
                for node in nodes[1:]
            ),
        ]
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=60.0)
        manifest_hashes = {result.manifest_hash for result in results}
        if len(manifest_hashes) != 1:
            raise RuntimeError("nodes produced different manifest hashes")
        if any(node.state is not NodeState.READY for node in nodes):
            raise RuntimeError("not every node reached READY")
        state.formation_ready(results[0].manifest_hash)
    finally:
        await asyncio.gather(*(node.stop() for node in nodes), return_exceptions=True)


async def _run_remote_formation(
    state: DashboardState,
    worker_urls: tuple[str, ...],
    round_count: int = 2,
) -> str:
    """Drive four independently containerized workers through their HTTP seam."""
    if len(worker_urls) != len(AXL_PORTS):
        raise RuntimeError("docker demo requires exactly four worker URLs")
    state.preparing("Starting four containerized Dromeus nodes")
    await _prepare_remote_workers(worker_urls)
    formation_deadline = time.monotonic() + 90.0

    # The invitation is an out-of-band artifact on the shared control volume.
    # Start the initiator first and wait for its fresh invitation so a
    # participant cannot consume the previous run's file.
    await asyncio.to_thread(
        _post_json,
        f"{worker_urls[0]}/start",
        {"round_count": round_count},
    )
    while time.monotonic() < formation_deadline:
        snapshot = await asyncio.to_thread(
            _get_json, f"{worker_urls[0]}/state"
        )
        state.ingest_remote_snapshot(0, snapshot)
        if snapshot.get("status") == "failed":
            raise RuntimeError(str(snapshot.get("error") or "worker failed"))
        events = snapshot.get("events")
        if isinstance(events, list) and any(
            isinstance(event, dict) and event.get("kind") == "invitation"
            for event in events
        ):
            break
        await asyncio.sleep(0.2)
    else:
        raise TimeoutError("initiator did not publish an invitation")

    for url in worker_urls[1:]:
        await asyncio.to_thread(
            _post_json,
            f"{url}/start",
            {"round_count": round_count},
        )

    while time.monotonic() < formation_deadline:
        snapshots = await asyncio.gather(
            *(
                asyncio.to_thread(_get_json, f"{url}/state")
                for url in worker_urls
            )
        )
        for index, snapshot in enumerate(snapshots):
            state.ingest_remote_snapshot(index, snapshot)
        failed = [
            snapshot.get("error")
            for snapshot in snapshots
            if snapshot.get("status") == "failed"
        ]
        if failed:
            error = next((item for item in failed if item), "worker failed")
            raise RuntimeError(str(error))
        if all(snapshot.get("status") == "formed" for snapshot in snapshots):
            manifest_hash = next(
                (
                    snapshot.get("manifest_hash")
                    for snapshot in snapshots
                    if isinstance(snapshot.get("manifest_hash"), str)
                ),
                None,
            )
            if not isinstance(manifest_hash, str):
                raise RuntimeError("workers completed without a manifest hash")
            return manifest_hash
        await asyncio.sleep(0.2)
    raise TimeoutError("containerized formation did not complete in 90 seconds")


async def _prepare_remote_workers(worker_urls: tuple[str, ...]) -> None:
    """Wait for worker APIs, stop stale runs, and verify restartability."""
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        try:
            await asyncio.gather(
                *(
                    asyncio.to_thread(_get_json, f"{url}/state")
                    for url in worker_urls
                )
            )
            break
        except (HTTPError, URLError, TimeoutError):
            await asyncio.sleep(0.2)
    else:
        raise TimeoutError("container worker APIs did not become ready")

    await asyncio.gather(
        *(
            asyncio.to_thread(_post_json, f"{url}/stop", {})
            for url in worker_urls
        )
    )
    stop_deadline = time.monotonic() + 10.0
    while time.monotonic() < stop_deadline:
        snapshots = await asyncio.gather(
            *(
                asyncio.to_thread(_get_json, f"{url}/state")
                for url in worker_urls
            )
        )
        if all(snapshot.get("can_start") is True for snapshot in snapshots):
            return
        await asyncio.sleep(0.1)
    raise TimeoutError("container workers did not stop before formation")


async def _run_remote_training(
    state: DashboardState,
    worker_urls: tuple[str, ...],
    round_count: int,
) -> None:
    """Start training on formed workers and stream their round state."""
    training_timeout = max(90.0, round_count * 15.0)
    training_deadline = time.monotonic() + training_timeout
    for url in worker_urls:
        await asyncio.to_thread(_post_json, f"{url}/train", {})
    while time.monotonic() < training_deadline:
        snapshots = await asyncio.gather(
            *(
                asyncio.to_thread(_get_json, f"{url}/state")
                for url in worker_urls
            )
        )
        for index, snapshot in enumerate(snapshots):
            state.ingest_remote_snapshot(index, snapshot)
        failed = [
            snapshot.get("error")
            for snapshot in snapshots
            if snapshot.get("status") == "failed"
        ]
        if failed:
            error = next((item for item in failed if item), "worker failed")
            raise RuntimeError(str(error))
        if all(snapshot.get("status") == "complete" for snapshot in snapshots):
            manifest_hash = next(
                (
                    snapshot.get("manifest_hash")
                    for snapshot in snapshots
                    if isinstance(snapshot.get("manifest_hash"), str)
                ),
                None,
            )
            if not isinstance(manifest_hash, str):
                raise RuntimeError("workers completed without a manifest hash")
            state.complete(
                manifest_hash,
                detail=(
                    f"Four nodes completed {round_count} D-PSGD rounds over real AXL"
                ),
            )
            return
        await asyncio.sleep(0.2)
    raise TimeoutError(
        "containerized training did not complete in "
        f"{int(training_timeout)} seconds"
    )


def _post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    body = json.dumps(payload, separators=(",", ":")).encode()
    request = Request(
        url,
        method="POST",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=5.0) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError(f"unexpected response from {url}")
    return cast(dict[str, object], value)


def _get_json(url: str) -> dict[str, object]:
    with urlopen(url, timeout=5.0) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError(f"unexpected response from {url}")
    return cast(dict[str, object], value)


def _draft(*, round_count: int = 2) -> DraftRunSpec:
    one = "1" * 64
    return DraftRunSpec.model_validate(
        {
            "run_id": "run-axl-demo",
            "algorithm_id": "dpsgd",
            "model_id": "resnet32",
            "model_definition_hash": MODEL_DEFINITION_HASH,
            "dataset": {
                "dataset_id": "cifar10",
                "version": DATASET_VERSION,
                "preprocessing_hash": PREPROCESSING_HASH,
                "iid_partition_seed": 7,
                "image_shape": [3, 32, 32],
                "class_count": 10,
                "sample_count": 50000,
                "partition_sample_counts": [12500, 12500, 12500, 12500],
                "node_index_partitions": [0, 1, 2, 3],
            },
            "environment": {
                "dromeus_version": "0.1.0",
                "dromeus_commit": "abcdef0",
                "pytorch_version": os.environ.get("DROMEUS_PYTORCH_VERSION", "2.7.1"),
                "axl_version": AXL_COMMIT,
                "model_definition_hash": MODEL_DEFINITION_HASH,
                "container_image_digest": f"sha256:{one}",
            },
            "local_steps": 1,
            "round_count": round_count,
            "learning_rate": 0.01,
            "peer_scheduler_seed": 8,
            "codec_id": "safetensors-v1",
            "transport": {
                "max_payload_bytes": MAX_ENVELOPE_BYTES,
                "max_retries": 3,
                "retry_timeout_seconds": 60.0,
            },
            "consensus_sketch": {"size": 4096, "seed": 9},
            "training": {
                "batch_size": 128,
                "momentum": 0.9,
                "weight_decay": 0.0001,
                "learning_rate_milestones": [8000, 12000],
                "learning_rate_gamma": 0.1,
                "crop_padding": 4,
                "normalize": True,
                "final_consensus_rounds": 2,
            },
        }
    )


def _write_checkpoint(path: Path) -> TensorSchema:
    shape = ((1024 * 1024 - 128) // 4,)
    save_file({"layer.weight": np.zeros(shape, dtype=np.float32)}, str(path))
    return TensorSchema(
        tensors=(Tensor(name="layer.weight", dtype="float32", shape=shape),)
    )


def _ensure_axl_binary(state: DashboardState) -> Path:
    binary = CACHE_ROOT / f"axl-node-{AXL_COMMIT[:8]}"
    if binary.is_file():
        return binary
    for command in ("git", "go"):
        if shutil.which(command) is None:
            raise RuntimeError(f"{command} is required to prepare the AXL demo")
    CACHE_ROOT.mkdir(exist_ok=True)
    source = CACHE_ROOT / f"axl-{AXL_COMMIT[:8]}"
    if not source.exists():
        state.preparing("Cloning the pinned AXL source (first run only)")
        subprocess.run(
            ["git", "clone", AXL_SOURCE, str(source)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            ["git", "checkout", AXL_COMMIT],
            check=True,
            cwd=source,
            stdout=subprocess.DEVNULL,
        )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        cwd=source,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != AXL_COMMIT:
        raise RuntimeError("cached AXL source is not at the pinned commit")
    state.preparing("Building the pinned AXL runtime (first run only)")
    subprocess.run(
        ["go", "build", "-o", str(binary), "./cmd/node"],
        check=True,
        cwd=source,
    )
    return binary


def _start_axl_nodes(binary: Path, run_root: Path) -> list[subprocess.Popen[str]]:
    openssl = _find_openssl()
    configs: list[dict[str, object]] = []
    for index, api_port in enumerate(AXL_PORTS):
        configs.append(
            {
                "PrivateKeyPath": str(run_root / f"private-{index}.pem"),
                "Peers": []
                if index == 0
                else [f"tls://127.0.0.1:{AXL_LISTEN_PORT}"],
                "Listen": [f"tls://127.0.0.1:{AXL_LISTEN_PORT}"]
                if index == 0
                else [],
                "api_port": api_port,
                "bridge_addr": "127.0.0.1",
                "max_message_size": MAX_ENVELOPE_BYTES,
            }
        )
    processes: list[subprocess.Popen[str]] = []
    for index, config in enumerate(configs):
        subprocess.run(
            [
                openssl,
                "genpkey",
                "-algorithm",
                "ed25519",
                "-out",
                cast(str, config["PrivateKeyPath"]),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        config_path = run_root / f"node-{index}.json"
        config_path.write_text(json.dumps(config))
        log_handle = (run_root / f"axl-{index}.log").open("w")
        process = subprocess.Popen(
            [str(binary), "-config", str(config_path)],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_handle.close()
        processes.append(process)
    return processes


def _find_openssl() -> str:
    for candidate in (
        "/opt/homebrew/opt/openssl/bin/openssl",
        "/usr/local/opt/openssl/bin/openssl",
        shutil.which("openssl"),
    ):
        if candidate:
            return candidate
    raise RuntimeError("openssl is required to create local AXL identities")


async def _wait_for_topology(port: int) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{port}/topology", timeout=1.0) as response:
                if response.status == HTTPStatus.OK:
                    return
        except (HTTPError, URLError):
            pass
        await asyncio.sleep(0.2)
    raise TimeoutError(f"AXL bridge {port} did not become ready")


async def _wait_for_peer_count(port: int, expected: int) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{port}/topology", timeout=1.0) as response:
                payload = cast(dict[str, object], json.load(response))
            peers = payload.get("peers")
            if isinstance(peers, list):
                peer_list = cast(list[object], peers)
                if len(peer_list) >= expected:
                    return
        except (HTTPError, URLError, ValueError):
            pass
        await asyncio.sleep(0.2)
    raise TimeoutError(f"AXL bridge {port} did not discover {expected} peers")


def _event_detail(envelope: Envelope) -> str:
    if envelope.message_type is MessageType.MANIFEST_SEALED:
        return f"manifest {envelope.manifest_hash[:12]}"
    if envelope.message_type in {
        MessageType.TRANSFER_BEGIN,
        MessageType.CHUNK,
        MessageType.CHUNK_ACK,
        MessageType.TRANSFER_COMPLETE,
    }:
        return "initial checkpoint"
    return envelope.message_id


def _short_key(public_key: str) -> str:
    if len(public_key) <= 20:
        return public_key
    return f"{public_key[:10]}…{public_key[-6:]}"


# Public demo helpers shared by the container worker.
draft_for_demo = _draft
event_detail_for_demo = _event_detail
wait_for_peer_count_for_demo = _wait_for_peer_count
wait_for_topology_for_demo = _wait_for_topology
write_checkpoint_for_demo = _write_checkpoint


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


utc_now_for_demo = _utc_now


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument(
        "--docker",
        action="store_true",
        help="drive four workers running in Docker containers",
    )
    args = parser.parse_args()

    worker_urls = tuple(
        item.strip()
        for item in os.environ.get("DROMEUS_WORKER_URLS", "").split(",")
        if item.strip()
    )
    if args.docker and not worker_urls:
        parser.error("--docker requires DROMEUS_WORKER_URLS")
    controller = DemoController(worker_urls if args.docker else ())
    DashboardHandler.controller = controller
    server = DashboardServer((args.host, args.port), DashboardHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Dromeus formation dashboard: {url}")
    if not args.no_open:
        threading.Timer(0.35, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
