"""One containerized Dromeus formation worker for the live demo."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import FrameType
from typing import cast

from dromeus.algorithms.dpsgd import DPSGDAdapter
from dromeus.manifests.models import Invitation
from dromeus.membership.protocol import create_invitation
from dromeus.persistence.run_store import RunStore
from dromeus.runtime import NodeRuntime, TrainingConfig
from dromeus.telemetry.events import JsonlEventSink
from dromeus.telemetry.metrics import (
    JsonlMetricsPublisher,
    RoundTiming,
)
from dromeus.training.cifar10 import (
    create_initial_checkpoint,
    create_trainer,
    load_cifar10,
)
from dromeus.transport.axl import AXLBridgeConfig
from dromeus.transport.envelope import Envelope
from dromeus.transport.transfer import ArtifactStore

from .server import (
    AXL_LISTEN_PORT,
    AXL_PORTS,
    MAX_ENVELOPE_BYTES,
    ObservedAXLTransport,
    draft_for_demo,
    event_detail_for_demo,
    utc_now_for_demo,
    wait_for_peer_count_for_demo,
    wait_for_topology_for_demo,
)


class WorkerMetricsPublisher:
    """Persist metrics and expose each committed round to the dashboard."""

    def __init__(
        self,
        *,
        state: WorkerState,
        sink: JsonlEventSink,
        run_id: str,
        manifest_hash: str,
        node_id: str,
    ) -> None:
        self._state = state
        self._publisher = JsonlMetricsPublisher(
            sink=sink,
            run_id=run_id,
            manifest_hash=manifest_hash,
            node_id=node_id,
        )

    async def start(self) -> None:
        await self._publisher.start()

    async def stop(self) -> None:
        await self._publisher.stop()

    def submit(self, timing: RoundTiming) -> bool:
        self._state.record_round(timing)
        return self._publisher.submit(timing)

    def submit_failure(self, *, round_id: int, error_type: str, reason: str) -> bool:
        self._state.log(
            f"round {round_id + 1} failed: {error_type}: {reason}",
            round_id=round_id,
        )
        return self._publisher.submit_failure(
            round_id=round_id,
            error_type=error_type,
            reason=reason,
        )


class WorkerState:
    """Thread-safe state and wire-event journal exposed to the dashboard."""

    def __init__(self, node_index: int) -> None:
        self.node_index = node_index
        self._lock = threading.Lock()
        self.status = "idle"
        self.error: str | None = None
        self.key: str | None = None
        self.manifest_hash: str | None = None
        self.round_count = 0
        self.completed_rounds = 0
        self.local_loss: float | None = None
        self.evaluation_loss: float | None = None
        self.evaluation_accuracy: float | None = None
        self._training_logs: list[dict[str, object]] = []
        self._next_sequence = 1
        self._events: list[dict[str, object]] = []

    def preparing(self) -> None:
        with self._lock:
            self.status = "preparing"
            self.error = None

    def reset(self, round_count: int) -> None:
        with self._lock:
            self.status = "idle"
            self.error = None
            self.key = None
            self.manifest_hash = None
            self.round_count = round_count
            self.completed_rounds = 0
            self.local_loss = None
            self.evaluation_loss = None
            self.evaluation_accuracy = None
            self._training_logs = []
            self._next_sequence = 1
            self._events = []

    def register(self, key: str) -> None:
        with self._lock:
            self.key = key

    def running(self) -> None:
        with self._lock:
            self.status = "running"

    def formed(self, manifest_hash: str) -> None:
        with self._lock:
            self.status = "formed"
            self.manifest_hash = manifest_hash
            self._log_locked("formation complete; waiting for training start")

    def training_started(self) -> None:
        with self._lock:
            self.status = "training"
            self._log_locked(
                f"training started: {self.round_count} D-PSGD rounds"
            )

    def record_round(self, timing: RoundTiming) -> None:
        with self._lock:
            self.completed_rounds = max(self.completed_rounds, timing.round_id + 1)
            self.local_loss = timing.local_loss
            self.evaluation_loss = timing.evaluation_loss
            self.evaluation_accuracy = timing.evaluation_accuracy
            detail = (
                f"round {timing.round_id + 1}/{self.round_count} committed "
                f"with peer {timing.peer_id[:8]}"
            )
            if timing.local_loss is not None:
                detail += f" · local loss {timing.local_loss:.4f}"
            if timing.evaluation_accuracy is not None:
                detail += f" · accuracy {timing.evaluation_accuracy:.4f}"
            self._log_locked(
                detail,
                round_id=timing.round_id,
                local_loss=timing.local_loss,
                evaluation_loss=timing.evaluation_loss,
                evaluation_accuracy=timing.evaluation_accuracy,
            )

    def log(self, message: str, *, round_id: int | None = None) -> None:
        with self._lock:
            self._log_locked(message, round_id=round_id)

    def complete(self, manifest_hash: str) -> None:
        with self._lock:
            self.status = "complete"
            self.manifest_hash = manifest_hash

    def fail(self, error: str) -> None:
        with self._lock:
            self.status = "failed"
            self.error = error

    def record_invitation(self, detail: str) -> None:
        with self._lock:
            self._append_locked(
                kind="invitation",
                source=self.key or "unknown",
                target="all-participants",
                type="INVITATION",
                message_id="invitation",
                payload_bytes=0,
                detail=detail,
                delivered=True,
            )

    def record_send(self, envelope: Envelope, destination: str) -> int:
        message_type = envelope.message_type
        message_id = envelope.message_id
        payload_length = envelope.payload_length
        detail = event_detail_for_demo(envelope)
        with self._lock:
            return self._append_locked(
                kind="send",
                source=envelope.sender_public_key,
                target=destination,
                type=str(message_type),
                message_id=message_id,
                payload_bytes=payload_length,
                detail=detail,
                delivered=False,
                bridge_accepted=False,
            )

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
            self._append_locked(
                kind="receive",
                source=envelope.sender_public_key,
                target=destination,
                type=str(envelope.message_type),
                message_id=envelope.message_id,
                payload_bytes=envelope.payload_length,
                detail=event_detail_for_demo(envelope),
                delivered=True,
            )

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "node_index": self.node_index,
                "status": self.status,
                "error": self.error,
                "key": self.key,
                "manifest_hash": self.manifest_hash,
                "training": {
                    "completed_rounds": self.completed_rounds,
                    "round_count": self.round_count,
                    "local_loss": self.local_loss,
                    "evaluation_loss": self.evaluation_loss,
                    "evaluation_accuracy": self.evaluation_accuracy,
                    "logs": [dict(log) for log in self._training_logs],
                },
                "events": [dict(event) for event in self._events],
            }

    def _log_locked(
        self,
        message: str,
        *,
        round_id: int | None = None,
        local_loss: float | None = None,
        evaluation_loss: float | None = None,
        evaluation_accuracy: float | None = None,
    ) -> None:
        self._training_logs.append(
            {
                "timestamp": utc_now_for_demo(),
                "message": message,
                "round_id": round_id,
                "local_loss": local_loss,
                "evaluation_loss": evaluation_loss,
                "evaluation_accuracy": evaluation_accuracy,
            }
        )
        del self._training_logs[:-200]

    def _append_locked(self, **event: object) -> int:
        sequence = self._next_sequence
        self._next_sequence += 1
        self._events.append(
            {
                "sequence": sequence,
                "timestamp": utc_now_for_demo(),
                "bridge_accepted": event.pop("bridge_accepted", True),
                "failed": False,
                **event,
            }
        )
        return sequence

    def _event_locked(self, sequence: int) -> dict[str, object] | None:
        return next(
            (event for event in self._events if event["sequence"] == sequence),
            None,
        )


class Worker:
    def __init__(
        self, node_index: int, control_dir: Path, default_round_count: int
    ) -> None:
        self.node_index = node_index
        self.control_dir = control_dir
        self.default_round_count = default_round_count
        self._round_count = default_round_count
        self.state = WorkerState(node_index)
        self._thread: threading.Thread | None = None
        self._axl: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._training_event = threading.Event()
        self._stopping = threading.Event()

    def start(self, round_count: int | None = None) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            if self._axl is not None and self._axl.poll() is None:
                self.stop()
            self._round_count = round_count or self.default_round_count
            if not 1 <= self._round_count <= 100:
                raise ValueError("round_count must be between 1 and 100")
            self._training_event.clear()
            self._stopping.clear()
            self.state.reset(self._round_count)
            self.state.preparing()
            self._thread = threading.Thread(
                target=self._run,
                name=f"dromeus-worker-{self.node_index}",
                daemon=True,
            )
            self._thread.start()
            return True

    def start_training(self) -> bool:
        if self._stopping.is_set():
            return False
        if self.state.snapshot().get("status") != "formed":
            return False
        self._training_event.set()
        return True

    def can_start(self) -> bool:
        with self._lock:
            return self._thread is None or not self._thread.is_alive()

    def stop(self) -> None:
        self._stopping.set()
        self._training_event.set()
        process = self._axl
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()

    def _run(self) -> None:
        try:
            asyncio.run(self._run_formation())
        except BaseException as error:
            self.state.fail(str(error))
            self.stop()

    async def _run_formation(self) -> None:
        data_root = Path(os.environ.get("DROMEUS_DATA_DIR", "/var/lib/dromeus"))
        run_root = data_root / f"session-{time.time_ns()}"
        run_root.mkdir(parents=True, exist_ok=True)
        self.control_dir.mkdir(parents=True, exist_ok=True)
        binary = Path(os.environ.get("DROMEUS_AXL_BINARY", "/opt/axl/node"))
        self._axl = _start_axl_node(binary, self.node_index, run_root)
        api_port = _api_port(self.node_index)
        await wait_for_topology_for_demo(api_port)
        if self.node_index != 0:
            await wait_for_peer_count_for_demo(api_port, 1)

        transport = ObservedAXLTransport(
            AXLBridgeConfig(base_url=f"http://127.0.0.1:{api_port}"),
            self.state,
        )
        local_key = await transport.local_public_key()
        self.state.register(local_key)
        draft = draft_for_demo(round_count=self._round_count)
        checkpoint = run_root / "checkpoint.safetensors"
        prepared = await asyncio.to_thread(
            create_initial_checkpoint, checkpoint, seed=17
        )
        tensor_schema = prepared.tensor_schema
        event_sink = JsonlEventSink(run_root / "logs" / "dromeus.jsonl")
        runtime = NodeRuntime(
            transport=transport,
            draft=draft,
            environment=draft.environment,
            dataset=draft.dataset,
            artifact_store=ArtifactStore(run_root / "formation-artifacts"),
            event_sink=event_sink,
        )
        self.state.running()
        try:
            if self.node_index == 0:
                invitation = create_invitation(
                    draft=draft,
                    initiator_public_key=local_key,
                    bootstrap_uri=f"tls://node-0:{AXL_LISTEN_PORT}",
                )
                _write_invitation(self.control_dir, invitation)
                self.state.record_invitation(
                    f"draft {invitation.draft_hash[:12]}"
                )
                result = await runtime.initiate(
                    bootstrap_uri=invitation.bootstrap_uri,
                    checkpoint_path=checkpoint,
                    tensor_schema=tensor_schema,
                )
            else:
                invitation = await _read_invitation(self.control_dir)
                result = await runtime.join(invitation=invitation)
            self.state.formed(result.manifest_hash)
            await asyncio.to_thread(self._training_event.wait)
            if self._stopping.is_set():
                return
            self.state.log("loading CIFAR-10")
            cache_dir = Path(
                os.environ.get(
                    "DROMEUS_CIFAR_CACHE", "/var/cache/dromeus/cifar10"
                )
            )
            train_data = await asyncio.to_thread(
                load_cifar10,
                cache_dir=cache_dir,
                train=True,
            )
            test_data = await asyncio.to_thread(
                load_cifar10,
                cache_dir=cache_dir,
                train=False,
            )
            partitions = await asyncio.to_thread(
                train_data.split_iid,
                participant_count=len(result.manifest.participants),
                seed=result.manifest.dataset.iid_partition_seed,
            )
            node_indices = {
                participant.public_key: participant.node_index
                for participant in result.manifest.participants
            }
            node_index = node_indices[local_key]
            partition_index = result.manifest.dataset.node_index_partitions[node_index]
            trainer = await asyncio.to_thread(
                create_trainer,
                train_data=partitions[partition_index],
                test_data=test_data,
                seed=17 + node_index,
                learning_rate=result.manifest.learning_rate,
            )
            metrics = WorkerMetricsPublisher(
                state=self.state,
                sink=event_sink,
                run_id=result.manifest.run_id,
                manifest_hash=result.manifest_hash,
                node_id=local_key,
            )
            runtime.configure_training(
                TrainingConfig(
                    algorithm=DPSGDAdapter(
                        trainer=trainer,
                        tensor_schema=prepared.tensor_schema,
                        local_steps=result.manifest.local_steps,
                        learning_rate=result.manifest.learning_rate,
                        training_round_count=result.manifest.round_count,
                    ),
                    load_checkpoint=trainer.load_checkpoint,
                    run_store=RunStore(run_root / "run-store"),
                    artifact_root=run_root / "rounds",
                    metrics_publisher=metrics,
                )
            )
            self.state.training_started()
            await runtime.run()
            self.state.complete(result.manifest_hash)
        finally:
            await runtime.stop()


class WorkerHandler(BaseHTTPRequestHandler):
    worker: Worker

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] == "/state":
            snapshot = self.worker.state.snapshot()
            snapshot["can_start"] = self.worker.can_start()
            self._json(HTTPStatus.OK, snapshot)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/start":
            try:
                payload = self._read_json()
                round_count = payload.get("round_count")
                if round_count is not None and (
                    not isinstance(round_count, int) or isinstance(round_count, bool)
                ):
                    raise ValueError("round_count must be an integer")
                started = self.worker.start(round_count)
            except ValueError as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            self._json(
                HTTPStatus.ACCEPTED if started else HTTPStatus.CONFLICT,
                {"started": started},
            )
            return
        if path == "/train":
            started = self.worker.start_training()
            self._json(
                HTTPStatus.ACCEPTED if started else HTTPStatus.CONFLICT,
                {"started": started},
            )
            return
        if path == "/stop":
            self.worker.stop()
            self._json(HTTPStatus.OK, {"stopped": True})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, status: HTTPStatus, payload: object) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Cache-Control", "no-store")
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


class WorkerServer(ThreadingHTTPServer):
    daemon_threads = True


def _start_axl_node(
    binary: Path, node_index: int, run_root: Path
) -> subprocess.Popen[str]:
    if not binary.is_file():
        raise RuntimeError(f"AXL binary does not exist: {binary}")
    openssl = shutil.which("openssl")
    if openssl is None:
        raise RuntimeError("openssl is required to create the AXL identity")
    private_key = run_root / "private.pem"
    subprocess.run(
        [openssl, "genpkey", "-algorithm", "ed25519", "-out", str(private_key)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    config = {
        "PrivateKeyPath": str(private_key),
        "Peers": [] if node_index == 0 else [f"tls://node-0:{AXL_LISTEN_PORT}"],
        "Listen": [] if node_index else [f"tls://0.0.0.0:{AXL_LISTEN_PORT}"],
        "api_port": _api_port(node_index),
        "bridge_addr": "127.0.0.1",
        "max_message_size": MAX_ENVELOPE_BYTES,
    }
    config_path = run_root / "axl.json"
    config_path.write_text(json.dumps(config))
    return subprocess.Popen(
        [str(binary), "-config", str(config_path)],
        stdout=None,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _api_port(node_index: int) -> int:
    return AXL_PORTS[node_index]


def _write_invitation(control_dir: Path, invitation: Invitation) -> None:
    temporary = control_dir / "invitation.json.tmp"
    temporary.write_text(invitation.model_dump_json())
    temporary.replace(control_dir / "invitation.json")


async def _read_invitation(control_dir: Path) -> Invitation:
    path = control_dir / "invitation.json"
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            return Invitation.model_validate_json(path.read_text())
        except (FileNotFoundError, ValueError):
            await asyncio.sleep(0.2)
    raise TimeoutError("timed out waiting for the initiator invitation")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--node-index",
        type=int,
        default=int(os.environ.get("DROMEUS_NODE_INDEX", "0")),
    )
    parser.add_argument(
        "--control-dir",
        type=Path,
        default=Path(os.environ.get("DROMEUS_CONTROL_DIR", "/run/dromeus")),
    )
    args = parser.parse_args()
    if not 0 <= args.node_index < len(AXL_PORTS):
        parser.error("node index must be 0 through 3")
    worker = Worker(
        args.node_index,
        args.control_dir,
        int(os.environ.get("DROMEUS_ROUND_COUNT", "2")),
    )
    WorkerHandler.worker = worker
    server = WorkerServer(
        ("0.0.0.0", int(os.environ.get("DROMEUS_WORKER_PORT", "9400"))),
        WorkerHandler,
    )

    def shutdown(_signal: int, _frame: FrameType | None) -> None:
        worker.stop()
        server.shutdown()

    signal.signal(signal.SIGTERM, shutdown)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
