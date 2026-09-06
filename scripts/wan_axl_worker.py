#!/usr/bin/env python3
"""One-off worker harness for the four-region WAN AXL formation test.

This harness intentionally composes the production ``NodeRuntime`` and transport
classes without changing their semantics.  It is not an M2 benchmark runner.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_SUPPORT_ROOT = REPO_ROOT / "tests"
for import_root in (REPO_ROOT / "src", TEST_SUPPORT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from support.sample_manifest import (  # noqa: E402
    manifest_data,
    write_checkpoint,
)

from dromeus.manifests.canonical import (  # noqa: E402
    canonical_hash,
    canonical_json,
    file_sha256,
)
from dromeus.manifests.models import (  # noqa: E402
    DraftRunSpec,
    Invitation,
    SealedManifest,
)
from dromeus.membership.formation import create_invitation  # noqa: E402
from dromeus.protocol.codec import (  # noqa: E402
    decode_envelope,
    decode_envelope_sender,
    decode_message,
)  # noqa: E402
from dromeus.protocol.models import (  # noqa: E402
    Chunk,
    ChunkAck,
    MessageType,
    TransferBegin,
    TransferComplete,
)  # noqa: E402
from dromeus.runtime import NodeRuntime, NodeState  # noqa: E402
from dromeus.telemetry.events import JsonlEventSink  # noqa: E402
from dromeus.transport.axl import AXLBridgeConfig, AXLTransport  # noqa: E402
from dromeus.transport.interface import (  # noqa: E402
    AsyncTransport,
    ReceivedBytes,
)

AXL_COMMIT = "628e28ace077f26dfe8d0259009b357216a9d8d4"
AXL_BINARY_SHA256 = "8ddebe8ceecc630cb8f0cf0c560c2c343697fb82ef86ea098dd758f7e0256385"
CHECKPOINT_BYTES = 45 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024

_EXCLUDED_DIRS = frozenset(
    {
        ".codex",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "aws",
        "graphify-out",
        "logs",
    }
)


def _excluded_source_path(relative: Path) -> bool:
    parts = relative.parts
    if any(part in _EXCLUDED_DIRS for part in parts):
        return True
    if parts[:2] == ("benchmarks", "results"):
        return True
    name = relative.name
    return name.endswith((".DS_Store", ".env", ".key", ".pem", ".secret"))


def source_tree_hash(root: Path) -> str:
    """Hash the deployable source tree, including paths and exact file bytes."""
    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and not _excluded_source_path(path.relative_to(root))
    ]
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda value: value.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as handle:
            data = handle.read()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    _atomic_write(
        path,
        (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(),
    )


def _write_canonical_model(path: Path, model: Any) -> None:
    _atomic_write(path, canonical_json(model) + b"\n")


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _record_event(path: Path, event: str, **fields: object) -> None:
    record: dict[str, object] = {
        "timestamp": _iso_now(),
        "monotonic_seconds": time.monotonic(),
        "event": event,
    }
    record.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
        handle.flush()


class FaultInjectingTransport:
    """The local test transport wrapper plus auditable wire observations."""

    def __init__(
        self,
        transport: AsyncTransport,
        *,
        max_payload_bytes: int,
        event_path: Path,
        drop_first_ack: bool = False,
        duplicate_first_chunk: bool = False,
    ) -> None:
        self._transport = transport
        self._max_payload_bytes = max_payload_bytes
        self._event_path = event_path
        self._drop_first_ack = drop_first_ack
        self._duplicate_first_chunk = duplicate_first_chunk
        self._dropped_ack_count = 0
        self._duplicated_chunk_count = 0
        self._seen_inbound_chunks: set[tuple[str, str, int, str]] = set()

    async def local_public_key(self) -> str:
        return await self._transport.local_public_key()

    def _decode(self, payload: bytes) -> tuple[str, Any]:
        sender = decode_envelope_sender(
            payload,
            max_payload_bytes=self._max_payload_bytes,
        )
        envelope = decode_envelope(
            payload,
            authenticated_sender=sender,
            participant_keys=None,
            max_payload_bytes=self._max_payload_bytes,
        )
        return sender, envelope

    def _details(self, envelope: Any) -> dict[str, object]:
        details: dict[str, object] = {
            "message_type": envelope.message_type.value,
            "message_id": envelope.message_id,
            "correlation_id": envelope.correlation_id,
            "sender_public_key": envelope.sender_public_key,
            "payload_bytes": len(envelope.payload),
        }
        if envelope.message_type is MessageType.TRANSFER_BEGIN:
            begin = decode_message(
                envelope.payload,
                TransferBegin,
                max_bytes=self._max_payload_bytes,
            )
            details.update(
                {
                    "transfer_id": begin.transfer_id,
                    "total_size_bytes": begin.total_size_bytes,
                    "total_sha256": begin.total_sha256,
                    "chunk_count": begin.chunk_count,
                }
            )
        elif envelope.message_type is MessageType.CHUNK:
            chunk = decode_message(
                envelope.payload,
                Chunk,
                max_bytes=self._max_payload_bytes,
            )
            details.update(
                {
                    "transfer_id": chunk.transfer_id,
                    "chunk_index": chunk.chunk_index,
                    "chunk_count": chunk.chunk_count,
                    "chunk_sha256": chunk.chunk_sha256,
                }
            )
        elif envelope.message_type is MessageType.CHUNK_ACK:
            ack = decode_message(
                envelope.payload,
                ChunkAck,
                max_bytes=self._max_payload_bytes,
            )
            details.update(
                {
                    "transfer_id": ack.transfer_id,
                    "chunk_index": ack.chunk_index,
                    "chunk_sha256": ack.chunk_sha256,
                }
            )
        elif envelope.message_type is MessageType.TRANSFER_COMPLETE:
            complete = decode_message(
                envelope.payload,
                TransferComplete,
                max_bytes=self._max_payload_bytes,
            )
            details.update(
                {
                    "transfer_id": complete.transfer_id,
                    "total_sha256": complete.total_sha256,
                }
            )
        return details

    async def send(self, destination: str, payload: bytes) -> None:
        _, envelope = self._decode(payload)
        details = self._details(envelope)
        details["destination"] = destination
        if self._drop_first_ack and envelope.message_type is MessageType.CHUNK_ACK:
            self._drop_first_ack = False
            self._dropped_ack_count += 1
            _record_event(
                self._event_path,
                "fault_drop_first_chunk_ack",
                **details,
            )
            return
        await self._transport.send(destination, payload)
        _record_event(self._event_path, "send", **details)
        if self._duplicate_first_chunk and envelope.message_type is MessageType.CHUNK:
            self._duplicate_first_chunk = False
            self._duplicated_chunk_count += 1
            _record_event(
                self._event_path,
                "fault_duplicate_first_chunk",
                **details,
            )
            await self._transport.send(destination, payload)
            _record_event(self._event_path, "send_duplicate", **details)

    async def recv(self, timeout_seconds: float) -> ReceivedBytes | None:
        inbound = await self._transport.recv(timeout_seconds)
        if inbound is None:
            return None
        try:
            _, envelope = self._decode(inbound.payload)
            details = self._details(envelope)
            details["transport_sender_public_key"] = inbound.sender_public_key
            if envelope.message_type is MessageType.CHUNK:
                key = (
                    inbound.sender_public_key,
                    cast(str, details["transfer_id"]),
                    cast(int, details["chunk_index"]),
                    cast(str, details["chunk_sha256"]),
                )
                if key in self._seen_inbound_chunks:
                    _record_event(
                        self._event_path,
                        "duplicate_chunk_received",
                        **details,
                    )
                self._seen_inbound_chunks.add(key)
            _record_event(self._event_path, "recv", **details)
        except Exception as error:
            _record_event(
                self._event_path,
                "wire_observation_failed",
                error_type=type(error).__name__,
                error=str(error)[:512],
            )
        return inbound

    @property
    def fault_status(self) -> dict[str, object]:
        return {
            "duplicate_first_chunk_configured": self._duplicated_chunk_count > 0
            or self._duplicate_first_chunk,
            "duplicate_first_chunk_applied": self._duplicated_chunk_count == 1,
            "duplicate_first_chunk_count": self._duplicated_chunk_count,
            "drop_first_chunk_ack_configured": self._dropped_ack_count > 0
            or self._drop_first_ack,
            "drop_first_chunk_ack_applied": self._dropped_ack_count == 1,
            "drop_first_chunk_ack_count": self._dropped_ack_count,
        }


def _build_draft() -> tuple[DraftRunSpec, SealedManifest]:
    manifest = SealedManifest.model_validate(manifest_data())
    draft_data = manifest.model_dump(mode="python")
    for field in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del draft_data[field]
    transport = cast(dict[str, object], draft_data["transport"])
    transport.update(
        {
            "max_payload_bytes": 64 * 1024 * 1024,
            "chunk_size_bytes": CHUNK_BYTES,
            "window_size": 4,
            "max_message_payload_bytes": 2 * 1024 * 1024,
            "max_artifact_bytes": 64 * 1024 * 1024,
            "max_concurrent_transfers": 4,
            "max_inflight_bytes": 8 * 1024 * 1024,
            "transfer_lifetime_seconds": 30.0,
            "artifact_store_capacity_bytes": 128 * 1024 * 1024,
            "max_retries": 2,
            "retry_timeout_seconds": 1.0,
        }
    )
    return DraftRunSpec.model_validate(draft_data), manifest


async def _wait_for_topology(
    transport: AXLTransport, *, timeout_seconds: float
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    last_error = "AXL topology was not ready"
    while time.monotonic() < deadline:
        try:
            topology = await transport.topology()
            if isinstance(topology.get("our_public_key"), str):
                return topology
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
        await asyncio.sleep(0.5)
    raise TimeoutError(last_error)


async def _read_invitation(path: Path, *, timeout_seconds: float) -> Invitation:
    await _wait_for_file(path, timeout_seconds=timeout_seconds)
    payload = await asyncio.to_thread(path.read_bytes)
    return Invitation.model_validate_json(payload)


async def _wait_for_file(path: Path, *, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            await asyncio.to_thread(path.stat)
            return
        except FileNotFoundError:
            await asyncio.sleep(0.25)
    raise TimeoutError(f"timed out waiting for invitation: {path}")


async def _write_topology(transport: AXLTransport, path: Path) -> None:
    topology = await transport.topology()
    _write_json(path, topology)


def _store_state(root: Path) -> dict[str, object]:
    files: list[dict[str, object]] = []
    parts: list[str] = []
    if root.exists():
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            size = path.stat().st_size
            files.append({"path": relative, "size_bytes": size})
            if relative.startswith(".tmp/") and relative.endswith(".part"):
                parts.append(relative)
    return {"files": files, "temporary_part_files": parts}


def _checkpoint_info(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


async def _run_worker(args: argparse.Namespace) -> None:
    run_root = args.run_root
    run_root.mkdir(parents=True, exist_ok=True)
    draft, sample_manifest = _build_draft()
    _write_canonical_model(run_root / "draft.json", draft)
    local_source_hash = source_tree_hash(args.source_root)
    _write_json(
        run_root / "source-hash.json",
        {"source_root": str(args.source_root), "source_tree_sha256": local_source_hash},
    )
    if args.axl_build_record is not None and args.axl_build_record.is_file():
        build_record = json.loads(args.axl_build_record.read_text(encoding="utf-8"))
    else:
        build_record = {}

    base_transport = AXLTransport(AXLBridgeConfig(base_url=args.axl_bridge_url))
    topology_before = await _wait_for_topology(
        base_transport, timeout_seconds=args.timeout_seconds
    )
    local_key = await base_transport.local_public_key()
    _write_json(
        run_root / "identity.json",
        {
            "worker_index": args.worker_index,
            "role": args.role,
            "public_key": local_key,
            "region": args.region,
        },
    )
    _write_json(run_root / "topology-before.json", topology_before)
    fault_transport = FaultInjectingTransport(
        base_transport,
        max_payload_bytes=draft.transport.message_payload_limit,
        event_path=run_root / "wire-events.jsonl",
        drop_first_ack=args.drop_first_ack,
        duplicate_first_chunk=args.duplicate_first_chunk,
    )
    event_sink = JsonlEventSink(run_root / "logs" / "dromeus.jsonl")
    runtime = NodeRuntime(
        transport=fault_transport,
        draft=draft,
        environment=sample_manifest.environment,
        dataset=sample_manifest.dataset,
        artifact_root=run_root / "formation-artifacts",
        event_sink=event_sink,
    )
    result: Any = None
    error: BaseException | None = None
    state_before_stop: str | None = None
    state_after_stop: str | None = None
    initial_checkpoint: dict[str, object] | None = None
    try:
        if args.role == "initiator":
            checkpoint = run_root / "initial-checkpoint.safetensors"
            element_count = (CHECKPOINT_BYTES - 128) // 4
            await asyncio.to_thread(
                write_checkpoint,
                checkpoint,
                shape=(element_count,),
            )
            tensor_schema = sample_manifest.tensor_schema.model_copy(
                update={
                    "tensors": (
                        sample_manifest.tensor_schema.tensors[0].model_copy(
                            update={"shape": (element_count,)}
                        ),
                    )
                }
            )
            initial_checkpoint = _checkpoint_info(checkpoint)
            invitation = create_invitation(
                draft=draft,
                initiator_public_key=local_key,
                bootstrap_uri=args.bootstrap_uri,
            )
            _write_canonical_model(run_root / "invitation.json", invitation)
            _record_event(
                run_root / "orchestration-events.jsonl",
                "invitation_written",
                invitation_path=str(run_root / "invitation.json"),
                initiator_public_key=local_key,
            )
            if args.start_gate_path is not None:
                await _wait_for_file(
                    args.start_gate_path,
                    timeout_seconds=args.timeout_seconds,
                )
            result = await asyncio.wait_for(
                runtime.initiate(
                    bootstrap_uri=args.bootstrap_uri,
                    checkpoint_path=checkpoint,
                    tensor_schema=tensor_schema,
                ),
                timeout=args.timeout_seconds,
            )
        else:
            invitation = await _read_invitation(
                args.invitation_path,
                timeout_seconds=args.timeout_seconds,
            )
            if invitation.bootstrap_uri != args.bootstrap_uri:
                raise ValueError(
                    "invitation bootstrap URI does not match worker config"
                )
            if args.start_gate_path is not None:
                await _wait_for_file(
                    args.start_gate_path,
                    timeout_seconds=args.timeout_seconds,
                )
            result = await asyncio.wait_for(
                runtime.join(invitation=invitation),
                timeout=args.timeout_seconds,
            )
        state_before_stop = runtime.state.value
        await _write_topology(base_transport, run_root / "topology-after.json")
    except BaseException as caught:
        error = caught
        _record_event(
            run_root / "orchestration-events.jsonl",
            "worker_failed",
            error_type=type(caught).__name__,
            error=str(caught)[:1024],
        )
    finally:
        state_before_stop = state_before_stop or runtime.state.value
        try:
            await runtime.stop()
        except BaseException as caught:
            if error is None:
                error = caught
        state_after_stop = runtime.state.value

    result_data: dict[str, object] = {
        "status": "passed" if error is None else "failed",
        "worker_index": args.worker_index,
        "role": args.role,
        "region": args.region,
        "public_key": local_key,
        "source_tree_sha256": local_source_hash,
        "axl_commit": build_record.get("source_commit", AXL_COMMIT),
        "axl_binary_sha256": build_record.get("binary_sha256"),
        "axl_binary_sha256_expected": AXL_BINARY_SHA256,
        "draft_hash": canonical_hash(draft),
        "manifest_hash": None,
        "state_before_stop": state_before_stop,
        "state_after_stop": state_after_stop,
        "transport": draft.transport.model_dump(mode="json"),
        "faults": fault_transport.fault_status,
        "initial_checkpoint": initial_checkpoint,
        "checkpoint": None,
        "store_state": _store_state(run_root / "formation-artifacts"),
    }
    if result is not None:
        _write_canonical_model(run_root / "sealed-manifest.json", result.manifest)
        _atomic_write(
            run_root / "manifest-hash.txt",
            f"{result.manifest_hash}\n".encode(),
        )
        checkpoint_info = _checkpoint_info(result.checkpoint_path)
        result_data["manifest_hash"] = result.manifest_hash
        result_data["checkpoint"] = checkpoint_info
        result_data["manifest"] = result.manifest.model_dump(mode="json")
        if args.role == "participant":
            size = cast(int, checkpoint_info["size_bytes"])
            if not CHECKPOINT_BYTES - 256 <= size <= CHECKPOINT_BYTES:
                raise AssertionError(
                    "participant checkpoint size is outside local contract"
                )
        if state_before_stop != NodeState.READY.value:
            raise AssertionError("worker did not reach READY before shutdown")
    if error is not None:
        result_data["error"] = {
            "type": type(error).__name__,
            "message": str(error)[:1024],
        }
        traceback.print_exception(error)
    _write_json(run_root / "fault-status.json", fault_transport.fault_status)
    _write_json(run_root / "result.json", result_data)
    if error is not None:
        raise error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("initiator", "participant"))
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--region")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--axl-bridge-url", default="http://127.0.0.1:9302")
    parser.add_argument("--bootstrap-uri")
    parser.add_argument("--invitation-path", type=Path)
    parser.add_argument("--axl-build-record", type=Path, default=None)
    parser.add_argument("--start-gate-path", type=Path, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--drop-first-ack", action="store_true")
    parser.add_argument("--duplicate-first-chunk", action="store_true")
    parser.add_argument("--print-source-hash", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.print_source_hash:
        if args.source_root is None:
            parser.error("--source-root is required with --print-source-hash")
        print(source_tree_hash(args.source_root))
        return 0
    required = {
        "--role": args.role,
        "--worker-index": args.worker_index,
        "--region": args.region,
        "--run-root": args.run_root,
        "--source-root": args.source_root,
        "--bootstrap-uri": args.bootstrap_uri,
        "--invitation-path": args.invitation_path,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))
    try:
        asyncio.run(_run_worker(args))
    except BaseException:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
