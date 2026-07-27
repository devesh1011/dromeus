"""Reliable artifact transfer."""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self, cast

import msgpack  # pyright: ignore[reportMissingTypeStubs]
from pydantic import Field
from safetensors import SafetensorError, safe_open

from dromeus.manifests.models import (
    AlgorithmId,
    DomainModel,
    Identifier,
    MessageId,
    PublicKey,
    RoundId,
    RunId,
    Sha256,
    TensorSchema,
    TransferId,
    TransportLimits,
)
from dromeus.telemetry.events import EventSink, emit_event
from dromeus.transport.envelope import (
    Envelope,
    MessageType,
    create_envelope,
    encode_envelope,
)
from dromeus.transport.receiver import MessageChannel, Receiver
from dromeus.transport.sender import OutboundScheduler, Priority, SendTiming


class TransferError(RuntimeError):
    """Artifact transfer failed terminally."""


class TransferBegin(DomainModel):
    transfer_id: TransferId
    artifact_name: Identifier
    total_size_bytes: int = Field(gt=0)
    total_sha256: Sha256
    chunk_count: int = Field(gt=0)
    codec_id: Identifier
    tensor_schema: TensorSchema


class Chunk(DomainModel):
    transfer_id: TransferId
    chunk_index: int = Field(ge=0)
    chunk_count: int = Field(gt=0)
    chunk_sha256: Sha256
    data: bytes


class ChunkAck(DomainModel):
    transfer_id: TransferId
    chunk_index: int = Field(ge=0)
    chunk_sha256: Sha256


class TransferComplete(DomainModel):
    transfer_id: TransferId
    total_sha256: Sha256


def _pack(model: DomainModel) -> bytes:
    value = model.model_dump(mode="python")
    return cast(bytes, msgpack.packb(value))  # pyright: ignore[reportUnknownMemberType]


def _unpack(data: bytes) -> object:
    return cast(
        object,
        msgpack.unpackb(  # pyright: ignore[reportUnknownMemberType]
            data, raw=False, strict_map_key=True
        ),
    )


@dataclass(frozen=True)
class ArtifactReceipt:
    transfer_id: TransferId
    sender_public_key: PublicKey
    artifact_name: Identifier
    path: Path
    sha256: Sha256
    size_bytes: int
    round_id: RoundId | None


@dataclass(frozen=True)
class TransferTiming:
    """Observable duration and retry count for one outbound artifact."""

    elapsed_seconds: float
    retry_count: int


class ArtifactStore:
    """Atomic temp-file based artifact finalization."""

    def __init__(self, root: Path, *, max_bytes: int = 128 * 1024 * 1024) -> None:
        self._root = root
        self._tmp_root = root / ".tmp"
        self._root.mkdir(parents=True, exist_ok=True)
        self._tmp_root.mkdir(parents=True, exist_ok=True)
        self._max_bytes = max_bytes
        self._reserved_bytes = 0

    def reserve(self, size_bytes: int) -> None:
        if self._reserved_bytes + size_bytes > self._max_bytes:
            raise TransferError("artifact store capacity exceeded")
        self._reserved_bytes += size_bytes

    def release(self, size_bytes: int) -> None:
        self._reserved_bytes = max(0, self._reserved_bytes - size_bytes)

    def temp_path(self, transfer_id: TransferId) -> Path:
        return self._tmp_root / f"{transfer_id}.part"

    def final_path(self, transfer_id: TransferId, artifact_name: Identifier) -> Path:
        safe_name = artifact_name.replace("/", "_")
        return self._root / f"{transfer_id}-{safe_name}.bin"

    def finalize(self, transfer_id: TransferId, artifact_name: Identifier) -> Path:
        temp = self.temp_path(transfer_id)
        final = self.final_path(transfer_id, artifact_name)
        temp.replace(final)
        return final

    def abort(self, transfer_id: TransferId, size_bytes: int) -> None:
        self.temp_path(transfer_id).unlink(missing_ok=True)
        self.release(size_bytes)


@dataclass
class _IncomingTransfer:
    sender_public_key: PublicKey
    begin: TransferBegin
    round_id: RoundId | None
    written_chunk_indices: set[int]
    started_at: float


_SAFETENSORS_DTYPES = {
    "float16": "F16",
    "float32": "F32",
    "float64": "F64",
    "int8": "I8",
    "int32": "I32",
    "int64": "I64",
}


class _TensorSlice(Protocol):
    def get_dtype(self) -> str: ...

    def get_shape(self) -> list[int]: ...


class _SafeTensors(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def keys(self) -> list[str]: ...

    def get_slice(self, name: str) -> _TensorSlice: ...


def _file_sha256(path: Path) -> Sha256:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _validate_safetensors(path: Path, schema: TensorSchema) -> None:
    try:
        reader = cast(_SafeTensors, safe_open(path, framework="numpy"))
        with reader as tensors:
            expected = {tensor.name: tensor for tensor in schema.tensors}
            if set(tensors.keys()) != set(expected):
                raise TransferError("safetensors names do not match tensor schema")
            for name, tensor in expected.items():
                view = tensors.get_slice(name)
                if view.get_dtype() != _SAFETENSORS_DTYPES[tensor.dtype]:
                    raise TransferError(
                        "safetensors dtype does not match tensor schema"
                    )
                if tuple(view.get_shape()) != tensor.shape:
                    raise TransferError(
                        "safetensors shape does not match tensor schema"
                    )
    except SafetensorError as error:
        raise TransferError("invalid safetensors artifact") from error


class TransferManager:
    """Single-chunk transfer protocol with retry and idempotent receive."""

    def __init__(
        self,
        *,
        local_public_key: PublicKey,
        run_id: RunId,
        manifest_hash: Sha256,
        algorithm_id: AlgorithmId,
        transport_limits: TransportLimits,
        receiver: Receiver,
        sender: OutboundScheduler,
        artifact_store: ArtifactStore,
        event_sink: EventSink | None = None,
    ) -> None:
        self._local_public_key = local_public_key
        self._run_id = run_id
        self._manifest_hash = manifest_hash
        self._algorithm_id = algorithm_id
        self._transport_limits = transport_limits
        self._receiver = receiver
        self._sender = sender
        self._artifact_store = artifact_store
        self._event_sink = event_sink
        self._incoming: dict[TransferId, _IncomingTransfer] = {}
        self._completed: dict[TransferId, ArtifactReceipt] = {}
        self._completed_futures: dict[TransferId, asyncio.Future[ArtifactReceipt]] = {}
        self._completed_queue: asyncio.Queue[ArtifactReceipt] = asyncio.Queue(
            maxsize=64
        )
        self._ack_waiters: dict[tuple[TransferId, int], asyncio.Future[ChunkAck]] = {}
        self._stop = asyncio.Event()
        self._transfer_task: asyncio.Task[None] | None = None
        self._ack_task: asyncio.Task[None] | None = None
        self._last_timing: TransferTiming | None = None

    @property
    def last_timing(self) -> TransferTiming | None:
        return self._last_timing

    async def start(self) -> None:
        if self._transfer_task is None:
            self._transfer_task = asyncio.create_task(
                self._transfer_loop(), name="dromeus-transfer-loop"
            )
            self._ack_task = asyncio.create_task(
                self._ack_loop(), name="dromeus-ack-loop"
            )

    async def stop(self) -> None:
        self._stop.set()
        for task in (self._transfer_task, self._ack_task):
            if task is not None:
                await task
        for transfer_id in tuple(self._incoming):
            self._abort_incoming(transfer_id)

    async def send_artifact(
        self,
        *,
        destination: PublicKey,
        artifact_name: Identifier,
        artifact_path: Path,
        codec_id: Identifier,
        tensor_schema: TensorSchema,
        round_id: RoundId | None = None,
    ) -> TransferId:
        started = time.perf_counter()
        retry_count = 0
        try:
            if codec_id != "safetensors-v1":
                raise TransferError("M1 only supports safetensors-v1 artifacts")
            _validate_safetensors(artifact_path, tensor_schema)
            payload_size = artifact_path.stat().st_size
            chunk_bytes = artifact_path.read_bytes()
            transfer_id = str(uuid.uuid4())
            total_sha256 = _file_sha256(artifact_path)
            begin = TransferBegin(
                transfer_id=transfer_id,
                artifact_name=artifact_name,
                total_size_bytes=payload_size,
                total_sha256=total_sha256,
                chunk_count=1,
                codec_id=codec_id,
                tensor_schema=tensor_schema,
            )
            chunk = Chunk(
                transfer_id=transfer_id,
                chunk_index=0,
                chunk_count=1,
                chunk_sha256=hashlib.sha256(chunk_bytes).hexdigest(),
                data=chunk_bytes,
            )
            begin_timing = await self._send_message(
                destination=destination,
                message_type=MessageType.TRANSFER_BEGIN,
                message_id=f"{transfer_id}-begin",
                correlation_id=transfer_id,
                payload=_pack(begin),
                round_id=round_id,
                priority=Priority.CONTROL,
            )
            retry_count += begin_timing.retry_count
            attempts = self._transport_limits.max_retries + 1
            for attempt in range(attempts):
                ack_key = (transfer_id, 0)
                loop = asyncio.get_running_loop()
                ack_future: asyncio.Future[ChunkAck] = loop.create_future()
                self._ack_waiters[ack_key] = ack_future
                chunk_timing = await self._send_message(
                    destination=destination,
                    message_type=MessageType.CHUNK,
                    message_id=f"{transfer_id}-chunk-0",
                    correlation_id=transfer_id,
                    payload=_pack(chunk),
                    round_id=round_id,
                    priority=Priority.DATA,
                )
                retry_count += chunk_timing.retry_count
                try:
                    ack = await asyncio.wait_for(
                        ack_future,
                        timeout=self._transport_limits.retry_timeout_seconds,
                    )
                except TimeoutError:
                    self._ack_waiters.pop(ack_key, None)
                    if attempt + 1 < attempts:
                        retry_count += 1
                    continue
                if ack.chunk_sha256 != chunk.chunk_sha256:
                    raise TransferError("chunk acknowledgement checksum mismatch")
                break
            else:
                raise TransferError("chunk acknowledgement retries exhausted")
            complete_timing = await self._send_message(
                destination=destination,
                message_type=MessageType.TRANSFER_COMPLETE,
                message_id=f"{transfer_id}-complete",
                correlation_id=transfer_id,
                payload=_pack(
                    TransferComplete(transfer_id=transfer_id, total_sha256=total_sha256)
                ),
                round_id=round_id,
                priority=Priority.CONTROL,
            )
            retry_count += complete_timing.retry_count
            self._last_timing = TransferTiming(
                elapsed_seconds=time.perf_counter() - started,
                retry_count=retry_count,
            )
            return transfer_id
        except BaseException:
            self._last_timing = TransferTiming(
                elapsed_seconds=time.perf_counter() - started,
                retry_count=retry_count,
            )
            raise

    async def wait_for_artifact(
        self, transfer_id: TransferId, *, timeout_seconds: float
    ) -> ArtifactReceipt:
        completed = self._completed.get(transfer_id)
        if completed is not None:
            return completed
        future = self._completed_futures.get(transfer_id)
        if future is None:
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            self._completed_futures[transfer_id] = future
        return await asyncio.wait_for(future, timeout=timeout_seconds)

    async def next_artifact(self, *, timeout_seconds: float) -> ArtifactReceipt:
        return await asyncio.wait_for(
            self._completed_queue.get(), timeout=timeout_seconds
        )

    async def _transfer_loop(self) -> None:
        while not self._stop.is_set():
            try:
                envelope = await self._receiver.receive(
                    MessageChannel.TRANSFER, timeout_seconds=0.1
                )
            except TimeoutError:
                self._expire_incoming()
                continue
            self._expire_incoming()
            try:
                await self._handle_transfer(envelope)
            except (TransferError, ValueError, msgpack.UnpackException) as error:
                emit_event(
                    "transfer_message_rejected",
                    run_id=envelope.run_id,
                    message_id=envelope.message_id,
                    transfer_id=envelope.correlation_id,
                    peer_id=envelope.sender_public_key,
                    round_id=envelope.round_id,
                    error=str(error),
                    sink=self._event_sink,
                )
                continue

    async def _ack_loop(self) -> None:
        while not self._stop.is_set():
            try:
                envelope = await self._receiver.receive(
                    MessageChannel.ACKNOWLEDGMENT, timeout_seconds=0.1
                )
            except TimeoutError:
                continue
            try:
                ack = ChunkAck.model_validate(_unpack(envelope.payload))
                waiter = self._ack_waiters.pop((ack.transfer_id, ack.chunk_index), None)
                if waiter is not None and not waiter.done():
                    waiter.set_result(ack)
            except (ValueError, msgpack.UnpackException) as error:
                emit_event(
                    "transfer_ack_rejected",
                    run_id=envelope.run_id,
                    message_id=envelope.message_id,
                    transfer_id=envelope.correlation_id,
                    peer_id=envelope.sender_public_key,
                    round_id=envelope.round_id,
                    error=str(error),
                    sink=self._event_sink,
                )

    async def _handle_transfer(self, envelope: Envelope) -> None:
        if envelope.message_type is MessageType.TRANSFER_BEGIN:
            begin = TransferBegin.model_validate(_unpack(envelope.payload))
            if begin.total_size_bytes > self._transport_limits.max_payload_bytes:
                raise TransferError("transfer exceeds manifest payload limit")
            if begin.chunk_count != 1 or begin.codec_id != "safetensors-v1":
                raise TransferError("unsupported M1 transfer encoding")
            incoming = self._incoming.get(begin.transfer_id)
            if incoming is None:
                self._artifact_store.reserve(begin.total_size_bytes)
                self._incoming[begin.transfer_id] = _IncomingTransfer(
                    sender_public_key=envelope.sender_public_key,
                    begin=begin,
                    round_id=envelope.round_id,
                    written_chunk_indices=set(),
                    started_at=time.monotonic(),
                )
            return
        if envelope.message_type is MessageType.CHUNK:
            chunk = Chunk.model_validate(_unpack(envelope.payload))
            incoming = self._incoming.get(chunk.transfer_id)
            if incoming is None:
                raise TransferError("received chunk without transfer begin")
            if envelope.sender_public_key != incoming.sender_public_key:
                raise TransferError("transfer sender changed")
            if chunk.chunk_count != 1 or chunk.chunk_index != 0:
                self._abort_incoming(chunk.transfer_id)
                raise TransferError("invalid M1 chunk index or count")
            if chunk.chunk_index in incoming.written_chunk_indices:
                await self._acknowledge_chunk(
                    envelope.sender_public_key, chunk, envelope.round_id
                )
                return
            if hashlib.sha256(chunk.data).hexdigest() != chunk.chunk_sha256:
                self._abort_incoming(chunk.transfer_id)
                raise TransferError("chunk checksum mismatch")
            temp = self._artifact_store.temp_path(chunk.transfer_id)
            with temp.open("ab") as handle:
                handle.write(chunk.data)
            incoming.written_chunk_indices.add(chunk.chunk_index)
            await self._acknowledge_chunk(
                envelope.sender_public_key, chunk, envelope.round_id
            )
            return
        complete = TransferComplete.model_validate(_unpack(envelope.payload))
        incoming = self._incoming.get(complete.transfer_id)
        if incoming is None:
            if complete.transfer_id in self._completed:
                return
            raise TransferError("received completion without transfer begin")
        temp = self._artifact_store.temp_path(complete.transfer_id)
        digest = _file_sha256(temp)
        if digest != complete.total_sha256 or digest != incoming.begin.total_sha256:
            self._abort_incoming(complete.transfer_id)
            raise TransferError("final artifact checksum mismatch")
        if temp.stat().st_size != incoming.begin.total_size_bytes:
            self._abort_incoming(complete.transfer_id)
            raise TransferError("final artifact size mismatch")
        try:
            _validate_safetensors(temp, incoming.begin.tensor_schema)
        except TransferError:
            self._abort_incoming(complete.transfer_id)
            raise
        final_path = self._artifact_store.finalize(
            complete.transfer_id, incoming.begin.artifact_name
        )
        receipt = ArtifactReceipt(
            transfer_id=complete.transfer_id,
            sender_public_key=incoming.sender_public_key,
            artifact_name=incoming.begin.artifact_name,
            path=final_path,
            sha256=digest,
            size_bytes=incoming.begin.total_size_bytes,
            round_id=incoming.round_id,
        )
        self._completed[complete.transfer_id] = receipt
        self._incoming.pop(complete.transfer_id, None)
        self._artifact_store.release(incoming.begin.total_size_bytes)
        future = self._completed_futures.get(complete.transfer_id)
        if future is not None and not future.done():
            future.set_result(receipt)
        await self._completed_queue.put(receipt)

    def _expire_incoming(self) -> None:
        lifetime = self._transport_limits.retry_timeout_seconds * (
            self._transport_limits.max_retries + 2
        )
        now = time.monotonic()
        for transfer_id, incoming in tuple(self._incoming.items()):
            if now - incoming.started_at >= lifetime:
                self._abort_incoming(transfer_id)

    def _abort_incoming(self, transfer_id: TransferId) -> None:
        incoming = self._incoming.pop(transfer_id, None)
        if incoming is not None:
            self._artifact_store.abort(transfer_id, incoming.begin.total_size_bytes)

    async def _acknowledge_chunk(
        self,
        destination: PublicKey,
        chunk: Chunk,
        round_id: RoundId | None,
    ) -> None:
        await self._send_message(
            destination=destination,
            message_type=MessageType.CHUNK_ACK,
            message_id=f"{chunk.transfer_id}-ack-{chunk.chunk_index}",
            correlation_id=chunk.transfer_id,
            payload=_pack(
                ChunkAck(
                    transfer_id=chunk.transfer_id,
                    chunk_index=chunk.chunk_index,
                    chunk_sha256=chunk.chunk_sha256,
                )
            ),
            round_id=round_id,
            priority=Priority.ACK,
        )

    async def _send_message(
        self,
        *,
        destination: PublicKey,
        message_type: MessageType,
        message_id: MessageId,
        correlation_id: MessageId | None,
        payload: bytes,
        round_id: RoundId | None,
        priority: Priority,
    ) -> SendTiming:
        envelope = create_envelope(
            message_type=message_type,
            message_id=message_id,
            run_id=self._run_id,
            manifest_hash=self._manifest_hash,
            sender_public_key=self._local_public_key,
            algorithm_id=self._algorithm_id,
            round_id=round_id,
            correlation_id=correlation_id,
            payload=payload,
        )
        timing = await self._sender.send(
            destination,
            encode_envelope(envelope),
            priority=priority,
            retries=self._transport_limits.max_retries,
            retry_delay_seconds=self._transport_limits.retry_timeout_seconds,
        )
        emit_event(
            "transfer_message_sent",
            run_id=self._run_id,
            manifest_hash=self._manifest_hash,
            node_id=self._local_public_key,
            message_id=message_id,
            transfer_id=correlation_id,
            peer_id=destination,
            round_id=round_id,
            message_type=message_type,
            payload_bytes=len(payload),
            queue_seconds=timing.queue_seconds,
            send_seconds=timing.send_seconds,
            retry_count=timing.retry_count,
            completion_seconds=timing.completion_seconds,
            sink=self._event_sink,
        )
        return timing
