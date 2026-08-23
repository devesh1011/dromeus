"""Reliable artifact transfer."""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from dromeus.manifests.canonical import file_sha256
from dromeus.manifests.models import (
    AlgorithmId,
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
from dromeus.protocol.codec import (
    ProtocolDecodeError,
    decode_message,
    encode_envelope,
    encode_message,
)
from dromeus.protocol.models import (
    Chunk,
    ChunkAck,
    Envelope,
    MessageType,
    TransferBegin,
    TransferComplete,
    create_envelope,
)
from dromeus.telemetry.events import EventSink, emit_event
from dromeus.telemetry.evidence import (
    TransferMessageSentEvidence,
    append_evidence,
)
from dromeus.transport.outbound_scheduler import (
    OutboundScheduler,
    Priority,
    SendTiming,
)
from dromeus.transport.receiver import MessageChannel, Receiver


class TransferError(RuntimeError):
    """Artifact transfer failed terminally."""


@dataclass(frozen=True)
class ArtifactReceipt:
    transfer_id: TransferId
    sender_public_key: PublicKey
    artifact_name: Identifier
    path: Path
    sha256: Sha256
    size_bytes: int
    round_id: RoundId | None
    codec_id: Identifier
    tensor_schema: TensorSchema


@dataclass(frozen=True)
class TransferTiming:
    """Observable duration and retry count for one outbound artifact."""

    elapsed_seconds: float
    retry_count: int


@dataclass
class _Reservation:
    size_bytes: int
    final_path: Path | None = None


class _ArtifactStore:
    """Private transfer storage with exact per-transfer reservations."""

    def __init__(self, root: Path, *, max_bytes: int = 128 * 1024 * 1024) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self._root = root
        self._tmp_root = root / ".tmp"
        self._root.mkdir(parents=True, exist_ok=True)
        self._tmp_root.mkdir(parents=True, exist_ok=True)
        self._max_bytes = max_bytes
        self._reserved_bytes = 0
        self._reservations: dict[TransferId, _Reservation] = {}

    def reserve(self, transfer_id: TransferId, size_bytes: int) -> None:
        if transfer_id in self._reservations:
            raise TransferError("transfer capacity already reserved")
        if self._reserved_bytes + size_bytes > self._max_bytes:
            raise TransferError("artifact store capacity exceeded")
        temp = self._temp_path(transfer_id)
        try:
            temp.unlink(missing_ok=True)
            temp.touch(exist_ok=False)
        except OSError as error:
            raise TransferError("failed to create transfer artifact") from error
        self._reservations[transfer_id] = _Reservation(size_bytes=size_bytes)
        self._reserved_bytes += size_bytes

    def append(self, transfer_id: TransferId, data: bytes) -> None:
        reservation = self._reservation(transfer_id)
        if reservation.final_path is not None:
            raise TransferError("cannot append to finalized transfer")
        try:
            with self._temp_path(transfer_id).open("ab") as handle:
                handle.write(data)
        except OSError as error:
            raise TransferError("failed to write transfer artifact") from error

    def write_chunk(
        self,
        transfer_id: TransferId,
        *,
        offset: int,
        data: bytes,
    ) -> None:
        reservation = self._reservation(transfer_id)
        if reservation.final_path is not None:
            raise TransferError("cannot write to finalized transfer")
        if offset < 0 or offset + len(data) > reservation.size_bytes:
            raise TransferError("chunk write exceeds transfer reservation")
        try:
            with self._temp_path(transfer_id).open("r+b") as handle:
                handle.seek(offset)
                handle.write(data)
        except OSError as error:
            raise TransferError("failed to write transfer chunk") from error

    def finalize(
        self,
        transfer_id: TransferId,
        artifact_name: Identifier,
        expected_size_bytes: int,
        expected_sha256: Sha256,
    ) -> tuple[Path, Sha256]:
        reservation = self._reservation(transfer_id)
        if reservation.size_bytes != expected_size_bytes:
            raise TransferError("transfer reservation size mismatch")
        temp = self._temp_path(transfer_id)
        try:
            if temp.stat().st_size != expected_size_bytes:
                raise TransferError("final artifact size mismatch")
            digest = file_sha256(temp)
            if digest != expected_sha256:
                raise TransferError("final artifact checksum mismatch")
            final = self._final_path(transfer_id, artifact_name)
            if final.exists():
                raise TransferError("final artifact already exists")
            temp.replace(final)
        except OSError as error:
            raise TransferError("failed to finalize transfer artifact") from error
        reservation.final_path = final
        return final, digest

    def commit(self, transfer_id: TransferId) -> None:
        reservation = self._reservation(transfer_id)
        if reservation.final_path is None:
            raise TransferError("cannot commit unfinished transfer")

    def claim(self, transfer_id: TransferId) -> None:
        reservation = self._reservation(transfer_id)
        if reservation.final_path is None:
            raise TransferError("cannot claim unfinished transfer")
        self._release(transfer_id)

    def abort(self, transfer_id: TransferId) -> None:
        reservation = self._reservation(transfer_id)
        try:
            self._temp_path(transfer_id).unlink(missing_ok=True)
            if reservation.final_path is not None:
                reservation.final_path.unlink(missing_ok=True)
        except OSError as error:
            raise TransferError("failed to remove transfer artifact") from error
        self._release(transfer_id)

    def _reservation(self, transfer_id: TransferId) -> _Reservation:
        reservation = self._reservations.get(transfer_id)
        if reservation is None:
            raise TransferError("transfer capacity is not reserved")
        return reservation

    def _release(self, transfer_id: TransferId) -> None:
        reservation = self._reservations.pop(transfer_id, None)
        if reservation is None:
            raise TransferError("transfer capacity already released")
        remaining = self._reserved_bytes - reservation.size_bytes
        if remaining < 0:
            raise TransferError("transfer capacity accounting underflow")
        self._reserved_bytes = remaining

    def _temp_path(self, transfer_id: TransferId) -> Path:
        return self._tmp_root / f"{transfer_id}.part"

    def _final_path(
        self, transfer_id: TransferId, artifact_name: Identifier
    ) -> Path:
        safe_name = artifact_name.replace("/", "_")
        return self._root / f"{transfer_id}-{safe_name}.bin"


@dataclass
class _IncomingTransfer:
    sender_public_key: PublicKey
    begin: TransferBegin
    round_id: RoundId | None
    written_chunk_hashes: dict[int, Sha256]
    started_at: float


def _artifact_metadata(path: Path) -> tuple[int, Sha256]:
    try:
        size_bytes = path.stat().st_size
    except OSError as error:
        raise TransferError("cannot inspect transfer artifact") from error
    if size_bytes <= 0:
        raise TransferError("transfer artifact must not be empty")
    return size_bytes, file_sha256(path)


def _read_chunk(path: Path, *, index: int, chunk_size: int) -> bytes:
    try:
        with path.open("rb") as handle:
            handle.seek(index * chunk_size)
            return handle.read(chunk_size)
    except OSError as error:
        raise TransferError("cannot read transfer artifact chunk") from error


class TransferManager:
    """Bounded multi-chunk transfer with sliding-window selective retry."""

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
        artifact_root: Path,
        max_inflight_bytes: int = 128 * 1024 * 1024,
        event_sink: EventSink | None = None,
    ) -> None:
        self._local_public_key = local_public_key
        self._run_id = run_id
        self._manifest_hash = manifest_hash
        self._algorithm_id = algorithm_id
        self._transport_limits = transport_limits
        self._receiver = receiver
        self._sender = sender
        self._artifact_store = _ArtifactStore(
            artifact_root,
            max_bytes=transport_limits.artifact_store_capacity_limit,
        )
        self._max_incoming_reservation_bytes = max_inflight_bytes
        self._incoming_reserved_bytes = 0
        self._event_sink = event_sink
        self._incoming: dict[TransferId, _IncomingTransfer] = {}
        self._completed: dict[TransferId, ArtifactReceipt] = {}
        self._completed_futures: dict[TransferId, asyncio.Future[ArtifactReceipt]] = {}
        self._completed_queue: asyncio.Queue[ArtifactReceipt] = asyncio.Queue(
            maxsize=64
        )
        self._discarded_round_transfers: set[tuple[PublicKey, RoundId]] = set()
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
        tasks = tuple(
            task
            for task in (self._transfer_task, self._ack_task)
            if task is not None
        )
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for transfer_id in tuple(self._incoming):
            self._abort_incoming(transfer_id)
        for waiter in self._ack_waiters.values():
            if not waiter.done():
                waiter.cancel()
        self._ack_waiters.clear()
        for future in self._completed_futures.values():
            if not future.done():
                future.cancel()
        for transfer_id in tuple(self._completed):
            await asyncio.to_thread(self._artifact_store.abort, transfer_id)
        self._completed.clear()
        self._completed_futures.clear()
        while True:
            try:
                self._completed_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

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
            payload_size, total_sha256 = await asyncio.to_thread(
                _artifact_metadata, artifact_path
            )
            if payload_size > self._transport_limits.artifact_size_limit:
                raise TransferError("transfer exceeds manifest artifact limit")
            chunk_size = self._transport_limits.effective_chunk_size
            chunk_count = math.ceil(payload_size / chunk_size)
            transfer_id = str(uuid.uuid4())
            begin = TransferBegin(
                transfer_id=transfer_id,
                artifact_name=artifact_name,
                total_size_bytes=payload_size,
                total_sha256=total_sha256,
                chunk_count=chunk_count,
                codec_id=codec_id,
                tensor_schema=tensor_schema,
            )
            begin_timing = await self._send_message(
                destination=destination,
                message_type=MessageType.TRANSFER_BEGIN,
                message_id=f"{transfer_id}-begin",
                correlation_id=transfer_id,
                payload=encode_message(begin),
                round_id=round_id,
                priority=Priority.CONTROL,
            )
            retry_count += begin_timing.retry_count
            semaphore = asyncio.Semaphore(
                self._transport_limits.effective_window_size
            )
            chunk_tasks = tuple(
                asyncio.create_task(
                    self._send_chunk_with_ack(
                        semaphore=semaphore,
                        destination=destination,
                        artifact_path=artifact_path,
                        transfer_id=transfer_id,
                        begin=begin,
                        chunk_index=chunk_index,
                        chunk_size=chunk_size,
                        round_id=round_id,
                    ),
                    name=f"dromeus-transfer-chunk-{chunk_index}",
                )
                for chunk_index in range(chunk_count)
            )
            try:
                chunk_retries = await asyncio.gather(*chunk_tasks)
            except BaseException:
                for task in chunk_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*chunk_tasks, return_exceptions=True)
                raise
            retry_count += sum(chunk_retries)
            complete_timing = await self._send_message(
                destination=destination,
                message_type=MessageType.TRANSFER_COMPLETE,
                message_id=f"{transfer_id}-complete",
                correlation_id=transfer_id,
                payload=encode_message(
                    TransferComplete(
                        transfer_id=transfer_id, total_sha256=total_sha256
                    )
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

    async def _send_chunk_with_ack(
        self,
        *,
        semaphore: asyncio.Semaphore,
        destination: PublicKey,
        artifact_path: Path,
        transfer_id: TransferId,
        begin: TransferBegin,
        chunk_index: int,
        chunk_size: int,
        round_id: RoundId | None,
    ) -> int:
        async with semaphore:
            chunk_bytes = await asyncio.to_thread(
                _read_chunk,
                artifact_path,
                index=chunk_index,
                chunk_size=chunk_size,
            )
            if not chunk_bytes:
                raise TransferError("transfer artifact changed during chunking")
            chunk = Chunk(
                transfer_id=transfer_id,
                chunk_index=chunk_index,
                chunk_count=begin.chunk_count,
                chunk_sha256=hashlib.sha256(chunk_bytes).hexdigest(),
                data=chunk_bytes,
            )
            retry_count = 0
            attempts = self._transport_limits.max_retries + 1
            for attempt in range(attempts):
                ack_key = (transfer_id, chunk_index)
                loop = asyncio.get_running_loop()
                ack_future: asyncio.Future[ChunkAck] = loop.create_future()
                self._ack_waiters[ack_key] = ack_future
                try:
                    if attempt > 0:
                        retry_count += 1
                        begin_timing = await self._send_message(
                            destination=destination,
                            message_type=MessageType.TRANSFER_BEGIN,
                            message_id=f"{transfer_id}-begin",
                            correlation_id=transfer_id,
                            payload=encode_message(begin),
                            round_id=round_id,
                            priority=Priority.CONTROL,
                        )
                        retry_count += begin_timing.retry_count
                    chunk_timing = await self._send_message(
                        destination=destination,
                        message_type=MessageType.CHUNK,
                        message_id=f"{transfer_id}-chunk-{chunk_index}",
                        correlation_id=transfer_id,
                        payload=encode_message(chunk),
                        round_id=round_id,
                        priority=Priority.DATA,
                    )
                    retry_count += chunk_timing.retry_count
                    ack = await asyncio.wait_for(
                        asyncio.shield(ack_future),
                        timeout=self._transport_limits.retry_timeout_seconds,
                    )
                except TimeoutError:
                    if self._ack_waiters.get(ack_key) is ack_future:
                        self._ack_waiters.pop(ack_key, None)
                    ack_future.cancel()
                    continue
                except BaseException:
                    if self._ack_waiters.get(ack_key) is ack_future:
                        self._ack_waiters.pop(ack_key, None)
                    ack_future.cancel()
                    raise
                if ack.chunk_sha256 != chunk.chunk_sha256:
                    raise TransferError("chunk acknowledgement checksum mismatch")
                return retry_count
            raise TransferError("chunk acknowledgement retries exhausted")

    async def next_artifact(self, *, timeout_seconds: float) -> ArtifactReceipt:
        while True:
            receipt = await asyncio.wait_for(
                self._completed_queue.get(), timeout=timeout_seconds
            )
            if (
                receipt.round_id is not None
                and (receipt.sender_public_key, receipt.round_id)
                in self._discarded_round_transfers
            ):
                await self.release_receipt(receipt)
                continue
            return receipt

    async def release_receipt(self, receipt: ArtifactReceipt) -> None:
        """Remove one materialized receipt and its artifact."""
        await asyncio.to_thread(receipt.path.unlink, missing_ok=True)
        self.claim_receipt(receipt)

    def claim_receipt(self, receipt: ArtifactReceipt) -> None:
        """Transfer ownership of one materialized artifact to its consumer."""
        completed = self._completed.pop(receipt.transfer_id, None)
        if completed is not None:
            self._artifact_store.claim(receipt.transfer_id)
        self._completed_futures.pop(receipt.transfer_id, None)
        retained: list[ArtifactReceipt] = []
        while True:
            try:
                queued = self._completed_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if queued.transfer_id != receipt.transfer_id:
                retained.append(queued)
        for queued in retained:
            self._completed_queue.put_nowait(queued)

    async def discard_round_transfers(
        self, *, sender: PublicKey, round_id: RoundId
    ) -> None:
        """Reject and remove all transfer state for one failed peer round."""
        self._discarded_round_transfers.add((sender, round_id))
        for transfer_id, incoming in tuple(self._incoming.items()):
            if incoming.sender_public_key == sender and incoming.round_id == round_id:
                self._abort_incoming(transfer_id)
        for receipt in tuple(self._completed.values()):
            if receipt.sender_public_key == sender and receipt.round_id == round_id:
                await self.release_receipt(receipt)
        retained: list[ArtifactReceipt] = []
        while True:
            try:
                receipt = self._completed_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if receipt.sender_public_key == sender and receipt.round_id == round_id:
                await self.release_receipt(receipt)
            else:
                retained.append(receipt)
        for receipt in retained:
            self._completed_queue.put_nowait(receipt)

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
            except (TransferError, ValueError, ProtocolDecodeError) as error:
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
                ack = decode_message(
                    envelope.payload,
                    ChunkAck,
                    max_bytes=self._transport_limits.message_payload_limit,
                )
                waiter = self._ack_waiters.pop((ack.transfer_id, ack.chunk_index), None)
                if waiter is not None and not waiter.done():
                    waiter.set_result(ack)
            except (ValueError, ProtocolDecodeError) as error:
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
            begin = decode_message(
                envelope.payload,
                TransferBegin,
                max_bytes=self._transport_limits.message_payload_limit,
            )
            if (
                envelope.round_id is not None
                and (envelope.sender_public_key, envelope.round_id)
                in self._discarded_round_transfers
            ):
                raise TransferError("round transfers were discarded")
            if begin.total_size_bytes > self._transport_limits.artifact_size_limit:
                raise TransferError("transfer exceeds manifest artifact limit")
            expected_chunk_count = math.ceil(
                begin.total_size_bytes
                / self._transport_limits.effective_chunk_size
            )
            if begin.chunk_count != expected_chunk_count:
                raise TransferError("transfer chunk count does not match manifest")
            incoming = self._incoming.get(begin.transfer_id)
            if incoming is not None:
                if (
                    incoming.sender_public_key != envelope.sender_public_key
                    or incoming.round_id != envelope.round_id
                    or incoming.begin != begin
                ):
                    raise TransferError("conflicting duplicate transfer begin")
                return
            if begin.transfer_id in self._completed:
                raise TransferError("transfer is already complete")
            if len(self._incoming) >= self._transport_limits.concurrent_transfer_limit:
                raise TransferError("concurrent transfer limit exceeded")
            if (
                self._incoming_reserved_bytes + begin.total_size_bytes
                > self._max_incoming_reservation_bytes
            ):
                raise TransferError("local incoming reservation limit exceeded")
            self._artifact_store.reserve(
                begin.transfer_id, begin.total_size_bytes
            )
            self._incoming_reserved_bytes += begin.total_size_bytes
            self._incoming[begin.transfer_id] = _IncomingTransfer(
                sender_public_key=envelope.sender_public_key,
                begin=begin,
                round_id=envelope.round_id,
                written_chunk_hashes={},
                started_at=time.monotonic(),
            )
            return
        if envelope.message_type is MessageType.CHUNK:
            chunk = decode_message(
                envelope.payload,
                Chunk,
                max_bytes=self._transport_limits.message_payload_limit,
            )
            incoming = self._incoming.get(chunk.transfer_id)
            if incoming is None:
                raise TransferError("received chunk without transfer begin")
            if envelope.sender_public_key != incoming.sender_public_key:
                raise TransferError("transfer sender changed")
            if (
                chunk.chunk_count != incoming.begin.chunk_count
                or chunk.chunk_index >= chunk.chunk_count
            ):
                self._abort_incoming(chunk.transfer_id)
                raise TransferError("invalid transfer chunk index or count")
            expected_size = min(
                self._transport_limits.effective_chunk_size,
                incoming.begin.total_size_bytes
                - (
                    chunk.chunk_index
                    * self._transport_limits.effective_chunk_size
                ),
            )
            if len(chunk.data) != expected_size:
                self._abort_incoming(chunk.transfer_id)
                raise TransferError("transfer chunk size does not match manifest")
            written_hash = incoming.written_chunk_hashes.get(chunk.chunk_index)
            if written_hash is not None:
                if written_hash != chunk.chunk_sha256:
                    self._abort_incoming(chunk.transfer_id)
                    raise TransferError("conflicting duplicate transfer chunk")
                await self._acknowledge_chunk(
                    envelope.sender_public_key, chunk, envelope.round_id
                )
                return
            if hashlib.sha256(chunk.data).hexdigest() != chunk.chunk_sha256:
                raise TransferError("chunk checksum mismatch")
            append_task = asyncio.create_task(
                asyncio.to_thread(
                    self._artifact_store.write_chunk,
                    chunk.transfer_id,
                    offset=(
                        chunk.chunk_index
                        * self._transport_limits.effective_chunk_size
                    ),
                    data=chunk.data,
                )
            )
            try:
                await asyncio.shield(append_task)
            except asyncio.CancelledError:
                try:
                    await append_task
                finally:
                    self._abort_incoming(chunk.transfer_id)
                raise
            except TransferError:
                self._abort_incoming(chunk.transfer_id)
                raise
            incoming.written_chunk_hashes[chunk.chunk_index] = chunk.chunk_sha256
            await self._acknowledge_chunk(
                envelope.sender_public_key, chunk, envelope.round_id
            )
            return
        complete = decode_message(
            envelope.payload,
            TransferComplete,
            max_bytes=self._transport_limits.message_payload_limit,
        )
        incoming = self._incoming.get(complete.transfer_id)
        if incoming is None:
            if complete.transfer_id in self._completed:
                return
            raise TransferError("received completion without transfer begin")
        if complete.total_sha256 != incoming.begin.total_sha256:
            self._abort_incoming(complete.transfer_id)
            raise TransferError("completion checksum does not match transfer begin")
        if len(incoming.written_chunk_hashes) != incoming.begin.chunk_count:
            raise TransferError("transfer completed before all chunks arrived")
        finalize_task = asyncio.create_task(
            asyncio.to_thread(
                self._artifact_store.finalize,
                complete.transfer_id,
                incoming.begin.artifact_name,
                incoming.begin.total_size_bytes,
                complete.total_sha256,
            )
        )
        try:
            final_path, digest = await asyncio.shield(finalize_task)
        except asyncio.CancelledError:
            try:
                await finalize_task
            finally:
                self._abort_incoming(complete.transfer_id)
            raise
        except TransferError:
            self._abort_incoming(complete.transfer_id)
            raise
        receipt = ArtifactReceipt(
            transfer_id=complete.transfer_id,
            sender_public_key=incoming.sender_public_key,
            artifact_name=incoming.begin.artifact_name,
            path=final_path,
            sha256=digest,
            size_bytes=incoming.begin.total_size_bytes,
            round_id=incoming.round_id,
            codec_id=incoming.begin.codec_id,
            tensor_schema=incoming.begin.tensor_schema,
        )
        self._artifact_store.commit(complete.transfer_id)
        self._incoming.pop(complete.transfer_id, None)
        self._incoming_reserved_bytes -= incoming.begin.total_size_bytes
        self._completed[complete.transfer_id] = receipt
        future = self._completed_futures.get(complete.transfer_id)
        if future is not None and not future.done():
            future.set_result(receipt)
        await self._completed_queue.put(receipt)

    def _expire_incoming(self) -> None:
        lifetime = self._transport_limits.transfer_lifetime_limit
        now = time.monotonic()
        for transfer_id, incoming in tuple(self._incoming.items()):
            if now - incoming.started_at >= lifetime:
                self._abort_incoming(transfer_id)

    def _abort_incoming(self, transfer_id: TransferId) -> None:
        incoming = self._incoming.pop(transfer_id, None)
        if incoming is not None:
            self._incoming_reserved_bytes -= incoming.begin.total_size_bytes
            if self._incoming_reserved_bytes < 0:
                raise TransferError("incoming reservation accounting underflow")
            self._artifact_store.abort(transfer_id)

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
            payload=encode_message(
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
        append_evidence(
            self._event_sink,
            TransferMessageSentEvidence(
                run_id=self._run_id,
                manifest_hash=self._manifest_hash,
                node_id=self._local_public_key,
                message_id=message_id,
                transfer_id=correlation_id,
                peer_id=destination,
                round_id=round_id,
                message_type=message_type.value,
                payload_bytes=len(payload),
                queue_seconds=float(timing.queue_seconds),
                send_seconds=float(timing.send_seconds),
                retry_count=timing.retry_count,
                completion_seconds=float(timing.completion_seconds),
            ),
        )
        return timing
