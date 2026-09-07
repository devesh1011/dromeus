"""AXL pair exchange and broadcasting over reliable opaque artifact transfer."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path

import numpy as np

from dromeus.algorithms.base import MaterializedArtifact, UpdateBundle
from dromeus.gossip._async import run_blocking as _run_blocking
from dromeus.gossip.interfaces import (
    ConsensusBroadcastResult,
    PairCommitError,
    PairExchangeResult,
    RunFailure,
)
from dromeus.manifests.canonical import (
    materialize_bundle_metadata,
    parse_bundle_metadata,
)
from dromeus.manifests.models import (
    AlgorithmId,
    MessageId,
    OpaqueUpdateBundleMetadata,
    PublicKey,
    RoundId,
    RunId,
    Sha256,
    TransportLimits,
)
from dromeus.protocol.codec import decode_message, encode_envelope, encode_message
from dromeus.protocol.models import (
    Envelope,
    MessageType,
    PairCommitMessage,
    RunFailedMessage,
    create_envelope,
)
from dromeus.telemetry.consensus import encode_sketch
from dromeus.transport.outbound_scheduler import (
    OutboundScheduler,
    Priority,
    SendTiming,
)
from dromeus.transport.receiver import MessageChannel, Receiver
from dromeus.transport.transfer import ArtifactReceipt, TransferError, TransferManager


def decode_run_failure(payload: bytes) -> RunFailure:
    """Decode validated terminal failure evidence from a control envelope."""
    message = decode_message(
        payload,
        RunFailedMessage,
        max_bytes=4096,
    )
    return RunFailure(
        round_id=message.round_id,
        error_type=message.error_type,
        reason=message.reason,
    )


class AXLFailureBroadcaster:
    """Failure-only control broadcaster with no artifact filesystem dependency."""

    def __init__(
        self,
        *,
        local_public_key: PublicKey,
        run_id: RunId,
        manifest_hash: Sha256,
        algorithm_id: AlgorithmId,
        transport_limits: TransportLimits,
        sender: OutboundScheduler,
        participant_keys: frozenset[PublicKey],
    ) -> None:
        self._local_public_key = local_public_key
        self._run_id = run_id
        self._manifest_hash = manifest_hash
        self._algorithm_id = algorithm_id
        self._transport_limits = transport_limits
        self._sender = sender
        self._participant_keys = participant_keys

    async def broadcast_run_failed(self, failure: RunFailure) -> None:
        """Best-effort failure broadcast using manifest-bounded sender retries."""
        peers = self._participant_keys - {self._local_public_key}
        if not peers:
            return
        payload = encode_message(
            RunFailedMessage(
                round_id=failure.round_id,
                error_type=failure.error_type[:128],
                reason=failure.reason[:1024],
            )
        )

        async def send(peer: PublicKey) -> None:
            envelope = create_envelope(
                message_type=MessageType.RUN_FAILED,
                message_id=f"run-failed-{failure.round_id}-{self._local_public_key[:8]}",
                run_id=self._run_id,
                manifest_hash=self._manifest_hash,
                sender_public_key=self._local_public_key,
                algorithm_id=self._algorithm_id,
                round_id=failure.round_id,
                correlation_id=f"run-failure-{failure.round_id}",
                payload=payload,
            )
            await self._sender.send(
                peer,
                encode_envelope(envelope),
                priority=Priority.CONTROL,
                retries=self._transport_limits.max_retries,
                retry_delay_seconds=self._transport_limits.retry_timeout_seconds,
            )

        await asyncio.gather(*(send(peer) for peer in peers), return_exceptions=True)


class AXLPairTransport:
    """Pair transport backed by reliable opaque artifact transfer."""

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
        transfer_manager: TransferManager,
        metadata_root: Path,
        participant_keys: frozenset[PublicKey] | None = None,
    ) -> None:
        self._local_public_key = local_public_key
        self._run_id = run_id
        self._manifest_hash = manifest_hash
        self._algorithm_id = algorithm_id
        self._transport_limits = transport_limits
        self._receiver = receiver
        self._sender = sender
        self._transfer_manager = transfer_manager
        self._metadata_root = metadata_root
        self._participant_keys = participant_keys or frozenset()
        self._failure_broadcaster = AXLFailureBroadcaster(
            local_public_key=local_public_key,
            run_id=run_id,
            manifest_hash=manifest_hash,
            algorithm_id=algorithm_id,
            transport_limits=transport_limits,
            sender=sender,
            participant_keys=self._participant_keys,
        )
        self._ready_cache: dict[RoundId, str] = {}
        self._committed_rounds: dict[RoundId, str] = {}

    async def broadcast_run_failed(self, failure: RunFailure) -> None:
        await self._failure_broadcaster.broadcast_run_failed(failure)

    async def broadcast_consensus_sketch(
        self, *, round_id: RoundId, sketch: np.ndarray
    ) -> ConsensusBroadcastResult:
        """Best-effort low-priority broadcast of one FP32 consensus sketch."""
        peers = self._participant_keys - {self._local_public_key}
        payload = encode_sketch(sketch)
        if not peers:
            return ConsensusBroadcastResult(
                payload_bytes=len(payload),
                recipient_count=0,
                successful_recipient_count=0,
                retry_count=0,
            )

        async def send(peer: PublicKey) -> SendTiming:
            envelope = create_envelope(
                message_type=MessageType.CONSENSUS_SKETCH,
                message_id=(
                    f"consensus-sketch-{round_id}-{self._local_public_key[:8]}"
                ),
                run_id=self._run_id,
                manifest_hash=self._manifest_hash,
                sender_public_key=self._local_public_key,
                algorithm_id=self._algorithm_id,
                round_id=round_id,
                correlation_id=f"consensus-round-{round_id}",
                payload=payload,
            )
            return await self._sender.send(
                peer,
                encode_envelope(envelope),
                priority=Priority.TELEMETRY,
                retries=self._transport_limits.max_retries,
                retry_delay_seconds=self._transport_limits.retry_timeout_seconds,
            )

        results = await asyncio.gather(
            *(send(peer) for peer in peers),
            return_exceptions=True,
        )
        timings = [result for result in results if isinstance(result, SendTiming)]
        return ConsensusBroadcastResult(
            payload_bytes=len(payload),
            recipient_count=len(peers),
            successful_recipient_count=len(timings),
            retry_count=sum(timing.retry_count for timing in timings),
        )

    async def exchange_update(
        self,
        *,
        peer: PublicKey,
        round_id: RoundId,
        bundle: UpdateBundle,
    ) -> PairExchangeResult:
        self._validate_bundle(
            bundle.metadata,
            sender=self._local_public_key,
            round_id=round_id,
        )
        await _run_blocking(
            bundle.validate_materialized,
            self._transport_limits.max_update_bundle_bytes,
        )
        receipts: list[ArtifactReceipt] = []
        metadata_receipt: ArtifactReceipt | None = None
        claimed = False
        carrier_task = asyncio.create_task(
            asyncio.to_thread(
                materialize_bundle_metadata,
                bundle.metadata,
                self._metadata_root,
            )
        )
        try:
            metadata_carrier = await asyncio.shield(carrier_task)
        except asyncio.CancelledError:
            metadata_carrier = await carrier_task
            await _run_blocking(metadata_carrier.path.unlink, True)
            raise
        try:
            retry_count = 0
            transfer_id = await self._transfer_manager.send_artifact(
                destination=peer,
                artifact_name="update-bundle-metadata",
                artifact_path=metadata_carrier.path,
                codec_id="safetensors-v1",
                tensor_schema=metadata_carrier.tensor_schema,
                round_id=round_id,
            )
            timing = self._transfer_manager.last_timing
            retry_count += timing.retry_count if timing is not None else 0
            metadata_receipt = await self._next_peer_receipt(
                peer=peer, round_id=round_id
            )
            if metadata_receipt.artifact_name != "update-bundle-metadata":
                raise PairCommitError("peer update metadata artifact is missing")
            peer_metadata = await _run_blocking(
                parse_bundle_metadata, metadata_receipt.path
            )
            self._validate_bundle(peer_metadata, sender=peer, round_id=round_id)
            await self._transfer_manager.release_receipt(metadata_receipt)
            metadata_receipt = None

            for artifact, materialized in zip(
                bundle.metadata.artifacts,
                bundle.artifacts,
                strict=True,
            ):
                transfer_id = (
                    await self._transfer_manager.send_artifact(
                        destination=peer,
                        artifact_name=artifact.name,
                        artifact_path=materialized.path,
                        codec_id=materialized.transfer_codec_id,
                        tensor_schema=materialized.transfer_schema,
                        round_id=round_id,
                    )
                )
                timing = self._transfer_manager.last_timing
                retry_count += timing.retry_count if timing is not None else 0
            for _ in peer_metadata.artifacts:
                receipt = await self._next_peer_receipt(
                    peer=peer, round_id=round_id
                )
                receipts.append(receipt)
            receipt_by_name = {receipt.artifact_name: receipt for receipt in receipts}
            if len(receipt_by_name) != len(receipts):
                raise PairCommitError("peer update contains duplicate artifacts")
            for artifact in peer_metadata.artifacts:
                receipt = receipt_by_name.get(artifact.name)
                if (
                    receipt is None
                    or receipt.size_bytes != artifact.size_bytes
                    or receipt.sha256 != artifact.sha256
                ):
                    raise PairCommitError("peer update artifact metadata mismatch")
            peer_bundle = UpdateBundle(
                metadata=peer_metadata,
                artifacts=tuple(
                    MaterializedArtifact(
                        path=receipt_by_name[artifact.name].path,
                        transfer_codec_id=receipt_by_name[
                            artifact.name
                        ].codec_id,
                        transfer_schema=receipt_by_name[
                            artifact.name
                        ].tensor_schema,
                    )
                    for artifact in peer_metadata.artifacts
                ),
            )
            await _run_blocking(
                peer_bundle.validate_materialized,
                self._transport_limits.max_update_bundle_bytes,
            )
            for receipt in receipts:
                self._transfer_manager.claim_receipt(receipt)
            claimed = True
            return PairExchangeResult(
                bundle=peer_bundle,
                transfer_id=transfer_id,
                retry_count=retry_count,
            )
        except (
            OSError,
            ValueError,
            TypeError,
            TransferError,
        ) as error:
            raise PairCommitError("peer update transfer failed") from error
        finally:
            if not claimed:
                await self._transfer_manager.discard_round_transfers(
                    sender=peer, round_id=round_id
                )
                for receipt in receipts:
                    await self._transfer_manager.release_receipt(receipt)
            if metadata_receipt is not None:
                await self._transfer_manager.release_receipt(metadata_receipt)
            await _run_blocking(metadata_carrier.path.unlink, True)

    async def _next_peer_receipt(
        self, *, peer: PublicKey, round_id: RoundId
    ) -> ArtifactReceipt:
        receipt = await self._transfer_manager.next_artifact(
            timeout_seconds=self._timeout_seconds
        )
        if receipt.sender_public_key != peer or receipt.round_id != round_id:
            await self._transfer_manager.release_receipt(receipt)
            raise PairCommitError("received unexpected peer update artifact")
        return receipt

    def _validate_bundle(
        self,
        metadata: OpaqueUpdateBundleMetadata,
        *,
        sender: PublicKey,
        round_id: RoundId,
    ) -> None:
        if (
            metadata.run_id != self._run_id
            or metadata.manifest_hash != self._manifest_hash
            or metadata.algorithm_id != self._algorithm_id
            or metadata.sender_public_key != sender
            or metadata.round_id != round_id
        ):
            raise PairCommitError("update bundle context mismatch")
        if (
            sum(artifact.size_bytes for artifact in metadata.artifacts)
            > self._transport_limits.max_update_bundle_bytes
        ):
            raise PairCommitError("update bundle exceeds manifest payload limit")

    async def exchange_update_ready(
        self,
        *,
        peer: PublicKey,
        round_id: RoundId,
        bundle_checksum: str,
    ) -> str:
        cached = self._ready_cache.get(round_id)
        if cached is not None:
            return cached
        envelope = await self._exchange_pair_message(
            destination=peer,
            message_type=MessageType.UPDATE_READY,
            message_id=f"update-ready-{round_id}",
            payload=PairCommitMessage(round_id=round_id, checksum=bundle_checksum),
            round_id=round_id,
            exchange_timeout_seconds=(
                self._transport_limits.transfer_lifetime_limit
            ),
        )
        message = decode_message(
            envelope.payload,
            PairCommitMessage,
            max_bytes=self._transport_limits.max_payload_bytes,
        )
        if message.round_id != round_id:
            raise PairCommitError("peer UPDATE_READY round mismatch")
        self._ready_cache[round_id] = message.checksum
        return message.checksum

    async def exchange_round_committed(
        self,
        *,
        peer: PublicKey,
        round_id: RoundId,
        state_checksum: str,
    ) -> None:
        committed_checksum = self._committed_rounds.get(round_id)
        if committed_checksum is not None:
            if committed_checksum != state_checksum:
                raise PairCommitError("duplicate ROUND_COMMITTED checksum mismatch")
            return
        envelope = await self._exchange_pair_message(
            destination=peer,
            message_type=MessageType.ROUND_COMMITTED,
            message_id=f"round-committed-{round_id}",
            payload=PairCommitMessage(round_id=round_id, checksum=state_checksum),
            round_id=round_id,
        )
        message = decode_message(
            envelope.payload,
            PairCommitMessage,
            max_bytes=self._transport_limits.max_payload_bytes,
        )
        if message.round_id != round_id:
            raise PairCommitError("peer ROUND_COMMITTED round mismatch")
        self._receiver.set_current_round(round_id + 1)
        await self._receiver.advance_round(round_id + 1)
        self._committed_rounds[round_id] = state_checksum

    @property
    def _timeout_seconds(self) -> float:
        return max(
            self._transport_limits.transfer_lifetime_limit,
            self._transport_limits.retry_timeout_seconds
            * (self._transport_limits.max_retries + 4),
        )

    async def _send_pair_message(
        self,
        *,
        destination: PublicKey,
        message_type: MessageType,
        message_id: MessageId,
        payload: PairCommitMessage,
        round_id: RoundId,
    ) -> None:
        envelope = create_envelope(
            message_type=message_type,
            message_id=message_id,
            run_id=self._run_id,
            manifest_hash=self._manifest_hash,
            sender_public_key=self._local_public_key,
            algorithm_id=self._algorithm_id,
            round_id=round_id,
            correlation_id=f"pair-round-{round_id}",
            payload=encode_message(payload),
        )
        await self._sender.send(
            destination,
            encode_envelope(envelope),
            priority=Priority.CONTROL,
            retries=self._transport_limits.max_retries,
            retry_delay_seconds=self._transport_limits.retry_timeout_seconds,
        )

    async def _exchange_pair_message(
        self,
        *,
        destination: PublicKey,
        message_type: MessageType,
        message_id: MessageId,
        payload: PairCommitMessage,
        round_id: RoundId,
        exchange_timeout_seconds: float | None = None,
    ) -> Envelope:
        attempts = (
            math.ceil(
                exchange_timeout_seconds
                / self._transport_limits.retry_timeout_seconds
            )
            if exchange_timeout_seconds is not None
            else self._transport_limits.max_retries + 1
        )
        for attempt in range(attempts):
            await self._send_pair_message(
                destination=destination,
                message_type=message_type,
                message_id=message_id,
                payload=payload,
                round_id=round_id,
            )
            try:
                envelope = await self._receive_pair_message(
                    peer=destination,
                    message_type=message_type,
                    round_id=round_id,
                    timeout_seconds=self._transport_limits.retry_timeout_seconds,
                )
            except PairCommitError as error:
                if (
                    not isinstance(error.__cause__, TimeoutError)
                    or attempt + 1 >= attempts
                ):
                    raise
                continue
            for _ in range(self._transport_limits.max_retries):
                await self._send_pair_message(
                    destination=destination,
                    message_type=message_type,
                    message_id=message_id,
                    payload=payload,
                    round_id=round_id,
                )
            return envelope
        raise PairCommitError("pair commit deadline exceeded")

    async def _receive_pair_message(
        self,
        *,
        peer: PublicKey,
        message_type: MessageType,
        round_id: RoundId,
        timeout_seconds: float | None = None,
    ) -> Envelope:
        try:
            envelope = await self._receiver.receive(
                MessageChannel.PAIR_COMMIT,
                timeout_seconds=(
                    timeout_seconds
                    if timeout_seconds is not None
                    else self._timeout_seconds
                ),
            )
        except TimeoutError as error:
            raise PairCommitError("pair commit deadline exceeded") from error
        if (
            envelope.sender_public_key != peer
            or envelope.message_type is not message_type
            or envelope.round_id != round_id
        ):
            raise PairCommitError("received unexpected pair commit message")
        return envelope
