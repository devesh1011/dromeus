"""Event-driven local training and pairwise commit orchestration."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np

from dromeus.algorithms.base import (
    AlgorithmSnapshot,
    MaterializedArtifact,
    UpdateBundle,
    ValidatedUpdate,
    checksum_tensors,
)
from dromeus.gossip.peer_scheduler import PeerScheduler
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
from dromeus.telemetry.metrics import MetricsPublisher, RoundTiming
from dromeus.transport.outbound_scheduler import OutboundScheduler, Priority
from dromeus.transport.receiver import MessageChannel, Receiver
from dromeus.transport.transfer import ArtifactReceipt, TransferError, TransferManager


class PairCommitError(RuntimeError):
    """A peer update or pair commit could not be completed safely."""


@dataclass(frozen=True, slots=True)
class RunFailure:
    """Terminal failure evidence for persistence and control-plane reporting."""

    round_id: RoundId
    error_type: str
    reason: str


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """One node's deterministic evaluation result for a committed round."""

    round_id: RoundId
    loss: float
    accuracy: float


class FailureBroadcaster(Protocol):
    async def broadcast_run_failed(self, failure: RunFailure) -> None: ...


class ConsensusPublisher(Protocol):
    def submit(
        self, *, round_id: RoundId, weights: Mapping[str, np.ndarray]
    ) -> bool: ...


EvaluationCallback = Callable[
    [EvaluationMetrics], None | Awaitable[None]
]


class GossipAlgorithm(Protocol):
    def pre_local(self, round_id: RoundId) -> None: ...

    def local_training(self) -> None: ...

    def post_local_bundle(self) -> UpdateBundle: ...

    def validate_peer(self, peer_bundle: UpdateBundle) -> ValidatedUpdate: ...

    def peer_apply(self, peer_update: ValidatedUpdate) -> AlgorithmSnapshot: ...

    def release_bundle(self, bundle: UpdateBundle) -> None: ...

    def checkpoint_tensors(self) -> dict[str, np.ndarray]: ...


class PairTransport(Protocol):
    """Transport seam for one peer's update and commit handshake."""

    async def exchange_update(
        self,
        *,
        peer: PublicKey,
        round_id: RoundId,
        bundle: UpdateBundle,
    ) -> UpdateBundle: ...

    async def exchange_update_ready(
        self,
        *,
        peer: PublicKey,
        round_id: RoundId,
        bundle_checksum: str,
    ) -> str: ...

    async def exchange_round_committed(
        self,
        *,
        peer: PublicKey,
        round_id: RoundId,
        state_checksum: str,
    ) -> None: ...


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
        self.last_transfer_id: str | None = None
        self.last_retry_count = 0

    async def broadcast_run_failed(self, failure: RunFailure) -> None:
        await self._failure_broadcaster.broadcast_run_failed(failure)

    async def broadcast_consensus_sketch(
        self, *, round_id: RoundId, sketch: np.ndarray
    ) -> None:
        """Best-effort low-priority broadcast of one FP32 consensus sketch."""
        peers = self._participant_keys - {self._local_public_key}
        if not peers:
            return
        payload = encode_sketch(sketch)

        async def send(peer: PublicKey) -> None:
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
            await self._sender.send(
                peer,
                encode_envelope(envelope),
                priority=Priority.TELEMETRY,
                retries=self._transport_limits.max_retries,
                retry_delay_seconds=self._transport_limits.retry_timeout_seconds,
            )

        await asyncio.gather(*(send(peer) for peer in peers), return_exceptions=True)

    async def exchange_update(
        self,
        *,
        peer: PublicKey,
        round_id: RoundId,
        bundle: UpdateBundle,
    ) -> UpdateBundle:
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
            self.last_transfer_id = await self._transfer_manager.send_artifact(
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
                self.last_transfer_id = (
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
            self.last_retry_count = retry_count
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
            return peer_bundle
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
        return self._transport_limits.retry_timeout_seconds * (
            self._transport_limits.max_retries + 4
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
    ) -> Envelope:
        attempts = self._transport_limits.max_retries + 1
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


def _algorithm_metric(algorithm: GossipAlgorithm, name: str) -> float | None:
    value = getattr(algorithm, name, None)
    if isinstance(value, (int, float)) and np.isfinite(value) and value >= 0:
        return float(value)
    return None


def _transport_retry_count(transport: PairTransport) -> int:
    value = getattr(transport, "last_retry_count", 0)
    return value if isinstance(value, int) and value >= 0 else 0


def _transport_transfer_id(transport: PairTransport) -> str | None:
    value = getattr(transport, "last_transfer_id", None)
    return value if isinstance(value, str) and value else None


@dataclass(frozen=True, slots=True)
class RoundCommit:
    """Evidence passed to the atomic persistence seam after peer validation."""

    round_id: RoundId
    peer_public_key: PublicKey
    local_bundle_digest: str
    peer_bundle_digest: str
    state_checksum: str
    phase: Literal["training", "final-consensus"] = "training"


CommitCallback = Callable[[RoundCommit], None | Awaitable[None]]


async def _run_blocking[T](callback: Callable[..., T], *args: object) -> T:
    task = asyncio.create_task(asyncio.to_thread(callback, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


class GossipEngine:
    """Run fixed-round local training without a group-wide barrier."""

    def __init__(
        self,
        *,
        local_public_key: PublicKey,
        round_count: int,
        scheduler: PeerScheduler,
        algorithm: GossipAlgorithm,
        transport: PairTransport,
        commit_callback: CommitCallback,
        confirm_callback: CommitCallback | None = None,
        timeout_seconds: float | None = None,
        transport_limits: TransportLimits | None = None,
        failure_callback: Callable[[RunFailure], None | Awaitable[None]] | None = None,
        failure_broadcaster: FailureBroadcaster | None = None,
        consensus_publisher: ConsensusPublisher | None = None,
        evaluation_interval: int = 5,
        evaluation_callback: EvaluationCallback | None = None,
        metrics_publisher: MetricsPublisher | None = None,
    ) -> None:
        if round_count <= 0:
            raise ValueError("round_count must be positive")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if timeout_seconds is not None and transport_limits is not None:
            raise ValueError("pass timeout_seconds or transport_limits, not both")
        if evaluation_interval <= 0:
            raise ValueError("evaluation_interval must be positive")
        evaluator = getattr(algorithm, "evaluate", None)
        if evaluation_callback is not None and not callable(evaluator):
            raise ValueError("evaluation callback requires an evaluatable algorithm")
        self._local_public_key = local_public_key
        self._round_count = round_count
        self._scheduler = scheduler
        self._algorithm = algorithm
        self._transport = transport
        self._commit_callback = commit_callback
        self._confirm_callback = confirm_callback
        self._timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else (
                transport_limits.retry_timeout_seconds
                * (transport_limits.max_retries + 4)
                if transport_limits is not None
                else None
            )
        )
        self._failure_callback = failure_callback
        if failure_broadcaster is None:
            candidate = getattr(transport, "broadcast_run_failed", None)
            if candidate is not None:
                failure_broadcaster = cast(FailureBroadcaster, transport)
        self._failure_broadcaster = failure_broadcaster
        self._consensus_publisher = consensus_publisher
        self._evaluation_interval = evaluation_interval
        self._evaluator = cast(Callable[[], tuple[float, float]] | None, evaluator)
        self._evaluation_callback = evaluation_callback
        self._metrics_publisher = metrics_publisher
        self._current_round = 0
        self._commits: list[RoundCommit] = []
        self._failure: RunFailure | None = None

    @property
    def current_round(self) -> RoundId:
        return self._current_round

    @property
    def commits(self) -> tuple[RoundCommit, ...]:
        return tuple(self._commits)

    @property
    def failure(self) -> RunFailure | None:
        return self._failure

    async def run(self) -> tuple[RoundCommit, ...]:
        """Train and commit every manifest round in order."""
        while self._current_round < self._round_count:
            await self.run_round(self._current_round)
        return self.commits

    async def run_round(self, round_id: RoundId) -> RoundCommit:
        """Complete one scheduled pair exchange and commit."""
        if self._failure is not None:
            raise PairCommitError("run has already failed")
        if round_id != self._current_round:
            raise PairCommitError(
                f"round {round_id} is not current; expected {self._current_round}"
            )
        try:
            return await self._run_round(round_id)
        except asyncio.CancelledError:
            raise
        except TimeoutError as error:
            failure = PairCommitError("pair round deadline exceeded")
            await self._record_failure(round_id, failure)
            raise failure from error
        except Exception as error:
            await self._record_failure(round_id, error)
            raise

    async def _run_round(self, round_id: RoundId) -> RoundCommit:
        pairing = self._scheduler.schedule(round_id)
        try:
            peer = pairing.peer_for(self._local_public_key)
        except KeyError as error:
            raise PairCommitError(str(error)) from error

        local_bundle: UpdateBundle | None = None
        peer_bundle: UpdateBundle | None = None
        try:
            local_started = time.perf_counter()
            await _run_blocking(self._algorithm.pre_local, round_id)
            await _run_blocking(self._algorithm.local_training)
            bundle_task = asyncio.create_task(
                asyncio.to_thread(self._algorithm.post_local_bundle)
            )
            try:
                materialized = await asyncio.shield(bundle_task)
            except asyncio.CancelledError:
                local_bundle = await bundle_task
                raise
            local_bundle = materialized
            if materialized.metadata.round_id != round_id:
                raise PairCommitError("local update round does not match current round")
            local_compute_seconds = time.perf_counter() - local_started

            transfer_started = time.perf_counter()
            peer_bundle = await self._with_pair_timeout(
                self._transport.exchange_update(
                    peer=peer,
                    round_id=round_id,
                    bundle=materialized,
                )
            )
            transfer_seconds = time.perf_counter() - transfer_started
            if peer_bundle.metadata.round_id != round_id:
                raise PairCommitError("peer update round does not match current round")
            try:
                peer_update = await _run_blocking(
                    self._algorithm.validate_peer, peer_bundle
                )
            except (ValueError, TypeError) as error:
                raise PairCommitError("peer update validation failed") from error

            peer_wait_started = time.perf_counter()
            remote_digest = await self._with_pair_timeout(
                self._transport.exchange_update_ready(
                    peer=peer,
                    round_id=round_id,
                    bundle_checksum=materialized.digest,
                )
            )
            if remote_digest != peer_bundle.digest:
                raise PairCommitError("peer UPDATE_READY bundle digest mismatch")
            peer_wait_seconds = time.perf_counter() - peer_wait_started

            mixing_started = time.perf_counter()
            try:
                post_mix = await _run_blocking(
                    self._algorithm.peer_apply,
                    peer_update,
                )
            except (ValueError, TypeError) as error:
                raise PairCommitError("peer update application failed") from error
            mixing_seconds = time.perf_counter() - mixing_started
            state_checksum = await _run_blocking(checksum_tensors, post_mix.weights)
            commit = RoundCommit(
                round_id=round_id,
                peer_public_key=peer,
                local_bundle_digest=materialized.digest,
                peer_bundle_digest=peer_bundle.digest,
                state_checksum=state_checksum,
                phase=pairing.phase,
            )
            result = await _run_blocking(self._commit_callback, commit)
            if inspect.isawaitable(result):
                await result
            commit_wait_started = time.perf_counter()
            await self._with_pair_timeout(
                self._transport.exchange_round_committed(
                    peer=peer,
                    round_id=round_id,
                    state_checksum=state_checksum,
                )
            )
            if self._confirm_callback is not None:
                result = await _run_blocking(self._confirm_callback, commit)
                if inspect.isawaitable(result):
                    await result
            peer_wait_seconds += time.perf_counter() - commit_wait_started

            evaluation_started = time.perf_counter()
            evaluation = await self._evaluate_if_due(round_id)
            evaluation_seconds = time.perf_counter() - evaluation_started
            self._commits.append(commit)
            self._current_round += 1
            if self._consensus_publisher is not None:
                try:
                    self._consensus_publisher.submit(
                        round_id=round_id,
                        weights=post_mix.weights,
                    )
                except Exception:
                    pass
            if self._metrics_publisher is not None:
                timing = RoundTiming(
                    round_id=round_id,
                    peer_id=peer,
                    local_compute_seconds=local_compute_seconds,
                    peer_wait_seconds=peer_wait_seconds,
                    transfer_seconds=transfer_seconds,
                    mixing_seconds=mixing_seconds,
                    evaluation_seconds=evaluation_seconds,
                    retries=_transport_retry_count(self._transport),
                    local_loss=_algorithm_metric(self._algorithm, "local_loss"),
                    evaluation_loss=(
                        evaluation.loss if evaluation is not None else None
                    ),
                    evaluation_accuracy=(
                        evaluation.accuracy if evaluation is not None else None
                    ),
                    transfer_id=_transport_transfer_id(self._transport),
                )
                try:
                    self._metrics_publisher.submit(timing)
                except Exception:
                    pass
            return commit
        finally:
            try:
                if peer_bundle is not None:
                    await _run_blocking(
                        self._algorithm.release_bundle, peer_bundle
                    )
            finally:
                if local_bundle is not None:
                    await _run_blocking(
                        self._algorithm.release_bundle, local_bundle
                    )

    async def _with_pair_timeout[T](self, operation: Awaitable[T]) -> T:
        if self._timeout_seconds is None:
            return await operation
        return await asyncio.wait_for(operation, timeout=self._timeout_seconds)

    async def _evaluate_if_due(self, round_id: RoundId) -> EvaluationMetrics | None:
        if self._evaluator is None:
            return None
        completed_round = round_id + 1
        if (
            completed_round % self._evaluation_interval != 0
            and completed_round != self._round_count
        ):
            return None
        loss, accuracy = await _run_blocking(self._evaluator)
        metrics = EvaluationMetrics(
            round_id=round_id,
            loss=loss,
            accuracy=accuracy,
        )
        if self._evaluation_callback is not None:
            try:
                result = self._evaluation_callback(metrics)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                pass
        return metrics

    async def _record_failure(self, round_id: RoundId, error: Exception) -> None:
        if self._failure is not None:
            return
        failure = RunFailure(
            round_id=round_id,
            error_type=type(error).__name__,
            reason=str(error)[:1024] or "pair round failed",
        )
        self._failure = failure
        if self._failure_callback is not None:
            try:
                result = self._failure_callback(failure)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                pass
        if self._metrics_publisher is not None:
            try:
                self._metrics_publisher.submit_failure(
                    round_id=failure.round_id,
                    error_type=failure.error_type,
                    reason=failure.reason,
                )
            except Exception:
                pass
        if self._failure_broadcaster is not None:
            try:
                await self._failure_broadcaster.broadcast_run_failed(failure)
            except Exception:
                pass


__all__ = [
    "AXLPairTransport",
    "AXLFailureBroadcaster",
    "GossipAlgorithm",
    "GossipEngine",
    "FailureBroadcaster",
    "ConsensusPublisher",
    "decode_run_failure",
    "EvaluationCallback",
    "EvaluationMetrics",
    "PairCommitError",
    "PairTransport",
    "RunFailedMessage",
    "RunFailure",
    "RoundCommit",
]
