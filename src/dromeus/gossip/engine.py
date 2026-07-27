"""Event-driven local training and pairwise commit orchestration."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import msgpack  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from safetensors.numpy import (
    load_file as _load_file,  # pyright: ignore[reportUnknownVariableType]
)
from safetensors.numpy import (
    save_file as _save_file,  # pyright: ignore[reportUnknownVariableType]
)

from dromeus.algorithms.base import (
    AlgorithmSnapshot,
    TrainedWeightsBundle,
    checksum_tensors,
)
from dromeus.gossip.scheduler import PeerScheduler
from dromeus.manifests.models import (
    AlgorithmId,
    DomainModel,
    MessageId,
    PublicKey,
    RoundId,
    RunId,
    Sha256,
    TensorSchema,
    TransportLimits,
)
from dromeus.telemetry.consensus import encode_sketch
from dromeus.telemetry.metrics import MetricsPublisher, RoundTiming
from dromeus.transport.envelope import (
    Envelope,
    MessageType,
    create_envelope,
    encode_envelope,
)
from dromeus.transport.receiver import MessageChannel, Receiver
from dromeus.transport.sender import OutboundScheduler, Priority
from dromeus.transport.transfer import TransferError, TransferManager

_load_safetensors = Callable[[str], dict[str, np.ndarray]]
_save_safetensors = Callable[[dict[str, np.ndarray], str], None]
load_file = cast(_load_safetensors, _load_file)
save_file = cast(_save_safetensors, _save_file)


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
    def pre_local(self, round_id: RoundId) -> AlgorithmSnapshot: ...

    def local_training(self) -> AlgorithmSnapshot: ...

    def post_local_bundle(self) -> TrainedWeightsBundle: ...

    def validate_peer(self, peer_bundle: TrainedWeightsBundle) -> None: ...

    def peer_apply(self, peer_bundle: TrainedWeightsBundle) -> AlgorithmSnapshot: ...


class PairTransport(Protocol):
    """Transport seam for one peer's update and commit handshake."""

    async def exchange_update(
        self,
        *,
        peer: PublicKey,
        round_id: RoundId,
        bundle: TrainedWeightsBundle,
    ) -> TrainedWeightsBundle: ...

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


class PairCommitMessage(DomainModel):
    """Validated metadata carried by pair-commit envelopes."""

    round_id: RoundId
    checksum: Sha256


class RunFailedMessage(DomainModel):
    """Validated control payload for terminal run failure."""

    round_id: RoundId
    error_type: str
    reason: str


def decode_run_failure(payload: bytes) -> RunFailure:
    """Decode validated terminal failure evidence from a control envelope."""
    message = RunFailedMessage.model_validate(_unpack(payload))
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
        payload = _pack(
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
    """Pair transport backed by reliable safetensors transfer and AXL envelopes."""

    def __init__(
        self,
        *,
        local_public_key: PublicKey,
        run_id: RunId,
        manifest_hash: Sha256,
        algorithm_id: AlgorithmId,
        tensor_schema: TensorSchema,
        transport_limits: TransportLimits,
        receiver: Receiver,
        sender: OutboundScheduler,
        transfer_manager: TransferManager,
        artifact_root: Path,
        participant_keys: frozenset[PublicKey] | None = None,
    ) -> None:
        self._local_public_key = local_public_key
        self._run_id = run_id
        self._manifest_hash = manifest_hash
        self._algorithm_id = algorithm_id
        self._tensor_schema = tensor_schema
        self._transport_limits = transport_limits
        self._receiver = receiver
        self._sender = sender
        self._transfer_manager = transfer_manager
        self._artifact_root = artifact_root
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
        self._artifact_root.mkdir(parents=True, exist_ok=True)
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
        bundle: TrainedWeightsBundle,
    ) -> TrainedWeightsBundle:
        if bundle.round_id != round_id:
            raise PairCommitError("local update round does not match current round")
        artifact_path = (
            self._artifact_root / f"round-{round_id}-trained-weights.safetensors"
        )
        await asyncio.to_thread(
            save_file,
            {
                name: np.ascontiguousarray(value)
                for name, value in bundle.tensors.items()
            },
            str(artifact_path),
        )
        try:
            self.last_transfer_id = await self._transfer_manager.send_artifact(
                destination=peer,
                artifact_name=f"round-{round_id}-trained-weights",
                artifact_path=artifact_path,
                codec_id="safetensors-v1",
                tensor_schema=self._tensor_schema,
                round_id=round_id,
            )
            timing = self._transfer_manager.last_timing
            self.last_retry_count = timing.retry_count if timing is not None else 0
            receipt = await self._transfer_manager.next_artifact(
                timeout_seconds=self._timeout_seconds
            )
            if (
                receipt.sender_public_key != peer
                or receipt.round_id != round_id
                or receipt.artifact_name != f"round-{round_id}-trained-weights"
            ):
                raise PairCommitError("received unexpected peer update artifact")
            values = await asyncio.to_thread(load_file, str(receipt.path))
            tensors = {
                name: np.ascontiguousarray(value) for name, value in values.items()
            }
            checksum = checksum_tensors(tensors)
            return TrainedWeightsBundle(
                round_id=round_id,
                tensors=tensors,
                checksum=checksum,
            )
        except (
            OSError,
            ValueError,
            TypeError,
            TransferError,
            msgpack.UnpackException,
        ) as error:
            raise PairCommitError("peer update transfer failed") from error
        finally:
            artifact_path.unlink(missing_ok=True)

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
        await self._send_pair_message(
            destination=peer,
            message_type=MessageType.UPDATE_READY,
            message_id=f"update-ready-{round_id}",
            payload=PairCommitMessage(round_id=round_id, checksum=bundle_checksum),
            round_id=round_id,
        )
        envelope = await self._receive_pair_message(
            peer=peer,
            message_type=MessageType.UPDATE_READY,
            round_id=round_id,
        )
        message = PairCommitMessage.model_validate(_unpack(envelope.payload))
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
        await self._send_pair_message(
            destination=peer,
            message_type=MessageType.ROUND_COMMITTED,
            message_id=f"round-committed-{round_id}",
            payload=PairCommitMessage(round_id=round_id, checksum=state_checksum),
            round_id=round_id,
        )
        envelope = await self._receive_pair_message(
            peer=peer,
            message_type=MessageType.ROUND_COMMITTED,
            round_id=round_id,
        )
        PairCommitMessage.model_validate(_unpack(envelope.payload))
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
            payload=_pack(payload),
        )
        await self._sender.send(
            destination,
            encode_envelope(envelope),
            priority=Priority.CONTROL,
            retries=self._transport_limits.max_retries,
            retry_delay_seconds=self._transport_limits.retry_timeout_seconds,
        )

    async def _receive_pair_message(
        self,
        *,
        peer: PublicKey,
        message_type: MessageType,
        round_id: RoundId,
    ) -> Envelope:
        try:
            envelope = await self._receiver.receive(
                MessageChannel.PAIR_COMMIT,
                timeout_seconds=self._timeout_seconds,
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


def _pack(model: DomainModel) -> bytes:
    return cast(
        bytes,
        msgpack.packb(  # pyright: ignore[reportUnknownMemberType]
            model.model_dump(mode="python")
        ),
    )


def _unpack(data: bytes) -> object:
    return cast(
        object,
        msgpack.unpackb(  # pyright: ignore[reportUnknownMemberType]
            data, raw=False, strict_map_key=True
        ),
    )


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
    pre_local: AlgorithmSnapshot
    local_bundle: TrainedWeightsBundle
    peer_bundle: TrainedWeightsBundle
    post_mix: AlgorithmSnapshot
    state_checksum: str


CommitCallback = Callable[[RoundCommit], None | Awaitable[None]]


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
            if self._timeout_seconds is None:
                return await self._run_round(round_id)
            return await asyncio.wait_for(
                self._run_round(round_id), timeout=self._timeout_seconds
            )
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

        local_started = time.perf_counter()
        pre_local = await asyncio.to_thread(self._algorithm.pre_local, round_id)
        await asyncio.to_thread(self._algorithm.local_training)
        local_bundle = await asyncio.to_thread(self._algorithm.post_local_bundle)
        local_checksum = await asyncio.to_thread(checksum_tensors, local_bundle.tensors)
        if local_checksum != local_bundle.checksum:
            raise PairCommitError("local update checksum mismatch")
        local_compute_seconds = time.perf_counter() - local_started

        transfer_started = time.perf_counter()
        peer_bundle = await self._transport.exchange_update(
            peer=peer,
            round_id=round_id,
            bundle=local_bundle,
        )
        transfer_seconds = time.perf_counter() - transfer_started
        if peer_bundle.round_id != round_id:
            raise PairCommitError("peer update round does not match current round")
        peer_checksum = await asyncio.to_thread(checksum_tensors, peer_bundle.tensors)
        if peer_checksum != peer_bundle.checksum:
            raise PairCommitError("peer update checksum mismatch")
        try:
            await asyncio.to_thread(self._algorithm.validate_peer, peer_bundle)
        except (ValueError, TypeError) as error:
            raise PairCommitError("peer update validation failed") from error

        peer_wait_started = time.perf_counter()
        remote_checksum = await self._transport.exchange_update_ready(
            peer=peer,
            round_id=round_id,
            bundle_checksum=local_bundle.checksum,
        )
        if remote_checksum != peer_bundle.checksum:
            raise PairCommitError("peer UPDATE_READY checksum mismatch")
        peer_wait_seconds = time.perf_counter() - peer_wait_started

        mixing_started = time.perf_counter()
        try:
            post_mix = await asyncio.to_thread(
                self._algorithm.peer_apply,
                peer_bundle,
            )
        except (ValueError, TypeError) as error:
            raise PairCommitError("peer update application failed") from error
        mixing_seconds = time.perf_counter() - mixing_started
        state_checksum = await asyncio.to_thread(checksum_tensors, post_mix.weights)
        commit = RoundCommit(
            round_id=round_id,
            peer_public_key=peer,
            pre_local=pre_local,
            local_bundle=local_bundle,
            peer_bundle=peer_bundle,
            post_mix=post_mix,
            state_checksum=state_checksum,
        )
        result = await asyncio.to_thread(self._commit_callback, commit)
        if inspect.isawaitable(result):
            await result
        commit_wait_started = time.perf_counter()
        await self._transport.exchange_round_committed(
            peer=peer,
            round_id=round_id,
            state_checksum=state_checksum,
        )
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
                evaluation_loss=evaluation.loss if evaluation is not None else None,
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

    async def _evaluate_if_due(self, round_id: RoundId) -> EvaluationMetrics | None:
        if self._evaluator is None:
            return None
        completed_round = round_id + 1
        if (
            completed_round % self._evaluation_interval != 0
            and completed_round != self._round_count
        ):
            return None
        loss, accuracy = await asyncio.to_thread(self._evaluator)
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
