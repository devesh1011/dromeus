"""Event-driven local training and pairwise commit orchestration."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable

from dromeus.algorithms.base import (
    AlgorithmObservations,
    UpdateBundle,
    checksum_tensors,
)
from dromeus.gossip._async import run_blocking as _run_blocking
from dromeus.gossip.interfaces import (
    CommitCallback,
    ConsensusPublisher,
    EvaluationCallback,
    EvaluationMetrics,
    FailureBroadcaster,
    GossipAlgorithm,
    PairCommitError,
    PairExchangeResult,
    PairTransport,
    RoundCommit,
    RunFailure,
)
from dromeus.gossip.peer_scheduler import PeerScheduler
from dromeus.manifests.models import PublicKey, RoundId, TransportLimits
from dromeus.telemetry.metrics import MetricsPublisher, RoundTiming


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
                max(
                    transport_limits.transfer_lifetime_limit,
                    transport_limits.retry_timeout_seconds
                    * (transport_limits.max_retries + 4),
                )
                + transport_limits.retry_timeout_seconds
                if transport_limits is not None
                else None
            )
        )
        self._failure_callback = failure_callback
        self._failure_broadcaster = failure_broadcaster
        self._consensus_publisher = consensus_publisher
        self._evaluation_interval = evaluation_interval
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

            peer_wait_started = time.perf_counter()
            remote_digest = await self._with_pair_timeout(
                self._transport.exchange_update_ready(
                    peer=peer,
                    round_id=round_id,
                    bundle_checksum=materialized.digest,
                )
            )
            peer_wait_seconds = time.perf_counter() - peer_wait_started

            transfer_started = time.perf_counter()
            exchange = await self._with_pair_timeout(
                self._transport.exchange_update(
                    peer=peer,
                    round_id=round_id,
                    bundle=materialized,
                )
            )
            peer_bundle = exchange.bundle
            transfer_seconds = time.perf_counter() - transfer_started
            if peer_bundle.metadata.round_id != round_id:
                raise PairCommitError("peer update round does not match current round")
            try:
                peer_update = await _run_blocking(
                    self._algorithm.validate_peer, peer_bundle
                )
            except (ValueError, TypeError) as error:
                raise PairCommitError("peer update validation failed") from error
            if remote_digest != peer_bundle.digest:
                raise PairCommitError("peer UPDATE_READY bundle digest mismatch")

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
            observations = await self._capture_algorithm_observations()
            commit = RoundCommit(
                round_id=round_id,
                peer_public_key=peer,
                local_bundle_digest=materialized.digest,
                peer_bundle_digest=peer_bundle.digest,
                state_checksum=state_checksum,
                phase=pairing.phase,
                local_loss=observations.local_loss,
                error_feedback_residual_l2_norm=(
                    observations.error_feedback_residual_l2_norm
                ),
                error_feedback_signal_l2_norm=observations.error_feedback_signal_l2_norm,
                error_feedback_residual_to_signal_ratio=(
                    observations.error_feedback_residual_to_signal_ratio
                ),
                transfer_id=exchange.transfer_id,
                transfer_retries=exchange.retry_count,
                encoded_artifact_bytes=sum(
                    artifact.size_bytes for artifact in materialized.metadata.artifacts
                ),
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
            evaluation_started = time.perf_counter()
            evaluation = await self._evaluate_if_due(round_id)
            evaluation_seconds = time.perf_counter() - evaluation_started
            if self._metrics_publisher is not None:
                timing = RoundTiming(
                    round_id=round_id,
                    peer_id=peer,
                    local_compute_seconds=local_compute_seconds,
                    peer_wait_seconds=peer_wait_seconds,
                    transfer_seconds=transfer_seconds,
                    mixing_seconds=mixing_seconds,
                    evaluation_seconds=evaluation_seconds,
                    retries=commit.transfer_retries,
                    local_loss=commit.local_loss,
                    error_feedback_residual_l2_norm=(
                        commit.error_feedback_residual_l2_norm
                    ),
                    error_feedback_signal_l2_norm=commit.error_feedback_signal_l2_norm,
                    error_feedback_residual_to_signal_ratio=(
                        commit.error_feedback_residual_to_signal_ratio
                    ),
                    evaluation_loss=(
                        evaluation.loss if evaluation is not None else None
                    ),
                    evaluation_metrics=evaluation.metrics
                    if evaluation is not None
                    else None,
                    evaluation_accuracy=(
                        evaluation.accuracy if evaluation is not None else None
                    ),
                    transfer_id=commit.transfer_id,
                    encoded_artifact_bytes=commit.encoded_artifact_bytes,
                )
                try:
                    self._metrics_publisher.submit(timing)
                except Exception:
                    pass
            return commit
        finally:
            try:
                if peer_bundle is not None:
                    await _run_blocking(self._algorithm.release_bundle, peer_bundle)
            finally:
                if local_bundle is not None:
                    await _run_blocking(self._algorithm.release_bundle, local_bundle)

    async def _capture_algorithm_observations(self) -> AlgorithmObservations:
        try:
            return await _run_blocking(self._algorithm.observations)
        except Exception:
            return AlgorithmObservations()

    async def _with_pair_timeout[T](self, operation: Awaitable[T]) -> T:
        if self._timeout_seconds is None:
            return await operation
        return await asyncio.wait_for(operation, timeout=self._timeout_seconds)

    async def _evaluate_if_due(self, round_id: RoundId) -> EvaluationMetrics | None:
        completed_round = round_id + 1
        if (
            completed_round % self._evaluation_interval != 0
            and completed_round != self._round_count
        ):
            return None
        try:
            evaluation = await _run_blocking(self._algorithm.evaluate)
        except Exception:
            return None
        if evaluation is None:
            return None
        metrics = EvaluationMetrics(
            round_id=round_id,
            loss=evaluation.loss,
            accuracy=evaluation.accuracy,
            metrics=evaluation.metrics,
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
    "CommitCallback",
    "ConsensusPublisher",
    "EvaluationCallback",
    "EvaluationMetrics",
    "FailureBroadcaster",
    "GossipAlgorithm",
    "GossipEngine",
    "PairCommitError",
    "PairExchangeResult",
    "PairTransport",
    "RoundCommit",
    "RunFailure",
]
