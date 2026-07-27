"""Node runtime lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import numpy as np

from dromeus.algorithms.dpsgd import DPSGDAdapter
from dromeus.gossip.engine import (
    AXLFailureBroadcaster,
    AXLPairTransport,
    GossipAlgorithm,
    GossipEngine,
    RoundCommit,
    RunFailure,
    decode_run_failure,
)
from dromeus.gossip.scheduler import PeerScheduler
from dromeus.manifests.models import (
    ConsensusSketchMessage,
    DatasetContract,
    DraftRunSpec,
    EnvironmentFingerprint,
    Invitation,
    TensorSchema,
)
from dromeus.membership.protocol import FormationProtocol, FormationResult
from dromeus.persistence.run_store import RunStore
from dromeus.telemetry.consensus import (
    ConsensusDistance,
    LiveConsensusTelemetry,
)
from dromeus.telemetry.events import EventSink, emit_event
from dromeus.telemetry.metrics import MetricsPublisher
from dromeus.training.pytorch import (
    InitialCheckpoint,
)
from dromeus.training.pytorch import (
    PreparedCIFARTraining as TrainingOwnedCIFAR,
)
from dromeus.training.pytorch import (
    prepare_cifar_training as prepare_training_owned_cifar,
)
from dromeus.transport.base import AsyncTransport
from dromeus.transport.envelope import MessageType
from dromeus.transport.receiver import MessageChannel
from dromeus.transport.transfer import ArtifactStore


class NodeRuntimeError(RuntimeError):
    """The node runtime lifecycle was used out of order."""


class PeerRunFailureError(RuntimeError):
    """A sealed peer announced terminal run failure."""


class NodeState(StrEnum):
    CREATED = "created"
    FORMING = "forming"
    READY = "ready"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    STOPPED = "stopped"


class MetricsService(MetricsPublisher, Protocol):
    """Metrics publisher with an optional runtime-owned task lifecycle."""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Dependencies needed to run the formed node's local training."""

    algorithm: GossipAlgorithm
    load_checkpoint: Callable[[Path], None]
    run_store: RunStore
    artifact_root: Path
    metrics_publisher: MetricsService | None = None


@dataclass(frozen=True, slots=True)
class FailureConfig:
    """Durable dependencies available immediately after formation."""

    run_store: RunStore
    artifact_root: Path

    @classmethod
    def for_run_root(cls, run_root: Path) -> FailureConfig:
        return cls(
            run_store=RunStore(run_root / "run-store"),
            artifact_root=run_root / "rounds",
        )


@dataclass(frozen=True, slots=True)
class PreparedCIFARTraining:
    """Runtime composition over training-owned CIFAR data."""

    _training: TrainingOwnedCIFAR

    def create_initial_checkpoint(self, path: Path) -> InitialCheckpoint:
        return self._training.create_initial_checkpoint(path)

    def build_config(
        self,
        *,
        result: FormationResult,
        local_public_key: str,
        run_root: Path,
        metrics_publisher: MetricsService,
    ) -> TrainingConfig:
        trainer = self._training.create_trainer(
            manifest=result.manifest,
            local_public_key=local_public_key,
        )
        return TrainingConfig(
            algorithm=DPSGDAdapter(
                trainer=trainer,
                tensor_schema=result.manifest.tensor_schema,
                local_steps=result.manifest.local_steps,
                learning_rate=result.manifest.learning_rate,
            ),
            load_checkpoint=trainer.load_checkpoint,
            run_store=RunStore(run_root / "run-store"),
            artifact_root=run_root / "rounds",
            metrics_publisher=metrics_publisher,
        )


def prepare_cifar_training(
    *,
    draft: DraftRunSpec,
    cifar_root: Path,
    benchmark_seed: int,
) -> PreparedCIFARTraining:
    """Prepare local data through training-owned interfaces."""
    return PreparedCIFARTraining(
        _training=prepare_training_owned_cifar(
            draft=draft,
            cifar_root=cifar_root,
            benchmark_seed=benchmark_seed,
        )
    )


class NodeRuntime:
    """Own formation and post-formation training lifecycle."""

    def __init__(
        self,
        *,
        transport: AsyncTransport,
        draft: DraftRunSpec,
        environment: EnvironmentFingerprint,
        dataset: DatasetContract,
        artifact_store: ArtifactStore,
        event_sink: EventSink | None = None,
        training: TrainingConfig | None = None,
        failure: FailureConfig | None = None,
    ) -> None:
        self._transport = transport
        self._formation = FormationProtocol(
            transport=transport,
            draft=draft,
            environment=environment,
            dataset=dataset,
            transport_limits=draft.transport,
            artifact_store=artifact_store,
            event_sink=event_sink,
        )
        self._state = NodeState.CREATED
        self._result: FormationResult | None = None
        self._training = training
        self._failure = failure
        self._engine: GossipEngine | None = None
        self._pair_transport: AXLPairTransport | None = None
        self._consensus_telemetry: LiveConsensusTelemetry | None = None
        self._event_sink = event_sink
        self._local_public_key: str | None = None
        self._commits: tuple[RoundCommit, ...] = ()
        self._run_task: asyncio.Task[tuple[RoundCommit, ...]] | None = None
        self._control_task: asyncio.Task[None] | None = None
        self._remote_failure: PeerRunFailureError | None = None
        self._terminal_lock = asyncio.Lock()

    @property
    def state(self) -> NodeState:
        return self._state

    @property
    def formation_result(self) -> FormationResult:
        if self._result is None:
            raise NodeRuntimeError("node has not completed formation")
        return self._result

    @property
    def training_commits(self) -> tuple[RoundCommit, ...]:
        return self._commits

    def configure_training(self, training: TrainingConfig) -> None:
        """Attach the local trainer once fixed membership has formed."""
        if self._state is not NodeState.READY:
            raise NodeRuntimeError(f"cannot configure training from {self._state}")
        if self._training is not None:
            raise NodeRuntimeError("training is already configured")
        self._training = training

    async def fail_before_run(self, error: BaseException) -> None:
        """Persist and announce a formed node failure before training starts."""
        assert self._result is not None
        run_store, _ = self._failure_dependencies()
        persistence_error: Exception | None = None
        try:
            async with self._terminal_lock:
                if self._state is not NodeState.READY:
                    raise NodeRuntimeError(
                        f"cannot fail node before run from {self._state}"
                    )
                self._state = NodeState.FAILED
                try:
                    await asyncio.to_thread(
                        run_store.initialize, self._result.manifest
                    )
                    await asyncio.to_thread(
                        run_store.record_terminal,
                        "failed",
                        {
                            "error_type": type(error).__name__,
                            "error": str(error)[:1024],
                        },
                    )
                except Exception as failure:
                    persistence_error = failure
            try:
                local_key = await self._transport.local_public_key()
                self._local_public_key = local_key
                failure_broadcaster = self._create_failure_broadcaster(local_key)
                await failure_broadcaster.broadcast_run_failed(
                    RunFailure(
                        round_id=0,
                        error_type=type(error).__name__,
                        reason=str(error)[:1024] or "node failed before training",
                    )
                )
            except Exception:
                pass
            try:
                await asyncio.to_thread(
                    emit_event,
                    "run_failed",
                    run_id=self._result.manifest.run_id,
                    manifest_hash=self._result.manifest_hash,
                    node_id=self._local_public_key,
                    message_id="run-failed-0",
                    round_id=0,
                    error_type=type(error).__name__,
                    reason=str(error)[:1024] or "node failed before training",
                    sink=self._event_sink,
                )
            except Exception:
                pass
        finally:
            await self._cleanup()
        if persistence_error is not None:
            raise persistence_error

    async def initiate(
        self,
        *,
        bootstrap_uri: str,
        checkpoint_path: Path,
        tensor_schema: TensorSchema,
    ) -> FormationResult:
        """Form as initiator using a checkpoint prepared by the local trainer."""
        await self._start_formation()
        try:
            result = await self._formation.initiate(
                bootstrap_uri=bootstrap_uri,
                checkpoint_path=checkpoint_path,
                tensor_schema=tensor_schema,
            )
        except BaseException:
            await self._fail()
            raise
        return self._ready(result)

    async def join(self, *, invitation: Invitation) -> FormationResult:
        await self._start_formation()
        try:
            result = await self._formation.join(invitation=invitation)
        except BaseException:
            await self._fail()
            raise
        return self._ready(result)

    async def run(self) -> tuple[RoundCommit, ...]:
        """Load formed checkpoint, run gossip rounds, and persist terminal state."""
        async with self._terminal_lock:
            if self._state is not NodeState.READY:
                raise NodeRuntimeError(f"cannot run node from {self._state}")
            if self._training is None:
                raise NodeRuntimeError("training is not configured")
            assert self._result is not None
            self._state = NodeState.RUNNING
        run_task = asyncio.current_task()
        assert run_task is not None
        self._run_task = run_task
        initialized = False
        try:
            await asyncio.to_thread(
                self._training.run_store.initialize, self._result.manifest
            )
            initialized = True
            await asyncio.to_thread(
                self._training.load_checkpoint, self._result.checkpoint_path
            )
            local_key = await self._transport.local_public_key()
            self._local_public_key = local_key
            participants = frozenset(
                participant.public_key
                for participant in self._result.manifest.participants
            )
            services = self._formation.services
            pair_transport = await self._create_pair_transport(
                local_key, artifact_root=self._training.artifact_root
            )
            self._pair_transport = pair_transport

            async def receive_consensus_sketch(
                timeout_seconds: float,
            ) -> ConsensusSketchMessage:
                envelope = await services.receiver.receive(
                    MessageChannel.TELEMETRY,
                    timeout_seconds=timeout_seconds,
                )
                if (
                    envelope.message_type is not MessageType.CONSENSUS_SKETCH
                    or envelope.round_id is None
                ):
                    raise ValueError("received invalid consensus sketch envelope")
                return ConsensusSketchMessage(
                    sender_public_key=envelope.sender_public_key,
                    round_id=envelope.round_id,
                    payload=envelope.payload,
                )

            async def publish_consensus_sketch(
                round_id: int, sketch: np.ndarray
            ) -> None:
                await pair_transport.broadcast_consensus_sketch(
                    round_id=round_id,
                    sketch=sketch,
                )

            self._consensus_telemetry = LiveConsensusTelemetry(
                local_public_key=local_key,
                participant_keys=tuple(sorted(participants)),
                seed=self._result.manifest.consensus_sketch.seed,
                receive=receive_consensus_sketch,
                publish=publish_consensus_sketch,
                on_distance=self._record_consensus,
                size=self._result.manifest.consensus_sketch.size,
            )
            self._engine = GossipEngine(
                local_public_key=local_key,
                round_count=self._result.manifest.round_count,
                scheduler=PeerScheduler(
                    sorted(participants),
                    seed=self._result.manifest.peer_scheduler_seed,
                ),
                algorithm=self._training.algorithm,
                transport=self._pair_transport,
                commit_callback=self._persist_commit,
                transport_limits=self._result.manifest.transport,
                failure_broadcaster=self._pair_transport,
                consensus_publisher=self._consensus_telemetry,
                metrics_publisher=self._training.metrics_publisher,
            )
            await self._start_metrics()
            await self._consensus_telemetry.start()
            commits = await self._engine.run()
            async with self._terminal_lock:
                if self._remote_failure is not None:
                    raise self._remote_failure
                write_task = asyncio.create_task(
                    asyncio.to_thread(
                        self._training.run_store.record_terminal,
                        "complete",
                        {"committed_rounds": len(commits)},
                    )
                )
                try:
                    await asyncio.shield(write_task)
                except asyncio.CancelledError:
                    await write_task
                self._commits = commits
                self._state = NodeState.COMPLETE
            return commits
        except asyncio.CancelledError as error:
            self._state = NodeState.FAILED
            failure = self._remote_failure or error
            await self._record_terminal_failure(
                initialized, result="failed", error=failure
            )
            if self._remote_failure is not None:
                raise self._remote_failure from error
            raise
        except Exception as error:
            self._state = NodeState.FAILED
            await self._record_terminal_failure(
                initialized, result="failed", error=error
            )
            raise
        finally:
            if self._run_task is run_task:
                self._run_task = None
            await self._cleanup()

    async def stop(self) -> None:
        if self._state is NodeState.STOPPED:
            return
        task = self._run_task
        if (
            self._state is NodeState.RUNNING
            and task is not None
            and task is not asyncio.current_task()
        ):
            task.cancel()
            try:
                await task
            except BaseException:
                pass
        await self._stop_control_monitor()
        if self._state is not NodeState.CREATED:
            await self._formation.stop()
        self._state = NodeState.STOPPED

    async def _start_formation(self) -> None:
        if self._state is not NodeState.CREATED:
            raise NodeRuntimeError(f"cannot form node from {self._state}")
        self._state = NodeState.FORMING
        try:
            await self._formation.start()
        except BaseException:
            self._state = NodeState.FAILED
            raise

    async def _fail(self) -> None:
        self._state = NodeState.FAILED
        await self._formation.stop()

    async def _start_metrics(self) -> None:
        if self._training is not None and self._training.metrics_publisher is not None:
            await self._training.metrics_publisher.start()

    async def _create_pair_transport(
        self, local_key: str, *, artifact_root: Path
    ) -> AXLPairTransport:
        assert self._result is not None
        participants = frozenset(
            participant.public_key for participant in self._result.manifest.participants
        )
        services = self._formation.services
        return await asyncio.to_thread(
            AXLPairTransport,
            local_public_key=local_key,
            run_id=self._result.manifest.run_id,
            manifest_hash=self._result.manifest_hash,
            algorithm_id=self._result.manifest.algorithm_id,
            tensor_schema=self._result.manifest.tensor_schema,
            transport_limits=self._result.manifest.transport,
            receiver=services.receiver,
            sender=services.sender,
            transfer_manager=services.transfer_manager,
            artifact_root=artifact_root,
            participant_keys=participants,
        )

    def _create_failure_broadcaster(
        self, local_key: str
    ) -> AXLFailureBroadcaster:
        assert self._result is not None
        participants = frozenset(
            participant.public_key for participant in self._result.manifest.participants
        )
        return AXLFailureBroadcaster(
            local_public_key=local_key,
            run_id=self._result.manifest.run_id,
            manifest_hash=self._result.manifest_hash,
            algorithm_id=self._result.manifest.algorithm_id,
            transport_limits=self._result.manifest.transport,
            sender=self._formation.services.sender,
            participant_keys=participants,
        )

    def _failure_dependencies(self) -> tuple[RunStore, Path]:
        if self._training is not None:
            return self._training.run_store, self._training.artifact_root
        if self._failure is not None:
            return self._failure.run_store, self._failure.artifact_root
        raise NodeRuntimeError("failure persistence is not configured")

    async def _monitor_run_failures(self) -> None:
        services = self._formation.services
        while self._state in {NodeState.READY, NodeState.RUNNING}:
            try:
                envelope = await services.receiver.receive(
                    MessageChannel.CONTROL,
                    timeout_seconds=0.1,
                )
            except TimeoutError:
                continue
            if envelope.message_type is not MessageType.RUN_FAILED:
                continue
            try:
                failure = decode_run_failure(envelope.payload)
            except ValueError:
                continue
            error = PeerRunFailureError(
                f"peer {envelope.sender_public_key} failed at round "
                f"{failure.round_id}: {failure.error_type}: {failure.reason}"
            )
            async with self._terminal_lock:
                if self._state not in {NodeState.READY, NodeState.RUNNING}:
                    return
                self._remote_failure = error
                task = self._run_task
                if self._state is NodeState.RUNNING and task is not None:
                    task.cancel()
                    return
                self._state = NodeState.FAILED
                try:
                    run_store, _ = self._failure_dependencies()
                    assert self._result is not None
                    await asyncio.to_thread(
                        run_store.initialize, self._result.manifest
                    )
                    await asyncio.to_thread(
                        run_store.record_terminal,
                        "failed",
                        {
                            "error_type": type(error).__name__,
                            "error": str(error)[:1024],
                        },
                    )
                finally:
                    await self._cleanup()
            return

    async def _stop_control_monitor(self) -> None:
        task = self._control_task
        if task is None:
            return
        self._control_task = None
        if task is asyncio.current_task():
            return
        task.cancel()
        try:
            await task
        except BaseException:
            pass

    async def _stop_metrics(self) -> None:
        if self._training is not None and self._training.metrics_publisher is not None:
            await self._training.metrics_publisher.stop()

    async def _stop_consensus(self) -> None:
        telemetry = self._consensus_telemetry
        if telemetry is None:
            return
        await telemetry.stop()
        if telemetry.dropped and self._result is not None:
            await asyncio.to_thread(
                emit_event,
                "consensus_telemetry_dropped",
                run_id=self._result.manifest.run_id,
                manifest_hash=self._result.manifest_hash,
                node_id=self._local_public_key,
                message_id="consensus-telemetry-dropped",
                dropped=telemetry.dropped,
                sink=self._event_sink,
            )

    async def _record_terminal_failure(
        self, initialized: bool, *, result: str, error: BaseException
    ) -> None:
        if not initialized or self._training is None:
            return
        async with self._terminal_lock:
            try:
                await asyncio.to_thread(
                    self._training.run_store.record_terminal,
                    result,
                    {
                        "error_type": type(error).__name__,
                        "error": str(error)[:1024],
                    },
                )
            except Exception:
                pass

    async def _cleanup(self) -> None:
        for cleanup in (
            self._stop_control_monitor,
            self._stop_consensus,
            self._stop_metrics,
            self._formation.stop,
        ):
            try:
                await cleanup()
            except BaseException:
                pass

    async def _record_consensus(self, distance: ConsensusDistance) -> None:
        if (
            self._training is None
            or self._result is None
            or self._local_public_key is None
        ):
            return
        await asyncio.to_thread(
            self._training.run_store.record_consensus,
            round_id=distance.round_id,
            normalized_rms=distance.normalized_rms,
            sketch_count=distance.sketch_count,
        )
        await asyncio.to_thread(
            emit_event,
            "consensus_distance",
            run_id=self._result.manifest.run_id,
            manifest_hash=self._result.manifest_hash,
            node_id=self._local_public_key,
            message_id=f"consensus-distance-{distance.round_id}",
            round_id=distance.round_id,
            normalized_rms=distance.normalized_rms,
            sketch_count=distance.sketch_count,
            sink=self._event_sink,
        )

    def _persist_commit(self, commit: RoundCommit) -> None:
        assert self._training is not None
        assert self._pair_transport is not None
        metrics: dict[str, object] = {"round_id": commit.round_id}
        local_loss = getattr(self._training.algorithm, "local_loss", None)
        if isinstance(local_loss, (int, float)):
            metrics["local_loss"] = float(local_loss)
        self._training.run_store.persist_commit(
            committed_round=commit.round_id,
            algorithm_state=commit.post_mix.weights,
            pre_mix_state=commit.local_bundle.tensors,
            post_mix_state=commit.post_mix.weights,
            state_checksum=commit.state_checksum,
            schedule={
                "round_id": commit.round_id,
                "peer": commit.peer_public_key,
            },
            metrics=metrics,
            transfer_diagnostics={
                "transfer_id": self._pair_transport.last_transfer_id,
                "retries": self._pair_transport.last_retry_count,
            },
        )

    def _ready(self, result: FormationResult) -> FormationResult:
        self._result = result
        self._state = NodeState.READY
        self._control_task = asyncio.create_task(
            self._monitor_run_failures(),
            name="dromeus-control-monitor",
        )
        return result
