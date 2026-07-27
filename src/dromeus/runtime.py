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
    AXLPairTransport,
    GossipAlgorithm,
    GossipEngine,
    RoundCommit,
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
    CIFAR10Data,
    CIFAR10Trainer,
    InitialCheckpoint,
    create_initial_checkpoint,
)
from dromeus.transport.base import AsyncTransport
from dromeus.transport.envelope import MessageType
from dromeus.transport.receiver import MessageChannel
from dromeus.transport.transfer import ArtifactStore


class NodeRuntimeError(RuntimeError):
    """The node runtime lifecycle was used out of order."""


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
class PreparedCIFARTraining:
    """Validated local CIFAR data ready for one formed benchmark node."""

    partitions: tuple[CIFAR10Data, ...]
    test_data: CIFAR10Data
    benchmark_seed: int

    def create_initial_checkpoint(self, path: Path) -> InitialCheckpoint:
        return create_initial_checkpoint(path, seed=self.benchmark_seed)

    def build_config(
        self,
        *,
        result: FormationResult,
        local_public_key: str,
        run_root: Path,
        metrics_publisher: MetricsService,
    ) -> TrainingConfig:
        node_indices = {
            participant.public_key: participant.node_index
            for participant in result.manifest.participants
        }
        node_index = node_indices[local_public_key]
        partition_index = result.manifest.dataset.node_index_partitions[node_index]
        trainer = CIFAR10Trainer(
            train_data=self.partitions[partition_index],
            test_data=self.test_data,
            seed=self.benchmark_seed + node_index,
            batch_size=32,
            learning_rate=result.manifest.learning_rate,
            device="cpu",
            augment=True,
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
    """Load and validate local CIFAR data before membership becomes ready."""
    train_data = CIFAR10Data.from_torchvision(
        root=cifar_root,
        train=True,
        download=False,
    )
    test_data = CIFAR10Data.from_torchvision(
        root=cifar_root,
        train=False,
        download=False,
    )
    if len(train_data) != draft.dataset.sample_count:
        raise ValueError("local CIFAR-10 sample count does not match draft")
    partitions = train_data.split_iid(
        participant_count=4,
        seed=draft.dataset.iid_partition_seed,
    )
    if tuple(len(partition) for partition in partitions) != (
        draft.dataset.partition_sample_counts
    ):
        raise ValueError("local CIFAR-10 partitions do not match draft")
    return PreparedCIFARTraining(
        partitions=partitions,
        test_data=test_data,
        benchmark_seed=benchmark_seed,
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
        self._engine: GossipEngine | None = None
        self._pair_transport: AXLPairTransport | None = None
        self._consensus_telemetry: LiveConsensusTelemetry | None = None
        self._event_sink = event_sink
        self._local_public_key: str | None = None
        self._commits: tuple[RoundCommit, ...] = ()
        self._run_task: asyncio.Task[tuple[RoundCommit, ...]] | None = None

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
            pair_transport = await asyncio.to_thread(
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
                artifact_root=self._training.artifact_root,
                participant_keys=participants,
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
            await asyncio.to_thread(
                self._training.run_store.record_terminal,
                "complete",
                {"committed_rounds": len(commits)},
            )
            self._commits = commits
            self._state = NodeState.COMPLETE
            return commits
        except asyncio.CancelledError as error:
            self._state = NodeState.FAILED
            await self._record_terminal_failure(
                initialized, result="failed", error=error
            )
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
        return result
