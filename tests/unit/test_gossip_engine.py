from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from support.in_memory_transport import (
    InMemoryFaults,
    InMemoryNetwork,
    InMemoryTransport,
)

from dromeus.algorithms.base import UpdateBundle
from dromeus.algorithms.codec import SafetensorsUpdateBundleCodec
from dromeus.algorithms.dpsgd import DPSGDAdapter, checksum_tensors
from dromeus.gossip.engine import (
    AXLPairTransport,
    EvaluationMetrics,
    GossipEngine,
    PairCommitError,
    RoundCommit,
    RunFailure,
)
from dromeus.gossip.peer_scheduler import PeerScheduler
from dromeus.manifests.models import Tensor, TensorSchema, TransportLimits
from dromeus.protocol.codec import encode_message
from dromeus.protocol.models import (
    Envelope,
    MessageType,
    PairCommitMessage,
    create_envelope,
)
from dromeus.telemetry.metrics import RoundTiming
from dromeus.transport.outbound_scheduler import OutboundScheduler
from dromeus.transport.receiver import Receiver
from dromeus.transport.transfer import TransferManager


class LinearTrainer:
    def __init__(self, value: float) -> None:
        self._weights = {"weight": np.array([value], dtype=np.float32)}

    def train_local_steps(self, step_count: int) -> None:
        self._weights["weight"] += np.float32(step_count)

    def weights(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self._weights.items()}

    def load_weights(self, weights: dict[str, np.ndarray]) -> None:
        self._weights = {name: value.copy() for name, value in weights.items()}

    def evaluate(self) -> tuple[float, float]:
        return float(self._weights["weight"][0]), 0.5


class ConvexTrainer(LinearTrainer):
    def train_local_steps(self, step_count: int) -> None:
        for _ in range(step_count):
            self._weights["weight"] *= np.float32(0.5)


class SlowEvaluationTrainer(LinearTrainer):
    def evaluate(self) -> tuple[float, float]:
        time.sleep(0.2)
        return super().evaluate()


class BlockingTrainer(LinearTrainer):
    def __init__(
        self,
        value: float,
        *,
        started: threading.Event,
        release: threading.Event,
    ) -> None:
        super().__init__(value)
        self._started = started
        self._release = release

    def train_local_steps(self, step_count: int) -> None:
        self._started.set()
        if not self._release.wait(timeout=1):
            raise RuntimeError("test did not release local training")
        super().train_local_steps(step_count)


@dataclass
class SharedPairChannel:
    updates: dict[tuple[str, str, int], asyncio.Queue[UpdateBundle]]
    ready: dict[tuple[str, str, int], asyncio.Queue[str]]
    committed: dict[tuple[str, str, int], asyncio.Queue[str]]

    @classmethod
    def create(cls) -> SharedPairChannel:
        return cls(updates={}, ready={}, committed={})

    async def exchange_update(
        self,
        local: str,
        peer: str,
        round_id: int,
        bundle: UpdateBundle,
    ) -> UpdateBundle:
        outgoing = self.updates.setdefault((local, peer, round_id), asyncio.Queue())
        incoming = self.updates.setdefault((peer, local, round_id), asyncio.Queue())
        await outgoing.put(bundle)
        return await incoming.get()

    async def exchange_ready(
        self,
        local: str,
        peer: str,
        round_id: int,
        checksum: str,
    ) -> str:
        outgoing = self.ready.setdefault((local, peer, round_id), asyncio.Queue())
        incoming = self.ready.setdefault((peer, local, round_id), asyncio.Queue())
        await outgoing.put(checksum)
        return await incoming.get()

    async def exchange_committed(
        self,
        local: str,
        peer: str,
        round_id: int,
        checksum: str,
    ) -> str:
        outgoing = self.committed.setdefault((local, peer, round_id), asyncio.Queue())
        incoming = self.committed.setdefault((peer, local, round_id), asyncio.Queue())
        await outgoing.put(checksum)
        return await incoming.get()


class InMemoryPairTransport:
    def __init__(self, local: str, channel: SharedPairChannel) -> None:
        self.local = local
        self.channel = channel

    async def exchange_update(
        self,
        *,
        peer: str,
        round_id: int,
        bundle: UpdateBundle,
    ) -> UpdateBundle:
        return await self.channel.exchange_update(self.local, peer, round_id, bundle)

    async def exchange_update_ready(
        self,
        *,
        peer: str,
        round_id: int,
        bundle_checksum: str,
    ) -> str:
        return await self.channel.exchange_ready(
            self.local, peer, round_id, bundle_checksum
        )

    async def exchange_round_committed(
        self,
        *,
        peer: str,
        round_id: int,
        state_checksum: str,
    ) -> None:
        remote_checksum = await self.channel.exchange_committed(
            self.local, peer, round_id, state_checksum
        )
        assert remote_checksum == state_checksum


class HangingPairTransport(InMemoryPairTransport):
    async def exchange_update(
        self,
        *,
        peer: str,
        round_id: int,
        bundle: UpdateBundle,
    ) -> UpdateBundle:
        await asyncio.sleep(1)
        return cast(UpdateBundle, None)


class StaticPairTransport:
    def __init__(self, peer_bundle: UpdateBundle) -> None:
        self.peer_bundle = peer_bundle

    async def exchange_update(
        self,
        *,
        peer: str,
        round_id: int,
        bundle: UpdateBundle,
    ) -> UpdateBundle:
        return self.peer_bundle

    async def exchange_update_ready(
        self,
        *,
        peer: str,
        round_id: int,
        bundle_checksum: str,
    ) -> str:
        return self.peer_bundle.digest

    async def exchange_round_committed(
        self,
        *,
        peer: str,
        round_id: int,
        state_checksum: str,
    ) -> None:
        return None


class RejectingCommitTransport(StaticPairTransport):
    async def exchange_round_committed(
        self,
        *,
        peer: str,
        round_id: int,
        state_checksum: str,
    ) -> None:
        raise PairCommitError("peer did not confirm")


class StubPairReceiver:
    def __init__(self, envelope: Envelope) -> None:
        self._envelope = envelope

    async def receive(
        self, channel: object, *, timeout_seconds: float
    ) -> Envelope:
        return self._envelope

    def set_current_round(self, round_id: int) -> None:
        return None

    async def advance_round(self, round_id: int) -> None:
        return None


class StubPairSender:
    async def send(self, *args: object, **kwargs: object) -> None:
        return None


class RecordingBundleCodec:
    def __init__(
        self,
        delegate: SafetensorsUpdateBundleCodec,
        *,
        validation_error: bool = False,
    ) -> None:
        self.delegate = delegate
        self.validation_error = validation_error
        self.released: list[str] = []

    def encode(
        self, *, round_id: int, tensors: Mapping[str, np.ndarray]
    ) -> UpdateBundle:
        return self.delegate.encode(round_id=round_id, tensors=tensors)

    def decode(self, bundle: UpdateBundle) -> dict[str, np.ndarray]:
        if self.validation_error:
            raise ValueError("forced validation error")
        return self.delegate.decode(bundle)

    def release(self, bundle: UpdateBundle) -> None:
        self.released.append(bundle.digest)
        self.delegate.release(bundle)

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state:
            raise ValueError("recording codec has no state")


@dataclass
class RecordingPublisher:
    rounds: list[int]

    def submit(
        self, *, round_id: int, weights: Mapping[str, np.ndarray]
    ) -> bool:
        self.rounds.append(round_id)
        return True


@dataclass
class RecordingMetricsPublisher:
    timings: list[RoundTiming]
    failures: list[tuple[int, str, str]]

    def submit(self, timing: RoundTiming) -> bool:
        self.timings.append(timing)
        return True

    def submit_failure(self, *, round_id: int, error_type: str, reason: str) -> bool:
        self.failures.append((round_id, error_type, reason))
        return True


class TimedPairTransport(InMemoryPairTransport):
    last_transfer_id = "transfer-0"
    last_retry_count = 2


@dataclass
class RecordingFailureBroadcaster:
    failures: list[RunFailure]

    async def broadcast_run_failed(self, failure: RunFailure) -> None:
        self.failures.append(failure)


def _algorithm(
    *,
    key: str,
    trainer: LinearTrainer,
    schema: TensorSchema,
    artifact_root: Path,
    training_round_count: int | None = None,
) -> DPSGDAdapter:
    return DPSGDAdapter(
        trainer=trainer,
        tensor_schema=schema,
        local_steps=1,
        training_round_count=training_round_count,
        bundle_codec=SafetensorsUpdateBundleCodec(
            artifact_root=artifact_root,
            run_id="test-run",
            manifest_hash="0" * 64,
            sender_public_key=key,
            algorithm_id="d-psgd",
            tensor_schema=schema,
        ),
    )


def test_pair_timeout_fails_once_and_reports_diagnostics(tmp_path: Path) -> None:
    failures: list[RunFailure] = []
    broadcasted: list[RunFailure] = []
    metrics = RecordingMetricsPublisher([], [])

    async def run() -> None:
        schema = TensorSchema(
            tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),)
        )
        codec = RecordingBundleCodec(
            SafetensorsUpdateBundleCodec(
                artifact_root=tmp_path / "peer-0",
                run_id="test-run",
                manifest_hash="0" * 64,
                sender_public_key="peer-0",
                algorithm_id="d-psgd",
                tensor_schema=schema,
            )
        )
        engine = GossipEngine(
            local_public_key="peer-0",
            round_count=1,
            scheduler=PeerScheduler(["peer-0", "peer-1"], seed=8),
            algorithm=DPSGDAdapter(
                trainer=LinearTrainer(1.0),
                tensor_schema=schema,
                local_steps=1,
                bundle_codec=codec,
            ),
            transport=HangingPairTransport("peer-0", SharedPairChannel.create()),
            commit_callback=lambda commit: None,
            timeout_seconds=0.01,
            failure_callback=failures.append,
            failure_broadcaster=RecordingFailureBroadcaster(broadcasted),
            metrics_publisher=metrics,
        )

        with pytest.raises(PairCommitError, match="deadline"):
            await engine.run()
        assert engine.failure == failures[0]
        assert len(failures) == 1
        assert broadcasted == failures
        assert metrics.failures[0][0] == 0
        assert len(codec.released) == 1

    asyncio.run(run())


def test_round_committed_rejects_payload_round_mismatch(tmp_path: Path) -> None:
    async def run() -> None:
        payload = encode_message(
            PairCommitMessage(round_id=1, checksum="1" * 64)
        )
        envelope = create_envelope(
            message_type=MessageType.ROUND_COMMITTED,
            message_id="round-committed-0",
            run_id="test-run",
            manifest_hash="0" * 64,
            sender_public_key="peer-1",
            algorithm_id="d-psgd",
            round_id=0,
            correlation_id="pair-round-0",
            payload=payload,
        )
        transport = AXLPairTransport(
            local_public_key="peer-0",
            run_id="test-run",
            manifest_hash="0" * 64,
            algorithm_id="d-psgd",
            transport_limits=TransportLimits(
                max_payload_bytes=1024,
                max_retries=0,
                retry_timeout_seconds=0.1,
            ),
            receiver=cast(Receiver, StubPairReceiver(envelope)),
            sender=cast(OutboundScheduler, StubPairSender()),
            transfer_manager=cast(TransferManager, object()),
            metadata_root=tmp_path,
        )

        with pytest.raises(PairCommitError, match="ROUND_COMMITTED round"):
            await transport.exchange_round_committed(
                peer="peer-1",
                round_id=0,
                state_checksum="2" * 64,
            )

    asyncio.run(run())


def test_round_committed_recovers_from_silent_message_loss(tmp_path: Path) -> None:
    async def run() -> None:
        network = InMemoryNetwork()
        raw_transports = (
            InMemoryTransport(
                network=network,
                public_key="peer-0",
                faults=InMemoryFaults(drop_send_calls=frozenset({1})),
            ),
            InMemoryTransport(network=network, public_key="peer-1"),
        )
        receivers = tuple(Receiver(raw) for raw in raw_transports)
        senders = tuple(OutboundScheduler(raw) for raw in raw_transports)
        limits = TransportLimits(
            max_payload_bytes=1024,
            max_retries=1,
            retry_timeout_seconds=0.05,
        )
        transports = tuple(
            AXLPairTransport(
                local_public_key=f"peer-{index}",
                run_id="test-run",
                manifest_hash="0" * 64,
                algorithm_id="d-psgd",
                transport_limits=limits,
                receiver=receivers[index],
                sender=senders[index],
                transfer_manager=cast(TransferManager, object()),
                metadata_root=tmp_path / f"peer-{index}",
            )
            for index in range(2)
        )
        for receiver in receivers:
            await receiver.start()
        for sender in senders:
            await sender.start()
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    transports[0].exchange_round_committed(
                        peer="peer-1",
                        round_id=0,
                        state_checksum="1" * 64,
                    ),
                    transports[1].exchange_round_committed(
                        peer="peer-0",
                        round_id=0,
                        state_checksum="1" * 64,
                    ),
                ),
                timeout=1.0,
            )
            await asyncio.sleep(0.05)
            assert all(receiver.stats.rejected_messages == 0 for receiver in receivers)
        finally:
            await asyncio.gather(*(receiver.stop() for receiver in receivers))
            await asyncio.gather(*(sender.stop() for sender in senders))

    asyncio.run(run())


def test_transport_cancellation_waits_for_active_bundle_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        started = threading.Event()
        release = threading.Event()
        schema = TensorSchema(
            tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),)
        )
        codec = SafetensorsUpdateBundleCodec(
            artifact_root=tmp_path / "local",
            run_id="test-run",
            manifest_hash="0" * 64,
            sender_public_key="peer-0",
            algorithm_id="d-psgd",
            tensor_schema=schema,
        )
        bundle = codec.encode(
            round_id=0,
            tensors={"weight": np.array([2.0], dtype=np.float32)},
        )
        original_validate = UpdateBundle.validate_materialized

        def blocking_validate(
            value: UpdateBundle, max_bundle_bytes: int
        ) -> None:
            started.set()
            if not release.wait(timeout=1):
                raise RuntimeError("test did not release bundle validation")
            original_validate(value, max_bundle_bytes)

        monkeypatch.setattr(UpdateBundle, "validate_materialized", blocking_validate)
        transport = AXLPairTransport(
            local_public_key="peer-0",
            run_id="test-run",
            manifest_hash="0" * 64,
            algorithm_id="d-psgd",
            transport_limits=TransportLimits(
                max_payload_bytes=1024,
                max_retries=0,
                retry_timeout_seconds=0.1,
            ),
            receiver=cast(Receiver, object()),
            sender=cast(OutboundScheduler, object()),
            transfer_manager=cast(TransferManager, object()),
            metadata_root=tmp_path / "metadata",
        )

        task = asyncio.create_task(
            transport.exchange_update(
                peer="peer-1",
                round_id=0,
                bundle=bundle,
            )
        )
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        try:
            await asyncio.sleep(0.01)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())


def test_engine_confirms_durability_only_after_peer_confirmation(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        schema = TensorSchema(
            tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),)
        )
        peer_codec = SafetensorsUpdateBundleCodec(
            artifact_root=tmp_path / "peer",
            run_id="test-run",
            manifest_hash="0" * 64,
            sender_public_key="peer-1",
            algorithm_id="d-psgd",
            tensor_schema=schema,
        )
        peer_bundle = peer_codec.encode(
            round_id=0,
            tensors={"weight": np.array([3.0], dtype=np.float32)},
        )
        prepared: list[RoundCommit] = []
        confirmed: list[RoundCommit] = []
        engine = GossipEngine(
            local_public_key="peer-0",
            round_count=1,
            scheduler=PeerScheduler(["peer-0", "peer-1"], seed=8),
            algorithm=_algorithm(
                key="peer-0",
                trainer=LinearTrainer(1.0),
                schema=schema,
                artifact_root=tmp_path / "local",
            ),
            transport=RejectingCommitTransport(peer_bundle),
            commit_callback=prepared.append,
            confirm_callback=confirmed.append,
        )

        with pytest.raises(PairCommitError, match="peer did not confirm"):
            await engine.run()
        assert len(prepared) == 1
        assert confirmed == []

    asyncio.run(run())


@pytest.mark.parametrize("failure", [None, "corruption", "validation"])
def test_release_runs_after_bundle_outcomes(
    tmp_path: Path, failure: str | None
) -> None:
    async def run() -> None:
        schema = TensorSchema(
            tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),)
        )
        local_delegate = SafetensorsUpdateBundleCodec(
            artifact_root=tmp_path / "local",
            run_id="test-run",
            manifest_hash="0" * 64,
            sender_public_key="peer-0",
            algorithm_id="d-psgd",
            tensor_schema=schema,
        )
        codec = RecordingBundleCodec(
            local_delegate, validation_error=failure == "validation"
        )
        peer_codec = SafetensorsUpdateBundleCodec(
            artifact_root=tmp_path / "peer",
            run_id="test-run",
            manifest_hash="0" * 64,
            sender_public_key="peer-1",
            algorithm_id="d-psgd",
            tensor_schema=schema,
        )
        peer_bundle = peer_codec.encode(
            round_id=0,
            tensors={"weight": np.array([3.0], dtype=np.float32)},
        )
        if failure == "corruption":
            with peer_bundle.artifacts[0].path.open("ab") as handle:
                handle.write(b"corrupt")
        engine = GossipEngine(
            local_public_key="peer-0",
            round_count=1,
            scheduler=PeerScheduler(["peer-0", "peer-1"], seed=8),
            algorithm=DPSGDAdapter(
                trainer=LinearTrainer(1.0),
                tensor_schema=schema,
                local_steps=1,
                bundle_codec=codec,
            ),
            transport=StaticPairTransport(peer_bundle),
            commit_callback=lambda commit: None,
        )
        if failure is None:
            await engine.run()
        else:
            with pytest.raises(PairCommitError, match="validation"):
                await engine.run()
        assert len(codec.released) == 2
        assert not any(path.is_file() for path in tmp_path.rglob("*"))

    asyncio.run(run())


def test_release_runs_after_cancellation(tmp_path: Path) -> None:
    async def run() -> None:
        schema = TensorSchema(
            tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),)
        )
        codec = RecordingBundleCodec(
            SafetensorsUpdateBundleCodec(
                artifact_root=tmp_path / "local",
                run_id="test-run",
                manifest_hash="0" * 64,
                sender_public_key="peer-0",
                algorithm_id="d-psgd",
                tensor_schema=schema,
            )
        )
        engine = GossipEngine(
            local_public_key="peer-0",
            round_count=1,
            scheduler=PeerScheduler(["peer-0", "peer-1"], seed=8),
            algorithm=DPSGDAdapter(
                trainer=LinearTrainer(1.0),
                tensor_schema=schema,
                local_steps=1,
                bundle_codec=codec,
            ),
            transport=HangingPairTransport("peer-0", SharedPairChannel.create()),
            commit_callback=lambda commit: None,
        )
        task = asyncio.create_task(engine.run())
        while not any(path.is_file() for path in tmp_path.rglob("*")):
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(codec.released) == 1
        assert not any(path.is_file() for path in tmp_path.rglob("*"))

    asyncio.run(run())


def test_cancellation_waits_for_active_model_mutation(tmp_path: Path) -> None:
    async def run() -> None:
        started = threading.Event()
        release = threading.Event()
        trainer = BlockingTrainer(1.0, started=started, release=release)
        schema = TensorSchema(
            tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),)
        )
        engine = GossipEngine(
            local_public_key="peer-0",
            round_count=1,
            scheduler=PeerScheduler(["peer-0", "peer-1"], seed=8),
            algorithm=_algorithm(
                key="peer-0",
                trainer=trainer,
                schema=schema,
                artifact_root=tmp_path / "local",
            ),
            transport=HangingPairTransport("peer-0", SharedPairChannel.create()),
            commit_callback=lambda commit: None,
        )

        task = asyncio.create_task(engine.run())
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        try:
            await asyncio.sleep(0.01)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert np.array_equal(
            trainer.weights()["weight"], np.array([2.0], dtype=np.float32)
        )

    asyncio.run(run())


def test_engine_publishes_round_timings_without_waiting_for_metric_writer(
    tmp_path: Path,
) -> None:
    schema = TensorSchema(tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),))
    channel = SharedPairChannel.create()
    metrics = RecordingMetricsPublisher([], [])
    evaluations: list[EvaluationMetrics] = []

    async def run() -> None:
        engines: list[GossipEngine] = []
        for key, value in (("peer-0", 1.0), ("peer-1", 3.0)):
            engines.append(
                GossipEngine(
                    local_public_key=key,
                    round_count=1,
                    scheduler=PeerScheduler(["peer-0", "peer-1"], seed=8),
                    algorithm=_algorithm(
                        key=key,
                        trainer=LinearTrainer(value),
                        schema=schema,
                        artifact_root=tmp_path / key,
                    ),
                    transport=TimedPairTransport(key, channel),
                    commit_callback=lambda commit: None,
                    evaluation_callback=evaluations.append,
                    metrics_publisher=metrics,
                )
            )
        await asyncio.gather(*(engine.run() for engine in engines))

    asyncio.run(run())
    assert len(metrics.timings) == 2
    assert all(timing.transfer_id == "transfer-0" for timing in metrics.timings)
    assert all(timing.retries == 2 for timing in metrics.timings)
    assert all(timing.transfer_seconds >= 0 for timing in metrics.timings)
    assert all(timing.peer_wait_seconds >= 0 for timing in metrics.timings)
    assert all(timing.mixing_seconds >= 0 for timing in metrics.timings)
    assert all(timing.evaluation_seconds >= 0 for timing in metrics.timings)
    assert all(timing.evaluation_accuracy == 0.5 for timing in metrics.timings)


def test_two_nodes_complete_pair_commit_without_group_barrier(
    tmp_path: Path,
) -> None:
    schema = TensorSchema(tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),))
    channel = SharedPairChannel.create()
    commits: dict[str, list[RoundCommit]] = {"peer-0": [], "peer-1": []}
    trainers: dict[str, LinearTrainer] = {}
    publisher = RecordingPublisher([])

    async def run() -> None:
        engines: list[GossipEngine] = []
        for key, value in (("peer-0", 1.0), ("peer-1", 3.0)):
            trainer = LinearTrainer(value)
            trainers[key] = trainer
            algorithm = _algorithm(
                key=key,
                trainer=trainer,
                schema=schema,
                artifact_root=tmp_path / key,
            )
            transport = InMemoryPairTransport(key, channel)
            engines.append(
                GossipEngine(
                    local_public_key=key,
                    round_count=1,
                    scheduler=PeerScheduler(["peer-0", "peer-1"], seed=8),
                    algorithm=algorithm,
                    transport=transport,
                    commit_callback=commits[key].append,
                    consensus_publisher=publisher if key == "peer-0" else None,
                )
            )
        await asyncio.gather(*(engine.run() for engine in engines))
        assert all(len(records) == 1 for records in commits.values())
        assert all(
            np.array_equal(trainer.weights()["weight"], np.array([3.0]))
            for trainer in trainers.values()
        )
        assert all(
            len(record.local_bundle_digest) == 64
            and len(record.peer_bundle_digest) == 64
            and len(record.state_checksum) == 64
            for records in commits.values()
            for record in records
        )
        assert all(
            not hasattr(record, snapshot_field)
            for records in commits.values()
            for record in records
            for snapshot_field in ("pre_local", "post_local", "post_mix")
        )
        assert publisher.rounds == [0]

    asyncio.run(run())


def test_evaluation_runs_every_five_rounds_and_on_final_round(
    tmp_path: Path,
) -> None:
    schema = TensorSchema(tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),))
    channel = SharedPairChannel.create()
    evaluations: dict[str, list[EvaluationMetrics]] = {"peer-0": [], "peer-1": []}

    async def run() -> None:
        engines: list[GossipEngine] = []
        for key, value in (("peer-0", 1.0), ("peer-1", 3.0)):
            algorithm = _algorithm(
                key=key,
                trainer=LinearTrainer(value),
                schema=schema,
                artifact_root=tmp_path / key,
            )
            engines.append(
                GossipEngine(
                    local_public_key=key,
                    round_count=6,
                    scheduler=PeerScheduler(["peer-0", "peer-1"], seed=8),
                    algorithm=algorithm,
                    transport=InMemoryPairTransport(key, channel),
                    commit_callback=lambda commit: None,
                    evaluation_callback=evaluations[key].append,
                )
            )
        await asyncio.gather(*(engine.run() for engine in engines))

    asyncio.run(run())
    assert [metric.round_id for metric in evaluations["peer-0"]] == [4, 5]
    assert [metric.round_id for metric in evaluations["peer-1"]] == [4, 5]
    assert all(
        metric.accuracy == 0.5
        for values in evaluations.values()
        for metric in values
    )


def test_evaluation_is_outside_pair_deadline(tmp_path: Path) -> None:
    schema = TensorSchema(tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),))
    channel = SharedPairChannel.create()

    async def run() -> None:
        engines = [
            GossipEngine(
                local_public_key=key,
                round_count=1,
                scheduler=PeerScheduler(["peer-0", "peer-1"], seed=8),
                algorithm=_algorithm(
                    key=key,
                    trainer=SlowEvaluationTrainer(value),
                    schema=schema,
                    artifact_root=tmp_path / key,
                ),
                transport=InMemoryPairTransport(key, channel),
                commit_callback=lambda commit: None,
                timeout_seconds=0.05,
                evaluation_callback=lambda metrics: None,
            )
            for key, value in (("peer-0", 1.0), ("peer-1", 3.0))
        ]
        await asyncio.gather(*(engine.run() for engine in engines))

    asyncio.run(run())


def test_four_in_memory_nodes_reduce_a_shared_convex_objective(
    tmp_path: Path,
) -> None:
    schema = TensorSchema(tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),))
    channel = SharedPairChannel.create()
    trainers = {
        f"peer-{index}": ConvexTrainer(float(8 - index * 2)) for index in range(4)
    }

    async def run() -> None:
        engines = [
            GossipEngine(
                local_public_key=key,
                round_count=3,
                scheduler=PeerScheduler(list(trainers), seed=8),
                algorithm=_algorithm(
                    key=key,
                    trainer=trainer,
                    schema=schema,
                    artifact_root=tmp_path / key,
                ),
                transport=InMemoryPairTransport(key, channel),
                commit_callback=lambda commit: None,
            )
            for key, trainer in trainers.items()
        ]
        await asyncio.gather(*(engine.run() for engine in engines))

    asyncio.run(run())
    assert all(
        abs(float(trainer.weights()["weight"][0])) < 8.0
        for trainer in trainers.values()
    )


def test_two_final_consensus_stages_exactly_average_four_nodes(
    tmp_path: Path,
) -> None:
    schema = TensorSchema(tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),))
    channel = SharedPairChannel.create()
    training_calls = {f"peer-{index}": 0 for index in range(4)}

    class CountingTrainer(LinearTrainer):
        def __init__(self, key: str, value: float) -> None:
            super().__init__(value)
            self._key = key

        def train_local_steps(self, step_count: int) -> None:
            training_calls[self._key] += 1
            super().train_local_steps(step_count)

    trainers = {
        key: CountingTrainer(key, float(index * 2))
        for index, key in enumerate(training_calls)
    }

    async def run() -> None:
        engines = [
            GossipEngine(
                local_public_key=key,
                round_count=3,
                scheduler=PeerScheduler(
                    list(trainers),
                    seed=8,
                    training_round_count=1,
                    final_consensus_rounds=2,
                ),
                algorithm=_algorithm(
                    key=key,
                    trainer=trainer,
                    schema=schema,
                    artifact_root=tmp_path / key,
                    training_round_count=1,
                ),
                transport=InMemoryPairTransport(key, channel),
                commit_callback=lambda commit: None,
            )
            for key, trainer in trainers.items()
        ]
        await asyncio.gather(*(engine.run() for engine in engines))
        assert all(len(engine.commits) == 3 for engine in engines)
        assert all(engine.commits[-1].phase == "final-consensus" for engine in engines)

    asyncio.run(run())
    assert training_calls == {key: 1 for key in trainers}
    assert {
        float(trainer.weights()["weight"][0]) for trainer in trainers.values()
    } == {4.0}


def test_tensor_checksum_is_stable() -> None:
    tensors = {"weight": np.array([2.0], dtype=np.float32)}
    assert (
        checksum_tensors(tensors)
        == "fd734a524f800f74e9b9ac0d0134f90237a95dbc2854a847382d01fed960bfb4"
    )
