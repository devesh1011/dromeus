from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
import pytest

from dromeus.algorithms.base import TrainedWeightsBundle
from dromeus.algorithms.dpsgd import DPSGDAdapter, checksum_tensors
from dromeus.gossip.engine import (
    EvaluationMetrics,
    GossipEngine,
    PairCommitError,
    RoundCommit,
    RunFailure,
)
from dromeus.gossip.scheduler import PeerScheduler
from dromeus.manifests.models import Tensor, TensorSchema
from dromeus.telemetry.metrics import RoundTiming


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


@dataclass
class SharedPairChannel:
    updates: dict[tuple[str, str, int], asyncio.Queue[TrainedWeightsBundle]]
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
        bundle: TrainedWeightsBundle,
    ) -> TrainedWeightsBundle:
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
        bundle: TrainedWeightsBundle,
    ) -> TrainedWeightsBundle:
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
        bundle: TrainedWeightsBundle,
    ) -> TrainedWeightsBundle:
        await asyncio.sleep(1)
        return cast(TrainedWeightsBundle, None)


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


def test_pair_timeout_fails_once_and_reports_diagnostics() -> None:
    failures: list[RunFailure] = []
    broadcasted: list[RunFailure] = []
    metrics = RecordingMetricsPublisher([], [])

    async def run() -> None:
        schema = TensorSchema(
            tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),)
        )
        engine = GossipEngine(
            local_public_key="peer-0",
            round_count=1,
            scheduler=PeerScheduler(["peer-0", "peer-1"], seed=8),
            algorithm=DPSGDAdapter(
                trainer=LinearTrainer(1.0), tensor_schema=schema, local_steps=1
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

    asyncio.run(run())


def test_engine_publishes_round_timings_without_waiting_for_metric_writer() -> None:
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
                    algorithm=DPSGDAdapter(
                        trainer=LinearTrainer(value),
                        tensor_schema=schema,
                        local_steps=1,
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


def test_two_nodes_complete_pair_commit_without_group_barrier() -> None:
    schema = TensorSchema(tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),))
    channel = SharedPairChannel.create()
    commits: dict[str, list[RoundCommit]] = {"peer-0": [], "peer-1": []}
    publisher = RecordingPublisher([])

    async def run() -> None:
        engines: list[GossipEngine] = []
        for key, value in (("peer-0", 1.0), ("peer-1", 3.0)):
            algorithm = DPSGDAdapter(
                trainer=LinearTrainer(value), tensor_schema=schema, local_steps=1
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
            np.array_equal(record.post_mix.weights["weight"], np.array([3.0]))
            for records in commits.values()
            for record in records
        )
        assert publisher.rounds == [0]

    asyncio.run(run())


def test_evaluation_runs_every_five_rounds_and_on_final_round() -> None:
    schema = TensorSchema(tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),))
    channel = SharedPairChannel.create()
    evaluations: dict[str, list[EvaluationMetrics]] = {"peer-0": [], "peer-1": []}

    async def run() -> None:
        engines: list[GossipEngine] = []
        for key, value in (("peer-0", 1.0), ("peer-1", 3.0)):
            algorithm = DPSGDAdapter(
                trainer=LinearTrainer(value), tensor_schema=schema, local_steps=1
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


def test_four_in_memory_nodes_reduce_a_shared_convex_objective() -> None:
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
                algorithm=DPSGDAdapter(
                    trainer=trainer, tensor_schema=schema, local_steps=1
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


def test_two_final_consensus_stages_exactly_average_four_nodes() -> None:
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
                algorithm=DPSGDAdapter(
                    trainer=trainer,
                    tensor_schema=schema,
                    local_steps=1,
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


def test_round_commit_checksum_is_independent_of_transport() -> None:
    tensors = {"weight": np.array([2.0], dtype=np.float32)}
    bundle = TrainedWeightsBundle(
        round_id=0, tensors=tensors, checksum=checksum_tensors(tensors)
    )

    assert (
        bundle.checksum
        == "fd734a524f800f74e9b9ac0d0134f90237a95dbc2854a847382d01fed960bfb4"
    )
