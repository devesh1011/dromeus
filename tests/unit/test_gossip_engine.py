from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
import pytest

from dromeus.algorithms.base import TrainedWeightsBundle
from dromeus.algorithms.dpsgd import DPSGDAdapter, checksum_tensors
from dromeus.gossip.engine import GossipEngine, PairCommitError, RoundCommit, RunFailure
from dromeus.gossip.scheduler import PeerScheduler
from dromeus.manifests.models import Tensor, TensorSchema


class LinearTrainer:
    def __init__(self, value: float) -> None:
        self._weights = {"weight": np.array([value], dtype=np.float32)}

    def train_local_steps(self, step_count: int) -> None:
        self._weights["weight"] += np.float32(step_count)

    def weights(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self._weights.items()}

    def load_weights(self, weights: dict[str, np.ndarray]) -> None:
        self._weights = {name: value.copy() for name, value in weights.items()}


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
class RecordingFailureBroadcaster:
    failures: list[RunFailure]

    async def broadcast_run_failed(self, failure: RunFailure) -> None:
        self.failures.append(failure)


def test_pair_timeout_fails_once_and_reports_diagnostics() -> None:
    failures: list[RunFailure] = []
    broadcasted: list[RunFailure] = []

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
        )

        with pytest.raises(PairCommitError, match="deadline"):
            await engine.run()
        assert engine.failure == failures[0]
        assert len(failures) == 1
        assert broadcasted == failures

    asyncio.run(run())


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


def test_round_commit_checksum_is_independent_of_transport() -> None:
    tensors = {"weight": np.array([2.0], dtype=np.float32)}
    bundle = TrainedWeightsBundle(
        round_id=0, tensors=tensors, checksum=checksum_tensors(tensors)
    )

    assert (
        bundle.checksum
        == "fd734a524f800f74e9b9ac0d0134f90237a95dbc2854a847382d01fed960bfb4"
    )
