from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np

from dromeus.algorithms.base import TrainedWeightsBundle
from dromeus.algorithms.dpsgd import DPSGDAdapter, checksum_tensors
from dromeus.gossip.engine import GossipEngine, RoundCommit
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


def test_two_nodes_complete_pair_commit_without_group_barrier() -> None:
    schema = TensorSchema(tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),))
    channel = SharedPairChannel.create()
    commits: dict[str, list[RoundCommit]] = {"peer-0": [], "peer-1": []}

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
                )
            )
        await asyncio.gather(*(engine.run() for engine in engines))
        assert all(len(records) == 1 for records in commits.values())
        assert all(
            np.array_equal(record.post_mix.weights["weight"], np.array([3.0]))
            for records in commits.values()
            for record in records
        )

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
