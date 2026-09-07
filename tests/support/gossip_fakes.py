from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from dromeus.algorithms.base import UpdateBundle
from dromeus.algorithms.codec import (
    NamedSafetensorsUpdateBundleCodec,
)
from dromeus.algorithms.dpsgd import DPSGDAdapter
from dromeus.gossip.interfaces import (
    PairCommitError,
    PairExchangeResult,
    RunFailure,
)
from dromeus.manifests.models import (
    TensorSchema,
    UpdateCodecBinding,
)
from dromeus.protocol.models import (
    Envelope,
)
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
        return abs(float(self._weights["weight"][0])), 0.5

    @property
    def local_loss(self) -> float:
        return 0.25


class ConvexTrainer(LinearTrainer):
    def train_local_steps(self, step_count: int) -> None:
        for _ in range(step_count):
            self._weights["weight"] *= np.float32(0.5)


class NoLoCoConvexTrainer(LinearTrainer):
    def train_local_steps(self, step_count: int) -> None:
        assert step_count == 50
        self._weights["weight"] *= np.float32(0.5)


class SlowEvaluationTrainer(LinearTrainer):
    def evaluate(self) -> tuple[float, float]:
        time.sleep(0.2)
        return super().evaluate()


class InvalidObservationTrainer(LinearTrainer):
    @property
    def local_loss(self) -> float:
        return float("nan")

    def evaluate(self) -> tuple[float, float]:
        raise RuntimeError("evaluation unavailable")


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
    ) -> PairExchangeResult:
        peer_bundle = await self.channel.exchange_update(
            self.local, peer, round_id, bundle
        )
        return PairExchangeResult(bundle=peer_bundle)

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


class DivergentPairTransport(InMemoryPairTransport):
    async def exchange_round_committed(
        self,
        *,
        peer: str,
        round_id: int,
        state_checksum: str,
    ) -> None:
        await self.channel.exchange_committed(
            self.local, peer, round_id, state_checksum
        )


class HangingPairTransport(InMemoryPairTransport):
    async def exchange_update(
        self,
        *,
        peer: str,
        round_id: int,
        bundle: UpdateBundle,
    ) -> PairExchangeResult:
        await asyncio.sleep(1)
        return cast(PairExchangeResult, None)


class StaticPairTransport:
    def __init__(self, peer_bundle: UpdateBundle) -> None:
        self.peer_bundle = peer_bundle

    async def exchange_update(
        self,
        *,
        peer: str,
        round_id: int,
        bundle: UpdateBundle,
    ) -> PairExchangeResult:
        return PairExchangeResult(bundle=self.peer_bundle)

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


class ReadinessGuardTransport(StaticPairTransport):
    def __init__(self, peer_bundle: UpdateBundle) -> None:
        super().__init__(peer_bundle)
        self.calls: list[str] = []
        self.ready_exchanged = False

    async def exchange_update_ready(
        self,
        *,
        peer: str,
        round_id: int,
        bundle_checksum: str,
    ) -> str:
        self.calls.append("ready")
        self.ready_exchanged = True
        return await super().exchange_update_ready(
            peer=peer,
            round_id=round_id,
            bundle_checksum=bundle_checksum,
        )

    async def exchange_update(
        self,
        *,
        peer: str,
        round_id: int,
        bundle: UpdateBundle,
    ) -> PairExchangeResult:
        self.calls.append("update")
        if not self.ready_exchanged:
            raise PairCommitError("bulk transfer started before peer readiness")
        return await super().exchange_update(
            peer=peer,
            round_id=round_id,
            bundle=bundle,
        )


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


class DelayedPairReceiver(StubPairReceiver):
    def __init__(self, envelope: Envelope, *, timeout_count: int) -> None:
        super().__init__(envelope)
        self.timeout_count = timeout_count
        self.receive_count = 0

    async def receive(
        self, channel: object, *, timeout_seconds: float
    ) -> Envelope:
        self.receive_count += 1
        if self.receive_count <= self.timeout_count:
            raise TimeoutError
        return await super().receive(channel, timeout_seconds=timeout_seconds)


class StubPairSender:
    async def send(self, *args: object, **kwargs: object) -> None:
        return None


class RecordingBundleCodec:
    def __init__(
        self,
        delegate: NamedSafetensorsUpdateBundleCodec,
        *,
        validation_error: bool = False,
    ) -> None:
        self.delegate = delegate
        self.validation_error = validation_error
        self.released: list[str] = []

    def encode(
        self,
        *,
        round_id: int,
        artifacts: Mapping[str, Mapping[str, np.ndarray]],
        codec_bindings: Mapping[str, UpdateCodecBinding] | None = None,
    ) -> UpdateBundle:
        return self.delegate.encode(
            round_id=round_id,
            artifacts=artifacts,
            codec_bindings=codec_bindings,
        )

    def decode(
        self,
        bundle: UpdateBundle,
        codec_bindings: Mapping[str, UpdateCodecBinding] | None = None,
    ) -> dict[str, dict[str, np.ndarray]]:
        if self.validation_error:
            raise ValueError("forced validation error")
        return self.delegate.decode(bundle, codec_bindings=codec_bindings)

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
    async def exchange_update(
        self,
        *,
        peer: str,
        round_id: int,
        bundle: UpdateBundle,
    ) -> PairExchangeResult:
        exchange = await super().exchange_update(
            peer=peer,
            round_id=round_id,
            bundle=bundle,
        )
        return PairExchangeResult(
            bundle=exchange.bundle,
            transfer_id="transfer-0",
            retry_count=2,
        )


@dataclass
class RecordingFailureBroadcaster:
    failures: list[RunFailure]

    async def broadcast_run_failed(self, failure: RunFailure) -> None:
        self.failures.append(failure)


def make_algorithm(
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
        bundle_codec=NamedSafetensorsUpdateBundleCodec(
            artifact_root=artifact_root,
            run_id="test-run",
            manifest_hash="0" * 64,
            sender_public_key=key,
            algorithm_id="d-psgd",
            artifact_schemas={"trained_weights": schema},
        ),
    )
