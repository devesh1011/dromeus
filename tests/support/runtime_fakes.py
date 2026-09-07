from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
from safetensors.numpy import (
    load_file as _load_file,  # pyright: ignore[reportUnknownVariableType]
)

from dromeus.gossip.interfaces import RoundCommit
from dromeus.manifests.models import SealedManifest, TensorSchema
from dromeus.membership.formation import (
    FormationResult,
)
from dromeus.persistence.run_store import RunStore
from dromeus.runtime import (
    FailureConfig,
    NodeRuntime,
    TrainingConfig,
)
from support.in_memory_transport import (
    InMemoryNetwork,
    InMemoryTransport,
)

_load_checkpoint = cast(Callable[[str], dict[str, np.ndarray]], _load_file)


class RuntimeTrainer:
    def __init__(self) -> None:
        self._weights = {"layer.weight": np.zeros((2, 2), dtype=np.float32)}

    def load_checkpoint(self, path: Path) -> None:
        self._weights = {
            name: np.ascontiguousarray(value)
            for name, value in _load_checkpoint(str(path)).items()
        }

    def train_local_steps(self, step_count: int) -> None:
        self._weights["layer.weight"] += np.float32(step_count)

    def weights(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self._weights.items()}

    def load_weights(self, weights: dict[str, np.ndarray]) -> None:
        self._weights = {name: value.copy() for name, value in weights.items()}

    def evaluate(self) -> tuple[float, float]:
        return 0.0, 0.5

    @property
    def local_loss(self) -> float:
        return 0.25


class LifecycleRuntime(NodeRuntime):
    def __init__(self, result: FormationResult) -> None:
        self.result = result
        self.events: list[str] = []
        self._failure = FailureConfig.for_run_root(result.checkpoint_path.parent)
        self._training = None

    async def initiate(
        self,
        *,
        bootstrap_uri: str,
        checkpoint_path: Path,
        tensor_schema: TensorSchema,
    ) -> FormationResult:
        self.events.append("initiate")
        return self.result

    def configure_training(self, training: TrainingConfig) -> None:
        self.events.append("configure")

    async def fail_before_run(self, error: BaseException) -> None:
        self.events.append(f"fail:{error}")

    async def run(self) -> tuple[RoundCommit, ...]:
        self.events.append("run")
        return ()

    async def stop(self) -> None:
        self.events.append("stop")


class RecordingInMemoryTransport(InMemoryTransport):
    def __init__(self, *, network: InMemoryNetwork, public_key: str) -> None:
        super().__init__(network=network, public_key=public_key)
        self.sent_payloads: list[tuple[str, bytes]] = []

    async def send(self, destination: str, payload: bytes) -> None:
        self.sent_payloads.append((destination, payload))
        await super().send(destination, payload)


class FailingRunStore(RunStore):
    def initialize(self, manifest: SealedManifest) -> str:
        raise OSError("run store unavailable")


class BlockingTransport:
    def __init__(self) -> None:
        self.send_started = asyncio.Event()
        self.release_send = asyncio.Event()

    async def local_public_key(self) -> str:
        return "peer-0"

    async def send(self, destination: str, payload: bytes) -> None:
        self.send_started.set()
        await self.release_send.wait()

    async def recv(self, timeout_seconds: float) -> None:
        return None
