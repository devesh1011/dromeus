"""Node runtime lifecycle."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from dromeus.manifests.models import (
    DatasetContract,
    DraftRunSpec,
    EnvironmentFingerprint,
    Invitation,
    TensorSchema,
)
from dromeus.membership.protocol import FormationProtocol, FormationResult
from dromeus.transport.base import AsyncTransport
from dromeus.transport.transfer import ArtifactStore


class NodeRuntimeError(RuntimeError):
    """The node runtime lifecycle was used out of order."""


class NodeState(StrEnum):
    CREATED = "created"
    FORMING = "forming"
    READY = "ready"
    FAILED = "failed"
    STOPPED = "stopped"


class NodeRuntime:
    """Own formation tasks and retain them for the running node lifetime."""

    def __init__(
        self,
        *,
        transport: AsyncTransport,
        draft: DraftRunSpec,
        environment: EnvironmentFingerprint,
        dataset: DatasetContract,
        artifact_store: ArtifactStore,
    ) -> None:
        self._formation = FormationProtocol(
            transport=transport,
            draft=draft,
            environment=environment,
            dataset=dataset,
            transport_limits=draft.transport,
            artifact_store=artifact_store,
        )
        self._state = NodeState.CREATED
        self._result: FormationResult | None = None

    @property
    def state(self) -> NodeState:
        return self._state

    @property
    def formation_result(self) -> FormationResult:
        if self._result is None:
            raise NodeRuntimeError("node has not completed formation")
        return self._result

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

    async def stop(self) -> None:
        if self._state is NodeState.STOPPED:
            return
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

    def _ready(self, result: FormationResult) -> FormationResult:
        self._result = result
        self._state = NodeState.READY
        return result
