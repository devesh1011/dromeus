"""Run one Dromeus node: the public `Node` and its machine-local config."""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self, cast
from urllib.parse import urlparse

import torch
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from dromeus.application import LocalData, check_environment, load_draft, prepare_task
from dromeus.errors import ConfigError, FormationError
from dromeus.manifests.canonical import (
    canonical_json,
)
from dromeus.manifests.models import (
    DraftRunSpec,
    Invitation,
    SealedManifestExpectation,
)
from dromeus.membership.formation import FormationResult, create_invitation
from dromeus.persistence.files import atomic_write_bytes
from dromeus.runtime import (
    FailureConfig,
    InitiatorFormation,
    NodeRunResult,
    NodeRuntime,
    ParticipantFormation,
    TrainingConfig,
    WorkloadFactory,
    private_preparation_for_node,
)
from dromeus.telemetry.events import (
    EventSink,
    JsonlEventSink,
    emit_event,
    event_log_path,
)
from dromeus.telemetry.liveness import hold_node_lock
from dromeus.telemetry.metrics import JsonlMetricsPublisher
from dromeus.training.task import TrainingTask
from dromeus.transport.axl import AXLBridgeConfig, AXLTransport
from dromeus.transport.interface import TransportError


class NodeRole(StrEnum):
    INITIATOR = "initiator"
    PARTICIPANT = "participant"


class NodeConfig(BaseModel):
    """Validated machine-local inputs for one non-interactive node."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: NodeRole
    draft_path: Path
    axl_bridge_url: Annotated[str, Field(min_length=1)]
    run_root: Path
    invitation_path: Path
    bootstrap_uri: Annotated[str, Field(min_length=1)]
    invitation_timeout_seconds: Annotated[float, Field(gt=0)] = 300.0
    enrollment_window_seconds: Annotated[float, Field(gt=0)] | None = None
    manifest_expectation: SealedManifestExpectation | None = None

    @model_validator(mode="after")
    def enrollment_window_is_initiator_only(self) -> Self:
        if (
            self.enrollment_window_seconds is not None
            and self.role is not NodeRole.INITIATOR
        ):
            raise ValueError("enrollment_window_seconds applies to the initiator only")
        return self

    @field_validator("axl_bridge_url")
    @classmethod
    def bridge_must_be_loopback(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "http" or parsed.hostname not in {
            "127.0.0.1",
            "::1",
            "localhost",
        }:
            raise ValueError("AXL bridge URL must use HTTP on loopback")
        return value.rstrip("/")


TrainingDecorator = Callable[
    [FormationResult, str, TrainingConfig],
    TrainingConfig,
]


def load_node_config(path: Path) -> NodeConfig:
    """Load one closed node configuration from YAML."""
    try:
        value = cast(object, yaml.safe_load(path.read_text(encoding="utf-8")))
        return NodeConfig.model_validate(value)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise ConfigError(f"invalid node config {path}: {error}") from error


@dataclass(frozen=True)
class RunResult:
    """What a node reports after the group finishes every round."""

    run_id: str
    manifest_hash: str
    committed_rounds: int


class Node:
    """One participant in a run, set up from its node config file.

    The config names this machine's AXL bridge, role, draft and run directory.
    Create it with `Node.from_file`, then `await node.run(task, local=...)`.
    """

    def __init__(self, config: NodeConfig) -> None:
        self._config = config

    @classmethod
    def from_file(cls, path: Path | str) -> Node:
        return cls(load_node_config(Path(path)))

    async def run(self, task: TrainingTask, *, local: LocalData) -> RunResult:
        """Join the group, train every round, and return when the run completes.

        Before contacting AXL, checks the draft against this task and against
        the installed Dromeus and PyTorch versions.
        """
        result = await run_node(
            self._config,
            prepare_training=lambda draft: prepare_task(
                draft=draft, task=task, local=local
            ),
        )
        return RunResult(
            run_id=result.formation.manifest.run_id,
            manifest_hash=result.formation.manifest_hash,
            committed_rounds=len(result.commits),
        )


async def run_node(
    config: NodeConfig,
    *,
    prepare_training: WorkloadFactory,
    training_decorator: TrainingDecorator | None = None,
    on_ready: Callable[[FormationResult, str, EventSink], None | Awaitable[None]]
    | None = None,
) -> NodeRunResult:
    """Run a developer-supplied workload; this module never selects models or data.

    The node holds its run root's lock for the whole run, so `dromeus watch`
    can tell when it stops, even without a logged result.
    """
    with hold_node_lock(config.run_root):
        return await _run_node(
            config,
            prepare_training=prepare_training,
            training_decorator=training_decorator,
            on_ready=on_ready,
        )


async def _run_node(
    config: NodeConfig,
    *,
    prepare_training: WorkloadFactory,
    training_decorator: TrainingDecorator | None,
    on_ready: Callable[[FormationResult, str, EventSink], None | Awaitable[None]]
    | None,
) -> NodeRunResult:
    draft, event_sink = await asyncio.to_thread(
        _prepare_node_start,
        config,
    )
    check_environment(draft)
    prepared_training = await asyncio.to_thread(prepare_training, draft)
    prepared_training.validate_draft(draft)
    private_trainer = private_preparation_for_node(draft, prepared_training)
    local_tensor_schema = prepared_training.tensor_schema
    transport = AXLTransport(AXLBridgeConfig(base_url=config.axl_bridge_url))
    local_key = await transport.local_public_key()
    failure = FailureConfig.for_run_root(config.run_root)
    runtime = NodeRuntime(
        transport=transport,
        draft=draft,
        environment=draft.environment,
        dataset=draft.dataset,
        artifact_root=config.run_root / "formation-artifacts",
        event_sink=event_sink,
        failure=failure,
        local_tensor_schema=local_tensor_schema,
        private_trainer=private_trainer,
    )
    event_sink = runtime.event_sink or event_sink

    if config.role is NodeRole.INITIATOR:
        checkpoint = await asyncio.to_thread(
            prepared_training.create_initial_checkpoint,
            config.run_root / "initial.safetensors",
        )
        expires_at = (
            None
            if config.enrollment_window_seconds is None
            else datetime.now(UTC)
            + timedelta(seconds=config.enrollment_window_seconds)
        )
        invitation = create_invitation(
            draft=draft,
            initiator_public_key=local_key,
            bootstrap_uri=config.bootstrap_uri,
            enrollment_expires_at=expires_at,
        )
        await asyncio.to_thread(_write_invitation, config.invitation_path, invitation)
        formation = InitiatorFormation(
            bootstrap_uri=config.bootstrap_uri,
            checkpoint_path=checkpoint.path,
            tensor_schema=checkpoint.tensor_schema,
            enrollment_expires_at=expires_at,
        )
    else:
        invitation = await _read_invitation(
            config.invitation_path,
            timeout_seconds=config.invitation_timeout_seconds,
        )
        if invitation.bootstrap_uri != config.bootstrap_uri:
            raise ConfigError("invitation bootstrap URI does not match node config")
        formation = ParticipantFormation(invitation=invitation)

    run_store_root = failure.run_store.root

    def build_training(result: FormationResult) -> TrainingConfig:
        metrics = JsonlMetricsPublisher(
            sink=event_sink,
            run_id=result.manifest.run_id,
            manifest_hash=result.manifest_hash,
            node_id=local_key,
        )
        training = prepared_training.build_config(
            result=result,
            local_public_key=local_key,
            run_root=config.run_root,
            metrics_publisher=metrics,
        )
        if training_decorator is not None:
            return training_decorator(result, local_key, training)
        return training

    async def record_ready(result: FormationResult) -> None:
        await _write_topology_snapshot(
            transport,
            run_store_root / "topology-ready.json",
            sink=event_sink,
            run_id=result.manifest.run_id,
        )
        await asyncio.to_thread(
            emit_event,
            "node_ready",
            run_id=result.manifest.run_id,
            manifest_hash=result.manifest_hash,
            node_id=local_key,
            sink=event_sink,
        )
        if on_ready is not None:
            notification = on_ready(result, local_key, event_sink)
            if inspect.isawaitable(notification):
                await notification

    async def record_complete(result: NodeRunResult) -> None:
        await _write_topology_snapshot(
            transport,
            run_store_root / "topology-complete.json",
            sink=event_sink,
            run_id=result.formation.manifest.run_id,
        )
        await asyncio.to_thread(
            emit_event,
            "node_complete",
            run_id=result.formation.manifest.run_id,
            manifest_hash=result.formation.manifest_hash,
            node_id=local_key,
            sink=event_sink,
            committed_rounds=len(result.commits),
        )

    return await runtime.run_to_completion(
        formation=formation,
        training_factory=build_training,
        manifest_expectation=config.manifest_expectation,
        ready_hook=record_ready,
        completion_hook=record_complete,
    )


def _write_invitation(path: Path, invitation: Invitation) -> None:
    _atomic_write(path, canonical_json(invitation))


def _prepare_node_start(
    config: NodeConfig,
) -> tuple[DraftRunSpec, JsonlEventSink]:
    draft = load_draft(config.draft_path)
    config.run_root.mkdir(parents=True, exist_ok=True)
    event_sink = JsonlEventSink(event_log_path(config.run_root))
    emit_event(
        "node_start",
        run_id=draft.run_id,
        sink=event_sink,
        role=config.role,
        pytorch_build=torch.__version__,  # drafts pin the release, not the build
    )
    return draft, event_sink


async def _read_invitation(path: Path, *, timeout_seconds: float) -> Invitation:
    """Wait for a complete invitation file.

    Users deliver the invitation by copying it, so the file can exist before all
    of its bytes arrive. An empty or unparseable file means "not yet".
    """
    deadline = time.monotonic() + timeout_seconds
    problem = "file does not exist"
    while time.monotonic() < deadline:
        try:
            data = await asyncio.to_thread(path.read_bytes)
            return Invitation.model_validate_json(data)
        except FileNotFoundError:
            problem = "file does not exist"
        except ValidationError as error:
            problem = f"file is incomplete or invalid: {error.errors()[0]['msg']}"
        await asyncio.sleep(0.2)
    raise FormationError(f"timed out waiting for invitation {path}: {problem}")


async def _write_topology_snapshot(
    transport: AXLTransport, path: Path, *, sink: EventSink, run_id: str
) -> None:
    """Record AXL's view of the network, as evidence only.

    A failed snapshot is logged, never raised: it must not fail a node that has
    formed, or one that has committed every round.
    """
    try:
        topology = await transport.topology()
    except TransportError as error:
        await asyncio.to_thread(
            emit_event,
            "topology_snapshot_failed",
            run_id=run_id,
            sink=sink,
            snapshot=path.name,
            error=str(error),
            cause=repr(error.__cause__),
        )
        return
    payload = (
        json.dumps(topology, allow_nan=False, sort_keys=True, indent=2) + "\n"
    ).encode()
    await asyncio.to_thread(_atomic_write, path, payload)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, payload)
