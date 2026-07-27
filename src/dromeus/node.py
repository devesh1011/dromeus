"""Internal Dromeus node entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, cast
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from dromeus.manifests.canonical import (
    canonical_json,
    parse_draft_yaml,
)
from dromeus.manifests.models import Invitation
from dromeus.membership.protocol import create_invitation
from dromeus.runtime import NodeRuntime, prepare_cifar_training
from dromeus.telemetry.events import JsonlEventSink, emit_event
from dromeus.telemetry.metrics import JsonlMetricsPublisher
from dromeus.transport.axl import AXLBridgeConfig, AXLTransport
from dromeus.transport.transfer import ArtifactStore


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
    cifar_root: Path
    invitation_path: Path
    bootstrap_uri: Annotated[str, Field(min_length=1)]
    benchmark_seed: int
    invitation_timeout_seconds: Annotated[float, Field(gt=0)] = 300.0

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


def load_node_config(path: Path) -> NodeConfig:
    """Load one closed node configuration from YAML."""
    value = cast(object, yaml.safe_load(path.read_text(encoding="utf-8")))
    return NodeConfig.model_validate(value)


async def run_node(config: NodeConfig) -> None:
    """Form, train, persist, and stop one production AXL-backed CIFAR node."""
    draft = parse_draft_yaml(config.draft_path)
    config.run_root.mkdir(parents=True, exist_ok=True)
    event_sink = JsonlEventSink(config.run_root / "logs" / "dromeus.jsonl")
    emit_event(
        "node_start",
        run_id=draft.run_id,
        sink=event_sink,
        role=config.role,
        benchmark_seed=config.benchmark_seed,
    )
    prepared_training = await asyncio.to_thread(
        prepare_cifar_training,
        draft=draft,
        cifar_root=config.cifar_root,
        benchmark_seed=config.benchmark_seed,
    )
    transport = AXLTransport(AXLBridgeConfig(base_url=config.axl_bridge_url))
    local_key = await transport.local_public_key()
    runtime = NodeRuntime(
        transport=transport,
        draft=draft,
        environment=draft.environment,
        dataset=draft.dataset,
        artifact_store=ArtifactStore(config.run_root / "formation-artifacts"),
        event_sink=event_sink,
    )
    try:
        if config.role is NodeRole.INITIATOR:
            checkpoint = await asyncio.to_thread(
                prepared_training.create_initial_checkpoint,
                config.run_root / "initial.safetensors",
            )
            invitation = create_invitation(
                draft=draft,
                initiator_public_key=local_key,
                bootstrap_uri=config.bootstrap_uri,
            )
            await asyncio.to_thread(
                _write_invitation, config.invitation_path, invitation
            )
            result = await runtime.initiate(
                bootstrap_uri=config.bootstrap_uri,
                checkpoint_path=checkpoint.path,
                tensor_schema=checkpoint.tensor_schema,
            )
        else:
            invitation = await _read_invitation(
                config.invitation_path,
                timeout_seconds=config.invitation_timeout_seconds,
            )
            if invitation.bootstrap_uri != config.bootstrap_uri:
                raise ValueError("invitation bootstrap URI does not match node config")
            result = await runtime.join(invitation=invitation)

        run_store_root = config.run_root / "run-store"
        await _write_topology_snapshot(
            transport,
            run_store_root / "topology-ready.json",
        )
        emit_event(
            "benchmark_node_ready",
            run_id=result.manifest.run_id,
            manifest_hash=result.manifest_hash,
            node_id=local_key,
            sink=event_sink,
            benchmark_seed=config.benchmark_seed,
            transport="axl",
        )
        metrics = JsonlMetricsPublisher(
            sink=event_sink,
            run_id=result.manifest.run_id,
            manifest_hash=result.manifest_hash,
            node_id=local_key,
        )
        training_config = await asyncio.to_thread(
            prepared_training.build_config,
            result=result,
            local_public_key=local_key,
            run_root=config.run_root,
            metrics_publisher=metrics,
        )
        runtime.configure_training(training_config)
        commits = await runtime.run()
        await _write_topology_snapshot(
            transport,
            run_store_root / "topology-complete.json",
        )
        emit_event(
            "node_complete",
            run_id=result.manifest.run_id,
            manifest_hash=result.manifest_hash,
            node_id=local_key,
            sink=event_sink,
            committed_rounds=len(commits),
        )
    finally:
        await runtime.stop()


def _write_invitation(path: Path, invitation: Invitation) -> None:
    _atomic_write(path, canonical_json(invitation))


async def _read_invitation(path: Path, *, timeout_seconds: float) -> Invitation:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            data = await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError:
            await asyncio.sleep(0.2)
            continue
        return Invitation.model_validate_json(data)
    raise TimeoutError(f"timed out waiting for invitation: {path}")


async def _write_topology_snapshot(transport: AXLTransport, path: Path) -> None:
    topology = await transport.topology()
    payload = (
        json.dumps(topology, allow_nan=False, sort_keys=True, indent=2) + "\n"
    ).encode()
    await asyncio.to_thread(_atomic_write, path, payload)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Dromeus node")
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path: Path = args.config
    if not config_path.is_file():
        _parser().error(f"config file does not exist: {config_path}")

    config = load_node_config(config_path)
    asyncio.run(run_node(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
