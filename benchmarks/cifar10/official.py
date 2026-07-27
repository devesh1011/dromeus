"""Frozen configuration guard for official M1 CIFAR-10 runs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, Self, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from benchmarks.cifar10.fedavg_reference import FedAvgConfig
from dromeus.manifests.canonical import parse_draft_yaml
from dromeus.manifests.models import (
    DatasetContract,
    DraftRunSpec,
    EnvironmentFingerprint,
    Identifier,
    Sha256,
)
from dromeus.node import NodeConfig, NodeRole, load_node_config


class OfficialBenchmarkError(ValueError):
    """Official benchmark inputs are incomplete or differ from frozen settings."""


class PilotEvidence(BaseModel):
    """Measured pilot settings required before benchmark values become frozen."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["complete"]
    model_definition_hash: Sha256
    dataset: DatasetContract
    data_source: Literal["torchvision-cifar10"]
    local_steps: Annotated[int, Field(gt=0)]
    round_count: Annotated[int, Field(gt=0)]
    learning_rate: Annotated[float, Field(gt=0)]
    node_ids: tuple[str, str, str, str]
    data_artifact_sha256: tuple[Sha256, Sha256, Sha256, Sha256]


class FrozenBenchmarkPlan(BaseModel):
    """Pilot-backed settings shared by three FedAvg and D-PSGD runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_seeds: tuple[int, int, int]
    local_steps: Annotated[int, Field(gt=0)]
    round_count: Annotated[int, Field(gt=0)]
    learning_rate: Annotated[float, Field(gt=0)]
    model_id: Identifier
    model_definition_hash: Sha256
    dataset: DatasetContract
    environment: EnvironmentFingerprint
    data_source: Literal["torchvision-cifar10"]
    optimizer: Literal["sgd"] = "sgd"
    weight_decay: Annotated[float, Field(ge=0.0, le=0.0)] = 0.0
    max_payload_bytes: Annotated[int, Field(gt=0)]
    max_retries: Annotated[int, Field(ge=0)]
    retry_timeout_seconds: Annotated[float, Field(gt=0)]
    pilot_artifact: Path
    cloud_provider: Literal["aws"]
    worker_instance_type: Annotated[str, Field(min_length=1)]
    worker_regions: tuple[str, str, str, str]
    bootstrap_region: Annotated[str, Field(min_length=1)]
    worker_root_volume_gib: Annotated[int, Field(gt=0)]
    parameter_count: Literal[5514] = 5514
    learning_rate_schedule: Literal["constant"] = "constant"
    checkpoint_interval: Literal[1] = 1
    receive_poll_seconds: Annotated[float, Field(ge=0.1, le=0.1)] = 0.1
    per_peer_in_flight: Literal[1] = 1
    batch_size: Literal[32] = 32
    evaluation_interval: Literal[5] = 5
    device: Literal["cpu"] = "cpu"
    augment: Literal[True] = True
    worker_count: Literal[4] = 4
    dpsgd_transport: Literal["axl"] = "axl"

    @model_validator(mode="after")
    def distinct_seeds(self) -> Self:
        if len(set(self.benchmark_seeds)) != 3:
            raise ValueError("benchmark seeds must be distinct")
        if len(set(self.worker_regions)) < 2:
            raise ValueError("official workers must span at least two regions")
        return self

    def fedavg_configs(self) -> tuple[FedAvgConfig, FedAvgConfig, FedAvgConfig]:
        """Build matching centralized controls without involving AXL."""
        configs = tuple(
            FedAvgConfig(
                local_steps=self.local_steps,
                round_count=self.round_count,
                learning_rate=self.learning_rate,
                model_id=self.model_id,
                model_definition_hash=self.model_definition_hash,
                dataset=self.dataset,
                environment=self.environment,
                data_source=self.data_source,
                test_sample_count=10_000,
                evaluation_interval=self.evaluation_interval,
                trainer_seed=seed,
                batch_size=self.batch_size,
                device=self.device,
                augment=self.augment,
            )
            for seed in self.benchmark_seeds
        )
        return configs[0], configs[1], configs[2]

    def validate_dpsgd_draft(self, draft: DraftRunSpec, *, seed: int) -> None:
        """Reject one D-PSGD run that differs from frozen comparison settings."""
        if seed not in self.benchmark_seeds:
            raise OfficialBenchmarkError("D-PSGD seed is not frozen")
        signature = (
            draft.model_id,
            draft.model_definition_hash,
            draft.dataset,
            draft.environment,
            draft.local_steps,
            draft.round_count,
            draft.optimizer,
            draft.learning_rate,
            draft.peer_scheduler_seed,
            draft.transport.max_payload_bytes,
            draft.transport.max_retries,
            draft.transport.retry_timeout_seconds,
            draft.algorithm_id,
        )
        expected = (
            self.model_id,
            self.model_definition_hash,
            self.dataset,
            self.environment,
            self.local_steps,
            self.round_count,
            self.optimizer,
            self.learning_rate,
            seed,
            self.max_payload_bytes,
            self.max_retries,
            self.retry_timeout_seconds,
            "dpsgd-v1",
        )
        if signature != expected:
            raise OfficialBenchmarkError(
                "D-PSGD draft does not match frozen configuration"
            )


def load_frozen_benchmark_plan(path: Path) -> FrozenBenchmarkPlan:
    """Load a closed plan only after its documented pilot artifact exists."""
    value = cast(object, yaml.safe_load(path.read_text(encoding="utf-8")))
    plan = FrozenBenchmarkPlan.model_validate(value)
    pilot_artifact = plan.pilot_artifact
    if not pilot_artifact.is_absolute():
        plan = plan.model_copy(
            update={"pilot_artifact": (path.parent / pilot_artifact).resolve()}
        )
    if not plan.pilot_artifact.is_file():
        raise OfficialBenchmarkError("documented pilot artifact is missing")
    try:
        pilot = PilotEvidence.model_validate_json(
            plan.pilot_artifact.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise OfficialBenchmarkError("documented pilot artifact is invalid") from error
    if (
        pilot.model_definition_hash,
        pilot.dataset,
        pilot.data_source,
        pilot.local_steps,
        pilot.round_count,
        pilot.learning_rate,
    ) != (
        plan.model_definition_hash,
        plan.dataset,
        plan.data_source,
        plan.local_steps,
        plan.round_count,
        plan.learning_rate,
    ):
        raise OfficialBenchmarkError(
            "documented pilot does not match frozen configuration"
        )
    return plan


def prepare_dpsgd_node_configs(
    *,
    plan_path: Path,
    draft_path: Path,
    seed: int,
    node_config_paths: Sequence[Path],
) -> tuple[NodeConfig, ...]:
    """Validate one frozen four-node D-PSGD launch before deployment."""
    plan = load_frozen_benchmark_plan(plan_path)
    draft = parse_draft_yaml(draft_path)
    plan.validate_dpsgd_draft(draft, seed=seed)
    configs = tuple(load_node_config(path) for path in node_config_paths)
    if len(configs) != 4:
        raise OfficialBenchmarkError("official D-PSGD requires four node configs")
    if (
        sum(config.role is NodeRole.INITIATOR for config in configs) != 1
        or sum(config.role is NodeRole.PARTICIPANT for config in configs) != 3
    ):
        raise OfficialBenchmarkError(
            "official D-PSGD requires one initiator and three participants"
        )
    if any(config.benchmark_seed != seed for config in configs):
        raise OfficialBenchmarkError("node config benchmark seed is not frozen")
    if len({config.bootstrap_uri for config in configs}) != 1:
        raise OfficialBenchmarkError("node configs do not share one bootstrap URI")
    if any(parse_draft_yaml(config.draft_path) != draft for config in configs):
        raise OfficialBenchmarkError("node configs do not share the frozen draft")
    return configs


__all__ = [
    "FrozenBenchmarkPlan",
    "OfficialBenchmarkError",
    "PilotEvidence",
    "load_frozen_benchmark_plan",
    "prepare_dpsgd_node_configs",
]
