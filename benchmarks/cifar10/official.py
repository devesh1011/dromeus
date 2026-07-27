"""Frozen configuration guard for official M1 CIFAR-10 runs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from benchmarks.cifar10.fedavg_reference import FedAvgConfig
from dromeus.manifests.models import DraftRunSpec


class OfficialBenchmarkError(ValueError):
    """Official benchmark inputs are incomplete or differ from frozen settings."""


class FrozenBenchmarkPlan(BaseModel):
    """Pilot-backed settings shared by three FedAvg and D-PSGD runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_seeds: tuple[int, int, int]
    partition_seed: int
    local_steps: Annotated[int, Field(gt=0)]
    round_count: Annotated[int, Field(gt=0)]
    learning_rate: Annotated[float, Field(gt=0)]
    pilot_artifact: Path
    batch_size: Literal[32] = 32
    device: Literal["cpu"] = "cpu"
    augment: Literal[True] = True
    worker_count: Literal[4] = 4
    dpsgd_transport: Literal["axl"] = "axl"

    @model_validator(mode="after")
    def distinct_seeds(self) -> Self:
        if len(set(self.benchmark_seeds)) != 3:
            raise ValueError("benchmark seeds must be distinct")
        return self

    def fedavg_configs(self) -> tuple[FedAvgConfig, FedAvgConfig, FedAvgConfig]:
        """Build matching centralized controls without involving AXL."""
        return tuple(
            FedAvgConfig(
                local_steps=self.local_steps,
                round_count=self.round_count,
                learning_rate=self.learning_rate,
                trainer_seed=seed,
                batch_size=self.batch_size,
                device=self.device,
                augment=self.augment,
            )
            for seed in self.benchmark_seeds
        )

    def validate_dpsgd_draft(self, draft: DraftRunSpec, *, seed: int) -> None:
        """Reject one D-PSGD run that differs from frozen comparison settings."""
        if seed not in self.benchmark_seeds:
            raise OfficialBenchmarkError("D-PSGD seed is not frozen")
        signature = (
            draft.local_steps,
            draft.round_count,
            draft.learning_rate,
            draft.dataset.iid_partition_seed,
            draft.peer_scheduler_seed,
            draft.algorithm_id,
        )
        expected = (
            self.local_steps,
            self.round_count,
            self.learning_rate,
            self.partition_seed,
            seed,
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
    return plan


__all__ = [
    "FrozenBenchmarkPlan",
    "OfficialBenchmarkError",
    "load_frozen_benchmark_plan",
]
