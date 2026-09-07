"""CIFAR benchmark composition using the public Dromeus runtime interface."""

from __future__ import annotations

# These machine-local inputs and evidence events belong to the benchmark client.
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from benchmarks.workloads.cifar10.dataset import (
    PreparedCIFAR10Training as TrainingOwnedCIFAR,
)
from benchmarks.workloads.cifar10.dataset import prepare_training
from dromeus.manifests.canonical import canonical_hash
from dromeus.manifests.models import DraftRunSpec, TensorSchema
from dromeus.membership.formation import FormationResult
from dromeus.node import NodeConfig, TrainingDecorator, run_node
from dromeus.persistence.run_store import RunStore
from dromeus.runtime import MetricsService, TrainingConfig, build_algorithm
from dromeus.telemetry.events import EventSink
from dromeus.telemetry.evidence import BenchmarkNodeReadyEvidence, append_evidence
from dromeus.training.state import InitialCheckpoint


@dataclass(frozen=True, slots=True)
class PreparedCIFARTraining:
    """Runtime composition over the deep training-owned CIFAR interface."""

    _training: TrainingOwnedCIFAR
    _draft_hash: str

    @property
    def tensor_schema(self) -> TensorSchema | None:
        return None

    def validate_draft(self, draft: DraftRunSpec) -> None:
        if canonical_hash(draft) != self._draft_hash:
            raise ValueError("prepared CIFAR workload does not match draft")

    def create_initial_checkpoint(self, path: Path) -> InitialCheckpoint:
        return self._training.create_initial_checkpoint(path)

    def build_config(
        self,
        *,
        result: FormationResult,
        local_public_key: str,
        run_root: Path,
        metrics_publisher: MetricsService | None = None,
    ) -> TrainingConfig:
        trainer = self._training.create_trainer(
            manifest=result.manifest,
            local_public_key=local_public_key,
        )
        return TrainingConfig(
            algorithm=build_algorithm(
                manifest=result.manifest,
                trainer=trainer,
            ),
            load_checkpoint=trainer.load_checkpoint,
            run_store=RunStore(run_root / "run-store"),
            artifact_root=run_root / "rounds",
            metrics_publisher=metrics_publisher,
        )


def prepare_cifar_training(
    *,
    draft: DraftRunSpec,
    dataset_cache: Path,
    benchmark_seed: int,
    device: str = "cpu",
) -> PreparedCIFARTraining:
    """Prepare local data through the deep training-owned interface."""
    return PreparedCIFARTraining(
        _draft_hash=canonical_hash(draft),
        _training=prepare_training(
            draft=draft,
            cache_dir=dataset_cache,
            benchmark_seed=benchmark_seed,
            device=device,
        ),
    )


class BenchmarkNodeConfig(NodeConfig):
    dataset_cache: Path
    benchmark_seed: int
    training_device: Literal["cpu", "cuda"] = "cpu"


def load_benchmark_node_config(path: Path) -> BenchmarkNodeConfig:
    return BenchmarkNodeConfig.model_validate(yaml.safe_load(path.read_text()))


async def run_benchmark_node(
    config: BenchmarkNodeConfig, *, training_decorator: TrainingDecorator | None = None
) -> None:
    def prepare(draft: DraftRunSpec) -> PreparedCIFARTraining:
        return prepare_cifar_training(
            draft=draft,
            dataset_cache=config.dataset_cache,
            benchmark_seed=config.benchmark_seed,
            device=config.training_device,
        )

    async def ready(result: FormationResult, local_key: str, sink: EventSink) -> None:
        await asyncio.to_thread(
            append_evidence,
            sink,
            BenchmarkNodeReadyEvidence(
                run_id=result.manifest.run_id,
                manifest_hash=result.manifest_hash,
                node_id=local_key,
                benchmark_seed=config.benchmark_seed,
                transport="axl",
            ),
        )

    await run_node(
        config,
        prepare_training=prepare,
        training_decorator=training_decorator,
        on_ready=ready,
    )
