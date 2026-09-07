"""Explicit opt-in classification workload composition."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from torch import nn

from dromeus.adapters.classification.data import LocalClassificationData
from dromeus.adapters.classification.trainer import (
    PreparedLocalTraining as TrainingOwnedLocal,
)
from dromeus.adapters.classification.trainer import (
    prepare_local_training as prepare_training_local,
)
from dromeus.manifests.models import DraftRunSpec, TensorSchema
from dromeus.membership.formation import FormationResult
from dromeus.persistence.run_store import RunStore
from dromeus.runtime import MetricsService, TrainingConfig, build_algorithm
from dromeus.training.state import InitialCheckpoint


@dataclass(frozen=True, slots=True)
class PreparedLocalTraining:
    """Compose a validated node-local workload with runtime durability."""

    _training: TrainingOwnedLocal

    @property
    def tensor_schema(self) -> TensorSchema:
        return self._training.tensor_schema

    def validate_draft(self, draft: DraftRunSpec) -> None:
        self._training.validate_draft(draft)

    def create_initial_checkpoint(self, path: Path) -> InitialCheckpoint:
        return self._training.create_initial_checkpoint(path)

    def build_config(
        self,
        *,
        result: FormationResult,
        local_public_key: str,
        run_root: Path,
        metrics_publisher: MetricsService | None = None,
        evaluation_interval: int = 1,
    ) -> TrainingConfig:
        trainer = self._training.create_trainer(
            manifest=result.manifest, local_public_key=local_public_key
        )
        return TrainingConfig(
            algorithm=build_algorithm(manifest=result.manifest, trainer=trainer),
            load_checkpoint=trainer.load_checkpoint,
            run_store=RunStore(run_root / "run-store"),
            artifact_root=run_root / "rounds",
            metrics_publisher=metrics_publisher,
            evaluation_interval=evaluation_interval,
        )


def prepare_local_training(
    *,
    draft: DraftRunSpec,
    data: LocalClassificationData,
    model: nn.Module,
    model_definition: str,
    seed: int,
    device: str = "cpu",
) -> PreparedLocalTraining:
    """Validate independent local data before the runtime enters formation."""
    return PreparedLocalTraining(
        prepare_training_local(
            draft=draft,
            data=data,
            model=model,
            model_definition=model_definition,
            seed=seed,
            device=device,
        )
    )


type LocalTrainingFactory = Callable[[DraftRunSpec], PreparedLocalTraining]
