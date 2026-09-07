"""Public composition of developer-owned training with the Dromeus runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from dromeus.manifests.canonical import canonical_hash, validate_sealed_draft
from dromeus.manifests.models import ApplicationTaskContract, DraftRunSpec, TensorSchema
from dromeus.membership.formation import FormationResult
from dromeus.persistence.run_store import RunStore
from dromeus.runtime import MetricsService, TrainingConfig, build_algorithm
from dromeus.training.state import InitialCheckpoint
from dromeus.training.trainer import PyTorchTrainer


def definition_hash(definition: str) -> str:
    """Identify an application-supplied, versioned semantic definition."""
    return hashlib.sha256(definition.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedApplication:
    """A validated application workload ready for fixed-group formation."""

    _draft_hash: str
    _trainer: PyTorchTrainer
    _model_definition: str
    _evaluation_interval: int

    @property
    def tensor_schema(self) -> TensorSchema:
        return self._trainer.tensor_schema

    def validate_draft(self, draft: DraftRunSpec) -> None:
        if canonical_hash(draft) != self._draft_hash:
            raise ValueError("prepared application draft does not match")

    def create_initial_checkpoint(self, path: Path) -> InitialCheckpoint:
        return self._trainer.create_initial_checkpoint(
            path, model_definition=self._model_definition
        )

    def build_config(
        self,
        *,
        result: FormationResult,
        local_public_key: str,
        run_root: Path,
        metrics_publisher: MetricsService | None = None,
    ) -> TrainingConfig:
        validate_sealed_draft(result.manifest, expected_draft_hash=self._draft_hash)
        if canonical_hash(result.manifest) != result.manifest_hash:
            raise ValueError("formed manifest hash does not match its contents")
        if result.manifest.tensor_schema != self.tensor_schema:
            raise ValueError("formed tensor schema does not match application")
        if local_public_key not in {
            member.public_key for member in result.manifest.participants
        }:
            raise ValueError("local key is not a formed participant")
        return TrainingConfig(
            algorithm=build_algorithm(manifest=result.manifest, trainer=self._trainer),
            load_checkpoint=self._trainer.load_checkpoint,
            run_store=RunStore(run_root / "run-store"),
            artifact_root=run_root / "rounds",
            metrics_publisher=metrics_publisher,
            evaluation_interval=self._evaluation_interval,
        )


def prepare_application(
    *,
    draft: DraftRunSpec,
    trainer: PyTorchTrainer,
    model_definition: str,
    task_definition: str,
    training_definition: str,
    evaluation_interval: int = 1,
) -> PreparedApplication:
    """Check shared semantics locally, before any AXL or formation side effects.

    Definitions should cover architecture, input/target meaning and preprocessing,
    and optimizer/loss/batching policy. The application separately validates its
    private data and binds it into the trainer's local state_identity.
    """
    if (
        not isinstance(draft.dataset, ApplicationTaskContract)
        or draft.application_training is None
    ):
        raise ValueError("application preparation requires an application manifest")
    for actual, expected in (
        (definition_hash(model_definition), draft.model_definition_hash),
        (definition_hash(task_definition), draft.dataset.definition_hash),
        (
            definition_hash(training_definition),
            draft.application_training.definition_hash,
        ),
    ):
        if actual != expected:
            raise ValueError("application definition does not match shared manifest")
    if evaluation_interval <= 0:
        raise ValueError("evaluation interval must be positive")
    if trainer.completed_steps:
        raise ValueError("prepare a fresh trainer before formation")
    trainer.weights()
    outer_policy = draft.model_dump(
        mode="json",
        include={
            "algorithm_id",
            "algorithm_config",
            "artifact_codecs",
            "codec_id",
            "local_steps",
        },
    )
    outer_identity = definition_hash(
        json.dumps(outer_policy, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    trainer.bind_contract(
        ":".join(
            (
                "application-restore-v2",
                draft.model_definition_hash,
                canonical_hash(draft.dataset),
                canonical_hash(draft.application_training),
                outer_identity,
            )
        )
    )
    return PreparedApplication(
        canonical_hash(draft), trainer, model_definition, evaluation_interval
    )
