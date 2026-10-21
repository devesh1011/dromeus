"""Public composition of developer-owned training with the Dromeus runtime."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from dromeus.manifests.canonical import canonical_hash, validate_sealed_draft
from dromeus.manifests.models import (
    ApplicationTaskContract,
    DraftRunSpec,
    PrivateOptimizerSpec,
    TaskTrainingSpec,
    TensorSchema,
)
from dromeus.membership.formation import FormationResult
from dromeus.persistence.run_store import RunStore
from dromeus.runtime import MetricsService, TrainingConfig, build_algorithm
from dromeus.training.optimizer_factory import PrivateOptimizerFactory
from dromeus.training.privacy import ReservationStore
from dromeus.training.private_model import describe_private_model
from dromeus.training.private_trainer import PrivateTrainer, private_training_definition
from dromeus.training.state import InitialCheckpoint
from dromeus.training.task import TrainingTask
from dromeus.training.trainer import PyTorchTrainer


def definition_hash(definition: str) -> str:
    """Identify an application-supplied, versioned semantic definition."""
    return hashlib.sha256(definition.encode()).hexdigest()


@dataclass(frozen=True)
class LocalTrainingOptions:
    """Local record lineage and execution settings, never peer dataset contents."""

    data_identity: str
    reservation_store: ReservationStore | None = None
    max_physical_batch_size: int = 1
    research_seed: int | None = None


def describe_task(
    task: TrainingTask,
    *,
    batch_size: int,
    optimizer: PrivateOptimizerSpec | None = None,
    sampling: Literal["poisson", "shuffle"] = "poisson",
    schedule: Literal["constant", "linear_decay"] = "constant",
    initialization: Literal[
        "public_untrained", "public_pretrained"
    ] = "public_untrained",
) -> TaskTrainingSpec:
    """Describe public task/model semantics without opening private records."""
    model = task.model_factory()
    report = describe_private_model(model)
    if task.optimizer_registration is not None:
        if optimizer is not None:
            raise ValueError(
                "select one optimizer registration or legacy specification"
            )
        selected = task.optimizer_registration.describe(model)
    else:
        if optimizer is None:
            raise ValueError("task requires an optimizer specification or registration")
        selected = optimizer
    return TaskTrainingSpec(
        optimizer=selected,
        batch_size=batch_size,
        sampling=sampling,
        schedule=schedule,
        initialization=initialization,
        evaluation_provenance="public_fixed_holdout"
        if task.public_evaluate
        else "disabled",
        task_definition_hash=definition_hash(task.task_definition),
        implementation_hash=task.implementation_hash,
        batch_definition_hash=task.batch_adapter.definition_hash,
        loss_definition_hash=definition_hash(task.loss_definition),
        record_unit=task.record_unit,
        trainable_parameter_names=report.trainable_parameter_names,
        parameter_aliases=report.parameter_aliases,
        public_buffer_hashes=dict(report.public_buffer_hashes),
        frozen_parameter_hashes=dict(report.frozen_parameter_hashes),
    )


def prepare_task(
    *,
    draft: DraftRunSpec,
    task: TrainingTask,
    local: LocalTrainingOptions,
    specification: TaskTrainingSpec | None = None,
    evaluation_interval: int = 1,
) -> PreparedApplication:
    """Prepare one application task using the group's ordinary/private policy."""
    if draft.manifest_version != 6 or not isinstance(
        draft.dataset, ApplicationTaskContract
    ):
        raise ValueError("generic tasks require a version 6 application manifest")
    selected = draft.private_training if draft.privacy is not None else specification
    if not isinstance(selected, TaskTrainingSpec):
        raise ValueError("generic task training specification is required")
    if specification is not None and specification != selected:
        raise ValueError("task specification differs from shared private policy")
    task.check_definition(selected)
    training_definition = private_training_definition(selected)
    if (
        definition_hash(task.model_definition) != draft.model_definition_hash
        or definition_hash(task.task_definition) != draft.dataset.definition_hash
        or draft.application_training is None
        or definition_hash(training_definition)
        != draft.application_training.definition_hash
    ):
        raise ValueError("application task definitions differ from shared manifest")
    trainer: PyTorchTrainer
    if draft.privacy is not None:
        if local.reservation_store is None:
            raise ValueError("private task requires a persistent reservation store")
        trainer = PrivateTrainer(
            task=task,
            policy=draft.privacy,
            specification=selected,
            run_id=draft.run_id,
            local_data_identity=local.data_identity,
            reservation_store=local.reservation_store,
            max_physical_batch_size=local.max_physical_batch_size,
            research_seed=local.research_seed,
        )
    else:
        from dromeus.training.ordinary_task import OrdinaryTaskTrainer

        trainer = OrdinaryTaskTrainer(
            task=task,
            specification=selected,
            local_data_identity=local.data_identity,
            max_logical_steps=draft.local_steps * draft.round_count,
            research_seed=local.research_seed if local.research_seed is not None else 0,
            max_physical_batch_size=local.max_physical_batch_size,
        )
    return prepare_application(
        draft=draft,
        trainer=trainer,
        model_definition=task.model_definition,
        task_definition=task.task_definition,
        training_definition=training_definition,
        evaluation_interval=evaluation_interval,
    )


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
        if draft.privacy is not None:
            if (
                not isinstance(self._trainer, PrivateTrainer)
                or self._trainer.policy != draft.privacy
                or self._trainer.specification != draft.private_training
            ):
                raise ValueError("private preparation does not match draft")
            self._trainer.weights()
        if canonical_hash(draft) != self._draft_hash:
            raise ValueError("prepared application draft does not match")

    def create_initial_checkpoint(self, path: Path) -> InitialCheckpoint:
        return self._trainer.create_initial_checkpoint(
            path, model_definition=self._model_definition
        )

    @property
    def private_trainer(self) -> PrivateTrainer | None:
        return self._trainer if isinstance(self._trainer, PrivateTrainer) else None

    def export_model(self, path: Path) -> None:
        """Explicit model-only private export; training archives remain local."""
        if not isinstance(self._trainer, PrivateTrainer):
            raise ValueError("model-only private export requires a private adapter")
        self._trainer.export_model(path)

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
    if draft.privacy is not None:
        if not isinstance(trainer, PrivateTrainer):
            raise ValueError("private policy requires the owned private adapter")
        if (
            draft.privacy != trainer.policy
            or draft.private_training != trainer.specification
        ):
            raise ValueError("private trainer policy or specification mismatch")
        if training_definition != private_training_definition(trainer.specification):
            raise ValueError("private training definition mismatch")
    elif isinstance(trainer, PrivateTrainer):
        raise ValueError("private adapter requires a private manifest")
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
    if draft.manifest_version in (5, 6):
        outer_policy.update(
            draft.model_dump(
                mode="json", include={"round_count", "privacy", "private_training"}
            )
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


def prepare_private_application(
    *,
    draft: DraftRunSpec,
    model_factory: Callable[[], nn.Module],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    model_definition: str,
    task_definition: str,
    local_data_identity: str,
    reservation_store: ReservationStore,
    max_physical_batch_size: int,
    research_seed: int | None = None,
    public_evaluate: Callable[[nn.Module], Mapping[str, float]] | None = None,
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer] | None = None,
    optimizer_registration: PrivateOptimizerFactory | None = None,
    evaluation_interval: int = 1,
) -> PreparedApplication:
    """Construct the sole supported private workload before formation/output."""
    if draft.privacy is None or draft.private_training is None:
        raise ValueError("explicit private policy and training specification required")
    if (
        not isinstance(draft.dataset, ApplicationTaskContract)
        or draft.application_training is None
    ):
        raise ValueError("private adapter requires an application task contract")
    definition = private_training_definition(draft.private_training)
    if (
        definition_hash(model_definition) != draft.model_definition_hash
        or definition_hash(task_definition) != draft.dataset.definition_hash
        or definition_hash(definition) != draft.application_training.definition_hash
    ):
        raise ValueError("private application definitions do not match draft")
    trainer = PrivateTrainer(
        model_factory=model_factory,
        inputs=inputs,
        targets=targets,
        policy=draft.privacy,
        specification=draft.private_training,
        run_id=draft.run_id,
        local_data_identity=local_data_identity,
        reservation_store=reservation_store,
        max_physical_batch_size=max_physical_batch_size,
        research_seed=research_seed,
        evaluate=public_evaluate,
        optimizer_factory=optimizer_factory,
        optimizer_registration=optimizer_registration,
    )
    return prepare_application(
        draft=draft,
        trainer=trainer,
        model_definition=model_definition,
        task_definition=task_definition,
        training_definition=definition,
        evaluation_interval=evaluation_interval,
    )
