"""Turn a training task into a shared draft and a node's prepared training."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import torch
import yaml
from pydantic import ValidationError
from torch.utils.data import Dataset

from dromeus.errors import ConfigError, DraftError, PrivacyError, TaskError
from dromeus.manifests.canonical import (
    canonical_hash,
    parse_draft_yaml,
    validate_sealed_draft,
)
from dromeus.manifests.models import (
    ApplicationTaskContract,
    ArtifactCodec,
    ConsensusSketchConfig,
    DivergencePolicy,
    DraftRunSpec,
    EnvironmentFingerprint,
    NoLoCoConfig,
    PrivacyPolicy,
    TaskTrainingSpec,
    TensorSchema,
    TransportLimits,
)
from dromeus.membership.formation import FormationResult
from dromeus.runtime import (
    FailureConfig,
    MetricsService,
    TrainingConfig,
    build_algorithm,
)
from dromeus.training.privacy import ReservationStore
from dromeus.training.private_model import describe_private_model
from dromeus.training.private_trainer import PrivateTrainer
from dromeus.training.state import InitialCheckpoint
from dromeus.training.task import Optimizer, TrainingTask
from dromeus.training.trainer import PyTorchTrainer

DEFAULT_TRANSPORT = TransportLimits(
    max_payload_bytes=8 * 1024 * 1024,
    max_retries=3,
    retry_timeout_seconds=5.0,
    chunk_size_bytes=64 * 1024,
    window_size=2,
)


def definition_hash(definition: str) -> str:
    """Identify an application-supplied semantic definition."""
    return hashlib.sha256(definition.encode()).hexdigest()


def code_hash(*modules: ModuleType | Path) -> str:
    """Identify the code that implements a task, for `implementation_hash`.

    Hashes the source files of the given modules (or paths) in order. Every node
    must install identical files. Agreement shows the files match; it does not
    verify what the code does.
    """
    if not modules:
        raise TaskError("code_hash needs at least one module or path")
    digest = hashlib.sha256()
    for module in modules:
        if isinstance(module, Path):
            path = module
        else:
            source = inspect.getsourcefile(module)
            if source is None:
                raise TaskError(f"module has no source file: {module.__name__}")
            path = Path(source)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


@dataclass(frozen=True)
class NoLoCo:
    """NoLoCo outer-step settings shared by the group.

    Set `top_k_fraction` to compress updates: the outer gradient sends its largest
    entries as int8 and slow weights are sent as int8. None sends exact values.
    """

    alpha: float = 0.5
    beta: float = 0.7
    gamma: float = 0.7
    top_k_fraction: float | None = None

    def codecs(self) -> tuple[ArtifactCodec, ...]:
        if self.top_k_fraction is None:
            return (
                ArtifactCodec(artifact_name="outer_gradient", codec_id="identity"),
                ArtifactCodec(artifact_name="slow_weights", codec_id="identity"),
            )
        return (
            ArtifactCodec(
                artifact_name="outer_gradient",
                codec_id="topk-bitmap-int8",
                top_k_fraction=self.top_k_fraction,
                lossy_allowed=True,
            ),
            ArtifactCodec(
                artifact_name="slow_weights",
                codec_id="dense-int8",
                lossy_allowed=True,
            ),
        )


@dataclass(frozen=True)
class Privacy:
    """Record-level differential privacy for every node's local training.

    The privacy horizon is the run's total local steps (rounds x local steps).
    Use `randomness="reproducible_public_benchmark"` only for seeded research
    runs on public data; real private runs use the default `"secure"`.
    """

    max_grad_norm: float
    noise_multiplier: float
    delta: float
    epsilon_limit: float | None = None
    randomness: Literal["secure", "reproducible_public_benchmark"] = "secure"

    def policy(self, max_logical_steps: int) -> PrivacyPolicy:
        try:
            return PrivacyPolicy(
                max_grad_norm=self.max_grad_norm,
                noise_multiplier=self.noise_multiplier,
                delta=self.delta,
                max_logical_steps=max_logical_steps,
                epsilon_limit=self.epsilon_limit,
                randomness_profile=self.randomness,
            )
        except ValidationError as error:
            raise PrivacyError(f"invalid privacy settings: {error}") from error


@dataclass(frozen=True)
class LocalData:
    """This node's own records and limits; never sent to peers.

    `data_identity` names this exact copy of the records (for example a file
    hash); checkpoints and privacy spending are bound to it. A private run needs
    `ledger`, the node's durable privacy-spending record. `physical_batch_size`
    caps how many records are processed at once. `research_seed` makes ordinary
    or public-benchmark runs reproducible; secure private runs refuse it.
    """

    dataset: Dataset[Any]
    data_identity: str
    ledger: ReservationStore | None = None
    physical_batch_size: int = 1
    research_seed: int | None = None


def installed_versions() -> tuple[str, str]:
    """Dromeus and PyTorch versions installed on this machine."""
    return importlib.metadata.version("dromeus"), torch.__version__


def describe_task(
    task: TrainingTask, *, sampling: Literal["poisson", "shuffle"]
) -> TaskTrainingSpec:
    """The shared training specification for a task, without opening records."""
    evidence = task.initialization_evidence
    provenance = evidence.provenance if evidence is not None else None
    task.check_initialization(task.initialization_mode, provenance)
    model = task.model_factory()
    report = describe_private_model(model, gradient_backend=task.gradient_backend)
    optimizer = task.optimizer
    if isinstance(optimizer, Optimizer):
        selected = optimizer.specification(report.trainable_parameter_names)
    else:
        selected = optimizer.describe(model)
    try:
        return TaskTrainingSpec(
            optimizer=selected,
            batch_size=task.batch_size,
            sampling=sampling,
            schedule=task.schedule,
            initialization=task.initialization_mode,
            initialization_provenance=provenance,
            evaluation_provenance="public_fixed_holdout"
            if task.public_evaluate
            else "disabled",
            task_definition_hash=definition_hash(task.task_definition),
            implementation_hash=task.implementation_hash,
            batch_definition_hash=task.batch_adapter.definition_hash,
            loss_definition_hash=definition_hash(task.loss_definition),
            record_unit=task.record_unit,
            gradient_backend=report.gradient_backend,
            trainable_parameter_names=report.trainable_parameter_names,
            parameter_aliases=report.parameter_aliases,
            public_buffer_hashes=dict(report.public_buffer_hashes),
            frozen_parameter_hashes=dict(report.frozen_parameter_hashes),
        )
    except ValidationError as error:
        raise TaskError(f"task cannot be described: {error}") from error


def draft_for_task(
    task: TrainingTask,
    *,
    run_id: str,
    participants: int,
    rounds: int,
    local_steps: int,
    axl_version: str,
    noloco: NoLoCo | None = None,
    privacy: Privacy | None = None,
    transport: TransportLimits | None = None,
    divergence: DivergencePolicy | None = None,
    seed: int = 0,
) -> DraftRunSpec:
    """Build the shared draft for a task; every hash is computed here."""
    noloco = noloco or NoLoCo()
    dromeus_version, pytorch_version = installed_versions()
    model_hash = definition_hash(task.model_definition)
    training = describe_task(task, sampling="poisson" if privacy else "shuffle")
    try:
        return DraftRunSpec(
            run_id=run_id,
            model_id="application",
            model_definition_hash=model_hash,
            dataset=ApplicationTaskContract(
                task_id="application",
                definition_hash=training.task_definition_hash,
            ),
            expected_participant_count=participants,
            environment=EnvironmentFingerprint(
                dromeus_version=dromeus_version,
                pytorch_version=pytorch_version,
                axl_version=axl_version,
                model_definition_hash=model_hash,
            ),
            local_steps=local_steps,
            round_count=rounds,
            peer_scheduler_seed=seed,
            codec_id="safetensors",
            transport=transport or DEFAULT_TRANSPORT,
            consensus_sketch=ConsensusSketchConfig(seed=seed),
            training=training,
            algorithm_config=NoLoCoConfig(
                alpha=noloco.alpha,
                beta=noloco.beta,
                gamma=noloco.gamma,
                inner_steps=local_steps,
            ),
            artifact_codecs=noloco.codecs(),
            privacy=privacy.policy(local_steps * rounds) if privacy else None,
            divergence_detection=divergence,
        )
    except ValidationError as error:
        raise DraftError(f"invalid draft settings: {error}") from error


def make_draft(
    task: TrainingTask,
    path: Path | str,
    *,
    run_id: str,
    participants: int,
    rounds: int,
    local_steps: int,
    axl_version: str,
    noloco: NoLoCo | None = None,
    privacy: Privacy | None = None,
    transport: TransportLimits | None = None,
    divergence: DivergencePolicy | None = None,
    seed: int = 0,
) -> str:
    """Write the group's shared draft for `task` to `path`; return its hash.

    Run once, by one person, then give the file to every node. Every hash and
    the Dromeus and PyTorch versions are filled in from this machine.
    """
    draft = draft_for_task(
        task,
        run_id=run_id,
        participants=participants,
        rounds=rounds,
        local_steps=local_steps,
        axl_version=axl_version,
        noloco=noloco,
        privacy=privacy,
        transport=transport,
        divergence=divergence,
        seed=seed,
    )
    text = yaml.safe_dump(
        draft.model_dump(mode="json", exclude_none=True), sort_keys=False
    )
    Path(path).write_text(text, encoding="utf-8")
    return canonical_hash(draft)


def load_draft(path: Path | str) -> DraftRunSpec:
    """Read and validate a draft file."""
    try:
        return parse_draft_yaml(Path(path))
    except OSError as error:
        raise ConfigError(f"cannot read draft {path}: {error}") from error


def check_environment(draft: DraftRunSpec) -> None:
    """Refuse a draft made for a different Dromeus or PyTorch version."""
    dromeus_version, pytorch_version = installed_versions()
    for name, declared, installed in (
        ("Dromeus", draft.environment.dromeus_version, dromeus_version),
        ("PyTorch", draft.environment.pytorch_version, pytorch_version),
    ):
        if declared != installed:
            raise DraftError(
                f"draft requires {name} {declared}; this machine has {installed}"
            )


def require_task_matches(draft: DraftRunSpec, task: TrainingTask) -> None:
    """Refuse a draft that was not made from this task."""
    if definition_hash(task.model_definition) != draft.model_definition_hash:
        raise DraftError("draft model definition differs from this task")
    local = describe_task(task, sampling=draft.training.sampling)
    if local != draft.training:
        shared = draft.training.model_dump()
        differing = sorted(
            name for name, value in local.model_dump().items() if shared[name] != value
        )
        raise DraftError(
            "draft training differs from this task: " + ", ".join(differing)
        )


def check_draft(path: Path | str, task: TrainingTask) -> None:
    """Check a draft file against this task and this machine before running.

    Raises `DraftError` naming what differs. Every node should run this check;
    `Node.run` performs it too.
    """
    draft = load_draft(path)
    check_environment(draft)
    require_task_matches(draft, task)


def prepare_task(
    *,
    draft: DraftRunSpec,
    task: TrainingTask,
    local: LocalData,
    evaluation_interval: int = 1,
) -> PreparedApplication:
    """Prepare this node's training for the group's draft."""
    require_task_matches(draft, task)
    specification = draft.training
    trainer: PyTorchTrainer
    if draft.privacy is not None:
        if local.ledger is None:
            raise PrivacyError("private task requires a persistent privacy ledger")
        trainer = PrivateTrainer(
            task=task,
            dataset=local.dataset,
            policy=draft.privacy,
            specification=specification,
            run_id=draft.run_id,
            local_data_identity=local.data_identity,
            reservation_store=local.ledger,
            max_physical_batch_size=local.physical_batch_size,
            research_seed=local.research_seed,
        )
    else:
        from dromeus.training.ordinary_task import OrdinaryTaskTrainer

        trainer = OrdinaryTaskTrainer(
            task=task,
            dataset=local.dataset,
            specification=specification,
            local_data_identity=local.data_identity,
            max_logical_steps=draft.local_steps * draft.round_count,
            research_seed=local.research_seed if local.research_seed is not None else 0,
            max_physical_batch_size=local.physical_batch_size,
        )
    return prepare_trainer(
        draft=draft,
        trainer=trainer,
        model_definition=task.model_definition,
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
                or self._trainer.specification != draft.training
            ):
                raise PrivacyError("private preparation does not match draft")
            self._trainer.weights()
        if canonical_hash(draft) != self._draft_hash:
            raise DraftError("prepared application draft does not match")

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
            raise PrivacyError("model-only private export requires a private adapter")
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
            raise DraftError("formed manifest hash does not match its contents")
        if result.manifest.tensor_schema != self.tensor_schema:
            raise DraftError("formed tensor schema does not match application")
        if local_public_key not in {
            member.public_key for member in result.manifest.participants
        }:
            raise DraftError("local key is not a formed participant")
        durable = FailureConfig.for_run_root(run_root)
        return TrainingConfig(
            algorithm=build_algorithm(manifest=result.manifest, trainer=self._trainer),
            load_checkpoint=self._trainer.load_checkpoint,
            run_store=durable.run_store,
            artifact_root=durable.artifact_root,
            metrics_publisher=metrics_publisher,
            evaluation_interval=self._evaluation_interval,
        )


def prepare_trainer(
    *,
    draft: DraftRunSpec,
    trainer: PyTorchTrainer,
    model_definition: str,
    evaluation_interval: int = 1,
) -> PreparedApplication:
    """Bind a fresh trainer to the draft, before any AXL or formation work."""
    if draft.privacy is not None:
        if not isinstance(trainer, PrivateTrainer):
            raise PrivacyError("private policy requires the owned private adapter")
        if draft.privacy != trainer.policy or draft.training != trainer.specification:
            raise PrivacyError("private trainer policy or specification mismatch")
    elif isinstance(trainer, PrivateTrainer):
        raise PrivacyError("private adapter requires a private manifest")
    if definition_hash(model_definition) != draft.model_definition_hash:
        raise DraftError("model definition does not match shared draft")
    if evaluation_interval <= 0:
        raise ConfigError("evaluation interval must be positive")
    if trainer.completed_steps:
        raise TaskError("prepare a fresh trainer before formation")
    trainer.weights()
    outer_policy = draft.model_dump(
        mode="json",
        include={
            "algorithm_id",
            "algorithm_config",
            "artifact_codecs",
            "codec_id",
            "local_steps",
            "round_count",
            "privacy",
            "training",
        },
    )
    outer_identity = definition_hash(
        json.dumps(outer_policy, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    trainer.bind_contract(
        ":".join(
            (
                "application-restore",
                draft.model_definition_hash,
                canonical_hash(draft.dataset),
                canonical_hash(draft.training),
                outer_identity,
            )
        )
    )
    return PreparedApplication(
        canonical_hash(draft), trainer, model_definition, evaluation_interval
    )
