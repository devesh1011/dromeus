"""Load and resolve immutable NoLoCo benchmark experiment artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal, Self, cast

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StringConstraints,
    ValidationError,
    model_validator,
)

from benchmarks.workloads.cifar10.models import resolve_model
from benchmarks.workloads.cifar10.partitioning import iid_partition_index_hashes
from dromeus.adapters.classification.torch_trainer import derive_benchmark_seed
from dromeus.gossip.peer_scheduler import PeerScheduler
from dromeus.manifests.canonical import canonical_hash, file_sha256
from dromeus.manifests.models import ArtifactCodec, WarmupCosineSchedule
from dromeus.protocol.models import Identifier, PublicKey, Sha256

_OFFICIAL_SEEDS = frozenset((17, 29, 41))
_OFFICIAL_WORLD_SIZES = frozenset((4, 8, 16))

Profile = Literal["smoke", "trajectory", "official"]
WorldSize = Literal[4, 8, 16]
AblationId = Literal["identity", "compressed"]
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]


class ExperimentError(ValueError):
    """A frozen experiment cannot be safely selected or compared."""


class ArtifactValidationError(ExperimentError):
    """The experiment schema or a referenced path is invalid."""


class RunSelectionError(ExperimentError):
    """The requested run is not present in the frozen matrix."""


class ArtifactIntegrityError(ExperimentError):
    """A pilot or checkpoint artifact differs from its frozen hash."""


class ComparabilityError(ExperimentError):
    """Derived workload semantics differ from the frozen declaration."""


class _ExperimentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunSelector(_ExperimentModel):
    profile: Profile
    world_size: WorldSize
    benchmark_seed: int


class SourceProvenance(_ExperimentModel):
    repository: Literal["gensyn-ai/noloco"]
    commit: Literal["a1b4a425bdc4050a356cf9f4bae7c383419703ab"]
    files: tuple[Annotated[str, Field(min_length=1)], ...] = Field(min_length=1)
    outer_gradient_convention: Literal["phi-minus-theta-v1"]


class ModelSpec(_ExperimentModel):
    model_id: Literal["resnet18-groupnorm-cifar10-v1"]
    definition_hash: Sha256
    tensor_schema_hash: Sha256
    parameter_count: PositiveInt
    dtype: Literal["float32"]


class DatasetSpec(_ExperimentModel):
    dataset_id: Literal["cifar10"]
    source: Literal["huggingface-uoft-cs-cifar10"]
    revision: Identifier
    preprocessing_hash: Sha256
    partition_seed: int
    sample_count: Literal[50_000]


class WorkloadSpec(_ExperimentModel):
    batch_size: PositiveInt
    crop_padding: Annotated[int, Field(ge=0)]
    normalize: Literal[True]
    augment: Literal[True]
    weight_decay: Annotated[float, Field(ge=0.0)]

    @model_validator(mode="after")
    def noloco_weight_decay(self) -> Self:
        if self.weight_decay != 0.0:
            raise ValueError("NoLoCo workload requires zero weight decay")
        return self


class NoLoCoSpec(_ExperimentModel):
    alpha: Annotated[float, Field(ge=0.0, lt=1.0)]
    beta: PositiveFloat
    gamma: PositiveFloat
    inner_steps: Literal[50]
    adam_learning_rate: PositiveFloat
    adam_beta1: Annotated[float, Field(ge=0.0, lt=1.0)]
    adam_beta2: Annotated[float, Field(ge=0.0, lt=1.0)]
    adam_epsilon: PositiveFloat
    gradient_clip_norm: PositiveFloat

    @model_validator(mode="after")
    def frozen_outer_settings(self) -> Self:
        if (self.alpha, self.beta, self.gamma, self.gradient_clip_norm) != (
            0.5,
            0.7,
            0.7,
            1.0,
        ):
            raise ValueError(
                "NoLoCo requires alpha=0.5, beta=gamma=0.7, and clipping=1.0"
            )
        return self


class ArtifactBinding(_ExperimentModel):
    path: Path
    sha256: Sha256


class TransferSpec(_ExperimentModel):
    chunk_size_bytes: PositiveInt
    window_size: PositiveInt


class CodecAblation(_ExperimentModel):
    ablation_id: AblationId
    artifact_codecs: tuple[ArtifactCodec, ArtifactCodec]

    @model_validator(mode="after")
    def valid_codecs(self) -> Self:
        names = tuple(codec.artifact_name for codec in self.artifact_codecs)
        if names != ("outer_gradient", "slow_weights"):
            raise ValueError("codec ablation artifacts must be ordered by name")
        pair = tuple(codec.codec_id for codec in self.artifact_codecs)
        expected = (
            {("identity-v1", "identity-v1")}
            if self.ablation_id == "identity"
            else {
                ("topk-int8-v1", "dense-int8-v1"),
                ("topk-bitmap-int8-v2", "dense-int8-v1"),
            }
        )
        if pair not in expected:
            raise ValueError("codec ablation does not match its declared identity")
        return self


class HardwareContract(_ExperimentModel):
    accelerator_class: Identifier
    instance_type: Literal["g5.xlarge"]
    container_image_digest: Annotated[
        str,
        StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$"),
    ]
    python_version: Literal["3.12.11"]
    pytorch_version: Literal["2.12.1+cu130"]
    cuda_version: Literal["13.0"]
    cudnn_version: Literal[92000]
    nccl_version: Literal["2.29.7"]
    driver_version: Literal["595.91.07"]
    axl_commit: Literal["628e28ace077f26dfe8d0259009b357216a9d8d4"]


class RankSeeds(_ExperimentModel):
    rank: Annotated[int, Field(ge=0, lt=16)]
    trainer_seed: Annotated[int, Field(ge=0)]
    augmentation_seed: Annotated[int, Field(ge=0)]
    loader_seed: Annotated[int, Field(ge=0)]


class BenchmarkParticipant(_ExperimentModel):
    rank: Annotated[int, Field(ge=0, lt=16)]
    node_index: Annotated[int, Field(ge=0, lt=16)]
    public_key: PublicKey


class EvidenceSpec(_ExperimentModel):
    trajectory_interval: PositiveInt
    absolute_tolerance: PositiveFloat
    relative_tolerance: PositiveFloat
    evaluation_interval_rounds: PositiveInt
    countsketch_interval_rounds: PositiveInt
    analysis_checkpoint_interval_rounds: PositiveInt
    analysis_storage_budget_bytes_per_node: PositiveInt
    residual_l2_bound: PositiveFloat
    residual_to_signal_ratio_bound: PositiveFloat
    overlap_enabled: Literal[False]

    @model_validator(mode="after")
    def frozen_tolerances(self) -> Self:
        if (self.absolute_tolerance, self.relative_tolerance) != (1e-6, 1e-6):
            raise ValueError("trajectory tolerances must both equal 1e-6")
        if (
            self.evaluation_interval_rounds,
            self.countsketch_interval_rounds,
            self.analysis_checkpoint_interval_rounds,
            self.analysis_storage_budget_bytes_per_node,
            self.residual_l2_bound,
            self.residual_to_signal_ratio_bound,
        ) != (25, 1, 50, 512 * 1024 * 1024, 6.0, 0.35):
            raise ValueError("official evidence values do not match frozen pilot")
        return self


class FrozenRun(_ExperimentModel):
    selector: RunSelector
    checkpoint: ArtifactBinding
    model_seed: Annotated[int, Field(ge=0)]
    scheduler_seed: int
    rank_seeds: tuple[RankSeeds, ...] = Field(min_length=4, max_length=16)
    participants: tuple[BenchmarkParticipant, ...] = Field(
        min_length=4,
        max_length=16,
    )
    partition_index_hashes: tuple[Sha256, ...] = Field(
        min_length=4,
        max_length=16,
    )
    round_count: PositiveInt
    transfer: TransferSpec
    schedule: WarmupCosineSchedule
    pairing_digest: Sha256
    evidence: EvidenceSpec

    @model_validator(mode="after")
    def matching_scale(self) -> Self:
        world_size = self.selector.world_size
        expected_ranks = tuple(range(world_size))
        if (
            len(self.rank_seeds) != world_size
            or len(self.participants) != world_size
            or len(self.partition_index_hashes) != world_size
        ):
            raise ValueError("run vectors must match world size")
        if tuple(item.rank for item in self.rank_seeds) != expected_ranks:
            raise ValueError("rank seeds must be ordered from zero through world size")
        if tuple(item.rank for item in self.participants) != expected_ranks:
            raise ValueError(
                "participants must be ordered from zero through world size"
            )
        if tuple(item.node_index for item in self.participants) != expected_ranks:
            raise ValueError("participant node index must equal rank")
        public_keys = tuple(item.public_key for item in self.participants)
        if len(set(public_keys)) != world_size:
            raise ValueError("participant public keys must be unique")
        if public_keys != tuple(sorted(public_keys)):
            raise ValueError(
                "participant public keys must follow formation sort order"
            )
        if self.schedule.total_inner_steps != self.round_count * 50:
            raise ValueError("schedule must contain exactly 50 inner steps per round")
        if self.selector.profile == "trajectory" and (
            world_size != 4 or self.evidence.trajectory_interval != 1
        ):
            raise ValueError("trajectory profile requires four nodes and every round")
        return self


class ResolvedRun(_ExperimentModel):
    experiment_sha256: Sha256
    run_config_sha256: Sha256
    selector: RunSelector
    source: SourceProvenance
    model: ModelSpec
    dataset: DatasetSpec
    workload: WorkloadSpec
    algorithm: NoLoCoSpec
    hardware: HardwareContract
    ablations: tuple[CodecAblation, CodecAblation]
    pilot_sha256: Sha256
    run: FrozenRun

    def codec_ablation(self, ablation_id: AblationId) -> CodecAblation:
        """Return one codec ablation declared by the frozen artifact."""
        for ablation in self.ablations:
            if ablation.ablation_id == ablation_id:
                return ablation
        raise ValueError("requested codec ablation is not frozen")


class FrozenExperiment(_ExperimentModel):
    schema_version: Literal[2]
    status: Literal["frozen"]
    source: SourceProvenance
    model: ModelSpec
    dataset: DatasetSpec
    workload: WorkloadSpec
    algorithm: NoLoCoSpec
    ablations: tuple[CodecAblation, CodecAblation]
    pilot: ArtifactBinding
    hardware: HardwareContract
    runs: tuple[FrozenRun, ...] = Field(min_length=1)

    _experiment_sha256: str = PrivateAttr(default="")

    @model_validator(mode="after")
    def complete_unique_matrix(self) -> Self:
        if {item.ablation_id for item in self.ablations} != {
            "identity",
            "compressed",
        }:
            raise ValueError("experiment must freeze identity and compressed ablations")
        selectors = tuple(run.selector for run in self.runs)
        selector_keys = {
            (item.profile, item.world_size, item.benchmark_seed) for item in selectors
        }
        if len(selector_keys) != len(selectors):
            raise ValueError("run selectors must be unique")
        official = {
            (item.world_size, item.benchmark_seed)
            for item in selectors
            if item.profile == "official"
        }
        expected_official = {
            (world_size, seed)
            for world_size in _OFFICIAL_WORLD_SIZES
            for seed in _OFFICIAL_SEEDS
        }
        if official != expected_official:
            raise ValueError("official profile must contain the complete 4/8/16 matrix")
        trajectory_runs = [
            run for run in self.runs if run.selector.profile == "trajectory"
        ]
        if (
            len(trajectory_runs) != 1
            or trajectory_runs[0].selector.world_size != 4
            or trajectory_runs[0].round_count != 2
        ):
            raise ValueError(
                "exactly one four-node two-round trajectory profile is required"
            )
        if any(
            run.schedule.peak_learning_rate != self.algorithm.adam_learning_rate
            for run in self.runs
        ):
            raise ValueError("run schedule peaks must match Adam learning rate")
        return self

    def resolve(self, selector: RunSelector) -> ResolvedRun:
        """Resolve one complete backend-independent run."""
        try:
            run = next(item for item in self.runs if item.selector == selector)
        except StopIteration as error:
            raise RunSelectionError(
                "requested run is not frozen in this experiment"
            ) from error
        identity = {
            "source": self.source.model_dump(mode="json"),
            "model": self.model.model_dump(mode="json"),
            "dataset": self.dataset.model_dump(mode="json"),
            "workload": self.workload.model_dump(mode="json"),
            "algorithm": self.algorithm.model_dump(mode="json"),
            "hardware": self.hardware.model_dump(mode="json"),
            "ablations": [
                item.model_dump(mode="json") for item in self.ablations
            ],
            "pilot_sha256": self.pilot.sha256,
            "run": {
                **run.model_dump(mode="json"),
                "checkpoint": {"sha256": run.checkpoint.sha256},
            },
        }
        return ResolvedRun(
            experiment_sha256=self._experiment_sha256,
            run_config_sha256=_value_hash(identity),
            selector=selector,
            source=self.source,
            model=self.model,
            dataset=self.dataset,
            workload=self.workload,
            algorithm=self.algorithm,
            hardware=self.hardware,
            ablations=self.ablations,
            pilot_sha256=self.pilot.sha256,
            run=run,
        )


def load_frozen_experiment(path: Path) -> FrozenExperiment:
    """Load and fully validate one pilot-backed immutable experiment."""
    try:
        raw = cast(object, yaml.safe_load(path.read_text(encoding="utf-8")))
        relative = FrozenExperiment.model_validate(raw)
    except (OSError, ValidationError, yaml.YAMLError) as error:
        raise ArtifactValidationError(
            f"invalid experiment artifact: {error}"
        ) from error

    artifact_root = path.parent.resolve()
    pilot = relative.pilot.model_copy(
        update={"path": _resolve_binding_path(artifact_root, relative.pilot.path)}
    )
    runs = tuple(
        run.model_copy(
            update={
                "checkpoint": run.checkpoint.model_copy(
                    update={
                        "path": _resolve_binding_path(
                            artifact_root,
                            run.checkpoint.path,
                        )
                    }
                )
            }
        )
        for run in relative.runs
    )
    experiment = relative.model_copy(update={"pilot": pilot, "runs": runs})
    object.__setattr__(experiment, "_experiment_sha256", canonical_hash(relative))
    _verify_model(experiment)
    _verify_binding(experiment.pilot, label="pilot artifact")
    for run in experiment.runs:
        _verify_binding(run.checkpoint, label="checkpoint")
        _verify_run(experiment, run)
    return experiment


def _verify_model(experiment: FrozenExperiment) -> None:
    try:
        recipe = resolve_model(
            experiment.model.model_id,
            definition_hash=experiment.model.definition_hash,
        )
    except ValueError as error:
        raise ComparabilityError("model definition is not a built-in recipe") from error
    if experiment.model.parameter_count != recipe.parameter_count:
        raise ComparabilityError("model parameter count does not match recipe")
    if experiment.model.tensor_schema_hash != recipe.tensor_schema_hash:
        raise ComparabilityError("model tensor schema does not match recipe")


def _resolve_binding_path(root: Path, path: Path) -> Path:
    if path.is_absolute():
        raise ArtifactValidationError("artifact paths must be relative")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ArtifactValidationError("artifact path escapes experiment directory")
    return resolved


def _verify_binding(binding: ArtifactBinding, *, label: str) -> None:
    if not binding.path.is_file():
        raise ArtifactIntegrityError(f"{label} is missing")
    if file_sha256(binding.path) != binding.sha256:
        raise ArtifactIntegrityError(f"{label} hash does not match")


def _verify_run(experiment: FrozenExperiment, run: FrozenRun) -> None:
    selector = run.selector
    expected_partitions = iid_partition_index_hashes(
        source_sample_count=experiment.dataset.sample_count,
        participant_count=selector.world_size,
        seed=experiment.dataset.partition_seed,
    )
    if run.partition_index_hashes != expected_partitions:
        raise ComparabilityError("partition hashes do not match canonical split")
    expected_model_seed = derive_benchmark_seed(
        selector.benchmark_seed,
        "model-initialization",
    )
    if run.model_seed != expected_model_seed:
        raise ComparabilityError("model seed does not match canonical derivation")
    if run.scheduler_seed != selector.benchmark_seed:
        raise ComparabilityError("scheduler seed must equal benchmark seed")
    base_trainer_seed = derive_benchmark_seed(
        selector.benchmark_seed,
        "local-training",
    )
    expected_rank_seeds = tuple(
        RankSeeds(
            rank=rank,
            trainer_seed=base_trainer_seed + rank,
            augmentation_seed=base_trainer_seed + rank + 1,
            loader_seed=base_trainer_seed + rank + 2,
        )
        for rank in range(selector.world_size)
    )
    if run.rank_seeds != expected_rank_seeds:
        raise ComparabilityError("rank seeds do not match canonical derivation")
    members = tuple(item.public_key for item in run.participants)
    if run.pairing_digest != pairing_digest(
        members,
        seed=run.scheduler_seed,
        rounds=run.round_count,
    ):
        raise ComparabilityError("pairing digest does not match canonical schedule")


def pairing_digest(members: tuple[str, ...], *, seed: int, rounds: int) -> str:
    """Return the canonical digest of all fixed-membership benchmark pairings."""
    scheduler = PeerScheduler(
        members,
        seed=seed,
        training_round_count=rounds,
        final_consensus_rounds=0,
    )
    value = [
        {
            "round_id": round_id,
            "pairs": scheduler.schedule(round_id).pairs,
        }
        for round_id in range(rounds)
    ]
    return _value_hash(value)


def _value_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "AblationId",
    "ArtifactIntegrityError",
    "ArtifactValidationError",
    "CodecAblation",
    "ComparabilityError",
    "ExperimentError",
    "FrozenExperiment",
    "ResolvedRun",
    "RunSelectionError",
    "RunSelector",
    "TransferSpec",
    "load_frozen_experiment",
    "pairing_digest",
]
