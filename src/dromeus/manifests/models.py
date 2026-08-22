"""Validated domain models for run formation and update exchange."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import (
    Field,
    StringConstraints,
    model_validator,
)

from dromeus.protocol.models import (
    AlgorithmId,
    DomainModel,
    Identifier,
    PublicKey,
    RoundId,
    RunId,
    Sha256,
    TensorSchema,
)
from dromeus.protocol.models import (
    MessageId as MessageId,
)
from dromeus.protocol.models import (
    Tensor as Tensor,
)
from dromeus.protocol.models import (
    TransferId as TransferId,
)
from dromeus.protocol.version import PROTOCOL_VERSION

MANIFEST_VERSION = 3
MIN_PARTICIPANT_COUNT = 4
MAX_PARTICIPANT_COUNT = 16
DPSGD_ALGORITHM_ID = "dpsgd"
NOLOCO_ALGORITHM_ID = "noloco"
RESNET32_MODEL_ID = "resnet32"

PackageVersion = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:+-]+$"),
]
NodeIndex = Annotated[int, Field(ge=0, lt=MAX_PARTICIPANT_COUNT)]


class ParticipantCountError(ValueError):
    """A run member count is outside the supported fixed-group range."""


def validate_participant_count(count: int) -> None:
    """Validate the fixed even membership range shared by manifests and gossip."""
    if count < MIN_PARTICIPANT_COUNT or count > MAX_PARTICIPANT_COUNT:
        raise ParticipantCountError(
            "participant count must be between 4 and 16 inclusive"
        )
    if count % 2:
        raise ParticipantCountError("participant count must be even")


class Participant(DomainModel):
    public_key: PublicKey
    node_index: NodeIndex


class ConsensusSketchMessage(DomainModel):
    sender_public_key: PublicKey
    round_id: RoundId
    payload: Annotated[bytes, Field(min_length=1)]


class DatasetContract(DomainModel):
    dataset_id: Literal["cifar10"]
    version: Identifier
    preprocessing_hash: Sha256
    iid_partition_seed: int
    image_shape: tuple[Annotated[int, Field(gt=0)], ...]
    class_count: Annotated[int, Field(gt=1)]
    sample_count: Annotated[int, Field(gt=0)]
    partition_sample_counts: tuple[Annotated[int, Field(gt=0)], ...] = Field(
        min_length=MIN_PARTICIPANT_COUNT,
        max_length=MAX_PARTICIPANT_COUNT,
    )
    node_index_partitions: tuple[NodeIndex, ...] = Field(
        min_length=MIN_PARTICIPANT_COUNT,
        max_length=MAX_PARTICIPANT_COUNT,
    )

    @property
    def participant_count(self) -> int:
        """Return the number of local partitions declared by the contract."""
        return len(self.node_index_partitions)

    @model_validator(mode="after")
    def partitions_cover_dataset(self) -> Self:
        if len(self.partition_sample_counts) != len(self.node_index_partitions):
            raise ValueError(
                "partition sample counts and node index partitions must have "
                "the same length"
            )
        validate_participant_count(len(self.partition_sample_counts))
        if sum(self.partition_sample_counts) != self.sample_count:
            raise ValueError("partition sample counts must equal sample count")
        if set(self.node_index_partitions) != set(
            range(len(self.partition_sample_counts))
        ):
            raise ValueError(
                "node index partitions must be exactly 0 through "
                f"{len(self.partition_sample_counts) - 1}"
            )
        return self


class EnvironmentFingerprint(DomainModel):
    dromeus_version: Identifier
    dromeus_commit: Annotated[
        str, StringConstraints(min_length=7, max_length=64, pattern=r"^[0-9a-f]+$")
    ]
    protocol_version: Literal[1] = PROTOCOL_VERSION
    pytorch_version: PackageVersion
    axl_version: Identifier
    model_definition_hash: Sha256
    container_image_digest: Annotated[
        str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")
    ]


class TransportLimits(DomainModel):
    max_payload_bytes: Annotated[int, Field(gt=0)]
    max_retries: Annotated[int, Field(ge=0)]
    retry_timeout_seconds: Annotated[float, Field(gt=0)]
    chunk_size_bytes: Annotated[int, Field(gt=0)] | None = None
    window_size: Annotated[int, Field(gt=0)] | None = None

    @property
    def max_update_bundle_bytes(self) -> int:
        """Return the v1 wire field with its bundle-total semantics."""
        return self.max_payload_bytes


class ConsensusSketchConfig(DomainModel):
    size: Literal[4096] = 4096
    seed: int


class WarmupCosineSchedule(DomainModel):
    """Versioned pre-step linear-warmup/cosine-decay schedule."""

    schedule_id: Literal["linear-warmup-cosine-v1"]
    total_inner_steps: Annotated[int, Field(gt=0)]
    warmup_inner_steps: Annotated[int, Field(gt=0)]
    start_learning_rate: Annotated[float, Field(gt=0.0)]
    peak_learning_rate: Annotated[float, Field(gt=0.0)]
    final_learning_rate: Annotated[float, Field(gt=0.0)]

    @model_validator(mode="after")
    def valid_schedule(self) -> Self:
        if self.warmup_inner_steps >= self.total_inner_steps:
            raise ValueError("warm-up steps must be less than total inner steps")
        if self.start_learning_rate > self.peak_learning_rate:
            raise ValueError("warm-up start learning rate must not exceed peak")
        if not math.isclose(
            self.final_learning_rate,
            self.peak_learning_rate / 10.0,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError("final learning rate must be one tenth of peak")
        return self

    def learning_rate(self, completed_inner_steps: int) -> float:
        """Return the rate applied before a zero-based inner optimizer step."""
        if not 0 <= completed_inner_steps <= self.total_inner_steps:
            raise ValueError("completed inner steps are outside the schedule")
        if completed_inner_steps == self.total_inner_steps:
            return self.final_learning_rate
        if completed_inner_steps < self.warmup_inner_steps:
            if self.warmup_inner_steps == 1:
                return self.peak_learning_rate
            progress = completed_inner_steps / (self.warmup_inner_steps - 1)
            return self.start_learning_rate + (
                self.peak_learning_rate - self.start_learning_rate
            ) * progress
        decay_steps = self.total_inner_steps - self.warmup_inner_steps
        progress = (completed_inner_steps - self.warmup_inner_steps + 1) / (
            decay_steps
        )
        return self.final_learning_rate + 0.5 * (
            self.peak_learning_rate - self.final_learning_rate
        ) * (1.0 + math.cos(math.pi * progress))


class TrainingPolicy(DomainModel):
    """Versioned local-optimizer and final-consensus settings."""

    batch_size: Annotated[int, Field(gt=0)]
    momentum: Annotated[float, Field(ge=0.0, lt=1.0)]
    weight_decay: Annotated[float, Field(ge=0.0)]
    learning_rate_milestones: tuple[Annotated[int, Field(gt=0)], ...] = ()
    learning_rate_gamma: Annotated[float, Field(gt=0.0, lt=1.0)]
    learning_rate_schedule: WarmupCosineSchedule | None = None
    crop_padding: Annotated[int, Field(ge=0)]
    normalize: bool
    final_consensus_rounds: Literal[0, 2] = 0

    @model_validator(mode="after")
    def increasing_milestones(self) -> Self:
        if any(
            right <= left
            for left, right in zip(
                self.learning_rate_milestones,
                self.learning_rate_milestones[1:],
                strict=False,
            )
        ):
            raise ValueError("learning-rate milestones must be strictly increasing")
        if self.learning_rate_schedule is not None and self.learning_rate_milestones:
            raise ValueError(
                "warmup-cosine and milestone schedules are mutually exclusive"
            )
        return self


class AdamSettings(DomainModel):
    """Explicit inner Adam settings carried by a manifest-v3 NoLoCo run."""

    learning_rate: Annotated[float, Field(gt=0)]
    beta1: Annotated[float, Field(ge=0.0, lt=1.0)]
    beta2: Annotated[float, Field(ge=0.0, lt=1.0)]
    epsilon: Annotated[float, Field(gt=0.0)]
    gradient_clip_norm: Annotated[float, Field(gt=0.0)]


class NoLoCoConfig(DomainModel):
    """Outer NoLoCo hyperparameters and the inner Adam configuration."""

    alpha: Annotated[float, Field(ge=0.0, lt=1.0)]
    beta: Annotated[float, Field(gt=0.0)]
    gamma: Annotated[float, Field(gt=0.0)]
    inner_steps: Annotated[int, Field(gt=0)]
    adam: AdamSettings

    @model_validator(mode="after")
    def frozen_hyperparameters(self) -> Self:
        if self.alpha != 0.5 or self.beta != 0.7 or self.gamma != 0.7:
            raise ValueError("NoLoCo requires alpha=0.5, beta=0.7, and gamma=0.7")
        if self.inner_steps != 50:
            raise ValueError("NoLoCo requires exactly 50 inner steps")
        if self.adam.gradient_clip_norm != 1.0:
            raise ValueError("NoLoCo requires Adam gradient clipping at 1.0")
        return self


class ArtifactCodec(DomainModel):
    """Codec identity bound to one logical update artifact."""

    artifact_name: Identifier
    codec_id: Identifier


class UpdateCodecBinding(DomainModel):
    """Logical codec identity and schema bound into one update artifact."""

    codec_id: Identifier
    codec_version: Annotated[int, Field(gt=0)]
    logical_schema: TensorSchema


class DraftRunSpec(DomainModel):
    manifest_version: Literal[3] = MANIFEST_VERSION
    protocol_version: Literal[1] = PROTOCOL_VERSION
    run_id: RunId
    algorithm_id: AlgorithmId
    model_id: Identifier
    model_definition_hash: Sha256
    dataset: DatasetContract
    environment: EnvironmentFingerprint
    local_steps: Annotated[int, Field(gt=0)]
    round_count: Annotated[int, Field(gt=0)]
    optimizer: Literal["sgd", "adam"] = "sgd"
    learning_rate: Annotated[float, Field(gt=0)]
    peer_scheduler_seed: int
    codec_id: Literal["safetensors-v1"]
    transport: TransportLimits
    consensus_sketch: ConsensusSketchConfig
    training: TrainingPolicy | None = None
    algorithm_config: NoLoCoConfig | None = None
    artifact_codecs: tuple[ArtifactCodec, ...] | None = None

    @model_validator(mode="after")
    def valid_manifest(self) -> Self:
        if self.environment.model_definition_hash != self.model_definition_hash:
            raise ValueError("environment model hash does not match draft")
        if self.training is None:
            raise ValueError("manifest v3 requires training policy")
        if self.algorithm_id == DPSGD_ALGORITHM_ID:
            if self.optimizer != "sgd":
                raise ValueError("dpsgd requires sgd")
            if self.training.learning_rate_schedule is not None:
                raise ValueError("warmup-cosine is only supported for NoLoCo")
            return self
        if self.algorithm_id != NOLOCO_ALGORITHM_ID:
            raise ValueError("manifest v3 requires dpsgd or noloco")
        if self.optimizer != "adam":
            raise ValueError("noloco requires adam")
        if self.training.final_consensus_rounds != 0:
            raise ValueError("NoLoCo requires final_consensus_rounds to be zero")
        if self.algorithm_config is None or self.artifact_codecs is None:
            raise ValueError(
                "NoLoCo requires algorithm and artifact codec configuration"
            )
        schedule = self.training.learning_rate_schedule
        if schedule is not None:
            expected_steps = self.round_count * self.algorithm_config.inner_steps
            if schedule.total_inner_steps != expected_steps:
                raise ValueError(
                    "warmup-cosine total steps must match round count"
                )
            peak = self.algorithm_config.adam.learning_rate
            if schedule.peak_learning_rate != peak or self.learning_rate != peak:
                raise ValueError(
                    "warmup-cosine peak must match NoLoCo Adam learning rate"
                )
        names = [codec.artifact_name for codec in self.artifact_codecs]
        if len(names) != len(set(names)) or set(names) != {
            "outer_gradient",
            "slow_weights",
        }:
            raise ValueError(
                "NoLoCo codec IDs must cover outer_gradient and slow_weights"
            )
        if (
            self.transport.chunk_size_bytes is None
            or self.transport.window_size is None
        ):
            raise ValueError("NoLoCo requires chunk size and window size")
        if self.transport.chunk_size_bytes > self.transport.max_payload_bytes:
            raise ValueError("chunk size must not exceed max payload bytes")
        return self


class Invitation(DomainModel):
    protocol_version: Literal[1] = PROTOCOL_VERSION
    run_id: RunId
    initiator_public_key: PublicKey
    bootstrap_uri: Annotated[str, StringConstraints(min_length=1, max_length=2048)]
    draft_hash: Sha256
    expected_participant_count: Annotated[
        int,
        Field(ge=MIN_PARTICIPANT_COUNT, le=MAX_PARTICIPANT_COUNT),
    ] = MIN_PARTICIPANT_COUNT
    enrollment_expires_at: datetime | None = None

    @model_validator(mode="after")
    def valid_expected_membership(self) -> Self:
        validate_participant_count(self.expected_participant_count)
        return self


class SealedManifest(DraftRunSpec):
    draft_hash: Sha256
    participants: tuple[Participant, ...] = Field(
        min_length=MIN_PARTICIPANT_COUNT,
        max_length=MAX_PARTICIPANT_COUNT,
    )
    initial_checkpoint_hash: Sha256
    tensor_schema: TensorSchema

    @model_validator(mode="after")
    def valid_membership(self) -> Self:
        participant_count = len(self.participants)
        validate_participant_count(participant_count)
        keys = {participant.public_key for participant in self.participants}
        indices = {participant.node_index for participant in self.participants}
        if len(keys) != participant_count:
            raise ValueError("participant public keys must be unique")
        if indices != set(range(participant_count)):
            raise ValueError(
                "participant node indices must be exactly 0 through "
                f"{participant_count - 1}"
            )
        if self.dataset.participant_count != participant_count:
            raise ValueError(
                "dataset partition count must match participant count"
            )
        return self


class SealedManifestExpectation(DomainModel):
    """Machine-local preflight constraints for one formed manifest."""

    draft_hash: Sha256
    participants: tuple[Participant, ...] = Field(
        min_length=MIN_PARTICIPANT_COUNT,
        max_length=MAX_PARTICIPANT_COUNT,
    )
    initial_checkpoint_hash: Sha256
    tensor_schema_hash: Sha256


class ArtifactMetadata(DomainModel):
    """Immutable bundle metadata v1 artifact."""

    name: Identifier
    size_bytes: Annotated[int, Field(gt=0)]
    sha256: Sha256
    tensor_schema: TensorSchema


class UpdateBundleMetadata(DomainModel):
    """Immutable historical bundle metadata v1."""

    version: Literal[1] = 1
    run_id: RunId
    manifest_hash: Sha256
    sender_public_key: PublicKey
    algorithm_id: AlgorithmId
    round_id: RoundId
    artifacts: tuple[ArtifactMetadata, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_artifacts(self) -> Self:
        names = [artifact.name for artifact in self.artifacts]
        if len(names) != len(set(names)):
            raise ValueError("artifact names must be unique")
        return self


class OpaqueArtifactMetadata(DomainModel):
    name: Identifier
    size_bytes: Annotated[int, Field(gt=0)]
    sha256: Sha256
    codec_id: Identifier
    codec_version: Annotated[int, Field(gt=0)]
    logical_schema_hash: Sha256
    encoded_schema_hash: Sha256


class OpaqueUpdateBundleMetadata(DomainModel):
    """Current independently versioned opaque bundle metadata."""

    version: Literal[2] = 2
    run_id: RunId
    manifest_hash: Sha256
    sender_public_key: PublicKey
    algorithm_id: AlgorithmId
    round_id: RoundId
    artifacts: tuple[OpaqueArtifactMetadata, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def unique_artifacts(self) -> Self:
        names = [artifact.name for artifact in self.artifacts]
        if len(names) != len(set(names)):
            raise ValueError("artifact names must be unique")
        return self
