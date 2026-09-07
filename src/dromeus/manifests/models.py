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
LOCAL_DATA_MANIFEST_VERSION = 4
MIN_PARTICIPANT_COUNT = 4
MAX_PARTICIPANT_COUNT = 16
DPSGD_ALGORITHM_ID = "dpsgd"
NOLOCO_ALGORITHM_ID = "noloco"
RESNET32_MODEL_ID = "resnet32"
RESNET18_GROUPNORM_MODEL_ID = "resnet18-groupnorm-cifar10-v1"

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


class ClassificationTaskContract(DomainModel):
    """Shared classification meaning, independent of each node's private records.

    Label position is the target integer. Nodes may hold different amounts of
    data and omit classes; each node has equal influence on the training objective.
    Local paths, counts, and fingerprints never enter this shared identity.
    """

    dataset_id: Literal["local-classification-v1"]
    input_shape: tuple[Annotated[int, Field(gt=0)], ...] = Field(min_length=1)
    input_dtype: Literal["float32"]
    label_names: tuple[Annotated[str, StringConstraints(min_length=1)], ...] = Field(
        min_length=2
    )
    preprocessing_hash: Sha256
    objective: Literal["equal-node"] = "equal-node"

    @property
    def class_count(self) -> int:
        return len(self.label_names)

    @model_validator(mode="after")
    def valid_label_meanings(self) -> Self:
        if any(not label.strip() for label in self.label_names):
            raise ValueError("label names must not be blank")
        if len(set(self.label_names)) != len(self.label_names):
            raise ValueError("label names must be unique and ordered")
        return self


class ApplicationTaskContract(DomainModel):
    """Application-defined input, target, preprocessing, and objective semantics.

    The definition hash identifies the shared task, never node-local records.
    """

    dataset_id: Literal["application-v1"] = "application-v1"
    task_id: Identifier
    definition_hash: Sha256
    objective: Literal["equal-node"] = "equal-node"


class ApplicationTrainingPolicy(DomainModel):
    """Identity of the application's local step, optimizer, and batching policy."""

    policy_id: Identifier
    definition_hash: Sha256


DataContract = DatasetContract | ClassificationTaskContract | ApplicationTaskContract


class EnvironmentFingerprint(DomainModel):
    dromeus_version: Identifier
    dromeus_commit: Annotated[
        str, StringConstraints(min_length=7, max_length=64, pattern=r"^[0-9a-f]+$")
    ]
    protocol_version: Literal[1] = PROTOCOL_VERSION
    pytorch_version: PackageVersion
    axl_version: Identifier
    model_definition_hash: Sha256
    container_image_digest: (
        Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")] | None
    ) = None


class TransportLimits(DomainModel):
    max_payload_bytes: Annotated[int, Field(gt=0)]
    max_retries: Annotated[int, Field(ge=0)]
    retry_timeout_seconds: Annotated[float, Field(gt=0)]
    chunk_size_bytes: Annotated[int, Field(gt=0)] | None = None
    window_size: Annotated[int, Field(gt=0)] | None = None
    max_message_payload_bytes: Annotated[int, Field(gt=0)] | None = None
    max_artifact_bytes: Annotated[int, Field(gt=0)] | None = None
    max_concurrent_transfers: Annotated[int, Field(gt=0)] | None = None
    max_inflight_bytes: Annotated[int, Field(gt=0)] | None = None
    transfer_lifetime_seconds: Annotated[float, Field(gt=0)] | None = None
    artifact_store_capacity_bytes: Annotated[int, Field(gt=0)] | None = None

    @model_validator(mode="after")
    def valid_transfer_bounds(self) -> Self:
        if self.effective_chunk_size > self.message_payload_limit:
            raise ValueError("chunk size must not exceed message payload limit")
        if self.message_payload_limit > self.max_payload_bytes:
            raise ValueError("message payload limit must not exceed bundle limit")
        if self.artifact_size_limit > self.max_payload_bytes:
            raise ValueError("artifact limit must not exceed bundle limit")
        if (
            self.effective_chunk_size * self.effective_window_size
            > self.inflight_byte_limit
        ):
            raise ValueError("window bytes must not exceed in-flight byte limit")
        if self.artifact_size_limit > self.artifact_store_capacity_limit:
            raise ValueError("artifact limit must not exceed store capacity")
        if self.inflight_byte_limit > self.artifact_store_capacity_limit:
            raise ValueError("in-flight limit must not exceed store capacity")
        minimum_lifetime = self.retry_timeout_seconds * (self.max_retries + 1)
        if self.transfer_lifetime_limit < minimum_lifetime:
            raise ValueError("transfer lifetime is shorter than retry budget")
        return self

    @property
    def max_update_bundle_bytes(self) -> int:
        """Return the v1 wire field with its bundle-total semantics."""
        return self.max_payload_bytes

    @property
    def message_payload_limit(self) -> int:
        return self.max_message_payload_bytes or self.max_payload_bytes

    @property
    def artifact_size_limit(self) -> int:
        return self.max_artifact_bytes or self.max_payload_bytes

    @property
    def effective_chunk_size(self) -> int:
        return self.chunk_size_bytes or self.artifact_size_limit

    @property
    def effective_window_size(self) -> int:
        return self.window_size or 1

    @property
    def concurrent_transfer_limit(self) -> int:
        return self.max_concurrent_transfers or 1

    @property
    def inflight_byte_limit(self) -> int:
        return self.max_inflight_bytes or self.artifact_size_limit

    @property
    def transfer_lifetime_limit(self) -> float:
        return self.transfer_lifetime_seconds or (
            self.retry_timeout_seconds * (self.max_retries + 2)
        )

    @property
    def artifact_store_capacity_limit(self) -> int:
        return self.artifact_store_capacity_bytes or self.artifact_size_limit


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
            return (
                self.start_learning_rate
                + (self.peak_learning_rate - self.start_learning_rate) * progress
            )
        decay_steps = self.total_inner_steps - self.warmup_inner_steps
        progress = (completed_inner_steps - self.warmup_inner_steps + 1) / (decay_steps)
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
    """Outer NoLoCo settings plus an optional legacy Adam recipe."""

    alpha: Annotated[float, Field(ge=0.0, lt=1.0)]
    beta: Annotated[float, Field(gt=0.0)]
    gamma: Annotated[float, Field(gt=0.0)]
    inner_steps: Annotated[int, Field(gt=0)]
    adam: AdamSettings | None = None

    def require_adam(self) -> AdamSettings:
        """Return the legacy recipe's optimizer settings."""
        if self.adam is None:
            raise ValueError("this recipe requires Adam settings")
        return self.adam

    def validate_legacy_recipe(self) -> None:
        """Preserve the frozen M2/classification recipe's exact constraints."""
        if self.alpha != 0.5 or self.beta != 0.7 or self.gamma != 0.7:
            raise ValueError("NoLoCo requires alpha=0.5, beta=0.7, and gamma=0.7")
        if self.inner_steps != 50:
            raise ValueError("NoLoCo requires exactly 50 inner steps")
        if self.require_adam().gradient_clip_norm != 1.0:
            raise ValueError("NoLoCo requires Adam gradient clipping at 1.0")


class ArtifactCodec(DomainModel):
    """Codec identity bound to one logical update artifact."""

    artifact_name: Identifier
    codec_id: Identifier
    top_k_fraction: Annotated[float, Field(gt=0.0, le=1.0)] | None = None
    lossy_allowed: bool | None = None

    @model_validator(mode="after")
    def valid_codec_settings(self) -> Self:
        if self.codec_id == "identity-v1":
            if self.top_k_fraction is not None or self.lossy_allowed is True:
                raise ValueError("identity codec cannot declare lossy settings")
            return self
        if self.codec_id in {"topk-int8-v1", "topk-bitmap-int8-v2"}:
            if self.artifact_name != "outer_gradient":
                raise ValueError("top-k codec is only valid for outer gradient")
            if self.top_k_fraction is None or not self.lossy_allowed:
                raise ValueError("top-k codec requires fraction and lossy permission")
            return self
        if self.codec_id == "dense-int8-v1":
            if self.artifact_name != "slow_weights":
                raise ValueError("dense int8 codec is only valid for slow weights")
            if self.top_k_fraction is not None or not self.lossy_allowed:
                raise ValueError("dense int8 codec requires lossy permission")
            return self
        raise ValueError("unsupported NoLoCo artifact codec")


class UpdateCodecBinding(DomainModel):
    """Logical codec identity and schema bound into one update artifact."""

    codec_id: Identifier
    codec_version: Annotated[int, Field(gt=0)]
    logical_schema: TensorSchema


class DraftRunSpec(DomainModel):
    manifest_version: Literal[3, 4] = MANIFEST_VERSION
    protocol_version: Literal[1] = PROTOCOL_VERSION
    run_id: RunId
    algorithm_id: AlgorithmId = NOLOCO_ALGORITHM_ID
    model_id: Identifier
    model_definition_hash: Sha256
    dataset: DataContract = Field(discriminator="dataset_id")
    expected_participant_count: (
        Annotated[int, Field(ge=MIN_PARTICIPANT_COUNT, le=MAX_PARTICIPANT_COUNT)] | None
    ) = None
    environment: EnvironmentFingerprint
    local_steps: Annotated[int, Field(gt=0)]
    round_count: Annotated[int, Field(gt=0)]
    optimizer: Literal["sgd", "adam", "application"] = "sgd"
    learning_rate: Annotated[float, Field(gt=0)] | None = None
    peer_scheduler_seed: int
    codec_id: Literal["safetensors-v1"]
    transport: TransportLimits
    consensus_sketch: ConsensusSketchConfig
    training: TrainingPolicy | None = None
    application_training: ApplicationTrainingPolicy | None = None
    algorithm_config: NoLoCoConfig | None = None
    artifact_codecs: tuple[ArtifactCodec, ...] | None = None

    @property
    def participant_count(self) -> int:
        """Resolve fixed membership without relying on private dataset sizes."""
        if isinstance(self.dataset, DatasetContract):
            return self.dataset.participant_count
        assert self.expected_participant_count is not None
        return self.expected_participant_count

    def require_iid_dataset(self) -> DatasetContract:
        """Return the legacy recipe contract or reject a local-data run."""
        if not isinstance(self.dataset, DatasetContract):
            raise ValueError("this recipe requires the manifest v3 CIFAR-10 contract")
        return self.dataset

    def require_learning_rate(self) -> float:
        if self.learning_rate is None:
            raise ValueError("this recipe requires a learning rate")
        return self.learning_rate

    @model_validator(mode="after")
    def valid_manifest(self) -> Self:
        if self.manifest_version == MANIFEST_VERSION:
            if self.environment.container_image_digest is None:
                raise ValueError("manifest v3 requires a container image digest")
            if not isinstance(self.dataset, DatasetContract):
                raise ValueError("manifest v3 requires the CIFAR-10 dataset contract")
            if self.expected_participant_count is not None:
                raise ValueError(
                    "manifest v3 derives membership from dataset partitions"
                )
        else:
            if not isinstance(
                self.dataset, (ClassificationTaskContract, ApplicationTaskContract)
            ):
                raise ValueError(
                    "manifest v4 requires a local classification task "
                    "or application task"
                )
            if self.expected_participant_count is None:
                raise ValueError("manifest v4 requires expected participant count")
            validate_participant_count(self.expected_participant_count)
        if self.environment.model_definition_hash != self.model_definition_hash:
            raise ValueError("environment model hash does not match draft")
        application = isinstance(self.dataset, ApplicationTaskContract)
        if application:
            if self.optimizer != "application" or self.application_training is None:
                raise ValueError(
                    "application tasks require application training policy"
                )
            if self.training is not None or self.learning_rate is not None:
                raise ValueError(
                    "application training owns its optimizer and learning rate"
                )
        else:
            if self.application_training is not None or self.optimizer == "application":
                raise ValueError("application training requires an application task")
            if self.training is None:
                raise ValueError("manifest requires training policy")
            self.require_learning_rate()
        if self.algorithm_id == DPSGD_ALGORITHM_ID:
            if not application:
                if self.optimizer != "sgd":
                    raise ValueError("dpsgd requires sgd")
                assert self.training is not None
                if self.training.learning_rate_schedule is not None:
                    raise ValueError("warmup-cosine is only supported for NoLoCo")
            if self.algorithm_config is not None or self.artifact_codecs is not None:
                raise ValueError("dpsgd does not use NoLoCo configuration")
            return self
        if self.algorithm_id != NOLOCO_ALGORITHM_ID:
            raise ValueError("manifest requires dpsgd or noloco")
        if self.algorithm_config is None or self.artifact_codecs is None:
            raise ValueError(
                "NoLoCo requires algorithm and artifact codec configuration"
            )
        if application:
            if self.algorithm_config.adam is not None:
                raise ValueError("application NoLoCo delegates its inner optimizer")
            if self.local_steps != self.algorithm_config.inner_steps:
                raise ValueError("local steps must match NoLoCo inner steps")
        else:
            if self.optimizer != "adam":
                raise ValueError("noloco requires adam")
            assert self.training is not None
            if self.training.final_consensus_rounds != 0:
                raise ValueError("NoLoCo requires final_consensus_rounds to be zero")
            self.algorithm_config.validate_legacy_recipe()
            schedule = self.training.learning_rate_schedule
            if schedule is not None:
                expected_steps = self.round_count * self.algorithm_config.inner_steps
                if schedule.total_inner_steps != expected_steps:
                    raise ValueError("warmup-cosine total steps must match round count")
                peak = self.algorithm_config.require_adam().learning_rate
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
        codecs = {codec.artifact_name: codec for codec in self.artifact_codecs}
        codec_pair = (
            codecs["outer_gradient"].codec_id,
            codecs["slow_weights"].codec_id,
        )
        if codec_pair not in {
            ("identity-v1", "identity-v1"),
            ("topk-int8-v1", "dense-int8-v1"),
            ("topk-bitmap-int8-v2", "dense-int8-v1"),
        }:
            raise ValueError("NoLoCo artifact codec combination is invalid")
        if (
            self.transport.chunk_size_bytes is None
            or self.transport.window_size is None
        ):
            raise ValueError("NoLoCo requires chunk size and window size")
        if self.transport.chunk_size_bytes > self.transport.message_payload_limit:
            raise ValueError("chunk size must not exceed message payload limit")
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
    # Defaults are a draft-authoring convenience; sealed authority is explicit.
    # Pydantic permits required overrides of defaulted, keyword-only fields.
    algorithm_id: AlgorithmId = Field(  # pyright: ignore[reportGeneralTypeIssues]
        ...
    )
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
        if self.participant_count != participant_count:
            raise ValueError(
                "declared participant count must match sealed participant count"
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
