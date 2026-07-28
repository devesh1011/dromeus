"""Validated domain models for run formation and update exchange."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

PROTOCOL_VERSION = 1
MANIFEST_VERSION = 2
M1_PARTICIPANT_COUNT = 4
DPSGD_ALGORITHM_ID = "dpsgd"
RESNET32_MODEL_ID = "resnet32"

Identifier = Annotated[
    str, StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
]
PackageVersion = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:+-]+$"),
]
RunId = Identifier
MessageId = Identifier
TransferId = Identifier
AlgorithmId = Identifier
RoundId = Annotated[int, Field(ge=0)]
NodeIndex = Annotated[int, Field(ge=0, lt=M1_PARTICIPANT_COUNT)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
PublicKey = Annotated[
    str, StringConstraints(min_length=1, max_length=512, pattern=r"^\S+$")
]


class DomainModel(BaseModel):
    """Closed, immutable value object used at protocol boundaries."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Participant(DomainModel):
    public_key: PublicKey
    node_index: NodeIndex


class ConsensusSketchMessage(DomainModel):
    sender_public_key: PublicKey
    round_id: RoundId
    payload: Annotated[bytes, Field(min_length=1)]


class Tensor(DomainModel):
    name: Identifier
    dtype: Literal["float16", "float32", "float64", "int8", "int32", "int64"]
    shape: tuple[Annotated[int, Field(gt=0)], ...]


class TensorSchema(DomainModel):
    tensors: tuple[Tensor, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_names(self) -> Self:
        names = [tensor.name for tensor in self.tensors]
        if len(names) != len(set(names)):
            raise ValueError("tensor names must be unique")
        return self


class DatasetContract(DomainModel):
    dataset_id: Literal["cifar10"]
    version: Identifier
    preprocessing_hash: Sha256
    iid_partition_seed: int
    image_shape: tuple[Annotated[int, Field(gt=0)], ...]
    class_count: Annotated[int, Field(gt=1)]
    sample_count: Annotated[int, Field(gt=0)]
    partition_sample_counts: tuple[
        Annotated[int, Field(gt=0)],
        Annotated[int, Field(gt=0)],
        Annotated[int, Field(gt=0)],
        Annotated[int, Field(gt=0)],
    ]
    node_index_partitions: tuple[NodeIndex, NodeIndex, NodeIndex, NodeIndex]

    @model_validator(mode="after")
    def partitions_cover_dataset(self) -> Self:
        if sum(self.partition_sample_counts) != self.sample_count:
            raise ValueError("partition sample counts must equal sample count")
        if set(self.node_index_partitions) != set(range(M1_PARTICIPANT_COUNT)):
            raise ValueError("node index partitions must be exactly 0 through 3")
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

    @property
    def max_update_bundle_bytes(self) -> int:
        """Return the v1 wire field with its bundle-total semantics."""
        return self.max_payload_bytes


class ConsensusSketchConfig(DomainModel):
    size: Literal[4096] = 4096
    seed: int


class TrainingPolicy(DomainModel):
    """Versioned local-optimizer and final-consensus settings."""

    batch_size: Annotated[int, Field(gt=0)]
    momentum: Annotated[float, Field(ge=0.0, lt=1.0)]
    weight_decay: Annotated[float, Field(ge=0.0)]
    learning_rate_milestones: tuple[Annotated[int, Field(gt=0)], ...] = ()
    learning_rate_gamma: Annotated[float, Field(gt=0.0, lt=1.0)]
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
        return self


class DraftRunSpec(DomainModel):
    manifest_version: Literal[1, 2] = MANIFEST_VERSION
    protocol_version: Literal[1] = PROTOCOL_VERSION
    run_id: RunId
    algorithm_id: AlgorithmId
    model_id: Identifier
    model_definition_hash: Sha256
    dataset: DatasetContract
    environment: EnvironmentFingerprint
    local_steps: Annotated[int, Field(gt=0)]
    round_count: Annotated[int, Field(gt=0)]
    optimizer: Literal["sgd"] = "sgd"
    learning_rate: Annotated[float, Field(gt=0)]
    peer_scheduler_seed: int
    codec_id: Literal["safetensors-v1"]
    transport: TransportLimits
    consensus_sketch: ConsensusSketchConfig
    training: TrainingPolicy | None = None

    @model_validator(mode="after")
    def compatible_environment(self) -> Self:
        if self.environment.model_definition_hash != self.model_definition_hash:
            raise ValueError("environment model hash does not match draft")
        if (self.manifest_version == 1) != (self.training is None):
            raise ValueError(
                "manifest version 1 excludes training policy; "
                "manifest version 2 requires it"
            )
        if self.manifest_version == 2 and (
            self.algorithm_id != DPSGD_ALGORITHM_ID
            or self.model_id != RESNET32_MODEL_ID
        ):
            raise ValueError(
                "manifest version 2 requires dpsgd and resnet32"
            )
        return self


class Invitation(DomainModel):
    protocol_version: Literal[1] = PROTOCOL_VERSION
    run_id: RunId
    initiator_public_key: PublicKey
    bootstrap_uri: Annotated[str, StringConstraints(min_length=1, max_length=2048)]
    draft_hash: Sha256
    expected_participant_count: Literal[4] = M1_PARTICIPANT_COUNT
    enrollment_expires_at: datetime | None = None


class SealedManifest(DraftRunSpec):
    draft_hash: Sha256
    participants: tuple[Participant, Participant, Participant, Participant]
    initial_checkpoint_hash: Sha256
    tensor_schema: TensorSchema

    @model_validator(mode="after")
    def valid_membership(self) -> Self:
        keys = {participant.public_key for participant in self.participants}
        indices = {participant.node_index for participant in self.participants}
        if len(keys) != M1_PARTICIPANT_COUNT:
            raise ValueError("participant public keys must be unique")
        if indices != set(range(M1_PARTICIPANT_COUNT)):
            raise ValueError("participant node indices must be exactly 0 through 3")
        return self


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
    artifacts: tuple[OpaqueArtifactMetadata, ...] = Field(
        min_length=1, max_length=16
    )

    @model_validator(mode="after")
    def unique_artifacts(self) -> Self:
        names = [artifact.name for artifact in self.artifacts]
        if len(names) != len(set(names)):
            raise ValueError("artifact names must be unique")
        return self
