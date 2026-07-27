"""Validated domain models for run formation and update exchange."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

PROTOCOL_VERSION = 1
MANIFEST_VERSION = 1
M1_PARTICIPANT_COUNT = 4

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


class ConsensusSketchConfig(DomainModel):
    size: Literal[4096] = 4096
    seed: int


class DraftRunSpec(DomainModel):
    manifest_version: Literal[1] = MANIFEST_VERSION
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

    @model_validator(mode="after")
    def compatible_environment(self) -> Self:
        if self.environment.model_definition_hash != self.model_definition_hash:
            raise ValueError("environment model hash does not match draft")
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
    name: Identifier
    size_bytes: Annotated[int, Field(gt=0)]
    sha256: Sha256
    tensor_schema: TensorSchema


class UpdateBundleMetadata(DomainModel):
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
