"""Validated domain models for run formation and update exchange."""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Annotated, Literal, Self, cast

from pydantic import (
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

from dromeus.protocol.models import (
    CHUNK_FRAMING_BYTES,
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

MANIFEST_VERSION = 7
MIN_PARTICIPANT_COUNT = 4
MAX_PARTICIPANT_COUNT = 16
NOLOCO_ALGORITHM_ID = "noloco"

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


class ApplicationTaskContract(DomainModel):
    """Application-defined input, target, preprocessing, and objective semantics.

    The definition hash identifies the shared task, never node-local records.
    """

    dataset_id: Literal["application"] = "application"
    task_id: Identifier
    definition_hash: Sha256
    objective: Literal["equal-node"] = "equal-node"


class ApplicationTrainingPolicy(DomainModel):
    """Identity of the application's local step, optimizer, and batching policy."""

    policy_id: Identifier
    definition_hash: Sha256


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
        if self.effective_chunk_size + CHUNK_FRAMING_BYTES > self.message_payload_limit:
            raise ValueError(
                "chunk size plus chunk framing must not exceed message payload limit"
            )
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
        """Return the byte limit for one whole update bundle."""
        return self.max_payload_bytes

    @property
    def message_payload_limit(self) -> int:
        return self.max_message_payload_bytes or self.max_payload_bytes

    @property
    def artifact_size_limit(self) -> int:
        return self.max_artifact_bytes or self.max_payload_bytes

    @property
    def effective_chunk_size(self) -> int:
        if self.chunk_size_bytes is not None:
            return self.chunk_size_bytes
        return max(
            1,
            min(
                self.artifact_size_limit,
                self.message_payload_limit - CHUNK_FRAMING_BYTES,
            ),
        )

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


class NoLoCoConfig(DomainModel):
    """Outer NoLoCo settings; the application owns its inner optimizer."""

    alpha: Annotated[float, Field(ge=0.0, lt=1.0)]
    beta: Annotated[float, Field(gt=0.0)]
    gamma: Annotated[float, Field(gt=0.0)]
    inner_steps: Annotated[int, Field(gt=0)]


class ArtifactCodec(DomainModel):
    """Codec identity bound to one logical update artifact."""

    artifact_name: Identifier
    codec_id: Identifier
    top_k_fraction: Annotated[float, Field(gt=0.0, le=1.0)] | None = None
    lossy_allowed: bool | None = None

    @model_validator(mode="after")
    def valid_codec_settings(self) -> Self:
        if self.codec_id == "identity":
            if self.top_k_fraction is not None or self.lossy_allowed is True:
                raise ValueError("identity codec cannot declare lossy settings")
            return self
        if self.codec_id == "topk-bitmap-int8":
            if self.artifact_name != "outer_gradient":
                raise ValueError("top-k codec is only valid for outer gradient")
            if self.top_k_fraction is None or not self.lossy_allowed:
                raise ValueError("top-k codec requires fraction and lossy permission")
            return self
        if self.codec_id == "dense-int8":
            if self.artifact_name != "slow_weights":
                raise ValueError("dense int8 codec is only valid for slow weights")
            if self.top_k_fraction is not None or not self.lossy_allowed:
                raise ValueError("dense int8 codec requires lossy permission")
            return self
        raise ValueError("unsupported NoLoCo artifact codec")


class UpdateCodecBinding(DomainModel):
    """Logical codec identity and schema bound into one update artifact."""

    codec_id: Identifier
    logical_schema: TensorSchema


class PrivacyPolicy(DomainModel):
    """Fixed record-level Poisson Gaussian mechanism and release policy."""

    mode: Literal["record_dp"] = "record_dp"
    adjacency: Literal["add_remove"] = "add_remove"
    sampling: Literal["poisson"] = "poisson"
    clipping: Literal["flat"] = "flat"
    max_grad_norm: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    noise_multiplier: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    accountant: Literal["prv"] = "prv"
    delta: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)]
    max_logical_steps: Annotated[int, Field(gt=0)]
    epsilon_limit: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    randomness_profile: Literal["secure", "reproducible_public_benchmark"]
    metric_policy: Literal["public_eval_only"] = "public_eval_only"
    prv_epsilon_error: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 0.01
    prv_delta_error_ratio: Annotated[float, Field(gt=0, lt=1)] = 0.001


class DivergencePolicy(DomainModel):
    """Explicit pilot thresholds; no universal detector defaults."""

    version: Literal[1] = 1
    mode: Literal["warn"] = "warn"
    threshold_set_id: Identifier
    warmup_rounds: Annotated[int, Field(ge=0)]
    window_rounds: Annotated[int, Field(ge=4, le=4096)]
    patience_windows: Annotated[int, Field(gt=0)]
    distance_floor: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    growth_ratio: Annotated[float, Field(gt=1, allow_inf_nan=False)]
    min_log_slope: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    severe_distance: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    severe_patience: Annotated[int, Field(gt=0)]
    recovery_windows: Annotated[int, Field(gt=0)]
    recovery_distance: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    max_lag_rounds: Annotated[int, Field(ge=0, le=4096)]
    absolute_distance_floor: (
        Annotated[float, Field(gt=0, allow_inf_nan=False)] | None
    ) = None
    absolute_severe_distance: (
        Annotated[float, Field(gt=0, allow_inf_nan=False)] | None
    ) = None
    absolute_recovery_distance: (
        Annotated[float, Field(gt=0, allow_inf_nan=False)] | None
    ) = None

    @model_validator(mode="after")
    def ordered_thresholds(self) -> Self:
        if self.recovery_distance >= self.severe_distance:
            raise ValueError("recovery threshold must be below severe threshold")
        absolute = (
            self.absolute_distance_floor,
            self.absolute_severe_distance,
            self.absolute_recovery_distance,
        )
        if any(x is not None for x in absolute):
            if any(x is None for x in absolute):
                raise ValueError("absolute scale thresholds must be provided together")
            assert (
                self.absolute_severe_distance is not None
                and self.absolute_recovery_distance is not None
            )
            if self.absolute_recovery_distance >= self.absolute_severe_distance:
                raise ValueError("absolute recovery must be below severe threshold")
        return self


class PrivateOptimizerSpec(DomainModel):
    """One complete named group, ordinary dense FP32 PyTorch execution."""

    name: Literal["sgd", "adam", "adamw"]
    torch_version: PackageVersion
    parameter_names: tuple[Identifier, ...] = Field(min_length=1)
    learning_rate: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    weight_decay: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0
    momentum: Annotated[float, Field(ge=0, lt=1)] = 0
    dampening: Annotated[float, Field(ge=0)] = 0
    nesterov: bool = False
    betas: tuple[
        Annotated[float, Field(ge=0, lt=1)], Annotated[float, Field(ge=0, lt=1)]
    ] = (0.9, 0.999)
    numerical_epsilon: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1e-8
    amsgrad: Literal[False] = False
    foreach: Literal[False] = False
    fused: Literal[False] = False
    capturable: Literal[False] = False
    differentiable: Literal[False] = False
    maximize: Literal[False] = False

    @model_validator(mode="after")
    def valid_options(self) -> Self:
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("parameter names must be unique and ordered")
        if self.name != "sgd" and (self.momentum or self.dampening or self.nesterov):
            raise ValueError("momentum options require SGD")
        if self.nesterov and (not self.momentum or self.dampening):
            raise ValueError("Nesterov requires momentum and zero dampening")
        return self


class OptimizerParameterGroup(DomainModel):
    """Ordered parameter membership and resolved JSON optimizer options."""

    parameter_names: tuple[Identifier, ...] = Field(min_length=1, max_length=65536)
    options: dict[str, JsonValue]

    @field_validator("options", mode="before")
    @classmethod
    def bounded_json_options(cls, value: object) -> object:
        """Bound untrusted metadata before Pydantic's recursive JSON parsing."""
        pending: list[tuple[object, int]] = [(value, 0)]
        remaining = 4096
        while pending:
            item, depth = pending.pop()
            remaining -= 1
            if depth > 8 or remaining < 0:
                raise ValueError(
                    "optimizer options exceed metadata depth or size limit"
                )
            if isinstance(item, dict):
                mapping = cast(dict[object, object], item)
                if len(mapping) > remaining:
                    raise ValueError("optimizer options exceed metadata size limit")
                for key, child in mapping.items():
                    if not isinstance(key, str) or not key:
                        raise ValueError(
                            "optimizer option keys must be nonempty strings"
                        )
                    pending.append((child, depth + 1))
            elif isinstance(item, list):
                sequence = cast(list[object], item)
                if len(sequence) > remaining:
                    raise ValueError("optimizer options exceed metadata size limit")
                pending.extend((child, depth + 1) for child in sequence)
            elif item is not None and type(item) not in (str, bool, int, float):
                raise ValueError("optimizer options require JSON values")
            elif isinstance(item, float) and not math.isfinite(item):
                raise ValueError("optimizer options must contain finite numbers")
        if len(json.dumps(value, allow_nan=False).encode()) > 65536:
            raise ValueError("optimizer options exceed metadata size limit")
        return value

    @model_validator(mode="after")
    def valid_options(self) -> Self:
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("parameter names must be unique and ordered")
        if "params" in self.options:
            raise ValueError("parameter membership belongs in parameter_names")
        learning_rate = self.options.get("lr")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or learning_rate <= 0
        ):
            raise ValueError("optimizer lr must be a positive finite scalar")
        for flag in ("foreach", "fused", "capturable", "differentiable", "maximize"):
            if flag in self.options and self.options[flag] is not False:
                raise ValueError(f"optimizer {flag} must be explicitly false")
        return self


class FactoryOptimizerSpec(DomainModel):
    """Group-agreed implementation and state-validator identity for a factory."""

    kind: Literal["factory"] = "factory"
    implementation_id: Annotated[
        str, StringConstraints(min_length=1, max_length=256, pattern=r"^\S+$")
    ]
    implementation_hash: Sha256
    torch_version: PackageVersion
    groups: tuple[OptimizerParameterGroup, ...] = Field(min_length=1, max_length=32)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(name for group in self.groups for name in group.parameter_names)

    @model_validator(mode="after")
    def unique_parameters(self) -> Self:
        names = self.parameter_names
        if len(set(names)) != len(names):
            raise ValueError("optimizer parameter groups must not overlap")
        return self


GradientBackend = Literal["opacus_hooks", "opacus_native_b1"]


class AccountedInitializationAuthority(DomainModel):
    """An existing protected lineage whose spending precedes initialization."""

    lineage_id: Annotated[
        str, StringConstraints(min_length=1, max_length=256, pattern=r"\S")
    ]
    ledger_sha256: Sha256
    reserved_steps: Annotated[int, Field(gt=0)]
    reservation_count: Annotated[int, Field(gt=0)]


class AccountedInitializationProvenance(DomainModel):
    """Bound ancestry of a sanitized model, never a fresh privacy allowance."""

    parent_run_id: RunId
    parent_model_sha256: Sha256
    parent_audit_sha256: Sha256
    accounting_authorities: tuple[AccountedInitializationAuthority, ...] = Field(
        min_length=1, max_length=16
    )

    @model_validator(mode="after")
    def distinct_lineages(self) -> Self:
        names = [row.lineage_id for row in self.accounting_authorities]
        if len(names) != len(set(names)):
            raise ValueError("initialization accounting lineages must be unique")
        return self


class TaskTrainingSpec(DomainModel):
    """Shared semantics for an application-defined finite-record training task."""

    optimizer: PrivateOptimizerSpec | FactoryOptimizerSpec
    batch_size: Annotated[int, Field(gt=0)]
    sampling: Literal["poisson", "shuffle"] = "poisson"
    schedule: Literal["constant", "linear_decay"] = "constant"
    initialization: Literal[
        "public_untrained", "public_pretrained", "accounted_private_pretrained"
    ] = "public_untrained"
    initialization_provenance: AccountedInitializationProvenance | None = None
    evaluation_provenance: Literal["public_fixed_holdout", "disabled"] = "disabled"
    task_definition_hash: Sha256
    implementation_hash: Sha256
    batch_definition_hash: Sha256
    loss_definition_hash: Sha256
    record_unit: Annotated[
        str, StringConstraints(min_length=1, max_length=256, pattern=r"\S")
    ]
    gradient_backend: GradientBackend = "opacus_hooks"
    trainable_parameter_names: tuple[Identifier, ...] = Field(
        min_length=1, max_length=65536
    )
    parameter_aliases: tuple[tuple[Identifier, ...], ...] = Field(
        default=(), max_length=65536
    )
    public_buffer_hashes: dict[Identifier, Sha256] = Field(
        default_factory=dict, max_length=65536
    )
    frozen_parameter_hashes: dict[Identifier, Sha256] = Field(
        default_factory=dict, max_length=65536
    )

    @model_validator(mode="after")
    def parameter_state_is_unambiguous(self) -> Self:
        if (self.initialization == "accounted_private_pretrained") != (
            self.initialization_provenance is not None
        ):
            raise ValueError(
                "accounted private initialization requires exclusive provenance"
            )
        trainable = set(self.trainable_parameter_names)
        if len(trainable) != len(self.trainable_parameter_names):
            raise ValueError("trainable parameter names must be unique")
        if trainable != set(self.optimizer.parameter_names):
            raise ValueError(
                "optimizer parameter names must match trainable parameters"
            )
        frozen = set(self.frozen_parameter_hashes)
        if trainable & frozen:
            raise ValueError("trainable and frozen parameter names must not overlap")
        canonical = trainable | frozen
        aliases: set[str] = set()
        for group in self.parameter_aliases:
            if len(group) < 2:
                raise ValueError("parameter alias groups require at least two names")
            names = set(group)
            if len(names) != len(group):
                raise ValueError("parameter alias names must be distinct")
            if aliases & names:
                raise ValueError("parameter alias groups must be disjoint")
            if len(names & canonical) != 1:
                raise ValueError(
                    "alias group must identify exactly one canonical parameter"
                )
            aliases.update(names)
        if set(self.public_buffer_hashes) & (canonical | aliases):
            raise ValueError("buffer and parameter names must not overlap")
        return self


class PrivacyReservation(DomainModel):
    version: Literal[1] = 1
    run_id: RunId
    policy_hash: Sha256
    sample_rate: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
    noise_multiplier: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    logical_steps: Annotated[int, Field(gt=0)]
    accountant_version: Literal["opacus-1.6.0-prv"] = "opacus-1.6.0-prv"


class DraftRunSpec(DomainModel):
    """Shared run definition every participant must agree on before formation."""

    manifest_version: Literal[7] = MANIFEST_VERSION
    protocol_version: Literal[1] = PROTOCOL_VERSION
    run_id: RunId
    algorithm_id: AlgorithmId = NOLOCO_ALGORITHM_ID
    model_id: Identifier
    model_definition_hash: Sha256
    dataset: ApplicationTaskContract
    expected_participant_count: Annotated[
        int, Field(ge=MIN_PARTICIPANT_COUNT, le=MAX_PARTICIPANT_COUNT)
    ]
    environment: EnvironmentFingerprint
    local_steps: Annotated[int, Field(gt=0)]
    round_count: Annotated[int, Field(gt=0)]
    peer_scheduler_seed: int
    codec_id: Literal["safetensors"]
    transport: TransportLimits
    consensus_sketch: ConsensusSketchConfig
    application_training: ApplicationTrainingPolicy
    algorithm_config: NoLoCoConfig
    artifact_codecs: tuple[ArtifactCodec, ...]

    privacy: PrivacyPolicy | None = None
    private_training: TaskTrainingSpec | None = None
    divergence_detection: DivergencePolicy | None = None

    @property
    def participant_count(self) -> int:
        return self.expected_participant_count

    @model_validator(mode="after")
    def valid_manifest(self) -> Self:
        validate_participant_count(self.expected_participant_count)
        if self.algorithm_id != NOLOCO_ALGORITHM_ID:
            raise ValueError("manifest requires the noloco algorithm")
        if self.environment.model_definition_hash != self.model_definition_hash:
            raise ValueError("environment model hash does not match draft")
        if (self.privacy is None) != (self.private_training is None):
            raise ValueError("privacy policy and private training must be set together")
        if self.privacy is not None:
            assert self.private_training is not None
            if self.private_training.sampling != "poisson":
                raise ValueError("private tasks require Poisson sampling")
            if self.privacy.max_logical_steps != self.local_steps * self.round_count:
                raise ValueError("privacy horizon must match local steps and rounds")
        if self.local_steps != self.algorithm_config.inner_steps:
            raise ValueError("local steps must match NoLoCo inner steps")
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
            ("identity", "identity"),
            ("topk-bitmap-int8", "dense-int8"),
        }:
            raise ValueError("NoLoCo artifact codec combination is invalid")
        if (
            self.transport.chunk_size_bytes is None
            or self.transport.window_size is None
        ):
            raise ValueError("NoLoCo requires chunk size and window size")
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


class OpaqueArtifactMetadata(DomainModel):
    name: Identifier
    size_bytes: Annotated[int, Field(gt=0)]
    sha256: Sha256
    codec_id: Identifier
    logical_schema_hash: Sha256
    encoded_schema_hash: Sha256


class OpaqueUpdateBundleMetadata(DomainModel):
    """Bundle metadata carried as the first artifact of every update."""

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
