"""Project one frozen NoLoCo run into Dromeus manifest inputs."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from benchmarks.noloco.experiment import AblationId, ResolvedRun
from dromeus.manifests.canonical import (
    canonical_hash,
    validate_sealed_expectation,
)
from dromeus.manifests.models import (
    AdamSettings,
    ConsensusSketchConfig,
    DatasetContract,
    DraftRunSpec,
    EnvironmentFingerprint,
    NoLoCoConfig,
    Participant,
    SealedManifest,
    SealedManifestExpectation,
    TrainingPolicy,
    TransportLimits,
)
from dromeus.training.trainer import derive_benchmark_seed


class DromeusAdapterError(ValueError):
    """A frozen run cannot be projected into valid Dromeus inputs."""


class DromeusComparabilityError(DromeusAdapterError):
    """Dromeus formation differs from the resolved frozen run."""


@dataclass(frozen=True, slots=True)
class DromeusExperimentAdapter:
    """One-way adapter from a resolved experiment to Dromeus values."""

    run: ResolvedRun

    def build_draft(
        self,
        *,
        run_id: str,
        environment: EnvironmentFingerprint,
        transport: TransportLimits,
        ablation_id: AblationId = "identity",
    ) -> DraftRunSpec:
        """Build one NoLoCo draft from frozen semantics and codec selection."""
        if environment.model_definition_hash != self.run.model.definition_hash:
            raise DromeusComparabilityError(
                "environment model definition does not match frozen run"
            )
        hardware = self.run.hardware
        if (
            environment.container_image_digest != hardware.container_image_digest
            or environment.pytorch_version != hardware.pytorch_version
            or environment.axl_version != hardware.axl_commit
        ):
            raise DromeusComparabilityError(
                "environment runtime does not match frozen run"
            )
        frozen_transfer = self.run.run.transfer
        if (
            transport.chunk_size_bytes != frozen_transfer.chunk_size_bytes
            or transport.window_size != frozen_transfer.window_size
        ):
            raise DromeusComparabilityError(
                "transport chunk/window does not match frozen run"
            )
        try:
            ablation = self.run.codec_ablation(ablation_id)
        except ValueError as error:
            raise DromeusAdapterError(str(error)) from error
        world_size = self.run.selector.world_size
        partition_size = self.run.dataset.sample_count // world_size
        try:
            return DraftRunSpec(
                run_id=run_id,
                algorithm_id="noloco",
                model_id=self.run.model.model_id,
                model_definition_hash=self.run.model.definition_hash,
                dataset=DatasetContract(
                    dataset_id="cifar10",
                    version=f"huggingface-{self.run.dataset.revision}",
                    preprocessing_hash=self.run.dataset.preprocessing_hash,
                    iid_partition_seed=self.run.dataset.partition_seed,
                    image_shape=(3, 32, 32),
                    class_count=10,
                    sample_count=self.run.dataset.sample_count,
                    partition_sample_counts=(partition_size,) * world_size,
                    node_index_partitions=tuple(range(world_size)),
                ),
                environment=environment,
                local_steps=self.run.algorithm.inner_steps,
                round_count=self.run.run.round_count,
                optimizer="adam",
                learning_rate=self.run.algorithm.adam_learning_rate,
                peer_scheduler_seed=self.run.run.scheduler_seed,
                codec_id="safetensors-v1",
                transport=transport,
                consensus_sketch=ConsensusSketchConfig(
                    seed=derive_benchmark_seed(
                        self.run.selector.benchmark_seed,
                        "consensus-sketch",
                    )
                ),
                training=TrainingPolicy(
                    batch_size=self.run.workload.batch_size,
                    momentum=0.0,
                    weight_decay=self.run.workload.weight_decay,
                    learning_rate_milestones=(),
                    learning_rate_gamma=0.1,
                    learning_rate_schedule=self.run.run.schedule,
                    crop_padding=self.run.workload.crop_padding,
                    normalize=self.run.workload.normalize,
                    final_consensus_rounds=0,
                ),
                algorithm_config=NoLoCoConfig(
                    alpha=self.run.algorithm.alpha,
                    beta=self.run.algorithm.beta,
                    gamma=self.run.algorithm.gamma,
                    inner_steps=self.run.algorithm.inner_steps,
                    adam=AdamSettings(
                        learning_rate=self.run.algorithm.adam_learning_rate,
                        beta1=self.run.algorithm.adam_beta1,
                        beta2=self.run.algorithm.adam_beta2,
                        epsilon=self.run.algorithm.adam_epsilon,
                        gradient_clip_norm=(
                            self.run.algorithm.gradient_clip_norm
                        ),
                    ),
                ),
                artifact_codecs=ablation.artifact_codecs,
            )
        except ValidationError as error:
            raise DromeusAdapterError(
                f"cannot project frozen run into Dromeus draft: {error}"
            ) from error

    def manifest_expectation(
        self,
        *,
        draft: DraftRunSpec,
    ) -> SealedManifestExpectation:
        """Return preflight constraints embedded into every node config."""
        return SealedManifestExpectation(
            draft_hash=canonical_hash(draft),
            participants=tuple(
                Participant(
                    public_key=item.public_key,
                    node_index=item.node_index,
                )
                for item in self.run.run.participants
            ),
            initial_checkpoint_hash=self.run.run.checkpoint.sha256,
            tensor_schema_hash=self.run.model.tensor_schema_hash,
        )

    def validate_sealed_manifest(
        self,
        manifest: SealedManifest,
        *,
        draft: DraftRunSpec,
    ) -> None:
        """Validate formed Dromeus authority before training starts."""
        try:
            sealed_draft = DraftRunSpec.model_validate(
                manifest.model_dump(
                    mode="python",
                    exclude={
                        "draft_hash",
                        "participants",
                        "initial_checkpoint_hash",
                        "tensor_schema",
                    },
                )
            )
        except ValidationError as error:
            raise DromeusComparabilityError(
                "sealed manifest draft fields are invalid"
            ) from error
        if sealed_draft != draft:
            raise DromeusComparabilityError(
                "sealed manifest draft fields do not match projected draft"
            )
        try:
            validate_sealed_expectation(
                self.manifest_expectation(draft=draft),
                manifest,
            )
        except ValueError as error:
            raise DromeusComparabilityError(str(error)) from error


def benchmark_transport_limits(run: ResolvedRun) -> TransportLimits:
    """Return the single frozen transport policy shared by benchmark launchers."""
    return TransportLimits(
        max_payload_bytes=128 * 1024 * 1024,
        max_retries=3,
        retry_timeout_seconds=5.0,
        chunk_size_bytes=run.run.transfer.chunk_size_bytes,
        window_size=run.run.transfer.window_size,
        max_message_payload_bytes=2 * 1024 * 1024,
        max_artifact_bytes=64 * 1024 * 1024,
        max_concurrent_transfers=4,
        max_inflight_bytes=8 * 1024 * 1024,
        transfer_lifetime_seconds=60.0,
        artifact_store_capacity_bytes=256 * 1024 * 1024,
    )


__all__ = [
    "DromeusAdapterError",
    "DromeusComparabilityError",
    "DromeusExperimentAdapter",
    "benchmark_transport_limits",
]
