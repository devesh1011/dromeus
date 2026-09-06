"""Pre-freeze four-node NoLoCo pilot candidate configuration."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import yaml

from dromeus.manifests.models import (
    AdamSettings,
    ArtifactCodec,
    ConsensusSketchConfig,
    DatasetContract,
    DraftRunSpec,
    EnvironmentFingerprint,
    NoLoCoConfig,
    TrainingPolicy,
    TransportLimits,
    WarmupCosineSchedule,
)
from dromeus.training.cifar10 import (
    DATASET_VERSION,
    PREPROCESSING_HASH,
)
from dromeus.training.resnet18_groupnorm import (
    MODEL_DEFINITION_HASH,
    MODEL_ID,
)
from dromeus.training.trainer import derive_benchmark_seed

AblationId = Literal["identity", "compressed"]
CompressedCodecId = Literal["topk-int8-v1", "topk-bitmap-int8-v2"]


@dataclass(frozen=True, slots=True)
class PilotCandidate:
    """One matched candidate evaluated before values become immutable."""

    benchmark_seed: int = 17
    round_count: int = 500
    top_k_fraction: float = 0.45
    chunk_size_bytes: int = 1024 * 1024
    window_size: int = 4
    warmup_inner_steps: int = 1000
    start_learning_rate: float = 0.0001
    peak_learning_rate: float = 0.001
    final_learning_rate: float = 0.0001
    compressed_codec_id: CompressedCodecId = "topk-bitmap-int8-v2"

    def __post_init__(self) -> None:
        if self.round_count <= 0:
            raise ValueError("pilot round count must be positive")
        if not 0 < self.top_k_fraction <= 1:
            raise ValueError("pilot top-k fraction must be in (0, 1]")
        if self.warmup_inner_steps >= self.round_count * 50:
            raise ValueError("pilot warm-up must end before the final inner step")

    def build_draft(
        self,
        *,
        run_id: str,
        ablation_id: AblationId,
        environment: EnvironmentFingerprint,
    ) -> DraftRunSpec:
        """Build one candidate draft; only the codec ablation may differ."""
        return DraftRunSpec(
            run_id=run_id,
            algorithm_id="noloco",
            model_id=MODEL_ID,
            model_definition_hash=MODEL_DEFINITION_HASH,
            dataset=DatasetContract(
                dataset_id="cifar10",
                version=DATASET_VERSION,
                preprocessing_hash=PREPROCESSING_HASH,
                iid_partition_seed=7,
                image_shape=(3, 32, 32),
                class_count=10,
                sample_count=50_000,
                partition_sample_counts=(12_500,) * 4,
                node_index_partitions=(0, 1, 2, 3),
            ),
            environment=environment,
            local_steps=50,
            round_count=self.round_count,
            optimizer="adam",
            learning_rate=self.peak_learning_rate,
            peer_scheduler_seed=self.benchmark_seed,
            codec_id="safetensors-v1",
            transport=TransportLimits(
                max_payload_bytes=128 * 1024 * 1024,
                max_retries=3,
                retry_timeout_seconds=5.0,
                chunk_size_bytes=self.chunk_size_bytes,
                window_size=self.window_size,
                max_message_payload_bytes=2 * 1024 * 1024,
                max_artifact_bytes=64 * 1024 * 1024,
                max_concurrent_transfers=4,
                max_inflight_bytes=8 * 1024 * 1024,
                transfer_lifetime_seconds=60.0,
                artifact_store_capacity_bytes=256 * 1024 * 1024,
            ),
            consensus_sketch=ConsensusSketchConfig(
                seed=derive_benchmark_seed(
                    self.benchmark_seed,
                    "consensus-sketch",
                )
            ),
            training=TrainingPolicy(
                batch_size=128,
                momentum=0.0,
                weight_decay=0.0,
                learning_rate_milestones=(),
                learning_rate_gamma=0.1,
                learning_rate_schedule=WarmupCosineSchedule(
                    schedule_id="linear-warmup-cosine-v1",
                    total_inner_steps=self.round_count * 50,
                    warmup_inner_steps=self.warmup_inner_steps,
                    start_learning_rate=self.start_learning_rate,
                    peak_learning_rate=self.peak_learning_rate,
                    final_learning_rate=self.final_learning_rate,
                ),
                crop_padding=4,
                normalize=True,
                final_consensus_rounds=0,
            ),
            algorithm_config=NoLoCoConfig(
                alpha=0.5,
                beta=0.7,
                gamma=0.7,
                inner_steps=50,
                adam=AdamSettings(
                    learning_rate=self.peak_learning_rate,
                    beta1=0.9,
                    beta2=0.999,
                    epsilon=1e-8,
                    gradient_clip_norm=1.0,
                ),
            ),
            artifact_codecs=_artifact_codecs(
                ablation_id,
                top_k_fraction=self.top_k_fraction,
                compressed_codec_id=self.compressed_codec_id,
            ),
        )


def _artifact_codecs(
    ablation_id: AblationId,
    *,
    top_k_fraction: float,
    compressed_codec_id: CompressedCodecId,
) -> tuple[ArtifactCodec, ArtifactCodec]:
    if ablation_id == "identity":
        return (
            ArtifactCodec(
                artifact_name="outer_gradient",
                codec_id="identity-v1",
            ),
            ArtifactCodec(
                artifact_name="slow_weights",
                codec_id="identity-v1",
            ),
        )
    return (
        ArtifactCodec(
            artifact_name="outer_gradient",
            codec_id=compressed_codec_id,
            top_k_fraction=top_k_fraction,
            lossy_allowed=True,
        ),
        ArtifactCodec(
            artifact_name="slow_weights",
            codec_id="dense-int8-v1",
            lossy_allowed=True,
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ablation", choices=("identity", "compressed"), required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    environment = EnvironmentFingerprint.model_validate_json(
        arguments.environment.read_text(encoding="utf-8")
    )
    draft = PilotCandidate().build_draft(
        run_id=arguments.run_id,
        ablation_id=cast(AblationId, arguments.ablation),
        environment=environment,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        yaml.safe_dump(draft.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PilotCandidate"]
