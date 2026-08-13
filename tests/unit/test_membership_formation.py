from __future__ import annotations

import pytest
from support.sample_manifest import manifest_data

from dromeus.manifests.models import DraftRunSpec, SealedManifest
from dromeus.membership.formation import (
    ReadyValidationError,
    create_invitation,
    seal_manifest,
    validate_ready,
)


def test_validate_ready_accepts_matching_environment() -> None:
    manifest = SealedManifest.model_validate(manifest_data())

    validate_ready(
        manifest=manifest,
        local_public_key="peer-0",
        environment=manifest.environment,
        dataset=manifest.dataset,
        checkpoint_hash=manifest.initial_checkpoint_hash,
    )


def test_environment_mismatch_prevents_ready() -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    environment = manifest.environment.model_copy(update={"axl_version": "2.0.0"})

    with pytest.raises(
        ReadyValidationError, match="environment fingerprint does not match manifest"
    ):
        validate_ready(
            manifest=manifest,
            local_public_key="peer-0",
            environment=environment,
            dataset=manifest.dataset,
            checkpoint_hash=manifest.initial_checkpoint_hash,
        )


def test_v3_sealing_assigns_deterministic_indices_for_eight_members() -> None:
    base = SealedManifest.model_validate(manifest_data())
    draft_data = base.model_dump(mode="python")
    for field in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del draft_data[field]
    draft_data.update(
        {
            "manifest_version": 3,
            "algorithm_id": "noloco",
            "model_id": "resnet18-gn",
            "optimizer": "adam",
            "training": {
                "batch_size": 128,
                "momentum": 0.0,
                "weight_decay": 0.0,
                "learning_rate_milestones": [],
                "learning_rate_gamma": 0.1,
                "crop_padding": 0,
                "normalize": True,
                "final_consensus_rounds": 0,
            },
            "algorithm_config": {
                "alpha": 0.5,
                "beta": 0.7,
                "gamma": 0.7,
                "inner_steps": 50,
                "adam": {
                    "learning_rate": 0.001,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "epsilon": 1e-8,
                    "gradient_clip_norm": 1.0,
                },
            },
            "artifact_codecs": [
                {"artifact_name": "outer_gradient", "codec_id": "identity-v1"},
                {"artifact_name": "slow_weights", "codec_id": "identity-v1"},
            ],
            "transport": {
                **draft_data["transport"],
                "chunk_size_bytes": 1_048_576,
                "window_size": 4,
            },
            "dataset": {
                **draft_data["dataset"],
                "partition_sample_counts": [6_250] * 8,
                "node_index_partitions": list(range(8)),
            },
        }
    )
    draft = DraftRunSpec.model_validate(draft_data)
    keys = {f"peer-{index}" for index in range(8)}

    invitation = create_invitation(
        draft=draft,
        initiator_public_key="peer-0",
        bootstrap_uri="axl://bootstrap",
    )
    manifest = seal_manifest(
        draft=draft,
        participant_keys=keys,
        initial_checkpoint_hash=base.initial_checkpoint_hash,
        tensor_schema=base.tensor_schema,
    )

    assert invitation.expected_participant_count == 8
    assert [participant.node_index for participant in manifest.participants] == list(
        range(8)
    )
    assert [participant.public_key for participant in manifest.participants] == sorted(
        keys
    )
