from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from support.sample_manifest import manifest_data

from dromeus.manifests.canonical import (
    canonical_hash,
    canonical_json,
    parse_draft_yaml,
    parse_sealed_json,
    update_bundle_digest,
)
from dromeus.manifests.models import (
    ArtifactMetadata,
    DraftRunSpec,
    Invitation,
    OpaqueArtifactMetadata,
    OpaqueUpdateBundleMetadata,
    SealedManifest,
    TrainingPolicy,
    UpdateBundleMetadata,
)

GOLDEN = Path(__file__).parents[1] / "golden" / "sealed_manifest.json"
GOLDEN_HASH = "dd9ef12063cd632283f8c5fad1570d106f2876d6575e6190d49aa2cce101583b"


def _artifact(name: str, marker: str = "a") -> OpaqueArtifactMetadata:
    return OpaqueArtifactMetadata(
        name=name,
        size_bytes=4,
        sha256=marker * 64,
        codec_id="safetensors",
        codec_version=1,
        logical_schema_hash="b" * 64,
        encoded_schema_hash="b" * 64,
    )


def test_update_bundle_digest_binds_context_and_canonical_artifact_order() -> None:
    first = OpaqueUpdateBundleMetadata(
        run_id="run-001",
        manifest_hash="1" * 64,
        sender_public_key="peer-0",
        algorithm_id="dpsgd",
        round_id=3,
        artifacts=(_artifact("zeta"), _artifact("alpha", "c")),
    )
    reversed_order = first.model_copy(
        update={"artifacts": tuple(reversed(first.artifacts))}
    )
    assert update_bundle_digest(first) == update_bundle_digest(reversed_order)
    changed_context = (
        first.model_copy(update={"run_id": "run-002"}),
        first.model_copy(update={"manifest_hash": "2" * 64}),
        first.model_copy(update={"sender_public_key": "peer-1"}),
        first.model_copy(update={"algorithm_id": "noloco"}),
        first.model_copy(update={"round_id": 4}),
    )
    changed_artifact = first.artifacts[0].model_copy(
        update={
            "size_bytes": 5,
            "sha256": "d" * 64,
            "codec_id": "quantized",
            "codec_version": 2,
            "logical_schema_hash": "e" * 64,
            "encoded_schema_hash": "f" * 64,
        }
    )
    changed_metadata = first.model_copy(
        update={"artifacts": (changed_artifact, first.artifacts[1])}
    )

    assert all(
        update_bundle_digest(first) != update_bundle_digest(changed)
        for changed in (*changed_context, changed_metadata)
    )


def test_update_bundle_metadata_bounds_artifact_count_and_version() -> None:
    assert len(
        OpaqueUpdateBundleMetadata(
            run_id="run-001",
            manifest_hash="1" * 64,
            sender_public_key="peer-0",
            algorithm_id="dpsgd",
            round_id=0,
            artifacts=(_artifact("only"),),
        ).artifacts
    ) == 1
    assert len(
        OpaqueUpdateBundleMetadata(
            run_id="run-001",
            manifest_hash="1" * 64,
            sender_public_key="peer-0",
            algorithm_id="dpsgd",
            round_id=0,
            artifacts=tuple(_artifact(f"artifact-{index}") for index in range(16)),
        ).artifacts
    ) == 16

    for artifacts in (
        (),
        tuple(_artifact(f"artifact-{index}") for index in range(17)),
    ):
        with pytest.raises(ValidationError):
            OpaqueUpdateBundleMetadata(
                run_id="run-001",
                manifest_hash="1" * 64,
                sender_public_key="peer-0",
                algorithm_id="dpsgd",
                round_id=0,
                artifacts=artifacts,
            )
    unsupported = {
        "version": 3,
        "run_id": "run-001",
        "manifest_hash": "1" * 64,
        "sender_public_key": "peer-0",
        "algorithm_id": "dpsgd",
        "round_id": 0,
        "artifacts": (_artifact("only"),),
    }
    with pytest.raises(ValidationError):
        OpaqueUpdateBundleMetadata.model_validate(unsupported)


def test_historical_update_bundle_metadata_v1_remains_parseable() -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    metadata = UpdateBundleMetadata(
        run_id="run-001",
        manifest_hash="1" * 64,
        sender_public_key="peer-0",
        algorithm_id="dpsgd",
        round_id=0,
        artifacts=(
            ArtifactMetadata(
                name="model-update",
                size_bytes=4,
                sha256="a" * 64,
                tensor_schema=manifest.tensor_schema,
            ),
        ),
    )

    assert metadata.version == 1
    assert OpaqueUpdateBundleMetadata(
        run_id="run-001",
        manifest_hash="1" * 64,
        sender_public_key="peer-0",
        algorithm_id="dpsgd",
        round_id=0,
        artifacts=(_artifact("model-update"),),
    ).version == 2


def test_canonical_manifest_matches_golden_file_and_hash() -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    golden = GOLDEN.read_bytes().rstrip(b"\n")

    assert canonical_json(manifest) == golden
    assert canonical_hash(manifest) == GOLDEN_HASH
    assert canonical_hash(parse_sealed_json(golden)) == canonical_hash(manifest)


def test_training_policy_validates_quality_recipe() -> None:
    policy = TrainingPolicy(
        batch_size=128,
        momentum=0.9,
        weight_decay=1e-4,
        learning_rate_milestones=(8_000, 12_000),
        learning_rate_gamma=0.1,
        crop_padding=4,
        normalize=True,
        final_consensus_rounds=2,
    )

    assert policy.batch_size == 128
    assert policy.learning_rate_milestones == (8_000, 12_000)

    with pytest.raises(ValidationError, match="strictly increasing"):
        TrainingPolicy(
            batch_size=128,
            momentum=0.9,
            weight_decay=1e-4,
            learning_rate_milestones=(12_000, 8_000),
            learning_rate_gamma=0.1,
            crop_padding=4,
            normalize=True,
            final_consensus_rounds=2,
        )


def test_active_manifest_requires_training_policy() -> None:
    data = manifest_data()
    for field in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del data[field]
    data["manifest_version"] = 2

    with pytest.raises(ValidationError, match="requires"):
        DraftRunSpec.model_validate(data)


def test_historical_manifest_rejects_training_policy() -> None:
    data = manifest_data()
    for field in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del data[field]
    data["training"] = {
        "batch_size": 128,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "learning_rate_milestones": [8000, 12000],
        "learning_rate_gamma": 0.1,
        "crop_padding": 4,
        "normalize": True,
        "final_consensus_rounds": 2,
    }

    with pytest.raises(ValidationError, match="excludes"):
        DraftRunSpec.model_validate(data)


def test_active_manifest_enforces_executable_identifiers() -> None:
    draft = manifest_data()
    for field in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del draft[field]
    draft["manifest_version"] = 2
    draft["algorithm_id"] = "other-algorithm"
    draft["model_id"] = "other-model"
    draft["training"] = {
        "batch_size": 128,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "learning_rate_milestones": [8000, 12000],
        "learning_rate_gamma": 0.1,
        "crop_padding": 4,
        "normalize": True,
        "final_consensus_rounds": 2,
    }

    with pytest.raises(ValidationError, match="requires dpsgd and resnet32"):
        DraftRunSpec.model_validate(draft)


def test_hash_is_stable_regardless_of_input_key_order() -> None:
    data = manifest_data()
    reversed_data = dict(reversed(tuple(data.items())))

    assert canonical_hash(SealedManifest.model_validate(data)) == canonical_hash(
        SealedManifest.model_validate(reversed_data)
    )


def test_invitation_is_emitted_as_canonical_json() -> None:
    invitation = Invitation(
        run_id="run-001",
        initiator_public_key="peer-0",
        bootstrap_uri="axl://bootstrap.example",
        draft_hash="3" * 64,
    )

    assert canonical_json(invitation) == (
        b'{"bootstrap_uri":"axl://bootstrap.example","draft_hash":"3333333333333333333333333333333333333333333333333333333333333333","expected_participant_count":4,"initiator_public_key":"peer-0","protocol_version":1,"run_id":"run-001"}'
    )


def test_draft_yaml_is_validated() -> None:
    data = manifest_data()
    for sealed_field in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del data[sealed_field]

    draft = parse_draft_yaml(json.dumps(data))

    assert isinstance(draft, DraftRunSpec)
    assert draft.run_id == "run-001"


def test_environment_accepts_cpu_wheel_version() -> None:
    data = manifest_data()
    environment = data["environment"]
    assert isinstance(environment, dict)
    environment["pytorch_version"] = "2.13.0+cpu"

    assert (
        SealedManifest.model_validate(data).environment.pytorch_version
        == "2.13.0+cpu"
    )


@pytest.mark.parametrize(
    ("field", "version"),
    [("protocol_version", 2), ("manifest_version", 3)],
)
def test_unknown_versions_are_rejected(field: str, version: int) -> None:
    data = manifest_data()
    data[field] = version

    with pytest.raises(ValidationError):
        SealedManifest.model_validate(data)


def test_duplicate_participants_are_rejected() -> None:
    data = manifest_data()
    participants = data["participants"]
    assert isinstance(participants, list)
    participants[3] = participants[0]

    with pytest.raises(ValidationError, match="public keys must be unique"):
        SealedManifest.model_validate(data)


def test_invalid_node_index_mapping_is_rejected() -> None:
    data = manifest_data()
    dataset = data["dataset"]
    assert isinstance(dataset, dict)
    dataset["node_index_partitions"] = [0, 1, 1, 3]

    with pytest.raises(
        ValidationError, match="node index partitions must be exactly 0 through 3"
    ):
        SealedManifest.model_validate(data)


def test_sealed_manifest_rejects_secrets_and_paths() -> None:
    data = manifest_data()
    data["private_key"] = "secret"
    data["dataset_path"] = "/private/cifar"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SealedManifest.model_validate(data)
