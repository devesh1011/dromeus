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
GOLDEN_HASH = "0019f8a0cf68272fffd536d51ddd491d9f80036a3ee3bdcfe008a9fc585b3907"


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


def _v3_manifest_data(participant_count: int = 8) -> dict[str, object]:
    data = manifest_data()
    data.update(
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
            "participants": [
                {"public_key": f"peer-{index}", "node_index": index}
                for index in range(participant_count)
            ],
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
        }
    )
    data["dataset"] = {
        **data["dataset"],  # type: ignore[misc]
        "sample_count": 50_000,
        "partition_sample_counts": [50_000 // participant_count]
        * participant_count,
        "node_index_partitions": list(range(participant_count)),
    }
    data["transport"] = {
        **data["transport"],  # type: ignore[misc]
        "chunk_size_bytes": 1_048_576,
        "window_size": 4,
    }
    for field in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del data[field]
    draft = DraftRunSpec.model_validate(data)
    sealed = manifest_data()
    sealed.update(data)
    sealed["draft_hash"] = canonical_hash(draft)
    sealed["participants"] = [
        {"public_key": f"peer-{index}", "node_index": index}
        for index in range(participant_count)
    ]
    sealed["initial_checkpoint_hash"] = "2" * 64
    sealed["tensor_schema"] = manifest_data()["tensor_schema"]
    return sealed


@pytest.mark.parametrize("participant_count", [4, 8, 16])
def test_manifest_v3_supports_deterministic_even_membership(
    participant_count: int,
) -> None:
    expected_hashes = {
        4: "4e154474ba978c6ce1eb0484cf7f24fff62e0e497768682315bc32375f5af775",
        8: "622057960f05a8c8e4750b85c0b8f094933f25b91d2f2a161e1c6d4527f85fa5",
        16: "e081033b465a4597472ce1f7e87dd4e07f99d45bd7f1080389afb456537c62a6",
    }
    first = SealedManifest.model_validate(_v3_manifest_data(participant_count))
    second = SealedManifest.model_validate(_v3_manifest_data(participant_count))

    assert first.manifest_version == 3
    assert len(first.participants) == participant_count
    assert first.dataset.participant_count == participant_count
    assert first.participants == second.participants
    assert (
        canonical_hash(first)
        == canonical_hash(second)
        == expected_hashes[participant_count]
    )
    assert parse_sealed_json(canonical_json(first)) == first


def test_manifest_v3_rejects_odd_membership_with_typed_validation_error() -> None:
    data = _v3_manifest_data(8)
    participants = data["participants"]
    dataset = data["dataset"]
    assert isinstance(participants, list)
    assert isinstance(dataset, dict)
    data["participants"] = participants[:5]
    dataset["sample_count"] = 50_000
    dataset["partition_sample_counts"] = [10_000] * 5
    dataset["node_index_partitions"] = list(range(5))

    with pytest.raises(ValidationError, match="even"):
        SealedManifest.model_validate(data)


def test_manifest_versions_before_v3_are_rejected() -> None:
    data = manifest_data()
    data["manifest_version"] = 2

    with pytest.raises(ValidationError, match="Input should be 3"):
        DraftRunSpec.model_validate(data)
    with pytest.raises(ValidationError, match="Input should be 3"):
        parse_draft_yaml(json.dumps(data))
    with pytest.raises(ValidationError, match="Input should be 3"):
        parse_sealed_json(json.dumps(data))


def test_manifest_v3_rejects_final_consensus_for_noloco() -> None:
    data = _v3_manifest_data(4)
    training = data["training"]
    assert isinstance(training, dict)
    training["final_consensus_rounds"] = 2
    with pytest.raises(ValidationError):
        SealedManifest.model_validate(data)


def test_dpsgd_rejects_adam_optimizer() -> None:
    data = manifest_data()
    data["optimizer"] = "adam"

    with pytest.raises(ValidationError, match="dpsgd requires sgd"):
        DraftRunSpec.model_validate(
            {
                key: value
                for key, value in data.items()
                if key not in {
                    "draft_hash",
                    "participants",
                    "initial_checkpoint_hash",
                    "tensor_schema",
                }
            }
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("alpha", 0.6, "alpha=0.5"),
        ("inner_steps", 49, "50 inner steps"),
    ],
)
def test_noloco_rejects_unfrozen_hyperparameters(
    field: str,
    value: float | int,
    message: str,
) -> None:
    data = _v3_manifest_data()
    algorithm_config = data["algorithm_config"]
    assert isinstance(algorithm_config, dict)
    algorithm_config[field] = value

    with pytest.raises(ValidationError, match=message):
        SealedManifest.model_validate(data)


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
    data["training"] = None

    with pytest.raises(ValidationError, match="requires training policy"):
        DraftRunSpec.model_validate(data)


def test_manifest_v3_enforces_executable_identifiers() -> None:
    draft = manifest_data()
    for field in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del draft[field]
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

    with pytest.raises(ValidationError, match="requires dpsgd or noloco"):
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
    [("protocol_version", 2), ("manifest_version", 4)],
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
