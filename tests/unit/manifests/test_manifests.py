from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError
from support.paths import TESTS_ROOT
from support.sample_manifest import manifest_data

from dromeus.manifests.canonical import (
    canonical_hash,
    canonical_json,
    parse_draft_yaml,
    parse_sealed_json,
    update_bundle_digest,
)
from dromeus.manifests.models import (
    DraftRunSpec,
    Invitation,
    OpaqueArtifactMetadata,
    OpaqueUpdateBundleMetadata,
    SealedManifest,
)

GOLDEN = TESTS_ROOT / "golden" / "sealed_manifest.json"
GOLDEN_HASH = "12ddaac3a7df7b6c98090695eb4ec3555e399b4c78db1eeb1ae4265e74a005e2"
SEALED_FIELDS = (
    "draft_hash",
    "participants",
    "initial_checkpoint_hash",
    "tensor_schema",
)


def _artifact(name: str, marker: str = "a") -> OpaqueArtifactMetadata:
    return OpaqueArtifactMetadata(
        name=name,
        size_bytes=4,
        sha256=marker * 64,
        codec_id="safetensors",
        logical_schema_hash="b" * 64,
        encoded_schema_hash="b" * 64,
    )


def _bundle(
    artifacts: tuple[OpaqueArtifactMetadata, ...],
) -> OpaqueUpdateBundleMetadata:
    return OpaqueUpdateBundleMetadata(
        run_id="run-001",
        manifest_hash="1" * 64,
        sender_public_key="peer-0",
        algorithm_id="noloco",
        round_id=3,
        artifacts=artifacts,
    )


def _draft_fields(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key not in SEALED_FIELDS}


def _participants(count: int) -> list[dict[str, Any]]:
    return [
        {"public_key": f"peer-{index}", "node_index": index} for index in range(count)
    ]


def _membership_manifest_data(participant_count: int) -> dict[str, Any]:
    """Sealed manifest data for one fixed group size, with a recomputed draft hash."""
    data = manifest_data()
    data["expected_participant_count"] = participant_count
    data["participants"] = _participants(participant_count)
    data["draft_hash"] = canonical_hash(
        DraftRunSpec.model_validate(_draft_fields(data))
    )
    return data


def test_update_bundle_digest_binds_context_and_canonical_artifact_order() -> None:
    first = _bundle((_artifact("zeta"), _artifact("alpha", "c")))
    reversed_order = first.model_copy(
        update={"artifacts": tuple(reversed(first.artifacts))}
    )
    assert update_bundle_digest(first) == update_bundle_digest(reversed_order)
    changed_context = (
        first.model_copy(update={"run_id": "run-002"}),
        first.model_copy(update={"manifest_hash": "2" * 64}),
        first.model_copy(update={"sender_public_key": "peer-1"}),
        first.model_copy(update={"algorithm_id": "other-algorithm"}),
        first.model_copy(update={"round_id": 4}),
    )
    changed_artifact = first.artifacts[0].model_copy(
        update={
            "size_bytes": 5,
            "sha256": "d" * 64,
            "codec_id": "quantized",
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


def test_update_bundle_metadata_bounds_artifact_count_and_rejects_unknown_fields() -> (
    None
):
    assert len(_bundle((_artifact("only"),)).artifacts) == 1
    assert (
        len(_bundle(tuple(_artifact(f"artifact-{i}") for i in range(16))).artifacts)
        == 16
    )
    for artifacts in ((), tuple(_artifact(f"artifact-{i}") for i in range(17))):
        with pytest.raises(ValidationError):
            _bundle(artifacts)
    retired_field = {**_bundle((_artifact("only"),)).model_dump(), "version": 2}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        OpaqueUpdateBundleMetadata.model_validate(retired_field)


def test_canonical_manifest_matches_golden_file_and_hash() -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    golden = GOLDEN.read_bytes().rstrip(b"\n")

    assert canonical_json(manifest) == golden
    assert canonical_hash(manifest) == GOLDEN_HASH
    assert canonical_hash(parse_sealed_json(golden)) == canonical_hash(manifest)


@pytest.mark.parametrize("participant_count", [4, 8, 16])
def test_manifest_supports_deterministic_even_membership(
    participant_count: int,
) -> None:
    expected_hashes = {
        4: "12ddaac3a7df7b6c98090695eb4ec3555e399b4c78db1eeb1ae4265e74a005e2",
        8: "5d15598a8d9c7f2d5c4b71e7a1c43c784c39c4a0adb183097b198091cdab4158",
        16: "3c20861100322a9d1159c9e0b326d7e908888c2969f1c71be321e8c026b6ec8d",
    }
    first = SealedManifest.model_validate(_membership_manifest_data(participant_count))
    second = SealedManifest.model_validate(_membership_manifest_data(participant_count))

    assert first.manifest_version == 7
    assert len(first.participants) == participant_count
    assert first.participant_count == participant_count
    assert first.participants == second.participants
    assert (
        canonical_hash(first)
        == canonical_hash(second)
        == expected_hashes[participant_count]
    )
    assert parse_sealed_json(canonical_json(first)) == first


def test_manifest_rejects_odd_membership_with_typed_validation_error() -> None:
    data = _membership_manifest_data(8)
    data["participants"] = _participants(5)

    with pytest.raises(ValidationError, match="even"):
        SealedManifest.model_validate(data)


@pytest.mark.parametrize("version", [3, 4, 5, 6])
def test_retired_manifest_versions_are_rejected(version: int) -> None:
    data = manifest_data()
    data["manifest_version"] = version

    with pytest.raises(ValidationError, match="Input should be 7"):
        DraftRunSpec.model_validate(data)
    with pytest.raises(ValidationError, match="Input should be 7"):
        parse_draft_yaml(json.dumps(data))
    with pytest.raises(ValidationError, match="Input should be 7"):
        parse_sealed_json(json.dumps(data))


@pytest.mark.parametrize(
    ("field", "version"),
    [("protocol_version", 2), ("manifest_version", 8)],
)
def test_unknown_versions_are_rejected(field: str, version: int) -> None:
    data = manifest_data()
    data[field] = version

    with pytest.raises(ValidationError):
        SealedManifest.model_validate(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("inner_steps", 49),
        ("alpha", 1.0),
        ("beta", 0.0),
    ],
)
def test_noloco_config_rejects_out_of_contract_values(field: str, value: float) -> None:
    data = manifest_data()
    data["algorithm_config"][field] = value

    with pytest.raises(ValidationError):
        SealedManifest.model_validate(data)


def test_noloco_local_steps_must_match_inner_steps() -> None:
    data = manifest_data()
    data["algorithm_config"]["inner_steps"] = 49

    with pytest.raises(ValidationError, match="local steps must match NoLoCo"):
        SealedManifest.model_validate(data)


def test_noloco_accepts_only_identity_or_complete_compressed_pair() -> None:
    data = manifest_data()
    data["artifact_codecs"] = [
        {
            "artifact_name": "outer_gradient",
            "codec_id": "topk-bitmap-int8",
            "top_k_fraction": 0.01,
            "lossy_allowed": True,
        },
        {
            "artifact_name": "slow_weights",
            "codec_id": "dense-int8",
            "lossy_allowed": True,
        },
    ]

    draft = DraftRunSpec.model_validate(_draft_fields(data))
    assert draft.artifact_codecs[0].lossy_allowed is True

    invalid = {
        **_draft_fields(data),
        "artifact_codecs": [
            {
                "artifact_name": "outer_gradient",
                "codec_id": "topk-bitmap-int8",
                "top_k_fraction": 0.01,
                "lossy_allowed": True,
            },
            {"artifact_name": "slow_weights", "codec_id": "identity"},
        ],
    }
    with pytest.raises(ValidationError, match="combination"):
        DraftRunSpec.model_validate(invalid)


@pytest.mark.parametrize(
    "codec_id", ["identity-v1", "dense-int8-v1", "topk-bitmap-int8-v2"]
)
def test_retired_artifact_codec_ids_are_rejected(codec_id: str) -> None:
    data = manifest_data()
    data["artifact_codecs"][0]["codec_id"] = codec_id

    with pytest.raises(ValidationError, match="unsupported NoLoCo artifact codec"):
        DraftRunSpec.model_validate(_draft_fields(data))


def test_retired_transfer_codec_id_is_rejected() -> None:
    data = manifest_data()
    data["codec_id"] = "safetensors-v1"

    with pytest.raises(ValidationError):
        SealedManifest.model_validate(data)


def test_manifest_requires_noloco_algorithm_and_executable_identifiers() -> None:
    draft = _draft_fields(manifest_data())
    draft["algorithm_id"] = "other-algorithm"
    draft["model_id"] = "other-model"

    with pytest.raises(ValidationError, match="manifest requires the noloco algorithm"):
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
    for sealed_field in SEALED_FIELDS:
        del data[sealed_field]

    draft = parse_draft_yaml(json.dumps(data))

    assert isinstance(draft, DraftRunSpec)
    assert draft.run_id == "run-001"


def test_environment_accepts_cpu_wheel_version() -> None:
    data = manifest_data()
    data["environment"]["pytorch_version"] = "2.13.0+cpu"

    assert (
        SealedManifest.model_validate(data).environment.pytorch_version == "2.13.0+cpu"
    )


def test_duplicate_participants_are_rejected() -> None:
    data = manifest_data()
    data["participants"][3] = data["participants"][0]

    with pytest.raises(ValidationError, match="public keys must be unique"):
        SealedManifest.model_validate(data)


def test_invalid_node_index_mapping_is_rejected() -> None:
    data = manifest_data()
    data["participants"][3]["node_index"] = 1

    with pytest.raises(
        ValidationError, match="participant node indices must be exactly 0 through 3"
    ):
        SealedManifest.model_validate(data)


def test_sealed_manifest_rejects_secrets_and_paths() -> None:
    data = manifest_data()
    data["private_key"] = "secret"
    data["dataset_path"] = "/private/node-0/data"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SealedManifest.model_validate(data)
