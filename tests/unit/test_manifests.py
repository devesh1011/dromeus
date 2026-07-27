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
)
from dromeus.manifests.models import (
    DraftRunSpec,
    Invitation,
    SealedManifest,
    TrainingPolicy,
)

GOLDEN = Path(__file__).parents[1] / "golden" / "sealed_manifest.json"
GOLDEN_HASH = "dd9ef12063cd632283f8c5fad1570d106f2876d6575e6190d49aa2cce101583b"


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


def test_resnet32_manifest_requires_versioned_training_policy() -> None:
    data = manifest_data()
    for field in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del data[field]
    data["model_id"] = "cifar-resnet32-v2"

    with pytest.raises(ValidationError, match="explicit training policy"):
        DraftRunSpec.model_validate(data)


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
