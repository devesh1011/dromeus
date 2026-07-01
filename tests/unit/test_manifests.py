from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from sample_manifest import manifest_data

from dromeus.manifests.canonical import (
    canonical_hash,
    canonical_json,
    parse_draft_yaml,
    parse_sealed_json,
)
from dromeus.manifests.models import DraftRunSpec, SealedManifest

GOLDEN = Path(__file__).parents[1] / "golden" / "sealed_manifest.json"
GOLDEN_HASH = "a62697d1a1ff8ca6f026f783989e22ed6468e86e57abbfecd1790d6620c88674"


def test_canonical_manifest_matches_golden_file_and_hash() -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    golden = GOLDEN.read_bytes().rstrip(b"\n")

    assert canonical_json(manifest) == golden
    assert canonical_hash(manifest) == GOLDEN_HASH
    assert canonical_hash(parse_sealed_json(golden)) == canonical_hash(manifest)


def test_hash_is_stable_regardless_of_input_key_order() -> None:
    data = manifest_data()
    reversed_data = dict(reversed(tuple(data.items())))

    assert canonical_hash(SealedManifest.model_validate(data)) == canonical_hash(
        SealedManifest.model_validate(reversed_data)
    )


def test_draft_yaml_is_validated() -> None:
    data = manifest_data()
    for sealed_field in ("participants", "initial_checkpoint_hash", "tensor_schema"):
        del data[sealed_field]

    draft = parse_draft_yaml(json.dumps(data))

    assert isinstance(draft, DraftRunSpec)
    assert draft.run_id == "run-001"


@pytest.mark.parametrize("field", ["protocol_version", "manifest_version"])
def test_unknown_versions_are_rejected(field: str) -> None:
    data = manifest_data()
    data[field] = 2

    with pytest.raises(ValidationError):
        SealedManifest.model_validate(data)


def test_duplicate_participants_are_rejected() -> None:
    data = manifest_data()
    participants = data["participants"]
    assert isinstance(participants, list)
    participants[3] = participants[0]

    with pytest.raises(ValidationError, match="public keys must be unique"):
        SealedManifest.model_validate(data)


def test_sealed_manifest_rejects_secrets_and_paths() -> None:
    data = manifest_data()
    data["private_key"] = "secret"
    data["dataset_path"] = "/private/cifar"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SealedManifest.model_validate(data)
