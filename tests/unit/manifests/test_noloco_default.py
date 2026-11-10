"""Outer algorithm selection must be explicit in sealed run identity."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from support.application_fixture import application_draft

from dromeus.manifests.canonical import canonical_hash, canonical_json
from dromeus.manifests.models import DraftRunSpec, SealedManifest, Tensor, TensorSchema
from dromeus.membership.formation import seal_manifest


def test_omitted_algorithm_defaults_to_noloco_with_identical_run_identity() -> None:
    explicit = application_draft()
    implicit = DraftRunSpec.model_validate(
        explicit.model_dump(mode="python", exclude={"algorithm_id"})
    )
    assert implicit.algorithm_id == "noloco"
    assert canonical_hash(implicit) == canonical_hash(explicit)
    assert canonical_json(implicit) == canonical_json(explicit)


@pytest.mark.parametrize("missing", ["algorithm_config", "artifact_codecs"])
def test_noloco_configuration_is_required_without_fallback(missing: str) -> None:
    value = application_draft().model_dump(
        mode="python", exclude={"algorithm_id", missing}
    )
    with pytest.raises(ValidationError, match=missing):
        DraftRunSpec.model_validate(value)


@pytest.mark.parametrize("algorithm", [None, "typo-noloco", "dpsgd"])
def test_explicit_non_noloco_algorithm_is_not_replaced_by_default(
    algorithm: str | None,
) -> None:
    value = application_draft().model_dump(mode="python")
    value["algorithm_id"] = algorithm
    with pytest.raises(ValueError):
        DraftRunSpec.model_validate(value)


def test_sealed_manifest_always_contains_algorithm_and_never_defaults_it() -> None:
    draft = application_draft()
    sealed = seal_manifest(
        draft=draft,
        participant_keys={f"peer-{index}" for index in range(4)},
        initial_checkpoint_hash="a" * 64,
        tensor_schema=TensorSchema(
            tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),)
        ),
    )
    assert sealed.algorithm_id == "noloco"
    assert b'"algorithm_id":"noloco"' in canonical_json(sealed)
    with pytest.raises(ValueError, match="algorithm_id"):
        SealedManifest.model_validate(
            sealed.model_dump(mode="python", exclude={"algorithm_id"})
        )
