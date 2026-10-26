"""The distributable digit policy remains bound through draft/sealed loading."""

from pathlib import Path

import pytest
import yaml
from support.application_fixture import application_draft
from support.sample_manifest import manifest_data

from dromeus.manifests.canonical import (
    canonical_hash,
    parse_draft_yaml,
    validate_sealed_draft,
)
from dromeus.manifests.models import DivergencePolicy, DraftRunSpec, SealedManifest


def test_validated_digit_policy_survives_deployment_loading(tmp_path: Path) -> None:
    # This synthetic draft checks configuration binding, not workload validation.
    path = Path(__file__).parents[3] / "examples/m3-digits-divergence-policy.json"
    policy = DivergencePolicy.model_validate_json(path.read_bytes())
    assert canonical_hash(policy) == (
        "d7a815d67ff04c9419231b89a400d63f2e951063c1c37ff67fef699cd407a448"
    )
    values = application_draft().model_dump(mode="python")
    values.update(manifest_version=5, divergence_detection=policy)
    draft = DraftRunSpec.model_validate(values)
    deployment_path = tmp_path / "digits-with-warnings.yaml"
    deployment_path.write_text(
        yaml.safe_dump(draft.model_dump(mode="json", exclude_none=True)),
        encoding="utf-8",
    )
    loaded = parse_draft_yaml(deployment_path)
    assert loaded.divergence_detection == policy
    assert canonical_hash(loaded) == canonical_hash(draft)

    original = manifest_data()
    values.update(
        {
            key: original[key]
            for key in ("participants", "initial_checkpoint_hash", "tensor_schema")
        }
    )
    values["draft_hash"] = canonical_hash(loaded)
    sealed = SealedManifest.model_validate(values)
    validate_sealed_draft(sealed, expected_draft_hash=canonical_hash(loaded))
    assert sealed.divergence_detection == policy

    # Substituting the former ratio under the same seal must fail.
    changed = policy.model_dump(mode="python")
    changed["growth_ratio"] = 2.0
    tampered = sealed.model_copy(
        update={"divergence_detection": DivergencePolicy.model_validate(changed)}
    )
    with pytest.raises(ValueError, match="sealed draft fields"):
        validate_sealed_draft(tampered)
