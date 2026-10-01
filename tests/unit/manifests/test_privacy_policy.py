"""M3 policy identity and legacy canonical compatibility."""

import pytest
from support.application_fixture import application_draft

from dromeus.manifests.canonical import canonical_hash
from dromeus.manifests.models import DraftRunSpec, PrivacyPolicy


def test_private_policy_is_bound_to_v5_draft() -> None:
    value = application_draft().model_dump(mode="python")
    value.update(
        manifest_version=5,
        privacy=PrivacyPolicy(
            max_grad_norm=1,
            noise_multiplier=1,
            delta=1e-5,
            max_logical_steps=9,
            randomness_profile="reproducible_public_benchmark",
        ),
    )
    draft = DraftRunSpec.model_validate(value)
    changed = dict(
        value, privacy=value["privacy"].model_copy(update={"noise_multiplier": 2})
    )
    assert canonical_hash(draft) != canonical_hash(DraftRunSpec.model_validate(changed))
    with pytest.raises(ValueError, match="version 5"):
        DraftRunSpec.model_validate(dict(value, manifest_version=4))


def test_sealed_draft_rejects_changed_private_policy() -> None:
    from support.sample_manifest import manifest_data

    from dromeus.manifests.canonical import validate_sealed_draft
    from dromeus.manifests.models import SealedManifest

    values = application_draft().model_dump(mode="python")
    values.update(
        manifest_version=5,
        privacy=PrivacyPolicy(
            max_grad_norm=1,
            noise_multiplier=1,
            delta=1e-5,
            max_logical_steps=9,
            randomness_profile="reproducible_public_benchmark",
        ),
    )
    draft = DraftRunSpec.model_validate(values)
    original = manifest_data()
    values.update(
        {
            k: original[k]
            for k in ("participants", "initial_checkpoint_hash", "tensor_schema")
        }
    )
    values["draft_hash"] = canonical_hash(draft)
    sealed = SealedManifest.model_validate(values)
    validate_sealed_draft(sealed)
    assert sealed.privacy is not None
    tampered = sealed.model_copy(
        update={"privacy": sealed.privacy.model_copy(update={"noise_multiplier": 2})}
    )
    with pytest.raises(ValueError, match="sealed draft fields"):
        validate_sealed_draft(tampered)
