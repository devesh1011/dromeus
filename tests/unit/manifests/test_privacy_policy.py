"""Privacy policy identity is bound to the draft and sealed manifest."""

import pytest
from support.application_fixture import application_draft
from support.sample_manifest import manifest_data

from dromeus.manifests.canonical import canonical_hash, validate_sealed_draft
from dromeus.manifests.models import (
    DraftRunSpec,
    PrivacyPolicy,
    PrivateOptimizerSpec,
    SealedManifest,
    TaskTrainingSpec,
)


def _private_values() -> dict[str, object]:
    """Application draft carrying the privacy policy and its paired task spec."""
    value: dict[str, object] = dict(application_draft().model_dump(mode="python"))
    value.update(
        privacy=PrivacyPolicy(
            max_grad_norm=1,
            noise_multiplier=1,
            delta=1e-5,
            max_logical_steps=9,
            randomness_profile="reproducible_public_benchmark",
        ),
        private_training=TaskTrainingSpec(
            optimizer=PrivateOptimizerSpec(
                name="sgd",
                torch_version="2.13.0",
                parameter_names=("weight",),
                learning_rate=0.01,
            ),
            batch_size=2,
            task_definition_hash="a" * 64,
            implementation_hash="b" * 64,
            batch_definition_hash="c" * 64,
            loss_definition_hash="d" * 64,
            record_unit="fixed-token-sequence",
            trainable_parameter_names=("weight",),
        ),
    )
    return value


def test_private_policy_is_bound_to_draft_identity() -> None:
    value = _private_values()
    draft = DraftRunSpec.model_validate(value)
    privacy = value["privacy"]
    assert isinstance(privacy, PrivacyPolicy)
    changed = dict(value, privacy=privacy.model_copy(update={"noise_multiplier": 2}))
    assert canonical_hash(draft) != canonical_hash(DraftRunSpec.model_validate(changed))


def test_sealed_draft_rejects_changed_private_policy() -> None:
    values = _private_values()
    draft = DraftRunSpec.model_validate(values)
    original = manifest_data()
    values.update(
        {
            key: original[key]
            for key in ("participants", "initial_checkpoint_hash", "tensor_schema")
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
