"""Generic task contracts bind local implementations to sealed run identity."""

import pytest
from pydantic import JsonValue
from support.application_fixture import application_draft

from dromeus.manifests.canonical import canonical_hash, canonical_json
from dromeus.manifests.models import (
    DraftRunSpec,
    PrivacyPolicy,
    PrivateOptimizerSpec,
    TaskTrainingSpec,
)


def task_spec() -> TaskTrainingSpec:
    return TaskTrainingSpec(
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
        parameter_aliases=(("weight", "head.weight"),),
    )


def task_draft() -> DraftRunSpec:
    values = application_draft().model_dump()
    values.update(
        privacy=PrivacyPolicy(
            max_grad_norm=1,
            noise_multiplier=1,
            delta=1e-5,
            max_logical_steps=9,
            randomness_profile="reproducible_public_benchmark",
        ),
        private_training=task_spec(),
    )
    return DraftRunSpec.model_validate(values)


def test_generic_private_task_roundtrip_preserves_model_contract() -> None:
    draft = task_draft()
    restored = DraftRunSpec.model_validate_json(canonical_json(draft))
    assert restored == draft
    assert isinstance(restored.private_training, TaskTrainingSpec)
    assert restored.private_training.parameter_aliases == (("weight", "head.weight"),)
    assert canonical_hash(restored) == canonical_hash(draft)


def test_privacy_and_private_training_must_be_set_together() -> None:
    value = task_draft().model_dump()
    message = "privacy policy and private training must be set together"
    with pytest.raises(ValueError, match=message):
        DraftRunSpec.model_validate(dict(value, privacy=None))
    with pytest.raises(ValueError, match=message):
        DraftRunSpec.model_validate(dict(value, private_training=None))
    ordinary = DraftRunSpec.model_validate(
        dict(value, private_training=None, privacy=None)
    )
    assert ordinary.private_training is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("trainable_parameter_names", ["weight", "weight"], "unique"),
        ("trainable_parameter_names", ["other"], "optimizer parameter"),
        ("parameter_aliases", [["weight"]], "at least two"),
        ("parameter_aliases", [["weight", "weight"]], "alias names"),
        (
            "parameter_aliases",
            [["weight", "head.weight"], ["weight", "other.weight"]],
            "alias groups",
        ),
        ("parameter_aliases", [["unknown", "other"]], "canonical parameter"),
        ("frozen_parameter_hashes", {"weight": "a" * 64}, "trainable and frozen"),
        ("frozen_parameter_hashes", {"head.weight": "a" * 64}, "alias"),
        ("public_buffer_hashes", {"weight": "a" * 64}, "buffer and parameter"),
        ("public_buffer_hashes", {"head.weight": "a" * 64}, "buffer and parameter"),
    ],
)
def test_ambiguous_parameter_state_rejects(
    field: str, value: JsonValue, message: str
) -> None:
    values = task_spec().model_dump()
    values[field] = value
    with pytest.raises(ValueError, match=message):
        TaskTrainingSpec.model_validate(values)


def test_private_task_rejects_shuffle_sampling() -> None:
    value = task_draft().model_dump()
    shuffled = task_spec().model_copy(update={"sampling": "shuffle"})
    with pytest.raises(ValueError, match="private tasks require Poisson sampling"):
        DraftRunSpec.model_validate(dict(value, private_training=shuffled))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_definition_hash", "e" * 64),
        ("implementation_hash", "e" * 64),
        ("batch_definition_hash", "e" * 64),
        ("loss_definition_hash", "e" * 64),
        ("record_unit", "document"),
        ("parameter_aliases", [["head.weight", "weight"]]),
        ("public_buffer_hashes", {"position_ids": "e" * 64}),
        ("frozen_parameter_hashes", {"frozen.weight": "e" * 64}),
        ("initialization", "public_pretrained"),
    ],
)
def test_task_semantics_are_hash_bound(field: str, value: JsonValue) -> None:
    original = task_draft()
    values = task_spec().model_dump()
    values[field] = value
    changed = original.model_copy(
        update={"private_training": TaskTrainingSpec.model_validate(values)}
    )
    assert canonical_hash(changed) != canonical_hash(original)


def test_canonical_identities_are_pinned() -> None:
    assert canonical_hash(application_draft()) == (
        "a4c70b59b131dd71ab9168440510c29415eb0409a59468dec114726935235576"
    )
    assert (
        canonical_hash(task_draft())
        == "82914d1051886ce801f048dc9dbcdf9e6ad91fe21eab4918f8b8df341b878139"
    )
