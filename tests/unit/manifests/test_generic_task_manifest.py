"""Generic task contracts bind local implementations to sealed run identity."""

import pytest
from pydantic import JsonValue
from support.sample_manifest import manifest_data

from dromeus.manifests.canonical import canonical_hash, canonical_json
from dromeus.manifests.models import (
    DraftRunSpec,
    PrivacyPolicy,
    PrivateOptimizerSpec,
    TaskTrainingSpec,
)


def base_draft() -> DraftRunSpec:
    """The fixed sample draft: no installed versions, so its hash is portable."""
    values = manifest_data()
    for field in (
        "participants",
        "draft_hash",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del values[field]
    return DraftRunSpec.model_validate(values)


def task_spec() -> TaskTrainingSpec:
    return TaskTrainingSpec(
        optimizer=PrivateOptimizerSpec(
            name="sgd",
            torch_version="2.13.0",
            parameter_names=("weight",),
            learning_rate=0.01,
        ),
        batch_size=2,
        # The run's training task must match its dataset contract.
        task_definition_hash=base_draft().dataset.definition_hash,
        implementation_hash="b" * 64,
        batch_definition_hash="c" * 64,
        loss_definition_hash="d" * 64,
        record_unit="fixed-token-sequence",
        trainable_parameter_names=("weight",),
        parameter_aliases=(("weight", "head.weight"),),
    )


def task_draft() -> DraftRunSpec:
    values = base_draft().model_dump()
    values.update(
        privacy=PrivacyPolicy(
            max_grad_norm=1,
            noise_multiplier=1,
            delta=1e-5,
            max_logical_steps=500,
            randomness_profile="reproducible_public_benchmark",
        ),
        training=task_spec(),
    )
    return DraftRunSpec.model_validate(values)


def test_generic_private_task_roundtrip_preserves_model_contract() -> None:
    draft = task_draft()
    restored = DraftRunSpec.model_validate_json(canonical_json(draft))
    assert restored == draft
    assert isinstance(restored.training, TaskTrainingSpec)
    assert restored.training.parameter_aliases == (("weight", "head.weight"),)
    assert canonical_hash(restored) == canonical_hash(draft)


def test_training_is_required_and_privacy_is_optional() -> None:
    value = task_draft().model_dump()
    without_training = {key: item for key, item in value.items() if key != "training"}
    with pytest.raises(ValueError, match="training"):
        DraftRunSpec.model_validate(without_training)
    ordinary = DraftRunSpec.model_validate(dict(value, privacy=None))
    assert ordinary.privacy is None
    assert ordinary.training.sampling == "poisson"


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
        DraftRunSpec.model_validate(dict(value, training=shuffled))


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
        update={"training": TaskTrainingSpec.model_validate(values)}
    )
    assert canonical_hash(changed) != canonical_hash(original)


def test_canonical_identities_are_pinned() -> None:
    assert (
        canonical_hash(task_draft())
        == "bebfc36e5bfcda2156f249bb7849760c53b8022ab1235f01b89a1a81dc34e11e"
    )
