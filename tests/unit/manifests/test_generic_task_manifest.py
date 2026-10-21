"""Generic task contracts bind local implementations without changing old runs."""

import pytest
from pydantic import JsonValue
from support.application_fixture import application_draft
from support.sample_manifest import manifest_data

from dromeus.manifests.canonical import canonical_hash, canonical_json
from dromeus.manifests.models import (
    DraftRunSpec,
    PrivacyPolicy,
    PrivateOptimizerSpec,
    PrivateTrainingSpec,
    SealedManifest,
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
        manifest_version=6,
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


def test_task_spec_requires_v6_application_and_private_mechanism() -> None:
    value = task_draft().model_dump()
    with pytest.raises(ValueError, match="generic tasks require manifest version 6"):
        DraftRunSpec.model_validate(dict(value, manifest_version=5))
    with pytest.raises(ValueError, match="requires privacy policy"):
        DraftRunSpec.model_validate(dict(value, privacy=None))
    legacy = PrivateTrainingSpec(
        optimizer=task_spec().optimizer, loss="mse", batch_size=2
    )
    with pytest.raises(ValueError, match="legacy private training requires version 5"):
        DraftRunSpec.model_validate(dict(value, private_training=legacy))
    ordinary = DraftRunSpec.model_validate(
        dict(value, private_training=None, privacy=None)
    )
    assert ordinary.manifest_version == 6


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


def test_private_task_rejects_shuffle_and_requires_complete_task_spec() -> None:
    value = task_draft().model_dump()
    shuffled = task_spec().model_copy(update={"sampling": "shuffle"})
    with pytest.raises(ValueError, match="private tasks require Poisson sampling"):
        DraftRunSpec.model_validate(dict(value, private_training=shuffled))
    with pytest.raises(
        ValueError, match="version 6 privacy requires a task specification"
    ):
        DraftRunSpec.model_validate(dict(value, private_training=None))


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


def test_legacy_manifest_hashes_remain_exact() -> None:
    assert canonical_hash(SealedManifest.model_validate(manifest_data())) == (
        "0019f8a0cf68272fffd536d51ddd491d9f80036a3ee3bdcfe008a9fc585b3907"
    )
    assert canonical_hash(application_draft()) == (
        "6d67d8aed45a44e25d9241136b1b774025d52afba646d9771c2d01a845e22e13"
    )
    values = task_draft().model_dump()
    values.update(
        manifest_version=5,
        private_training=PrivateTrainingSpec(
            optimizer=task_spec().optimizer, loss="mse", batch_size=2
        ),
    )
    assert canonical_hash(DraftRunSpec.model_validate(values)) == (
        "82769385773e1d159880e56bb8bf68e9180f91d3ae3854d3efcdc7469433a0aa"
    )
