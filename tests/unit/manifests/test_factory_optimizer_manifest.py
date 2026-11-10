"""Factory optimizer identity, admission boundaries and private optimizer identity."""

import pytest
from pydantic import JsonValue, ValidationError

from dromeus.manifests.canonical import canonical_hash, canonical_json
from dromeus.manifests.models import (
    FactoryOptimizerSpec,
    OptimizerParameterGroup,
    PrivateOptimizerSpec,
    TaskTrainingSpec,
)


def _optimizer() -> FactoryOptimizerSpec:
    return FactoryOptimizerSpec(
        implementation_id="torch.optim.rmsprop.RMSprop",
        implementation_hash="a" * 64,
        torch_version="2.13.0",
        groups=(
            OptimizerParameterGroup(
                parameter_names=("weight",),
                options={"lr": 0.01, "alpha": 0.99, "foreach": False},
            ),
            OptimizerParameterGroup(
                parameter_names=("bias",),
                options={"lr": 0.02, "alpha": 0.99, "foreach": False},
            ),
        ),
    )


def _task(optimizer: PrivateOptimizerSpec | FactoryOptimizerSpec) -> TaskTrainingSpec:
    return TaskTrainingSpec(
        optimizer=optimizer,
        batch_size=64,
        task_definition_hash="a" * 64,
        implementation_hash="b" * 64,
        batch_definition_hash="c" * 64,
        loss_definition_hash="d" * 64,
        record_unit="fixed-token-sequence",
        trainable_parameter_names=optimizer.parameter_names,
    )


def test_factory_roundtrip_preserves_group_identity() -> None:
    spec = _task(_optimizer())
    restored = TaskTrainingSpec.model_validate_json(canonical_json(spec))
    assert restored == spec
    assert isinstance(restored.optimizer, FactoryOptimizerSpec)
    assert restored.optimizer.parameter_names == ("weight", "bias")
    assert b'"kind":"factory"' in canonical_json(spec)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("implementation_id", "project.optim.CustomRMSprop"),
        ("implementation_hash", "b" * 64),
        ("torch_version", "2.13.1"),
    ],
)
def test_implementation_identity_is_hash_bound(field: str, value: str) -> None:
    original = _optimizer()
    payload = original.model_dump()
    payload[field] = value
    assert canonical_hash(
        FactoryOptimizerSpec.model_validate(payload)
    ) != canonical_hash(original)


def test_group_membership_order_and_options_are_hash_bound() -> None:
    original = _optimizer()
    changed_order = original.model_copy(update={"groups": original.groups[::-1]})
    changed_options = FactoryOptimizerSpec(
        implementation_id=original.implementation_id,
        implementation_hash=original.implementation_hash,
        torch_version=original.torch_version,
        groups=(
            OptimizerParameterGroup(
                parameter_names=("weight",),
                options={"lr": 0.01, "alpha": 0.95, "foreach": False},
            ),
            original.groups[1],
        ),
    )
    changed_membership = original.model_copy(
        update={
            "groups": (
                original.groups[0].model_copy(update={"parameter_names": ("bias",)}),
                original.groups[1].model_copy(update={"parameter_names": ("weight",)}),
            )
        }
    )
    for changed in (changed_order, changed_options, changed_membership):
        assert canonical_hash(changed) != canonical_hash(original)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"lr": True},
        {"lr": "0.1"},
        {"lr": 0},
        {"lr": -0.1},
        {"lr": float("nan")},
        {"lr": float("inf")},
        {"lr": 0.1, "params": []},
        {"lr": 0.1, "nested": {"bad": float("inf")}},
        {"lr": 0.1, "nested": [float("nan")]},
        {"lr": 0.1, "foreach": None},
        {"lr": 0.1, "foreach": 0},
        {"lr": 0.1, "fused": True},
        {"lr": 0.1, "capturable": True},
        {"lr": 0.1, "differentiable": True},
        {"lr": 0.1, "maximize": True},
    ],
)
def test_unsafe_or_ambiguous_options_reject(options: dict[str, JsonValue]) -> None:
    with pytest.raises(ValidationError):
        OptimizerParameterGroup(parameter_names=("weight",), options=options)


@pytest.mark.parametrize("names", [(), ("",), ("weight", "weight")])
def test_invalid_parameter_membership_rejects(names: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError):
        OptimizerParameterGroup(parameter_names=names, options={"lr": 0.1})


def test_group_overlap_and_missing_groups_reject() -> None:
    values = _optimizer().model_dump()
    for groups in ((), (_optimizer().groups[0],) * 2, _optimizer().groups * 17):
        with pytest.raises(ValidationError):
            FactoryOptimizerSpec.model_validate(dict(values, groups=groups))


def test_json_option_bounds_and_types_reject() -> None:
    nested: JsonValue = 1
    for _ in range(9):
        nested = [nested]
    for extra in (nested, [0] * 4096, "x" * 65536, object(), (0.9, 0.99)):
        with pytest.raises(ValidationError):
            OptimizerParameterGroup.model_validate(
                {"parameter_names": ["weight"], "options": {"lr": 0.1, "extra": extra}}
            )


def test_nested_json_options_roundtrip_without_type_coercion() -> None:
    group = OptimizerParameterGroup(
        parameter_names=("weight",),
        options={"lr": 0.1, "custom": {"values": [1, 2.0, False, None, "value"]}},
    )
    restored = OptimizerParameterGroup.model_validate_json(canonical_json(group))
    assert restored == group
    assert canonical_hash(restored) == canonical_hash(group)


def test_private_optimizer_identity_roundtrips_exactly() -> None:
    spec = PrivateOptimizerSpec(
        name="adam",
        torch_version="2.13.0",
        parameter_names=("weight", "bias"),
        learning_rate=0.001,
    )
    assert (
        canonical_hash(spec)
        == "a567e6091ce77e907d04dbc886f81fdf0de531ad2b80cb2858c2a47fb947fc35"
    )
    restored = PrivateOptimizerSpec.model_validate_json(canonical_json(spec))
    assert restored == spec
