"""New initialization provenance is sealed without changing prior canonical bytes."""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from dromeus.manifests.canonical import canonical_hash, canonical_json
from dromeus.manifests.models import (
    AccountedInitializationAuthority,
    AccountedInitializationProvenance,
    PrivateOptimizerSpec,
    TaskTrainingSpec,
)


def values() -> dict[str, Any]:
    return dict(
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
    )


def provenance() -> AccountedInitializationProvenance:
    return AccountedInitializationProvenance(
        parent_run_id="parent",
        parent_model_sha256="1" * 64,
        parent_audit_sha256="2" * 64,
        accounting_authorities=(
            AccountedInitializationAuthority(
                lineage_id="lineage",
                ledger_sha256="3" * 64,
                reserved_steps=4552,
                reservation_count=14,
            ),
        ),
    )


def test_old_spec_canonical_bytes_have_no_new_field() -> None:
    specification = TaskTrainingSpec(**values())
    old = specification.model_dump(mode="json")
    old.pop("initialization_provenance")
    expected = json.dumps(old, sort_keys=True, separators=(",", ":")).encode()
    assert canonical_json(specification) == expected
    assert TaskTrainingSpec.model_validate_json(expected) == specification


def test_accounted_provenance_is_required_and_exclusive() -> None:
    with pytest.raises(ValidationError, match="exclusive provenance"):
        TaskTrainingSpec(**values(), initialization="accounted_private_pretrained")
    for mode in ["public_untrained", "public_pretrained"]:
        with pytest.raises(ValidationError, match="exclusive provenance"):
            TaskTrainingSpec.model_validate(
                {
                    **values(),
                    "initialization": mode,
                    "initialization_provenance": provenance(),
                }
            )


def test_each_parent_binding_changes_canonical_identity() -> None:
    parent = provenance()
    specification = TaskTrainingSpec(
        **values(),
        initialization="accounted_private_pretrained",
        initialization_provenance=parent,
    )
    for field, changed in [
        ("parent_run_id", "different"),
        ("parent_model_sha256", "4" * 64),
        ("parent_audit_sha256", "5" * 64),
        (
            "accounting_authorities",
            (
                parent.accounting_authorities[0].model_copy(
                    update={"ledger_sha256": "6" * 64}
                ),
            ),
        ),
    ]:
        assert canonical_hash(
            specification.model_copy(
                update={
                    "initialization_provenance": parent.model_copy(
                        update={field: changed}
                    )
                }
            )
        ) != canonical_hash(specification)
    assert (
        TaskTrainingSpec.model_validate_json(canonical_json(specification))
        == specification
    )


def test_duplicate_or_empty_accounting_ancestry_rejected() -> None:
    parent = provenance()
    for rows in [(), parent.accounting_authorities * 2]:
        with pytest.raises(ValidationError):
            AccountedInitializationProvenance.model_validate(
                {
                    **parent.model_dump(),
                    "accounting_authorities": rows,
                }
            )
