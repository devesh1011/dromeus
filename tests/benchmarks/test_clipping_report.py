"""Clipping comparisons must preserve identities and avoid privacy mislabels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from benchmarks.m3.clipped_digits import ClippedRecipe, diagnostic_draft
from benchmarks.m3.clipping_report import (
    check_matching,
    check_nonprivate_summary,
    summarize_accuracy,
)
from benchmarks.m3.dp_digits import make_draft
from dromeus.manifests.canonical import canonical_hash, canonical_json
from dromeus.manifests.models import DraftRunSpec


def test_accuracy_own_vs_other_source_uses_training_source_rows() -> None:
    result = summarize_accuracy(
        [[0.8 if i == j else 0.4 for j in range(4)] for i in range(4)]
    )
    assert result == pytest.approx(
        {
            "mean_accuracy": 0.5,
            "mean_own_source_accuracy": 0.8,
            "mean_other_source_accuracy": 0.4,
        }
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("arm", "private"),
        ("epsilon", 0.0),
        ("privacy_claim", True),
        ("noise_multiplier", 1.0),
        ("accounting_history", []),
    ],
)
def test_diagnostic_rejects_private_or_zero_epsilon_claims(
    field: str, value: Any
) -> None:
    summary = {
        "arm": "non_private_clipping_diagnostic",
        "epsilon": None,
        "privacy_claim": False,
        "noise_multiplier": 0.0,
    }
    check_nonprivate_summary(summary)
    summary[field] = value
    with pytest.raises(ValueError):
        check_nonprivate_summary(summary)


@dataclass
class Manifest:
    draft: DraftRunSpec
    participants: tuple[str, ...] = ("peer0", "peer1", "peer2", "peer3")

    @property
    def run_id(self) -> str:
        return self.draft.run_id

    @property
    def draft_hash(self) -> str:
        return canonical_hash(self.draft)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        return self.draft.model_dump(**kwargs)


def nodes() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    reference = make_draft("a" * 40, run_id="private")
    assert reference.private_training is not None
    recipe = ClippedRecipe(
        batch_size=64,
        logical_steps=750,
        loss="cross_entropy",
        schedule="linear_decay",
        evaluation_interval_rounds=15,
        clipping=1.0,
        optimizer_definition=canonical_json(
            reference.private_training.optimizer
        ).decode(),
    )
    draft = diagnostic_draft(reference, recipe, run_id="clipped")
    shared: dict[str, Any] = {
        key: "same"
        for key in (
            "source",
            "reference_draft_hash",
            "initial_checkpoint_sha256",
            "training_data_sha256",
            "public_holdout_sha256",
            "physical_batch_limit",
            "evaluation_interval_rounds",
            "sample_rate",
            "expected_batch_size",
            "torch_version",
            "opacus_version",
        )
    }
    shared.update(rank=1, model_seed=17, sampling_seed=18)
    b = {"result": dict(shared), "node_id": "peer1", "manifest": Manifest(reference)}
    c = {
        "result": {**shared, "clipping": 1.0, "recipe": recipe.model_dump()},
        "node_id": "peer1",
        "manifest": Manifest(draft),
    }
    p = {
        "result": {**shared, "research_seed": 18},
        "node_id": "peer1",
        "manifest": Manifest(reference),
    }
    return b, c, p


def test_diagnostic_contract_matches_private_reference() -> None:
    check_matching(*nodes())


@pytest.mark.parametrize(
    "fault",
    [
        "identity",
        "membership",
        "source",
        "sampling",
        "clipping",
        "recipe",
        "privacy_policy",
    ],
)
def test_diagnostic_rejects_unmatched_comparison(fault: str) -> None:
    b, c, p = nodes()
    if fault == "identity":
        c["node_id"] = "peer0"
    elif fault == "membership":
        c["manifest"].participants = ("peer3", "peer2", "peer1", "peer0")
    elif fault == "source":
        c["result"]["training_data_sha256"] = "different"
    elif fault == "sampling":
        c["result"]["sampling_seed"] = 19
    elif fault == "clipping":
        c["result"]["clipping"] = 0.5
    elif fault == "recipe":
        c["result"]["recipe"]["schedule"] = "constant"
    else:
        p["manifest"].draft = p["manifest"].draft.model_copy(update={"privacy": None})
    with pytest.raises(ValueError):
        check_matching(b, c, p)
