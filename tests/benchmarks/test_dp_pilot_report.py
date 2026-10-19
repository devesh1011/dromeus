"""The report must reject incomplete training and mismatched comparison arms."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from benchmarks.m3.dp_digits import make_draft
from benchmarks.m3.dp_pilot_report import Metric, check_pair, check_rounds
from benchmarks.m3.heterogeneous_data import SOURCES
from benchmarks.m3.matched_baseline import BaselineRecipe, matched_draft
from dromeus.manifests.canonical import canonical_hash, canonical_json
from dromeus.telemetry.evidence import RoundMetricsEvidence, TaskRoundMetricsEvidence


def records() -> list[Metric]:
    base: dict[str, Any] = {
        "timestamp": datetime(2026, 10, 20, tzinfo=UTC),
        "run_id": "test",
        "manifest_hash": "a" * 64,
        "node_id": "node",
        "peer_id": "peer",
        "local_compute_seconds": 1.0,
        "peer_wait_seconds": 0.0,
        "transfer_seconds": 0.0,
        "mixing_seconds": 0.0,
        "evaluation_seconds": 0.0,
        "retries": 0,
    }
    return [
        RoundMetricsEvidence.model_validate({**base, "round_id": 0, "message_id": "0"}),
        TaskRoundMetricsEvidence.model_validate(
            {
                **base,
                "round_id": 1,
                "message_id": "1",
                "evaluation_metrics": {f"accuracy_{s}": 0.5 for s in SOURCES},
            }
        ),
    ]


def test_training_accepts_mixed_metric_versions() -> None:
    check_rounds(records(), rounds=2, interval=2, private=True)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "loss", "evaluation"])
def test_training_rejects_missing_or_unsafe_records(fault: str) -> None:
    rows = records()
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[1] = rows[0]
    elif fault == "loss":
        rows[0] = rows[0].model_copy(update={"local_loss": 0.5})
    else:
        rows[1] = rows[1].model_copy(update={"evaluation_metrics": {}})
    with pytest.raises(ValueError):
        check_rounds(rows, rounds=2, interval=2, private=True)


def pair() -> tuple[dict[str, Any], dict[str, Any]]:
    draft = make_draft("a" * 40, run_id="private")
    assert draft.private_training is not None
    spec = draft.private_training
    recipe = BaselineRecipe(
        batch_size=64,
        logical_steps=750,
        loss="cross_entropy",
        schedule="linear_decay",
        evaluation_interval_rounds=15,
        optimizer_definition=canonical_json(spec.optimizer).decode(),
    )
    ordinary = matched_draft(draft, recipe, run_id="baseline")

    # Only fields used at this seam; real load_node uses a strict sealed manifest.
    class Manifest:
        def __init__(self, value: Any) -> None:
            self.value = value
            self.run_id = value.run_id
            self.draft_hash = canonical_hash(value)
            self.participants = tuple(f"peer-{i}" for i in range(4))

        def model_dump(self, **kwargs: Any) -> dict[str, Any]:
            return self.value.model_dump(**kwargs)

    common: dict[str, Any] = {
        k: "same"
        for k in (
            "source",
            "rank",
            "reference_draft_hash",
            "initial_checkpoint_sha256",
            "training_data_sha256",
            "public_holdout_sha256",
            "model_seed",
            "physical_batch_limit",
            "evaluation_interval_rounds",
            "sample_rate",
            "expected_batch_size",
            "torch_version",
            "opacus_version",
        )
    }
    common.update(rank=2, model_seed=17)
    return (
        {
            "result": {**common, "sampling_seed": 19, "recipe": recipe.model_dump()},
            "node_id": "peer-2",
            "manifest": Manifest(ordinary),
        },
        {
            "result": {**common, "research_seed": 19},
            "node_id": "peer-2",
            "manifest": Manifest(draft),
        },
    )


def test_comparison_reconstructs_matched_contract() -> None:
    check_pair(*pair())


@pytest.mark.parametrize(
    "field",
    [
        "sample_rate",
        "training_data_sha256",
        "public_holdout_sha256",
        "reference_draft_hash",
    ],
)
def test_comparison_rejects_changed_source_or_sampler(field: str) -> None:
    baseline, private = pair()
    baseline["result"][field] = "different"
    with pytest.raises(ValueError, match=field):
        check_pair(baseline, private)


def test_comparison_rejects_changed_optimizer_recipe() -> None:
    baseline, private = pair()
    baseline["result"]["recipe"]["schedule"] = "constant"
    with pytest.raises(ValueError, match="recipe differs"):
        check_pair(baseline, private)


@pytest.mark.parametrize(
    "fault", ["node_id", "membership", "sampling", "research", "seed"]
)
def test_comparison_rejects_peer_mapping_or_seed_change(fault: str) -> None:
    baseline, private = pair()
    if fault == "node_id":
        baseline["node_id"] = "peer-1"
    elif fault == "membership":
        private["manifest"].participants = tuple(
            reversed(private["manifest"].participants)
        )
    elif fault == "sampling":
        baseline["result"]["sampling_seed"] = 20
    elif fault == "research":
        private["result"]["research_seed"] = 20
    else:
        baseline["result"]["model_seed"] = private["result"]["model_seed"] = 18
    with pytest.raises(ValueError, match="mapping|seeds"):
        check_pair(baseline, private)
