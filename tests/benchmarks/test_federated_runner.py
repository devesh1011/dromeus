from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from benchmarks.federated.controls import Trajectories
from benchmarks.federated.evidence import compare_trajectories
from benchmarks.federated.runner import RuntimeResult, assess_profile, run_suite
from benchmarks.federated.workload import (
    FROZEN_EXPERIMENT_SHA256,
    load_experiment,
    sample_budget,
)
from dromeus.manifests.canonical import file_sha256


def test_experiment_is_bound_before_execution() -> None:
    experiment = load_experiment()
    assert experiment.sha256 == FROZEN_EXPERIMENT_SHA256
    assert experiment.seed == 17
    assert experiment.rounds == 8
    assert experiment.settings["learning_rate_schedule"]["total_inner_steps"] == 400
    assert experiment.settings["top_k_fraction"] == 0.45
    assert experiment.settings["gates"]["common_heldout_mean_accuracy_min"] == 0.8


def _trajectory(value: float = 1.0) -> Trajectories:
    return [[{"weight": np.array([value], dtype=np.float32)}]]


@pytest.mark.parametrize(
    ("observed", "passed"),
    (
        (_trajectory(1.0000005), True),
        (_trajectory(1.001), False),
        (_trajectory(float("nan")), False),
        ([], False),
    ),
)
def test_trajectory_gate_uses_finite_per_element_tolerances(
    observed: Trajectories, passed: bool
) -> None:
    result = compare_trajectories(observed, _trajectory(), atol=1e-6, rtol=1e-6)
    assert result["passed"] is passed


@pytest.mark.parametrize(
    ("mean", "worst", "decision"),
    ((0.9, 0.8, "pass"), (0.79, 0.7, "refine"), (0.9, 0.59, "refine")),
)
def test_utility_gate_does_not_confuse_protocol_or_parity_with_learning(
    mean: float, worst: float, decision: str
) -> None:
    report: dict[str, Any] = {
        "status": "complete",
        "common_mean_accuracy": mean,
        "common_worst_accuracy": worst,
        "accepted_wire_bytes": {"round-protocol": 100},
    }
    result = RuntimeResult(report, _trajectory())
    assessment = assess_profile(load_experiment(), result, result, _trajectory())
    assert assessment["decision"] == decision


def test_failed_runtime_is_rejected_even_with_a_successful_reference() -> None:
    result = RuntimeResult({"status": "failed"}, None)
    assert (
        assess_profile(load_experiment(), result, result, _trajectory())["decision"]
        == "reject"
    )


def test_sample_budget_counts_partial_batches_and_repeated_passes() -> None:
    assert sample_budget(3, steps=5, batch_size=2) == 8
    assert sample_budget(1, steps=400, batch_size=16) == 400
    assert sample_budget(128, steps=400, batch_size=16) == 6400


def test_iid_suite_runs_real_runtime_controls_and_retains_reproducible_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "suite"
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = asyncio.run(run_suite(root, profiles=("iid",)))
    finally:
        torch.set_num_threads(previous)
    assert result["all_predeclared_gates_passed"] is True
    assert result["container_image_digest"] is None
    profile = cast(dict[str, Any], result["profiles"][0])
    assert (
        profile["assessment"]["identity_reference_parity"]["max_absolute_error"] == 0.0
    )
    assert profile["reference"]["nccl_acceptance_eligible"] is False
    assert len(set(key[:8] for key in result["membership"])) == 4
    for variant in ("identity", "bitmap-int8"):
        run = profile[variant]
        assert run["status"] == "complete"
        assert run["accepted_wire_bytes"]["round-protocol"] > 0
        assert run["accepted_wire_bytes"]["telemetry"] > 0
        assert run["encoded_artifact_bytes"] > 0
        assert run["round_retries"] == 0
        assert run["raw_update_tensor_bytes"] == 4 * 8 * 2 * 36 * 4
        assert len(run["weight_dispersion"]) == 9
        for rank, node in enumerate(run["nodes"]):
            assert node["optimizer_steps"] == 400
            assert node["sample_presentations"] == 6400
            assert len(node["metrics"]) == 8
            assert all(
                metric["local_compute_seconds"] > 0 for metric in node["metrics"]
            )
            assert len(node["common_heldout"]["per_class"]) == 4
            assert (
                len(
                    list(
                        (
                            root
                            / "runs"
                            / "iid"
                            / variant
                            / f"node-{rank}"
                            / "analysis"
                        ).glob("*.safetensors")
                    )
                )
                == 8
            )
    for line in (root / "checksums.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        assert file_sha256(root / relative) == expected
    before = (root / "results.json").read_bytes()
    with pytest.raises(FileExistsError):
        asyncio.run(run_suite(root, profiles=("iid",)))
    assert (root / "results.json").read_bytes() == before
