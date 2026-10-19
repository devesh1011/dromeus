"""Public forecasting contract: matched sampling and composed privacy expenditure."""

import numpy as np
import pytest

from benchmarks.m3.privacy_forecast import ForecastPlan, forecast

pytest.importorskip("opacus")


def plan(**updates: object) -> ForecastPlan:
    values: dict[str, object] = {
        "nodes": [{"node_id": "node-0", "sample_count": 16, "prior_history": ()}],
        "batch_size": 4,
        "noise_multipliers": [1.0],
        "logical_step_horizons": [4],
        "repetitions": 1,
        "delta": 1e-5,
        "delta_rationale": "Public synthetic test; fixed before comparison.",
        "randomness_profile": "reproducible_public_benchmark",
    }
    values.update(updates)
    import json

    return ForecastPlan.model_validate_json(json.dumps(values))


def test_forecast_matches_retained_w0_accountant_result() -> None:
    result = forecast(plan())
    node = result["nodes"][0]
    assert node["sample_rate"] == 0.25
    assert node["expected_batch_size"] == 4
    estimate = node["candidates"][0]["one_run"]
    np.testing.assert_allclose(estimate["prv_epsilon"], 4.216155971, atol=1e-7)
    assert node["prior"]["prv_epsilon"] == 0
    assert result["purpose"] == "mechanism_forecast_not_training_evidence"


def test_prior_repetitions_and_sweep_compose_mechanisms() -> None:
    from dromeus.training.privacy import opacus_module

    request = plan(
        nodes=[
            {
                "node_id": "node-0",
                "sample_count": 16,
                "prior_history": [
                    {"noise_multiplier": 2.0, "sample_rate": 0.125, "logical_steps": 3}
                ],
            }
        ],
        noise_multipliers=[1.0, 2.0],
        logical_step_horizons=[2],
        repetitions=3,
        epsilon_limit=1.0,
    )
    report = forecast(request)
    node = report["nodes"][0]
    candidate = node["candidates"][0]
    assert candidate["one_run"]["logical_steps"] == 2
    assert candidate["repetitions_only"]["logical_steps"] == 6
    composed = candidate["prior_plus_repetitions"]
    assert composed["logical_steps"] == 9
    assert composed["history"][0]["sample_rate"] == 0.125
    # Independent accountant use at the public boundary: append mechanisms step-by-step.
    for name, field in (
        ("PRVAccountant", "prv_epsilon"),
        ("RDPAccountant", "rdp_epsilon"),
    ):
        accountant = getattr(opacus_module("opacus.accountants"), name)()
        for _ in range(3):
            accountant.step(noise_multiplier=2.0, sample_rate=0.125)
        for _ in range(6):
            accountant.step(noise_multiplier=1.0, sample_rate=0.25)
        np.testing.assert_allclose(
            composed[field], accountant.get_epsilon(delta=1e-5), atol=1e-9
        )
        for _ in range(6):
            accountant.step(noise_multiplier=2.0, sample_rate=0.25)
        np.testing.assert_allclose(
            node["prior_plus_entire_sweep"][field],
            accountant.get_epsilon(delta=1e-5),
            atol=1e-9,
        )
    assert composed["within_epsilon_limit"] is False
    assert node["prior_plus_entire_sweep"]["logical_steps"] == 15
    assert report["group_privacy_bound"] is None
    assert report["inputs"] == request.model_dump(mode="json")


@pytest.mark.parametrize(
    "count,batch,expected",
    [
        (3000, 64, (47, 1 / 47, 63)),
        (10, 4, (3, 1 / 3, 3)),
        (16, 4, (4, 0.25, 4)),
        (2, 8, (1, 1.0, 2)),
    ],
)
def test_sampling_matches_actual_pinned_loader(
    count: int, batch: int, expected: tuple[int, float, int]
) -> None:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from benchmarks.m3.privacy_forecast import resolve_sampling
    from dromeus.training.privacy import opacus_module

    loader = opacus_module("opacus.data_loader").DPDataLoader.from_data_loader(
        DataLoader(TensorDataset(torch.zeros(count, 1)), batch_size=batch)
    )
    assert resolve_sampling(count, batch) == expected
    assert expected[1] == loader.sample_rate


def test_unequal_nodes_use_local_rates_and_worst_local_is_not_average() -> None:
    report = forecast(
        plan(
            nodes=[
                {"node_id": "small", "sample_count": 8, "prior_history": ()},
                {"node_id": "large", "sample_count": 16, "prior_history": ()},
            ],
            epsilon_limit=10.0,
        )
    )
    small, large = report["nodes"]
    assert small["sample_rate"] == 0.5 and large["sample_rate"] == 0.25
    a = small["candidates"][0]["prior_plus_repetitions"]
    b = large["candidates"][0]["prior_plus_repetitions"]
    assert a["prv_epsilon"] > b["prv_epsilon"]
    assert a["within_epsilon_limit"] is True
    assert (
        report["candidate_maxima"][0]["max_local_composed_prv_epsilon"]
        == a["prv_epsilon"]
    )
    assert report["group_privacy_bound"] is None


@pytest.mark.parametrize(
    "update",
    [
        {"noise_multipliers": [0.0]},
        {"noise_multipliers": [1.0, 1.0]},
        {"logical_step_horizons": [0]},
        {"logical_step_horizons": [4, 4]},
        {"delta": 0.0},
        {"delta": 1.0},
        {"delta": float("nan")},
        {"repetitions": True},
        {"batch_size": 0},
        {"epsilon_limit": -1.0},
        {"delta_rationale": " "},
        {"prv_epsilon_error": 0},
        {"nodes": [{"node_id": "a", "sample_count": 4}]},
        {"nodes": [{"node_id": "a", "sample_count": 4, "prior_history": ()}] * 2},
        {"nodes": [{"node_id": "a", "sample_count": 0, "prior_history": ()}]},
        {
            "nodes": [
                {
                    "node_id": "a",
                    "sample_count": 4,
                    "prior_history": [
                        {
                            "noise_multiplier": 1.0,
                            "sample_rate": 2.0,
                            "logical_steps": 4,
                        }
                    ],
                }
            ]
        },
    ],
)
def test_invalid_forecast_inputs_reject(update: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        plan(**update)


def test_prv_is_primary_even_when_rdp_exceeds_selected_cap() -> None:
    unbounded = forecast(plan())
    estimate = unbounded["nodes"][0]["candidates"][0]["one_run"]
    assert estimate["rdp_epsilon"] > estimate["prv_epsilon"]
    cap = (estimate["rdp_epsilon"] + estimate["prv_epsilon"]) / 2
    bounded = forecast(plan(epsilon_limit=cap))
    assert (
        bounded["nodes"][0]["candidates"][0]["prior_plus_repetitions"][
            "within_epsilon_limit"
        ]
        is True
    )


@pytest.mark.parametrize("failure", ["exception", "nan"])
def test_accountant_failure_never_falls_back_to_rdp(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    from dromeus.training.privacy import opacus_module

    def broken(*args: object, **kwargs: object) -> float:
        if failure == "exception":
            raise ValueError("simulated upstream numerical failure")
        return float("nan")

    monkeypatch.setattr(
        opacus_module("opacus.accountants").PRVAccountant, "get_epsilon", broken
    )
    with pytest.raises(ValueError, match="forecast failed"):
        forecast(plan())


def test_numerical_warnings_are_retained_with_the_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import warnings
    from typing import Any

    from dromeus.training.privacy import opacus_module

    kind = opacus_module("opacus.accountants").RDPAccountant
    original = kind.get_epsilon

    def warned(self: Any, *args: Any, **kwargs: Any) -> float:
        warnings.warn("Synthetic upstream convergence diagnostic", UserWarning)
        return float(original(self, *args, **kwargs))

    monkeypatch.setattr(kind, "get_epsilon", warned)
    result = forecast(plan())
    estimate = result["nodes"][0]["candidates"][0]["one_run"]
    assert any(
        "Synthetic upstream" in message for message in estimate["numerical_warnings"]
    )
