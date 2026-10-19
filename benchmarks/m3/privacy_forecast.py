"""Prospective per-node Poisson Gaussian budgets; no training or ledger writes."""

from __future__ import annotations

import importlib.metadata
import json
import math
import warnings
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dromeus.manifests.canonical import canonical_hash
from dromeus.manifests.models import PrivacyPolicy
from dromeus.training.privacy import opacus_module, privacy_epsilon

PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
PositiveSteps = Annotated[int, Field(gt=0, le=1_000_000_000)]


class ForecastInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, allow_inf_nan=False
    )


class MechanismHistory(ForecastInput):
    noise_multiplier: PositiveFloat
    sample_rate: Annotated[float, Field(gt=0, le=1)]
    logical_steps: PositiveSteps

    def mechanism(self) -> tuple[float, float, int]:
        return self.noise_multiplier, self.sample_rate, self.logical_steps


class ForecastNode(ForecastInput):
    node_id: str = Field(min_length=1, max_length=128, pattern=r"^\S+$")
    sample_count: int = Field(gt=0, le=1_000_000_000)
    # Required: [] explicitly acknowledges the no-prior-spending assumption.
    prior_history: tuple[MechanismHistory, ...] = Field(max_length=64)


class ForecastPlan(ForecastInput):
    version: Literal[1] = 1
    nodes: tuple[ForecastNode, ...] = Field(min_length=1, max_length=16)
    batch_size: int = Field(gt=0, le=1_000_000_000)
    noise_multipliers: tuple[PositiveFloat, ...] = Field(min_length=1, max_length=16)
    logical_step_horizons: tuple[PositiveSteps, ...] = Field(
        min_length=1, max_length=16
    )
    repetitions: int = Field(gt=0, le=100)
    delta: Annotated[float, Field(gt=0, lt=1)]
    delta_rationale: str = Field(min_length=1, max_length=2048)
    randomness_profile: Literal["secure", "reproducible_public_benchmark"]
    epsilon_limit: PositiveFloat | None = None
    prv_epsilon_error: PositiveFloat = 0.01
    prv_delta_error_ratio: Annotated[float, Field(gt=0, lt=1)] = 0.001

    @model_validator(mode="after")
    def unambiguous_grid(self) -> Self:
        if not self.delta_rationale.strip():
            raise ValueError("delta rationale must not be blank")
        if len({n.node_id for n in self.nodes}) != len(self.nodes):
            raise ValueError("duplicate node identity")
        if len(set(self.noise_multipliers)) != len(self.noise_multipliers) or len(
            set(self.logical_step_horizons)
        ) != len(self.logical_step_horizons):
            raise ValueError("duplicate forecast candidate")
        if len(self.noise_multipliers) * len(self.logical_step_horizons) > 64:
            raise ValueError("forecast grid exceeds 64 alternatives")
        if self.delta * self.prv_delta_error_ratio == 0:
            raise ValueError("PRV delta error underflows")
        return self


def parse_plan(source: bytes) -> ForecastPlan:
    """Reject ambiguous duplicate JSON fields at every nesting level."""

    def unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"duplicate forecast field: {name}")
            result[name] = value
        return result

    value = json.loads(source, object_pairs_hook=unique_fields)
    return ForecastPlan.model_validate_json(json.dumps(value, allow_nan=False))


def resolve_sampling(sample_count: int, batch_size: int) -> tuple[int, float, int]:
    """Match fixed-tensor DataLoader(drop_last=False) -> Opacus Poisson loader."""
    if (
        type(sample_count) is not int
        or type(batch_size) is not int
        or min(sample_count, batch_size) <= 0
    ):
        raise ValueError("positive integer sample count and batch request required")
    draws = (sample_count + batch_size - 1) // batch_size
    rate = 1 / draws
    return draws, rate, int(sample_count * rate)


def _history_json(
    history: tuple[tuple[float, float, int], ...],
) -> list[dict[str, float | int]]:
    return [
        dict(noise_multiplier=s, sample_rate=q, logical_steps=t) for s, q, t in history
    ]


def forecast(plan: ForecastPlan) -> dict[str, Any]:
    """Compute independent alternatives and full sweep using existing accounting.

    Prior histories must include all reserved horizons and external uses of each
    lineage. This forecast neither reserves budget nor certifies completeness.
    """
    # Revalidate nested caller-owned values and enforce the pinned accountant version.
    plan = ForecastPlan.model_validate_json(plan.model_dump_json())
    opacus_module()
    cache: dict[tuple[tuple[float, float, int], ...], dict[str, Any]] = {}

    def estimate(history: tuple[tuple[float, float, int], ...]) -> dict[str, Any]:
        # Adjacent identical mechanisms may be compressed without changing composition.
        merged: list[tuple[float, float, int]] = []
        for sigma, rate, steps in history:
            if merged and merged[-1][:2] == (sigma, rate):
                merged[-1] = (sigma, rate, merged[-1][2] + steps)
            else:
                merged.append((sigma, rate, steps))
        key = tuple(merged)
        if key not in cache:
            # C cancels at fixed noise multiplier. No mechanism is run.
            policy = PrivacyPolicy(
                max_grad_norm=1,
                noise_multiplier=1,
                delta=plan.delta,
                max_logical_steps=max(1, sum(t for _, _, t in key)),
                randomness_profile=plan.randomness_profile,
                prv_epsilon_error=plan.prv_epsilon_error,
                prv_delta_error_ratio=plan.prv_delta_error_ratio,
            )
            try:
                with warnings.catch_warnings(record=True) as notices:
                    warnings.simplefilter("always")
                    prv = privacy_epsilon(key, policy)
                    rdp = privacy_epsilon(key, policy, accountant="rdp")
            except Exception as error:
                raise ValueError(
                    "privacy forecast failed; no valid complete report produced"
                ) from error
            if any(not math.isfinite(x) or x < 0 for x in (prv, rdp)):
                raise ValueError("privacy forecast returned invalid accountant results")
            cache[key] = {
                "prv_epsilon": prv,
                "rdp_epsilon": rdp,
                "numerical_warnings": sorted({str(w.message) for w in notices}),
                "within_epsilon_limit": prv <= plan.epsilon_limit
                if plan.epsilon_limit is not None
                else None,
                "logical_steps": sum(t for _, _, t in key),
                "history": _history_json(key),
            }
        return cache[key]

    nodes: list[dict[str, Any]] = []
    for node in plan.nodes:
        draws, rate, expected_batch = resolve_sampling(
            node.sample_count, plan.batch_size
        )
        prior = tuple(h.mechanism() for h in node.prior_history)
        candidates: list[dict[str, Any]] = []
        sweep: list[tuple[float, float, int]] = []
        for sigma in plan.noise_multipliers:
            for steps in plan.logical_step_horizons:
                single = ((sigma, rate, steps),)
                repeated = ((sigma, rate, steps * plan.repetitions),)
                sweep.extend(repeated)
                candidates.append(
                    {
                        "noise_multiplier": sigma,
                        "logical_steps_per_run": steps,
                        "repetitions": plan.repetitions,
                        "expected_sampled_examples_per_run": node.sample_count
                        * rate
                        * steps,
                        "one_run": estimate(single),
                        "repetitions_only": estimate(repeated),
                        "prior_plus_repetitions": estimate(prior + repeated),
                    }
                )
        nodes.append(
            {
                "node_id": node.node_id,
                "sample_count": node.sample_count,
                "requested_batch_size": plan.batch_size,
                "loader_length": draws,
                "sample_rate": rate,
                "expected_batch_size": expected_batch,
                "delta_times_sample_count": plan.delta * node.sample_count,
                "prior": estimate(prior),
                "candidates": candidates,
                "prior_plus_entire_sweep": estimate(prior + tuple(sweep)),
            }
        )
    candidate_maxima = [
        {
            "noise_multiplier": candidate["noise_multiplier"],
            "logical_steps_per_run": candidate["logical_steps_per_run"],
            "max_local_composed_prv_epsilon": max(
                n["candidates"][index]["prior_plus_repetitions"]["prv_epsilon"]
                for n in nodes
            ),
        }
        for index, candidate in enumerate(nodes[0]["candidates"])
    ]
    return {
        "schema_version": 1,
        "purpose": "mechanism_forecast_not_training_evidence",
        "input_hash": canonical_hash(plan),
        "inputs": plan.model_dump(mode="json"),
        "accounting": {
            "primary": "prv",
            "cross_check": "rdp",
            "delta": plan.delta,
            "prv_epsilon_error": plan.prv_epsilon_error,
            "prv_delta_error": plan.delta * plan.prv_delta_error_ratio,
            "rdp_orders": "opacus-1.6.0-default",
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("opacus", "torch", "numpy", "scipy")
        },
        "nodes": nodes,
        "candidate_maxima": candidate_maxima,
        "max_local_entire_sweep_prv_epsilon": max(
            n["prior_plus_entire_sweep"]["prv_epsilon"] for n in nodes
        ),
        "new_logical_steps_per_node_entire_sweep": sum(plan.logical_step_horizons)
        * len(plan.noise_multipliers)
        * plan.repetitions,
        "group_privacy_bound": None,
        "interpretation": [
            "Each candidate is an alternative; running all candidates costs "
            "the entire-sweep composition.",
            "Repetitions reuse the same records and compose, including different "
            "seeds and failed reserved runs.",
            "Max-local values are diagnostics, not a certified cross-node "
            "transcript bound; record overlap is unverified.",
            "Prior histories are supplied assumptions; this tool neither reads "
            "nor changes live ledgers.",
            "Zero epsilon for empty prior history means no supplied spending, "
            "not a non-private baseline.",
            "Forecasts assume the declared Poisson Gaussian mechanism; secure "
            "RNG and achieved training are not verified.",
            "Runtime, GPU memory and cost need measured workload-specific "
            "pilots and are not estimated here.",
        ],
    }
