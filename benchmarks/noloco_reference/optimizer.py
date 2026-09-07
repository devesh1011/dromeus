"""Independent Torch port of the pinned upstream NoLoCo outer step."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

PINNED_UPSTREAM_COMMIT = "a1b4a425bdc4050a356cf9f4bae7c383419703ab"
PINNED_UPSTREAM_SOURCE = "src/noloco/sparse_optimizer_c.py"

TensorMap = Mapping[str, Tensor]


@dataclass(frozen=True, slots=True)
class ReferenceNoLoCoConfig:
    alpha: float
    beta: float
    gamma: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha < 1.0 or not math.isfinite(self.alpha):
            raise ValueError("alpha must be finite in [0, 1)")
        if self.beta <= 0.0 or not math.isfinite(self.beta):
            raise ValueError("beta must be positive and finite")
        if self.gamma <= 0.0 or not math.isfinite(self.gamma):
            raise ValueError("gamma must be positive and finite")


@dataclass(frozen=True, slots=True)
class ReferenceOuterResult:
    outer_momentum: dict[str, Tensor]
    slow_weights: dict[str, Tensor]


def outer_gradient(
    slow_weights: TensorMap,
    fast_weights: TensorMap,
) -> dict[str, Tensor]:
    """Return the descent-oriented ``g = phi - theta`` in FP32."""
    names = _validate_tensor_maps(slow_weights, fast_weights)
    with torch.no_grad():
        return {
            name: (slow_weights[name] - fast_weights[name]).detach().clone()
            for name in names
        }


def apply_outer_step(
    *,
    slow_weights: TensorMap,
    outer_momentum: TensorMap,
    local_outer_gradient: TensorMap,
    peer_outer_gradient: TensorMap,
    peer_slow_weights: TensorMap,
    config: ReferenceNoLoCoConfig,
) -> ReferenceOuterResult:
    """Apply the source-equivalent two-worker modified Nesterov step."""
    names = _validate_tensor_maps(
        slow_weights,
        outer_momentum,
        local_outer_gradient,
        peer_outer_gradient,
        peer_slow_weights,
    )
    next_momentum: dict[str, Tensor] = {}
    next_slow: dict[str, Tensor] = {}
    with torch.no_grad():
        for name in names:
            momentum = (
                config.alpha * outer_momentum[name]
                - (config.beta / 2.0)
                * (local_outer_gradient[name] + peer_outer_gradient[name])
                - (config.gamma / 2.0)
                * (slow_weights[name] - peer_slow_weights[name])
            ).detach()
            next_momentum[name] = momentum.clone()
            next_slow[name] = (slow_weights[name] + momentum).detach().clone()
    return ReferenceOuterResult(
        outer_momentum=next_momentum,
        slow_weights=next_slow,
    )


def _validate_tensor_maps(*maps: TensorMap) -> tuple[str, ...]:
    if not maps:
        raise ValueError("at least one tensor map is required")
    names = tuple(sorted(maps[0]))
    if not names:
        raise ValueError("tensor maps must not be empty")
    if any(set(values) != set(names) for values in maps):
        raise ValueError("tensor names do not match")
    for name in names:
        reference = maps[0][name]
        if reference.dtype != torch.float32:
            raise ValueError(f"tensor {name} must use float32")
        if not bool(torch.isfinite(reference).all()):
            raise ValueError(f"tensor {name} must be finite")
        for values in maps[1:]:
            value = values[name]
            if (
                value.dtype != reference.dtype
                or value.shape != reference.shape
                or value.device != reference.device
            ):
                raise ValueError(f"tensor {name} schema does not match")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"tensor {name} must be finite")
    return names


__all__ = [
    "PINNED_UPSTREAM_COMMIT",
    "PINNED_UPSTREAM_SOURCE",
    "ReferenceNoLoCoConfig",
    "ReferenceOuterResult",
    "apply_outer_step",
    "outer_gradient",
]
