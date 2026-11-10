"""Example group-owned RMSprop registration for private training.

This module's hash binds the factory and validator; Torch's pinned version binds
its RMSprop implementation. Keep this source identical on every participant.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from dromeus.training.optimizer_factory import PrivateOptimizerFactory


def create_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.RMSprop(
        model.parameters(),
        lr=0.01,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0,
        momentum=0,
        centered=False,
        capturable=False,
        foreach=False,
        maximize=False,
        differentiable=False,
    )


def validate_optimizer_state(
    state: dict[str, Any],
    parameters: tuple[torch.Tensor, ...],
    logical_steps: int,
) -> None:
    """Validate the exact RMSprop profile above, including step-zero empty state."""
    entries = cast(dict[int, dict[str, Any]], state["state"])
    expected_ids: set[int] = set(range(len(parameters))) if logical_steps else set()
    if set(entries) != expected_ids:
        raise ValueError("RMSprop state coverage mismatch")
    for index, values in entries.items():
        if set(values) != {"step", "square_avg"}:
            raise ValueError("RMSprop state fields mismatch")
        step, average = values["step"], values["square_avg"]
        if (
            not isinstance(step, torch.Tensor)
            or step.shape != ()
            or step.dtype != torch.float32
            or float(step) != logical_steps
        ):
            raise ValueError("RMSprop counter mismatch")
        if (
            not isinstance(average, torch.Tensor)
            or average.shape != parameters[index].shape
            or average.dtype != torch.float32
            or not torch.isfinite(average).all()
            or bool((average < 0).any())
        ):
            raise ValueError("RMSprop accumulator mismatch")


def registration() -> PrivateOptimizerFactory:
    digest = hashlib.sha256(
        Path(__file__).read_bytes() + torch.__version__.encode()
    ).hexdigest()
    return PrivateOptimizerFactory(
        optimizer_type=torch.optim.RMSprop,
        implementation_hash=digest,
        create=create_optimizer,
        validate_state=validate_optimizer_state,
    )
