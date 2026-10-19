"""Trusted optimizer registrations used by private conformance tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast

import torch
from torch import nn

from dromeus.training.optimizer_factory import PrivateOptimizerFactory


def public_model() -> nn.Module:
    model = nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.fill_(0.25)
        model.bias.fill_(0.1)
    return model


class CustomOptimizer(torch.optim.Optimizer):
    """Two rates, scalar history, and integer counters outside built-in profiles."""

    check_sanitized = True

    def __init__(self, model: nn.Module) -> None:
        parameters = list(model.parameters())
        super().__init__(
            [
                {"params": parameters[:1], "lr": 0.03},
                {"params": parameters[1:], "lr": 0.01},
            ],
            {"lr": 0.03, "decay": 0.8},
        )

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> None:  # type: ignore[override]
        assert closure is None
        for group in self.param_groups:
            for parameter in group["params"]:
                if self.check_sanitized:
                    assert getattr(parameter, "grad_sample", None) is None
                    assert getattr(parameter, "summed_grad", None) is None
                gradient = parameter.grad
                assert gradient is not None
                state = self.state[parameter]
                if not state:
                    state["count"] = 0
                    state["energy"] = torch.zeros(1, device=parameter.device)
                state["count"] += 1
                state["energy"].mul_(group["decay"]).add_(gradient.square().mean())
                parameter.add_(
                    gradient / (1 + state["energy"].sqrt()), alpha=-group["lr"]
                )


def _state_validator(kind: str) -> Callable[..., None]:
    def validate(
        state: dict[str, Any], parameters: tuple[torch.Tensor, ...], steps: int
    ) -> None:
        states = cast(dict[int, dict[str, Any]], state["state"])
        expected_ids: set[int] = (
            set(range(len(parameters))) if steps or kind == "adagrad" else set()
        )
        if set(states) != expected_ids:
            raise ValueError("optimizer state coverage")
        for index, fields in states.items():
            parameter = parameters[index]
            if kind == "custom":
                if (
                    set(fields) != {"count", "energy"}
                    or type(fields["count"]) is not int
                    or fields["count"] != steps
                ):
                    raise ValueError("custom state counter/fields")
                tensor = fields["energy"]
                if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (1,):
                    raise ValueError("custom state shape")
            else:
                buffer = "sum" if kind == "adagrad" else "square_avg"
                if set(fields) != {"step", buffer}:
                    raise ValueError("optimizer state fields")
                counter = fields["step"]
                if not isinstance(counter, torch.Tensor) or counter.numel() != 1:
                    raise ValueError("optimizer state counter shape")
                if float(counter.item()) != steps:
                    raise ValueError("optimizer state counter")
                tensor = fields[buffer]
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.shape != parameter.shape
                ):
                    raise ValueError("optimizer state shape")
            if tensor.dtype != parameter.dtype or not torch.isfinite(tensor).all():
                raise ValueError("optimizer state dtype/value")
            if bool((tensor < 0).any()):
                raise ValueError("optimizer state negative accumulator")

    return validate


def optimizer_registration(
    kind: Literal["custom", "rmsprop", "adagrad"] = "custom",
) -> PrivateOptimizerFactory:
    factories: dict[
        str,
        tuple[
            type[torch.optim.Optimizer], Callable[[nn.Module], torch.optim.Optimizer]
        ],
    ] = {
        "custom": (CustomOptimizer, CustomOptimizer),
        "rmsprop": (
            torch.optim.RMSprop,
            lambda model: torch.optim.RMSprop(
                model.parameters(), lr=0.03, alpha=0.8, foreach=False
            ),
        ),
        "adagrad": (
            torch.optim.Adagrad,
            lambda model: torch.optim.Adagrad(
                model.parameters(), lr=0.03, foreach=False, fused=False
            ),
        ),
    }
    optimizer_type, create = factories[kind]
    return PrivateOptimizerFactory(
        optimizer_type=optimizer_type,
        implementation_hash={"custom": "a", "rmsprop": "b", "adagrad": "c"}[kind] * 64,
        create=create,
        validate_state=_state_validator(kind),
    )
