"""Construction and state validation for built-in and group-selected optimizers.

Factories and validators are trusted application code. Their implementation hash
must bind the optimizer, factory and state validator. Optimizers may consume only
sanitized gradients, model parameters, public settings and their own numeric state.
"""

from __future__ import annotations

import inspect
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from dromeus.manifests.models import (
    FactoryOptimizerSpec,
    OptimizerParameterGroup,
    PrivateOptimizerSpec,
)
from dromeus.training.private_state import decode_state, encode_state


def build_private_optimizer(
    model: nn.Module, spec: PrivateOptimizerSpec | FactoryOptimizerSpec
) -> torch.optim.Optimizer:
    """Build the declared ordinary update rule, before optional Opacus wrapping."""
    if isinstance(spec, FactoryOptimizerSpec):
        raise ValueError("factory optimizer requires a local optimizer registration")
    parameters = [p for p in model.parameters() if p.requires_grad]
    common: dict[str, Any] = dict(
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
        foreach=False,
        fused=False,
        differentiable=False,
        maximize=False,
    )
    if spec.name == "sgd":
        return torch.optim.SGD(
            parameters,
            momentum=spec.momentum,
            dampening=spec.dampening,
            nesterov=spec.nesterov,
            **common,
        )
    factory = torch.optim.Adam if spec.name == "adam" else torch.optim.AdamW
    return factory(
        parameters,
        betas=spec.betas,
        eps=spec.numerical_epsilon,
        amsgrad=False,
        capturable=False,
        **common,
    )


def validate_builtin_optimizer_checkpoint(
    state: dict[str, Any],
    *,
    spec: PrivateOptimizerSpec,
    parameters: tuple[torch.Tensor, ...],
    initial_groups: list[dict[str, Any]],
    logical_steps: int,
    horizon: int,
    schedule: str,
) -> None:
    """Validate built-in state against the frozen recipe without mutating it.

    Parameters retain model order and groups come from the initial state_dict,
    before scheduling. Factory-defined state uses its registered validator.
    """
    if set(state) != {"state", "param_groups"}:
        raise ValueError("optimizer checkpoint fields mismatch")
    groups = state["param_groups"]
    if len(groups) != len(initial_groups):
        raise ValueError("optimizer checkpoint groups mismatch")
    for group, initial in zip(groups, initial_groups, strict=True):
        expected = dict(initial)
        if schedule == "linear_decay":
            expected["lr"] *= 1 - logical_steps / horizon
        if group != expected:
            raise ValueError("optimizer checkpoint options or schedule mismatch")
    fields: set[str] = (
        ({"momentum_buffer"} if spec.momentum else set())
        if spec.name == "sgd"
        else {"step", "exp_avg", "exp_avg_sq"}
    )
    ids: set[int] = set(range(len(parameters))) if logical_steps and fields else set()
    if set(state["state"]) != ids:
        raise ValueError("optimizer checkpoint state coverage mismatch")
    for index in ids:
        values = state["state"][index]
        if set(values) != fields:
            raise ValueError("optimizer checkpoint state fields mismatch")
        for name, tensor in values.items():
            shape = () if name == "step" else parameters[index].shape
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != shape
                or tensor.dtype != torch.float32
            ):
                raise ValueError("optimizer checkpoint tensor mismatch")
            if name == "step" and float(tensor) != logical_steps:
                raise ValueError("optimizer checkpoint step mismatch")


def _json_option(value: Any, depth: int = 0) -> Any:
    if depth > 32:
        raise ValueError("optimizer option nesting exceeds bound")
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, (tuple, list)):
        return [_json_option(item, depth + 1) for item in cast(list[Any], value)]
    if isinstance(value, dict) and all(
        type(k) is str for k in cast(dict[Any, Any], value)
    ):
        return {
            key: _json_option(item, depth + 1)
            for key, item in cast(dict[str, Any], value).items()
        }
    raise ValueError("optimizer options must contain finite JSON values")


def _canonical_options(value: Any) -> str:
    return json.dumps(_json_option(value), sort_keys=True, separators=(",", ":"))


def _implementation_id(optimizer_type: type[torch.optim.Optimizer]) -> str:
    return f"{optimizer_type.__module__}.{optimizer_type.__qualname__}"


@dataclass(frozen=True)
class _ActualGroup:
    parameter_names: tuple[str, ...]
    options: dict[str, Any]


def _groups(
    optimizer: torch.optim.Optimizer, model: nn.Module
) -> tuple[_ActualGroup, ...]:
    named = tuple((name, p) for name, p in model.named_parameters() if p.requires_grad)
    names = {id(parameter): name for name, parameter in named}
    seen: set[int] = set()
    result: list[_ActualGroup] = []
    for group in optimizer.param_groups:
        members: list[str] = []
        for parameter in group["params"]:
            identity = id(parameter)
            if identity not in names or identity in seen:
                raise ValueError(
                    "optimizer parameters must cover the model exactly once"
                )
            seen.add(identity)
            members.append(names[identity])
        options = _json_option({k: v for k, v in group.items() if k != "params"})
        result.append(_ActualGroup(parameter_names=tuple(members), options=options))
    if seen != set(names):
        raise ValueError("optimizer parameters must cover the model exactly once")
    return tuple(result)


def _numeric_copy(state: dict[str, Any]) -> dict[str, Any]:
    """Reject nonnumeric payloads before giving a detached copy to trusted code."""

    def bounded(value: Any, depth: int = 0) -> None:
        if depth > 32:
            raise ValueError("optimizer state nesting exceeds bound")
        if isinstance(value, dict):
            for key, item in cast(dict[Any, Any], value).items():
                bounded(key, depth + 1)
                bounded(item, depth + 1)
        elif isinstance(value, (tuple, list)):
            for item in cast(list[Any], value):
                bounded(item, depth + 1)

    bounded(state)
    return cast(dict[str, Any], decode_state(encode_state(state)))


@dataclass(frozen=True)
class PrivateOptimizerFactory:
    """Attested optimizer construction and optimizer-specific restore validation.

    ``validate_state`` receives a detached numeric state dictionary, parameters in
    optimizer group order, and the completed logical step count. It must reject
    invalid field layouts, tensor shapes/dtypes and counters. Public eager state
    at step zero is allowed; prior private training state is not public initialization.
    """

    optimizer_type: type[torch.optim.Optimizer]
    implementation_hash: str
    create: Callable[[nn.Module], torch.optim.Optimizer]
    validate_state: Callable[[dict[str, Any], tuple[torch.Tensor, ...], int], None]

    def _validate_type(self, optimizer: torch.optim.Optimizer) -> None:
        if not issubclass(cast(Any, self.optimizer_type), torch.optim.Optimizer) or (
            type(optimizer) is not self.optimizer_type
        ):
            raise ValueError("optimizer factory returned a different class")
        try:
            inspect.signature(optimizer.step).bind()
        except (TypeError, ValueError):
            raise ValueError(
                "optimizer step must work without closures or arguments"
            ) from None

    def describe(self, model: nn.Module) -> FactoryOptimizerSpec:
        """Describe a fresh optimizer constructed against a public model."""
        optimizer = self.create(model)
        self._validate_type(optimizer)
        spec = FactoryOptimizerSpec(
            implementation_id=_implementation_id(self.optimizer_type),
            implementation_hash=self.implementation_hash,
            torch_version=torch.__version__,
            groups=tuple(
                OptimizerParameterGroup(
                    parameter_names=g.parameter_names, options=g.options
                )
                for g in _groups(optimizer, model)
            ),
        )
        self.validate_checkpoint(
            optimizer,
            model,
            spec,
            optimizer.state_dict(),
            logical_steps=0,
            horizon=1,
            schedule="constant",
        )
        return spec

    def build(
        self, model: nn.Module, spec: FactoryOptimizerSpec
    ) -> torch.optim.Optimizer:
        """Construct and validate a fresh public optimizer before privacy spending."""
        optimizer = self.create(model)
        self.check(
            optimizer, model, spec, logical_steps=0, horizon=1, schedule="constant"
        )
        self.validate_checkpoint(
            optimizer,
            model,
            spec,
            optimizer.state_dict(),
            logical_steps=0,
            horizon=1,
            schedule="constant",
        )
        return optimizer

    def check(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        spec: FactoryOptimizerSpec,
        *,
        logical_steps: int,
        horizon: int,
        schedule: str,
    ) -> None:
        """Reject changed optimizer identity, parameter mapping, options or rates."""
        self._validate_type(optimizer)
        if (
            spec.implementation_id != _implementation_id(self.optimizer_type)
            or spec.implementation_hash != self.implementation_hash
            or spec.torch_version != torch.__version__
        ):
            raise ValueError(
                "optimizer implementation differs from shared specification"
            )
        actual = _groups(optimizer, model)
        if len(actual) != len(spec.groups):
            raise ValueError("optimizer parameter groups changed")
        for group, expected in zip(actual, spec.groups, strict=True):
            rate = _rate(expected, logical_steps, horizon, schedule)
            if (
                group.parameter_names != expected.parameter_names
                or not _matching_options(group.options, expected, rate)
            ):
                raise ValueError(
                    "optimizer groups or options differ from shared specification"
                )

    def validate_checkpoint(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        spec: FactoryOptimizerSpec,
        state: dict[str, Any],
        *,
        logical_steps: int,
        horizon: int,
        schedule: str,
    ) -> None:
        """Validate restore candidates without mutating the live optimizer."""
        self._validate_type(optimizer)
        if (
            spec.implementation_id != _implementation_id(self.optimizer_type)
            or spec.implementation_hash != self.implementation_hash
            or spec.torch_version != torch.__version__
        ):
            raise ValueError("optimizer checkpoint implementation mismatch")
        actual = _groups(optimizer, model)
        if tuple(g.parameter_names for g in actual) != tuple(
            g.parameter_names for g in spec.groups
        ):
            raise ValueError("optimizer checkpoint model mapping mismatch")
        candidate = _numeric_copy(state)
        if set(candidate) != {"state", "param_groups"}:
            raise ValueError("invalid optimizer checkpoint fields")
        groups = candidate["param_groups"]
        reference = optimizer.state_dict()["param_groups"]
        if not isinstance(groups, list) or len(cast(list[Any], groups)) != len(
            spec.groups
        ):
            raise ValueError("optimizer checkpoint group count mismatch")
        known: set[int] = set()
        for group, current, expected in zip(
            cast(list[Any], groups), reference, spec.groups, strict=True
        ):
            if not isinstance(group, dict) or set(cast(dict[str, Any], group)) != set(
                current
            ):
                raise ValueError("optimizer checkpoint group fields mismatch")
            group = cast(dict[str, Any], group)
            ids = group["params"]
            if (
                not isinstance(ids, list)
                or any(type(index) is not int for index in cast(list[Any], ids))
                or ids != current["params"]
            ):
                raise ValueError("optimizer checkpoint parameter mapping mismatch")
            known.update(cast(list[int], ids))
            rate = _rate(expected, logical_steps, horizon, schedule)
            if not _matching_options(
                {k: v for k, v in group.items() if k != "params"}, expected, rate
            ):
                raise ValueError("optimizer checkpoint options or schedule mismatch")
        saved_state = candidate["state"]
        if not isinstance(saved_state, dict) or any(
            type(index) is not int or index not in known
            for index in cast(dict[Any, Any], saved_state)
        ):
            raise ValueError("optimizer checkpoint state references unknown parameters")
        parameters = tuple(
            parameter.detach().clone()
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
        self.validate_state(candidate, parameters, logical_steps)


def _matching_options(
    actual: dict[str, Any], expected: OptimizerParameterGroup, rate: float
) -> bool:
    current_rate = actual.get("lr")
    return (
        type(current_rate) in (int, float)
        and math.isfinite(cast(float, current_rate))
        and current_rate == rate
        and _canonical_options({k: v for k, v in actual.items() if k != "lr"})
        == _canonical_options({k: v for k, v in expected.options.items() if k != "lr"})
    )


def _rate(
    group: OptimizerParameterGroup, steps: int, horizon: int, schedule: str
) -> float:
    if type(steps) is not int or not 0 <= steps <= horizon or horizon <= 0:
        raise ValueError("invalid optimizer schedule counters")
    if schedule not in {"constant", "linear_decay"}:
        raise ValueError("unsupported optimizer schedule")
    rate = cast(float, group.options["lr"])
    return rate * (1 - steps / horizon) if schedule == "linear_decay" else rate


class SanitizedOptimizer(torch.optim.Optimizer):
    """Expose sanitized gradients to the selected optimizer's update only.

    Opacus owns clipping/noise/accumulation. Its private parameter attributes are
    temporarily hidden during the update and restored even when the update fails.
    This is a lifecycle guard for trusted code, not a malicious-code sandbox.
    """

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.original_optimizer = optimizer
        super().__init__(optimizer.param_groups, optimizer.defaults)
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state

    def step(self, closure: Callable[[], float] | None = None) -> Any:
        if closure is not None:
            raise ValueError("private optimizers cannot evaluate closures")
        hidden: list[tuple[torch.Tensor, str, Any]] = []
        try:
            for group in self.param_groups:
                for parameter in group["params"]:
                    for name in ("grad_sample", "summed_grad"):
                        if hasattr(parameter, name):
                            hidden.append((parameter, name, getattr(parameter, name)))
                            delattr(parameter, name)
            return self.original_optimizer.step()  # pyright: ignore[reportUnknownMemberType]
        finally:
            for parameter, name, value in hidden:
                setattr(parameter, name, value)

    def zero_grad(self, set_to_none: bool = True) -> None:
        torch.optim.Optimizer.zero_grad(self, set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return self.original_optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.original_optimizer.load_state_dict(state_dict)
        self.param_groups = self.original_optimizer.param_groups
        self.state = self.original_optimizer.state
