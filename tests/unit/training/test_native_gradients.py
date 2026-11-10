"""Owned native-B1 sampler parity, isolation, and strict coverage negatives."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from dromeus.training.native_gradients import (
    NativeB1SamplerToken,
    check_native_b1_samplers,
    install_native_b1_samplers,
    validate_native_b1_model,
)
from dromeus.training.privacy import opacus_module
from dromeus.training.private_model import ensure_grad_samples

pytest.importorskip("opacus")
Base = opacus_module("opacus.grad_sample").GradSampleModule


@pytest.fixture(autouse=True)
def isolated_cpu_rng():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with cast(Any, torch.random).fork_rng(devices=[]):
        cast(Any, torch).manual_seed(17)
        yield
    torch.set_num_threads(previous)


class Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.norm = nn.LayerNorm(8)
        self.head = nn.Linear(8, 16, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(self.embedding(x)))


class Repeated(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x) + self.layer(x)


def wrap(model: nn.Module) -> tuple[Any, NativeB1SamplerToken]:
    validate_native_b1_model(model)
    wrapper = Base(model, batch_first=True, loss_reduction="mean")
    return wrapper, install_native_b1_samplers(wrapper)


def mean_loss(y: torch.Tensor) -> torch.Tensor:
    return y.mean()


def square_loss(y: torch.Tensor) -> torch.Tensor:
    return y.square().mean()


def identity_forward(x: torch.Tensor) -> torch.Tensor:
    return x


def foreign_sampler(*args: Any) -> dict[Any, Any]:
    return {}


@pytest.mark.parametrize(
    "kind",
    [
        "linear-cancellation",
        "norm-cancellation",
        "linear-noncontiguous",
        "norm-noncontiguous",
        "embedding",
        "padding",
        "tied",
        "repeated",
    ],
)
def test_native_matches_independent_autograd_without_mutating_state(kind: str):
    coefficients = torch.tensor([1000.0, 0.001, -1000.0, 0.001] * 256).reshape(
        1, 1024, 1
    )
    loss: Callable[[torch.Tensor], torch.Tensor]
    if kind == "linear-cancellation":
        model = nn.Linear(1, 1, bias=False)
        x = coefficients
        loss = mean_loss
    elif kind == "norm-cancellation":
        model = nn.LayerNorm(1)
        x = torch.ones_like(coefficients)

        def weighted_loss(y: torch.Tensor) -> torch.Tensor:
            return (y * coefficients).mean()

        loss = weighted_loss
    elif kind in ("linear-noncontiguous", "norm-noncontiguous"):
        model = nn.Linear(8, 4) if kind.startswith("linear") else nn.LayerNorm(8)
        x = torch.randn(1, 8, 32).transpose(1, 2)
        assert not x.is_contiguous()
        loss = square_loss
    elif kind in ("embedding", "padding", "tied"):
        model = (
            Tied()
            if kind == "tied"
            else nn.Embedding(16, 8, padding_idx=0 if kind == "padding" else None)
        )
        x = (torch.arange(32) % 16).reshape(1, 32)
        loss = square_loss
    else:
        model = Repeated()
        x = coefficients
        loss = mean_loss
    reference = copy.deepcopy(model)
    expected = torch.autograd.grad(loss(reference(x)), tuple(reference.parameters()))
    values = {n: p.detach().clone() for n, p in model.named_parameters()}
    versions = {n: p._version for n, p in model.named_parameters()}
    registry = dict(Base.GRAD_SAMPLERS)
    rng = torch.get_rng_state().clone()
    wrapper, token = wrap(model)
    try:
        cast(Any, loss(wrapper(x))).backward()
        ensure_grad_samples(model, 1)
        check_native_b1_samplers(wrapper, token)
        for (name, p), gradient in zip(model.named_parameters(), expected, strict=True):
            sample = cast(torch.Tensor, getattr(p, "grad_sample"))
            assert torch.allclose(sample[0], gradient, rtol=1e-4, atol=1e-6)
            assert torch.equal(p, values[name]) and p._version == versions[name]
        assert torch.equal(rng, torch.get_rng_state())
        assert dict(Base.GRAD_SAMPLERS) == registry
    finally:
        wrapper.remove_hooks()


def test_actual_physical1_token_cancellation_goes_red_then_green():
    x = torch.tensor([1000.0, 0.001, -1000.0, 0.001] * 256).reshape(1, 1024, 1)
    original = nn.Linear(1, 1, bias=False)
    legacy = Base(original, batch_first=True, loss_reduction="mean")
    try:
        legacy(x).mean().backward()
        with pytest.raises(ValueError, match="aggregate gradient"):
            ensure_grad_samples(original, 1)
    finally:
        legacy.remove_hooks()
    model = nn.Linear(1, 1, bias=False)
    wrapper, _ = wrap(model)
    try:
        wrapper(x).mean().backward()
        ensure_grad_samples(model, 1)
        assert model.weight.grad is not None
        assert torch.equal(
            cast(torch.Tensor, getattr(model.weight, "grad_sample"))[0],
            model.weight.grad,
        )
    finally:
        wrapper.remove_hooks()


class Uncovered(nn.Module):
    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind
        self.leaf = {
            "linear": lambda: nn.Linear(1, 1, bias=False),
            "norm": lambda: nn.LayerNorm(1),
            "embedding": lambda: nn.Embedding(4, 1),
        }[kind]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leaf = cast(Any, self.leaf)
        if self.kind == "linear":
            return leaf(x * 0) + F.linear(torch.ones_like(x), leaf.weight) * 5e-6
        if self.kind == "norm":
            return leaf(x) * 0 + leaf.bias * 5e-6
        return leaf(x) * 0 + leaf.weight[0] * 5e-6


@pytest.mark.parametrize("kind", ["linear", "norm", "embedding"])
def test_small_uncovered_contribution_remains_rejected(kind: str):
    model = Uncovered(kind)
    wrapper, _ = wrap(model)
    x = (
        torch.zeros(1, 1, dtype=torch.long)
        if kind == "embedding"
        else torch.ones(1, 1, 1)
    )
    try:
        wrapper(x).mean().backward()
        with pytest.raises(ValueError, match="aggregate gradient"):
            ensure_grad_samples(model, 1)
    finally:
        wrapper.remove_hooks()


@pytest.mark.parametrize("batch", [0, 2])
def test_callbacks_refuse_empty_or_multiple_records(batch: int):
    model = nn.Linear(1, 1)
    wrapper, _ = wrap(model)
    try:
        callback = wrapper.GRAD_SAMPLERS[nn.Linear]
        with pytest.raises(ValueError, match="one nonempty physical"):
            callback(model, [torch.ones(batch, 1)], torch.ones(batch, 1))
    finally:
        wrapper.remove_hooks()


def test_existing_zero_count_guard_path_does_not_require_native_callback():
    model = nn.Linear(1, 1)
    wrapper, token = wrap(model)
    try:
        ensure_grad_samples(model, 0)
        check_native_b1_samplers(wrapper, token)
    finally:
        wrapper.remove_hooks()


@pytest.mark.parametrize(
    "kind", ["mapping", "model", "configuration", "parameter", "forward", "registry"]
)
def test_token_rejects_tampering(kind: str, monkeypatch: pytest.MonkeyPatch):
    model = nn.LayerNorm(4)
    wrapper, token = wrap(model)
    try:
        if kind == "mapping":
            wrapper.GRAD_SAMPLERS = dict(wrapper.GRAD_SAMPLERS)
        elif kind == "model":
            wrapper._module = nn.LayerNorm(4)
        elif kind == "configuration":
            model.eps *= 2
        elif kind == "parameter":
            model.weight = nn.Parameter(model.weight.detach().clone())
        elif kind == "forward":
            monkeypatch.setattr(model, "forward", identity_forward)
        else:
            monkeypatch.setitem(Base.GRAD_SAMPLERS, nn.LayerNorm, foreign_sampler)
        with pytest.raises(ValueError):
            check_native_b1_samplers(wrapper, token)
    finally:
        wrapper.remove_hooks()


def test_mapping_and_token_are_immutable_and_instance_owned():
    first, token = wrap(nn.Linear(1, 1))
    second, other = wrap(nn.Linear(1, 1))
    try:
        assert first.GRAD_SAMPLERS is not second.GRAD_SAMPLERS
        assert first.GRAD_SAMPLERS is not Base.GRAD_SAMPLERS
        with pytest.raises(TypeError):
            first.GRAD_SAMPLERS[nn.Linear] = foreign_sampler
        with pytest.raises(FrozenInstanceError):
            setattr(token, "mapping", {})
        with pytest.raises(ValueError):
            check_native_b1_samplers(first, other)
        with pytest.raises(ValueError, match="existing instance"):
            install_native_b1_samplers(first)
    finally:
        first.remove_hooks()
        second.remove_hooks()


@pytest.mark.parametrize(
    "kind",
    ["dtype", "subclass", "forward", "max_norm", "sparse", "frequency", "unsupported"],
)
def test_model_admission_is_narrow(kind: str, monkeypatch: pytest.MonkeyPatch):
    class Foreign(nn.Linear):
        pass

    model: nn.Module = nn.Linear(2, 2)
    if kind == "dtype":
        model = model.double()
    elif kind == "subclass":
        model = Foreign(2, 2)
    elif kind == "forward":
        monkeypatch.setattr(model, "forward", identity_forward)
    elif kind in ("max_norm", "sparse", "frequency"):
        model = nn.Embedding(
            4,
            2,
            max_norm=1 if kind == "max_norm" else None,
            sparse=kind == "sparse",
            scale_grad_by_freq=kind == "frequency",
        )
    else:
        model = nn.Conv1d(1, 1, 1)
    with pytest.raises(ValueError):
        validate_native_b1_model(model)


@pytest.mark.parametrize(
    "option",
    [{"batch_first": False}, {"loss_reduction": "sum"}, {"force_functorch": True}],
)
def test_nonstandard_wrapper_is_rejected(option: dict[str, Any]):
    kwargs = {"batch_first": True, "loss_reduction": "mean", **option}
    wrapper = Base(nn.Linear(2, 2), **kwargs)
    try:
        with pytest.raises(ValueError, match="standard batch-first"):
            install_native_b1_samplers(wrapper)
    finally:
        wrapper.remove_hooks()


def test_versions_are_explicit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch, "__version__", "2.12.0")
    with pytest.raises(ValueError, match="Torch 2.13"):
        validate_native_b1_model(nn.Linear(1, 1))


def test_module_import_keeps_optional_dependencies_lazy():
    root = Path(__file__).resolve().parents[3]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    command = (
        "import sys;import dromeus.training.native_gradients;"
        'assert "torch" not in sys.modules;assert "opacus" not in sys.modules'
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_opacus_version_is_pinned(monkeypatch: pytest.MonkeyPatch):
    opacus = opacus_module()
    monkeypatch.setattr(opacus, "__version__", "1.5.0")
    with pytest.raises(ValueError, match="Opacus 1.6.0"):
        validate_native_b1_model(nn.Linear(1, 1))


def test_missing_optional_runtime_fails_only_on_backend_use(
    monkeypatch: pytest.MonkeyPatch,
):
    import importlib

    original = importlib.import_module

    def unavailable(name: str, package: str | None = None) -> Any:
        if name == "torch":
            raise ImportError("public fixture unavailable Torch")
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", unavailable)
    with pytest.raises(ValueError, match="requires pinned Torch"):
        validate_native_b1_model(nn.Linear(1, 1))
