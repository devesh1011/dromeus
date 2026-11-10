"""Public numerical fixture at the actual aggregate-coverage rejection seam.

Opacus hooks produce every gradient sample. Independent single-record autograd
checks their correctness; float64 is a reference, not an admitted private mode.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from torch import nn

from dromeus.training.privacy import opacus_module
from dromeus.training.private_model import ensure_grad_samples

pytest.importorskip("opacus")
GradSampleModule = opacus_module("opacus.grad_sample").GradSampleModule


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _linear(dtype: torch.dtype) -> nn.Linear:
    with cast(Any, torch.random).fork_rng(devices=[]):
        cast(Any, torch).manual_seed(17)
        model = nn.Linear(1, 1, bias=False, device="cpu", dtype=dtype)
    with torch.no_grad():
        model.weight.fill_(0.125)
    return model


def _backward(
    values: list[list[float]], *, dtype: torch.dtype = torch.float32
) -> tuple[nn.Module, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = torch.tensor(values, dtype=dtype).unsqueeze(-1)
    model = _linear(dtype)
    wrapper = GradSampleModule(model, batch_first=True, loss_reduction="mean")
    wrapper(inputs).mean(dim=(1, 2)).mean().backward()
    parameter = next(model.parameters())
    samples = cast(torch.Tensor, getattr(parameter, "grad_sample"))
    aggregate = parameter.grad
    assert aggregate is not None
    plain = _linear(dtype)
    independent = torch.stack(
        [
            torch.autograd.grad(plain(inputs[index : index + 1]).mean(), plain.weight)[
                0
            ]
            for index in range(len(inputs))
        ]
    )
    return model, samples, aggregate, independent


class _KnownFp32CancellationRejection(Exception):
    """Only the retained, independently verified coverage false rejection."""


@pytest.mark.xfail(
    strict=True,
    raises=_KnownFp32CancellationRejection,
    reason=(
        "Known FP32 token/batch reduction cancellation false rejection; "
        "core guard intentionally retained while physical1 mitigation is conditional"
    ),
)
def test_valid_fp32_sequence_gradients_are_not_rejected_as_missing_coverage() -> None:
    model, samples, aggregate, independent = _backward(
        [[10000.0, 0.001], [-10000.0, 0.001]]
    )
    assert torch.equal(samples, independent)
    assert bool(torch.isfinite(samples).all())
    assert not torch.allclose(samples.mean(0), aggregate, rtol=1e-4, atol=1e-6)
    # No generic ValueError is expected: a different failure remains a hard FAIL.
    try:
        ensure_grad_samples(model, 2)
    except ValueError as error:
        traceback = error.__traceback__
        while traceback is not None and traceback.tb_next is not None:
            traceback = traceback.tb_next
        if (
            type(error) is ValueError
            and str(error)
            == "weight: per-record gradients do not cover aggregate gradient"
            and traceback is not None
            and traceback.tb_frame.f_code.co_name == "ensure_grad_samples"
            and traceback.tb_frame.f_code.co_filename.endswith(
                "/src/dromeus/training/private_model.py"
            )
        ):
            raise _KnownFp32CancellationRejection from error
        raise


def test_fp64_reference_removes_the_same_reduction_discrepancy() -> None:
    model, samples, aggregate, independent = _backward(
        [[10000.0, 0.001], [-10000.0, 0.001]], dtype=torch.float64
    )
    assert torch.equal(samples, independent)
    assert torch.allclose(samples.mean(0), aggregate, rtol=0, atol=1e-12)
    expected = torch.tensor([[0.0005]], dtype=torch.float64)
    assert torch.allclose(aggregate, expected, rtol=0, atol=1e-12)
    ensure_grad_samples(model, 2)


def test_removing_token_reduction_removes_the_fp32_false_rejection() -> None:
    model, samples, aggregate, independent = _backward([[10000.0], [-10000.0]])
    assert torch.equal(samples, independent)
    assert torch.equal(samples.mean(0), aggregate)
    ensure_grad_samples(model, 2)


def test_actual_partly_unhooked_parameter_contribution_remains_rejected() -> None:
    class PartlyHooked(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = nn.Linear(2, 1, bias=False)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.layer(inputs) + nn.functional.linear(inputs, self.layer.weight)

    with cast(Any, torch.random).fork_rng(devices=[]):
        cast(Any, torch).manual_seed(17)
        model = PartlyHooked()
    wrapper = GradSampleModule(model, batch_first=True, loss_reduction="mean")
    try:
        wrapper(torch.tensor([[1.0, 2.0], [3.0, 4.0]])).mean().backward()
        with pytest.raises(ValueError, match="aggregate"):
            ensure_grad_samples(model, 2)
    finally:
        wrapper.remove_hooks()
