"""Built-in optimizer construction preserves the selected PyTorch update rule."""

from typing import Literal

import pytest
import torch
from torch import nn

from dromeus.manifests.models import PrivateOptimizerSpec
from dromeus.training.optimizer_factory import build_private_optimizer


@pytest.mark.parametrize("name", ["sgd", "adam", "adamw"])
def test_builtin_optimizer_matches_torch_updates(
    name: Literal["sgd", "adam", "adamw"],
) -> None:
    model = nn.Linear(2, 1, bias=False)
    reference = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
        reference.weight.fill_(1)
    spec = PrivateOptimizerSpec(
        name=name,
        torch_version=torch.__version__,
        parameter_names=("weight",),
        learning_rate=0.1,
        weight_decay=0.2,
        momentum=0.9 if name == "sgd" else 0,
    )
    actual = build_private_optimizer(model, spec)
    if name == "sgd":
        expected = torch.optim.SGD(
            reference.parameters(), lr=0.1, weight_decay=0.2, momentum=0.9
        )
    elif name == "adam":
        expected = torch.optim.Adam(reference.parameters(), lr=0.1, weight_decay=0.2)
    else:
        expected = torch.optim.AdamW(reference.parameters(), lr=0.1, weight_decay=0.2)
    for gradient in ([[0.5, -0.25]], [[-0.1, 0.8]], [[0.3, 0.1]]):
        model.weight.grad = torch.tensor(gradient)
        reference.weight.grad = torch.tensor(gradient)
        actual.step()
        expected.step()  # pyright: ignore[reportUnknownMemberType]
        torch.testing.assert_close(model.weight, reference.weight, rtol=0, atol=0)
    actual_state = actual.state[model.weight]
    expected_state = expected.state[reference.weight]
    assert actual_state.keys() == expected_state.keys()
    for key in expected_state:
        torch.testing.assert_close(actual_state[key], expected_state[key])
