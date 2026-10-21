"""Public model compatibility checks against independent record gradients."""

from typing import Any, cast

import pytest
import torch
from torch import nn

from dromeus.training.private_model import validate_private_model

pytest.importorskip("opacus")


class BatchAdapter:
    def size(self, batch: Any) -> int:
        return len(batch)

    def slice(self, batch: Any, start: int, end: int) -> Any:
        return batch[start:end]

    def to_device(self, batch: Any, device: torch.device) -> Any:
        return batch.to(device)


class UserRegression(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(2, 3)
        self.activation = nn.GELU()
        self.last = nn.Linear(3, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.last(self.activation(self.first(inputs)))


def regression_loss(model: nn.Module, batch: Any) -> torch.Tensor:
    return model(batch).square().flatten(1).mean(1)


def test_custom_model_is_checked_without_changing_model_or_attaching_hooks() -> None:
    model = UserRegression()
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    report = validate_private_model(
        model,
        regression_loss,
        BatchAdapter(),
        torch.tensor([[1.0, 2.0], [-0.5, 0.25], [0.1, -1.0]]),
    )
    assert report.trainable_parameter_names == (
        "first.weight",
        "first.bias",
        "last.weight",
        "last.bias",
    )
    assert report.gradient_backend == "opacus_hooks_v1"
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not hasattr(model, "autograd_grad_sample_hooks")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[name], rtol=0, atol=0)


class SharedEmbedding(nn.Module):
    def __init__(self, *, bad_positions: bool = False) -> None:
        super().__init__()
        self.tokens = nn.Embedding(7, 4)
        self.positions = nn.Embedding(5, 4)
        self.norm = nn.LayerNorm(4)
        self.head = nn.Linear(4, 7, bias=False)
        self.head.weight = self.tokens.weight
        self.bad_positions = bad_positions

    def forward(self, indices: torch.Tensor) -> tuple[torch.Tensor, None]:
        positions = torch.arange(indices.shape[1], device=indices.device)
        if not self.bad_positions:
            positions = positions.unsqueeze(0).expand_as(indices)
        hidden = self.tokens(indices) + self.positions(positions)
        return self.head(self.norm(hidden)), None


def language_loss(model: nn.Module, batch: Any) -> torch.Tensor:
    logits, _ = model(batch)
    return torch.nn.functional.cross_entropy(
        logits.transpose(1, 2), batch, reduction="none"
    ).mean(1)


def test_shared_embedding_contributions_are_combined_per_sequence() -> None:
    model = SharedEmbedding()
    report = validate_private_model(
        model, language_loss, BatchAdapter(), torch.tensor([[1, 2, 1], [3, 2, 4]])
    )
    assert report.parameter_aliases == (("head.weight", "tokens.weight"),)
    assert "tokens.weight" in report.trainable_parameter_names
    assert "head.weight" not in report.trainable_parameter_names
    assert model.tokens.weight is model.head.weight


def test_unbatched_position_embedding_fails_gradient_compatibility() -> None:
    with pytest.raises((ValueError, RuntimeError), match="gradient|shape|size|batch"):
        validate_private_model(
            SharedEmbedding(bad_positions=True),
            language_loss,
            BatchAdapter(),
            torch.tensor([[1, 2, 1], [3, 2, 4]]),
        )


class ModelWithPublicState(nn.Module):
    scale: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.frozen = nn.Linear(2, 2)
        self.frozen.requires_grad_(False)
        self.output = nn.Linear(2, 1)
        self.register_buffer("scale", torch.tensor([2.0]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.frozen(inputs)) * self.scale


def test_declared_public_buffers_and_frozen_parameters_have_guarded_identity() -> None:
    from dromeus.training.private_model import ModelStateGuard

    model = ModelWithPublicState()
    report = validate_private_model(
        model, regression_loss, BatchAdapter(), torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    )
    assert set(report.public_buffer_hashes) == {"scale"}
    assert set(report.frozen_parameter_hashes) == {"frozen.weight", "frozen.bias"}
    guard = ModelStateGuard(model, report)
    with torch.no_grad():
        model.output.weight.add_(0.1)
    guard.check()
    with torch.no_grad():
        model.frozen.weight.add_(0.1)
    with pytest.raises(ValueError, match="frozen"):
        guard.check()


def test_guard_rejects_public_buffer_mutation_and_broken_parameter_ties() -> None:
    from dromeus.training.private_model import ModelStateGuard, describe_private_model

    model = ModelWithPublicState()
    guard = ModelStateGuard(model, describe_private_model(model))
    model.scale.add_(1)
    with pytest.raises(ValueError, match="public buffers"):
        guard.check()
    shared = SharedEmbedding()
    guard = ModelStateGuard(shared, describe_private_model(shared))
    shared.head.weight = nn.Parameter(shared.head.weight.detach().clone())
    with pytest.raises(ValueError, match="parameter identity"):
        guard.check()


def test_unused_branch_parameters_remain_part_of_every_record_gradient() -> None:
    from dromeus.training.private_model import ensure_grad_samples

    class Branch(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.used = nn.Linear(2, 1)
            self.unused = nn.Linear(2, 1)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.used(inputs)

    model = Branch()
    report = validate_private_model(
        model, regression_loss, BatchAdapter(), torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    )
    assert "unused.weight" in report.trainable_parameter_names
    ensure_grad_samples(model, 0)
    for parameter in model.parameters():
        assert getattr(parameter, "grad_sample").shape == (0, *parameter.shape)


def test_batch_dependent_loss_fails_independent_record_parity() -> None:
    def batch_loss(model: nn.Module, batch: Any) -> torch.Tensor:
        predictions = model(batch)
        return (predictions - predictions.mean(0)).square().flatten(1).mean(1)

    with pytest.raises(ValueError, match="other records"):
        validate_private_model(
            UserRegression(),
            batch_loss,
            BatchAdapter(),
            torch.tensor([[1.0, 2.0], [8.0, -4.0]]),
        )


def test_private_probe_rejects_forward_mutation_of_model_state() -> None:
    class Mutating(ModelWithPublicState):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            self.scale.add_(1)
            return super().forward(inputs)

    model = Mutating()
    with pytest.raises(ValueError, match="mutated"):
        validate_private_model(
            model,
            regression_loss,
            BatchAdapter(),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        )
    torch.testing.assert_close(model.scale, torch.tensor([2.0]))


def test_no_implicit_functorch_or_stochastic_model_fallback() -> None:
    class CustomParameter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.coefficient = nn.Parameter(torch.ones(2))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value * self.coefficient

    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="<root>: no registered"):
        validate_private_model(
            CustomParameter(), regression_loss, BatchAdapter(), inputs
        )
    with pytest.raises(ValueError, match="stochastic layer"):
        validate_private_model(
            nn.Sequential(nn.Linear(2, 2), nn.Dropout()),
            regression_loss,
            BatchAdapter(),
            inputs,
        )


def test_storage_aliases_must_be_one_parameter_object() -> None:
    from dromeus.training.private_model import describe_private_model

    class DistinctAliasedParameters(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(2, 1, bias=False)
            self.second = nn.Linear(2, 1, bias=False)
            self.second.weight = nn.Parameter(self.first.weight.detach())

    with pytest.raises(ValueError, match="one Parameter object"):
        describe_private_model(DistinctAliasedParameters())


def test_backend_rejects_unhooked_use_of_a_shared_parameter() -> None:
    from dromeus.training.privacy import opacus_module
    from dromeus.training.private_model import ensure_grad_samples

    class PartlyHooked(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = nn.Linear(2, 1, bias=False)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.layer(inputs) + nn.functional.linear(inputs, self.layer.weight)

    model = PartlyHooked()
    wrapper = opacus_module("opacus.grad_sample").GradSampleModule(model)
    try:
        wrapper(torch.tensor([[1.0, 2.0], [3.0, 4.0]])).mean().backward()
        with pytest.raises(ValueError, match="aggregate"):
            ensure_grad_samples(model, 2)
    finally:
        wrapper.remove_hooks()


def test_undeclared_extra_serialized_tensor_is_rejected() -> None:
    from dromeus.training.private_model import describe_private_model

    class ExtraSerializedState(UserRegression):
        def get_extra_state(self) -> Any:
            return torch.tensor([42.0])

    with pytest.raises(ValueError, match="registered parameters and buffers"):
        describe_private_model(ExtraSerializedState())


def test_custom_serialization_cannot_substitute_registered_model_values() -> None:
    from dromeus.training.private_model import describe_private_model

    class RewrittenState(UserRegression):
        def state_dict(self, *args: Any, **kwargs: Any) -> Any:
            state = cast(dict[str, torch.Tensor], super().state_dict(*args, **kwargs))
            state["first.weight"] = state["first.weight"] + 1
            return state

    with pytest.raises(ValueError, match="serialized state differs"):
        describe_private_model(RewrittenState())


def test_nonpersistent_public_buffers_are_guarded_but_not_serialized() -> None:
    from dromeus.training.private_model import (
        ModelStateGuard,
        registered_model_state,
    )

    model = ModelWithPublicState()
    model.register_buffer("position_ids", torch.arange(3), persistent=False)
    report = validate_private_model(
        model, regression_loss, BatchAdapter(), torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    )
    assert "position_ids" in report.public_buffer_hashes
    registered = registered_model_state(model)
    assert "position_ids" not in registered
    assert set(registered) == set(model.state_dict())
    guard = ModelStateGuard(model, report)
    model.get_buffer("position_ids").add_(1)
    with pytest.raises(ValueError, match="public buffers"):
        guard.check()
