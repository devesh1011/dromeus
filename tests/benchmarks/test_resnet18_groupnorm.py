from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch import nn

from benchmarks.workloads.cifar10.dataset import create_initial_checkpoint
from benchmarks.workloads.cifar10.models import resolve_model
from benchmarks.workloads.cifar10.resnet18_groupnorm import (
    MODEL_DEFINITION,
    MODEL_DEFINITION_HASH,
    MODEL_ID,
    PARAMETER_COUNT,
    RAW_FP32_PARAMETER_BYTES,
    TENSOR_SCHEMA_HASH,
    ResNet18GroupNorm,
    build_model,
    floating_model_state,
    tensor_schema_for_model,
)

EXPECTED_DEFINITION = (
    "resnet18-groupnorm-cifar10-v1:"
    "torchvision=0fba2e84fe255a2fcd81bd0b10c74d7fca99a89f:"
    "input-channels=3:stem=conv3x3-64-stride1-padding1:no-maxpool:"
    "basic-blocks=2,2,2,2:channels=64,128,256,512:"
    "downsample=conv1x1:groupnorm=32-affine:gap:linear=10"
)
EXPECTED_HASH = "c5792864784d4c27d51a34e94e1457b2b4f4a3c38970c2d3340e7094be0be178"


def test_resnet18_groupnorm_definition_and_registry_are_frozen() -> None:
    recipe = resolve_model(MODEL_ID)

    assert MODEL_DEFINITION == EXPECTED_DEFINITION
    assert MODEL_DEFINITION_HASH == EXPECTED_HASH
    assert PARAMETER_COUNT == 11_173_962
    assert RAW_FP32_PARAMETER_BYTES == 44_695_848
    assert TENSOR_SCHEMA_HASH == (
        "2c0b3aacfde1526af9458f8804895d442ad32be5cddd982b1d27185c07480706"
    )
    assert recipe.model_id == MODEL_ID
    assert recipe.definition == MODEL_DEFINITION
    assert recipe.definition_hash == MODEL_DEFINITION_HASH
    assert recipe.parameter_count == PARAMETER_COUNT
    assert recipe.tensor_schema_hash == TENSOR_SCHEMA_HASH


def test_resnet18_groupnorm_has_exact_cifar_topology_and_size() -> None:
    model = build_model(seed=17)
    norms = [module for module in model.modules() if isinstance(module, nn.GroupNorm)]
    convolutions = [
        module for module in model.modules() if isinstance(module, nn.Conv2d)
    ]
    parameters = sum(parameter.numel() for parameter in model.parameters())
    state = floating_model_state(model)
    raw_bytes = sum(value.numel() * value.element_size() for value in state.values())

    assert isinstance(model, ResNet18GroupNorm)
    assert parameters == 11_173_962
    assert raw_bytes == 44_695_848
    assert len(norms) == 20
    assert all(norm.num_groups == 32 and norm.affine for norm in norms)
    assert all(convolution.bias is None for convolution in convolutions)
    assert set(state) == {name for name, _ in model.named_parameters()}
    assert not any("running_" in name for name in state)
    schema = tensor_schema_for_model(model)
    schema_elements = sum(math.prod(tensor.shape) for tensor in schema.tensors)
    assert schema_elements * 4 == 44_695_848


def test_resnet18_groupnorm_forward_and_seed_are_deterministic() -> None:
    torch.manual_seed(99)  # pyright: ignore[reportUnknownMemberType]
    expected_next_random = torch.rand(1)
    torch.manual_seed(99)  # pyright: ignore[reportUnknownMemberType]

    first = build_model(seed=17)
    second = build_model(seed=17)

    assert torch.equal(torch.rand(1), expected_next_random)
    assert all(
        torch.equal(value, second.state_dict()[name])
        for name, value in first.state_dict().items()
    )
    assert first(torch.zeros(2, 3, 32, 32)).shape == (2, 10)


def test_model_registry_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="unsupported training model"):
        resolve_model("unknown-model")


def test_cifar_checkpoint_selects_resnet18_recipe(tmp_path: Path) -> None:
    checkpoint = create_initial_checkpoint(
        tmp_path / "resnet18.safetensors",
        seed=17,
        model_id=MODEL_ID,
    )

    assert checkpoint.sha256
    assert checkpoint.tensor_schema == tensor_schema_for_model(build_model(seed=17))
    assert checkpoint.path.stat().st_size > 44_695_848
