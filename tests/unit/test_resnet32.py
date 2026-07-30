from __future__ import annotations

import hashlib

import torch

from dromeus.training.resnet32 import (
    MODEL_DEFINITION,
    MODEL_DEFINITION_HASH,
    MODEL_ID,
    ResNet32,
    build_model,
    model_definition,
    tensor_schema_for_model,
)


def test_resnet32_is_generic_and_reproducible() -> None:
    first = build_model(seed=17, input_channels=1, num_classes=7)
    second = build_model(seed=17, input_channels=1, num_classes=7)

    assert isinstance(first, ResNet32)
    assert MODEL_ID == "resnet32"
    assert MODEL_DEFINITION_HASH == hashlib.sha256(
        MODEL_DEFINITION.encode()
    ).hexdigest()
    assert sum(parameter.numel() for parameter in first.parameters()) >= 450_000
    assert torch.equal(
        first(torch.zeros(2, 1, 32, 32)),
        second(torch.zeros(2, 1, 32, 32)),
    )
    assert first(torch.zeros(2, 1, 32, 32)).shape == (2, 7)
    assert model_definition(input_channels=1, num_classes=7) != MODEL_DEFINITION


def test_resnet32_schema_includes_floating_batchnorm_state() -> None:
    schema = tensor_schema_for_model(build_model(seed=17))
    names = {tensor.name for tensor in schema.tensors}

    assert "stages.0.0.bn1.running_mean" in names
    assert "stages.0.0.bn1.num_batches_tracked" not in names
