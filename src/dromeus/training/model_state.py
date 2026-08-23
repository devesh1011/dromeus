"""Shared exchangeable model-state and tensor-schema utilities."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from dromeus.manifests.models import Tensor as TensorSpec
from dromeus.manifests.models import TensorSchema

_TORCH_TO_SCHEMA_DTYPE: dict[
    torch.dtype,
    Literal["float16", "float32", "float64"],
] = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
}


def floating_model_state(model: nn.Module) -> dict[str, Tensor]:
    """Return exchangeable floating parameters and model buffers."""
    return {
        name: value
        for name, value in model.state_dict().items()
        if value.is_floating_point()
    }


def tensor_schema_for_model(model: nn.Module) -> TensorSchema:
    """Derive the canonical wire schema from exchangeable model state."""
    tensors: list[TensorSpec] = []
    for name, value in floating_model_state(model).items():
        dtype = _TORCH_TO_SCHEMA_DTYPE.get(value.dtype)
        if dtype is None:
            raise ValueError(f"unsupported model state dtype: {value.dtype}")
        tensors.append(
            TensorSpec(
                name=name,
                dtype=dtype,
                shape=tuple(value.shape),
            )
        )
    return TensorSchema(tensors=tuple(tensors))


__all__ = ["floating_model_state", "tensor_schema_for_model"]
