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


type _StateLayout = tuple[int, int, torch.dtype, tuple[int, ...], tuple[int, ...]]


def model_state_alias_groups(model: nn.Module) -> tuple[tuple[str, ...], ...]:
    """Identify exact aliases and reject unsupported overlapping write layouts.

    Byte spans are conservative for strided views: ambiguous overlaps reject.
    Ordinary transposes, exact ties, and disjoint spans remain supported.
    """
    devices: dict[str, list[tuple[_StateLayout, str]]] = {}
    for name, value in model.state_dict().items():
        if value.numel() == 0:
            continue
        shape, strides = tuple(value.shape), tuple(value.stride())
        span = 1
        for stride, size in sorted(zip(strides, shape, strict=True)):
            if size <= 1:
                continue
            if stride < span:
                raise ValueError(f"unsupported overlapping model state: {name}")
            span += (size - 1) * stride
        start = (
            value.untyped_storage().data_ptr()
            + value.storage_offset() * value.element_size()
        )
        end = start + span * value.element_size()
        layout = (start, end, value.dtype, shape, strides)
        devices.setdefault(str(value.device), []).append((layout, name))
    aliases: list[tuple[str, ...]] = []
    for values in devices.values():
        previous: _StateLayout | None = None
        group: list[str] = []
        for layout, name in sorted(
            values, key=lambda item: (item[0][0], item[0][1], item[1])
        ):
            if previous is not None and layout[0] < previous[1]:
                if layout != previous:
                    raise ValueError(
                        f"unsupported overlapping model state: {group[0]} and {name}"
                    )
                group.append(name)
            else:
                if len(group) > 1:
                    aliases.append(tuple(sorted(group)))
                previous, group = layout, [name]
        if len(group) > 1:
            aliases.append(tuple(sorted(group)))
    return tuple(sorted(aliases))


__all__ = ["floating_model_state", "tensor_schema_for_model"]
