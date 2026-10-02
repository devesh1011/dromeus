"""Bounded numeric-array encoding for owned research state, without pickle."""

from __future__ import annotations

import json
from typing import Any, cast

import numpy as np
import torch


def encode_state(value: Any) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}

    def encode(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            key = f"tensor.{len(arrays)}"
            arrays[key] = item.detach().cpu().numpy().copy()
            return {"tensor": key}
        if isinstance(item, dict):
            return {
                "dict": [
                    [encode(k), encode(v)]
                    for k, v in cast(dict[Any, Any], item).items()
                ]
            }
        if isinstance(item, tuple):
            return {"tuple": [encode(x) for x in cast(list[Any], item)]}
        if isinstance(item, list):
            return {"list": [encode(x) for x in cast(list[Any], item)]}
        if item is None or type(item) in (str, bool, int, float):
            return item
        raise ValueError("unsupported research state value")

    payload = json.dumps(encode(value), allow_nan=False, separators=(",", ":")).encode()
    if len(payload) > 1024 * 1024:
        raise ValueError("research state metadata exceeds bound")
    arrays["structure"] = np.frombuffer(payload, dtype=np.uint8).copy()
    return arrays


def decode_state(arrays: dict[str, np.ndarray]) -> Any:
    metadata = arrays.get("structure")
    if (
        metadata is None
        or metadata.dtype != np.uint8
        or metadata.ndim != 1
        or metadata.size > 1024 * 1024
    ):
        raise ValueError("invalid research state structure")
    used = {"structure"}

    def decode(item: Any, depth: int = 0) -> Any:
        if depth > 32:
            raise ValueError("research state nesting exceeds bound")
        if isinstance(item, dict):
            if len(cast(dict[Any, Any], item)) != 1:
                raise ValueError("invalid research state tag")
            tag, values = next(iter(cast(dict[Any, Any], item).items()))
            if tag == "tensor":
                if values not in arrays or values in used:
                    raise ValueError("invalid research state tensor")
                used.add(values)
                value = arrays[values]
                if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                    raise ValueError("invalid research state array")
                return torch.from_numpy(value.copy())  # pyright: ignore[reportUnknownMemberType]
            if tag == "dict":
                return {decode(k, depth + 1): decode(v, depth + 1) for k, v in values}
            if tag in ("tuple", "list"):
                result = [decode(x, depth + 1) for x in values]
                return tuple(result) if tag == "tuple" else result
            raise ValueError("invalid research state tag")
        if item is None or type(item) in (str, bool, int, float):
            return item
        raise ValueError("invalid research state value")

    try:
        result = decode(json.loads(metadata.tobytes()))
    except Exception:
        raise ValueError("invalid research checkpoint encoding") from None
    if set(arrays) != used:
        raise ValueError("unexpected research checkpoint arrays")
    return result
