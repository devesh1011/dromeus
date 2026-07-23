"""Update codecs and their serializable local state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

TensorMap = dict[str, np.ndarray]
StateMap = Mapping[str, object]


class UpdateCodec(Protocol):
    """Encode/decode an algorithm update without owning transport concerns."""

    @property
    def codec_id(self) -> str: ...

    def encode(self, tensors: Mapping[str, np.ndarray]) -> TensorMap: ...

    def decode(self, tensors: Mapping[str, np.ndarray]) -> TensorMap: ...

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: StateMap) -> None: ...


@dataclass(frozen=True, slots=True)
class IdentityCodec:
    """M1 codec: preserve named tensors and keep no codec state."""

    _codec_id: str = "safetensors-v1"

    @property
    def codec_id(self) -> str:
        return self._codec_id

    def encode(self, tensors: Mapping[str, np.ndarray]) -> TensorMap:
        return _copy_tensors(tensors)

    def decode(self, tensors: Mapping[str, np.ndarray]) -> TensorMap:
        return _copy_tensors(tensors)

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: StateMap) -> None:
        if state:
            raise ValueError("identity codec has no state")


def _copy_tensors(tensors: Mapping[str, np.ndarray]) -> TensorMap:
    return {
        name: np.ascontiguousarray(value).copy() for name, value in tensors.items()
    }


__all__ = ["IdentityCodec", "StateMap", "TensorMap", "UpdateCodec"]
