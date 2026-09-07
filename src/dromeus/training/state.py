"""Model-state initialization and deterministic experiment seed helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from safetensors.torch import (
    save_file as _save_file,  # pyright: ignore[reportUnknownVariableType]
)
from torch import nn

from dromeus.manifests.canonical import file_sha256
from dromeus.manifests.models import TensorSchema
from dromeus.training.model_state import floating_model_state, tensor_schema_for_model

_save_checkpoint = cast(Callable[..., None], _save_file)


@dataclass(frozen=True, slots=True)
class InitialCheckpoint:
    """Canonical checkpoint handoff data for initiator formation."""

    path: Path
    tensor_schema: TensorSchema
    sha256: str


def derive_benchmark_seed(benchmark_seed: int, purpose: str) -> int:
    """Derive one stable RNG seed for a named benchmark concern."""
    digest = hashlib.sha256(f"{benchmark_seed}:{purpose}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def create_initial_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    model_definition: str,
) -> InitialCheckpoint:
    """Write and describe the canonical checkpoint before formation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_checkpoint(
        {
            name: value.detach().cpu().contiguous().clone()
            for name, value in floating_model_state(model).items()
        },
        str(path),
        metadata={"model_definition": model_definition},
    )
    return InitialCheckpoint(
        path=path,
        tensor_schema=tensor_schema_for_model(model),
        sha256=checkpoint_hash(path),
    )


def checkpoint_hash(path: Path) -> str:
    """Return a checkpoint's SHA-256 digest."""
    return file_sha256(path)
