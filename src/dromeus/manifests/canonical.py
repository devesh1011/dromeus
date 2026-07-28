"""Canonical JSON encoding, hashing, and draft YAML parsing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from uuid import uuid4

import numpy as np
import yaml
from pydantic import BaseModel
from safetensors.numpy import (
    load_file as _load_file,  # pyright: ignore[reportUnknownVariableType]
)
from safetensors.numpy import (
    save_file as _save_file,  # pyright: ignore[reportUnknownVariableType]
)

from dromeus.manifests.models import (
    DraftRunSpec,
    OpaqueUpdateBundleMetadata,
    SealedManifest,
    Tensor,
    TensorSchema,
)

_LoadSafetensors = Callable[[str], dict[str, np.ndarray]]
_SaveSafetensors = Callable[[dict[str, np.ndarray], str], None]
load_safetensors = cast(_LoadSafetensors, _load_file)
save_safetensors = cast(_SaveSafetensors, _save_file)


def canonical_json(model: BaseModel) -> bytes:
    """Encode a validated model as deterministic UTF-8 JSON."""
    value = model.model_dump(mode="json", exclude_none=True)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def canonical_hash(model: BaseModel) -> str:
    return hashlib.sha256(canonical_json(model)).hexdigest()


def update_bundle_digest(metadata: OpaqueUpdateBundleMetadata) -> str:
    """Hash bundle identity and canonically ordered artifact descriptors."""
    ordered = tuple(sorted(metadata.artifacts, key=lambda artifact: artifact.name))
    return canonical_hash(metadata.model_copy(update={"artifacts": ordered}))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class EncodedBundleMetadata:
    path: Path
    tensor_schema: TensorSchema


def materialize_bundle_metadata(
    metadata: OpaqueUpdateBundleMetadata, root: Path
) -> EncodedBundleMetadata:
    """Atomically encode opaque metadata into a v1-compatible artifact."""
    root.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(metadata)
    values = np.frombuffer(payload, dtype=np.uint8).view(np.int8).copy()
    identifier = uuid4().hex
    temp = root / f".{identifier}.tmp"
    final = root / f"{identifier}.safetensors"
    try:
        save_safetensors({"metadata": values}, str(temp))
        temp.replace(final)
    except BaseException:
        temp.unlink(missing_ok=True)
        final.unlink(missing_ok=True)
        raise
    return EncodedBundleMetadata(
        path=final,
        tensor_schema=TensorSchema(
            tensors=(Tensor(name="metadata", dtype="int8", shape=values.shape),)
        ),
    )


def parse_bundle_metadata(path: Path) -> OpaqueUpdateBundleMetadata:
    """Decode and validate current bundle metadata from its carrier artifact."""
    values = load_safetensors(str(path))
    if set(values) != {"metadata"} or values["metadata"].dtype != np.dtype(np.int8):
        raise ValueError("invalid update bundle metadata artifact")
    payload = np.ascontiguousarray(values["metadata"]).view(np.uint8).tobytes()
    return OpaqueUpdateBundleMetadata.model_validate_json(payload)


def parse_draft_yaml(source: str | bytes | Path) -> DraftRunSpec:
    """Parse and validate a draft from YAML text or a file path."""
    text = source.read_text() if isinstance(source, Path) else source
    value = cast(object, yaml.safe_load(text))
    return DraftRunSpec.model_validate(value)


def parse_sealed_json(data: str | bytes) -> SealedManifest:
    return SealedManifest.model_validate_json(data)
