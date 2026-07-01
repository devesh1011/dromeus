"""Canonical JSON encoding, hashing, and draft YAML parsing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import yaml
from pydantic import BaseModel

from dromeus.manifests.models import DraftRunSpec, SealedManifest


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


def parse_draft_yaml(source: str | bytes | Path) -> DraftRunSpec:
    """Parse and validate a draft from YAML text or a file path."""
    text = source.read_text() if isinstance(source, Path) else source
    value = cast(object, yaml.safe_load(text))
    return DraftRunSpec.model_validate(value)


def parse_sealed_json(data: str | bytes) -> SealedManifest:
    return SealedManifest.model_validate_json(data)
