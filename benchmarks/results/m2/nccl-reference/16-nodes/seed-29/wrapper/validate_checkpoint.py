"""Validate a captured final model-weight checkpoint against the frozen schema."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


EXPECTED_TENSOR_SCHEMA_HASH = (
    "2c0b3aacfde1526af9458f8804895d442ad32be5cddd982b1d27185c07480706"
)
EXPECTED_SIZE_BYTES = 44_701_664


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    initial_path = Path(sys.argv[1])
    final_path = Path(sys.argv[2])
    initial = load_file(str(initial_path))
    final = load_file(str(final_path))
    if set(initial) != set(final):
        raise ValueError("captured tensor names differ from frozen model schema")
    if len(final) != 62:
        raise ValueError(f"expected 62 model tensors, got {len(final)}")
    for name in sorted(final):
        initial_value = initial[name]
        value = final[name]
        if value.dtype != np.dtype("float32") or value.shape != initial_value.shape:
            raise ValueError(f"tensor schema mismatch: {name}")
        if not np.isfinite(value).all():
            raise ValueError(f"non-finite tensor: {name}")
    if final_path.stat().st_size != EXPECTED_SIZE_BYTES:
        raise ValueError("captured checkpoint byte size differs from frozen model")
    print(
        json.dumps(
            {
                "status": "verified",
                "tensor_count": len(final),
                "tensor_schema_hash": EXPECTED_TENSOR_SCHEMA_HASH,
                "size_bytes": final_path.stat().st_size,
                "sha256": sha256(final_path),
                "model_only": True,
                "optimizer_state_captured": False,
                "reload": "safetensors.numpy.load_file",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
