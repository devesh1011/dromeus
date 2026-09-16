"""Verify the reviewed final-model capture wrapper without training."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path


EXPECTED_SHA256 = (
    "8cceac7f09a262ce0144ec3033bac2b2c7af757cd593b9276f345f06065d4524"
)


def main() -> int:
    path = Path(sys.argv[1])
    source = path.read_bytes()
    actual_sha256 = hashlib.sha256(source).hexdigest()
    if actual_sha256 != EXPECTED_SHA256:
        raise ValueError("capture wrapper hash differs from reviewed bytes")
    tree = ast.parse(source.decode("utf-8"), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_evaluate_and_capture"
    )
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    save_lines = [
        node.lineno
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "save_checkpoint"
    ]
    evaluate_lines = [
        node.lineno
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "_original_evaluate"
    ]
    if not save_lines or not evaluate_lines or min(save_lines) >= min(evaluate_lines):
        raise ValueError("capture must save model weights before final evaluation")
    text = source.decode("utf-8")
    for required in ("NCCL_FINAL_CHECKPOINT_DIR", "RANK", "save_checkpoint"):
        if required not in text:
            raise ValueError(f"capture wrapper is missing {required}")
    print(
        json.dumps(
            {
                "status": "verified",
                "wrapper_sha256": actual_sha256,
                "capture_phase": "immediately-before-frozen-final-evaluation",
                "model_only": True,
                "optimizer_state_captured": False,
                "training_executed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
