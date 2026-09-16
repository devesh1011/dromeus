"""Operational wrapper that preserves the reference runner's final model state."""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from pathlib import Path

import benchmarks.noloco_reference.runner as reference
from benchmarks.noloco_reference.runner import main
from dromeus.training.trainer import PyTorchTrainer


_original_evaluate = PyTorchTrainer.evaluate
_original_outer_step = reference.apply_outer_step
_completed_outer_steps = 0


def _outer_step_with_progress(*args: object, **kwargs: object) -> object:
    """Report completed outer updates without changing inputs or results."""
    global _completed_outer_steps
    result = _original_outer_step(*args, **kwargs)
    _completed_outer_steps += 1
    if _completed_outer_steps == 1 or _completed_outer_steps % 25 == 0:
        print(json.dumps({"event": "outer_step_completed", "rank": int(os.environ["RANK"]), "completed_outer_steps": _completed_outer_steps, "timestamp": datetime.now(timezone.utc).isoformat()}), flush=True)
    return result


def _evaluate_and_capture(
    self: PyTorchTrainer,
    *args: object,
    **kwargs: object,
) -> tuple[float, float]:
    output_root = Path(os.environ["NCCL_FINAL_CHECKPOINT_DIR"])
    rank = os.environ["RANK"]
    self.save_checkpoint(output_root / f"rank-{rank}.safetensors")
    return _original_evaluate(self, *args, **kwargs)


PyTorchTrainer.evaluate = _evaluate_and_capture
reference.apply_outer_step = _outer_step_with_progress

raise SystemExit(main())
