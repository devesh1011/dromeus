"""Operational wrapper that preserves the reference runner's final model state."""

from __future__ import annotations

import os
from pathlib import Path

from benchmarks.noloco_reference.runner import main
from dromeus.training.trainer import PyTorchTrainer


_original_evaluate = PyTorchTrainer.evaluate


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

raise SystemExit(main())
