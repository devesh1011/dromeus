"""Regression where each node trains on its own NPZ file.

Write the shared draft once, on one machine:

    python -m examples.custom_training draft examples/custom-regression-draft.yaml

Then run every node, each with its own data file:

    DROMEUS_LOCAL_DATA=/path/to/local.npz python -m dromeus.node \\
        --config examples/node.yaml --task examples.custom_training:setup

The local NPZ holds `inputs` [N, 3] and `targets` [N, 1], both float32.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

import dromeus

MODEL_DEFINITION = "regression: Linear(3,7), Tanh, Linear(7,1); float32"
TASK_DEFINITION = "regression: three float32 features; scalar float32 target"
LOSS_DEFINITION = "Huber loss, delta 1, one value per record"


def build_model() -> nn.Module:
    return nn.Sequential(nn.Linear(3, 7), nn.Tanh(), nn.Linear(7, 1))


def huber_per_record(model: nn.Module, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
    inputs, targets = batch
    losses = nn.functional.huber_loss(model(inputs), targets, reduction="none")
    return losses.mean(dim=1)


def _public_holdout() -> tuple[torch.Tensor, torch.Tensor]:
    """A fixed synthetic evaluation set; public, so every node may use it."""
    inputs = np.random.default_rng(0).normal(size=(32, 3)).astype(np.float32)
    return torch.from_numpy(inputs), torch.from_numpy(inputs[:, :1] * 0.3)


def evaluate(model: nn.Module) -> dict[str, float]:
    inputs, targets = _public_holdout()
    with torch.no_grad():
        error = model(inputs) - targets
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(error.square().mean().sqrt()),
    }


def task() -> dromeus.TrainingTask:
    """The task every node builds identically."""
    return dromeus.TrainingTask(
        model_factory=build_model,
        batch_adapter=dromeus.TensorBatchAdapter(
            example=(torch.zeros(2, 3), torch.zeros(2, 1)),
            definition="inputs [B,3] float32, targets [B,1] float32",
        ),
        loss_per_record=huber_per_record,
        model_definition=MODEL_DEFINITION,
        task_definition=TASK_DEFINITION,
        loss_definition=LOSS_DEFINITION,
        implementation_hash=dromeus.code_hash(Path(__file__)),
        optimizer=dromeus.Optimizer.adam(learning_rate=0.005),
        batch_size=8,
        public_evaluate=evaluate,
    )


def local_data(path: Path) -> dromeus.LocalData:
    """This node's own records; they never leave the machine."""
    with np.load(path, allow_pickle=False) as data:
        inputs, targets = data["inputs"], data["targets"]
    if inputs.dtype != np.float32 or targets.dtype != np.float32:
        raise ValueError("inputs and targets must be float32")
    if inputs.ndim != 2 or inputs.shape[1] != 3 or targets.shape != (len(inputs), 1):
        raise ValueError("expected inputs [N,3] and targets [N,1]")
    return dromeus.LocalData(
        dataset=TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets)),
        data_identity=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def setup() -> tuple[dromeus.TrainingTask, dromeus.LocalData]:
    """Entry point for `python -m dromeus.node --task`."""
    return task(), local_data(Path(os.environ["DROMEUS_LOCAL_DATA"]))


def write_draft(path: Path) -> str:
    return dromeus.make_draft(
        task(),
        path,
        run_id="custom-regression",
        participants=4,
        rounds=20,
        local_steps=10,
        axl_version="replace-with-your-axl-version",
    )


if __name__ == "__main__":
    if sys.argv[1:2] != ["draft"] or len(sys.argv) != 3:
        raise SystemExit("usage: python -m examples.custom_training draft PATH")
    print(write_draft(Path(sys.argv[2])))
