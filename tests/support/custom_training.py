"""Custom regression over each node's private NPZ data.

DROMEUS_LOCAL_DATA=/path/to/local.npz uv run python -m dromeus.node \
    --config node.yaml --factory examples.custom_training:prepare_training

The local NPZ contains inputs [N, 3] and targets [N, 1], both FP32. Optional
independent held-out arrays are evaluation_inputs and evaluation_targets. No
shared dataset, image library, model registry, or classification loss is used.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import nn

from dromeus.application import PreparedApplication, prepare_application
from dromeus.manifests.canonical import file_sha256
from dromeus.manifests.models import DraftRunSpec
from dromeus.training.trainer import PyTorchTrainer

MODEL_DEFINITION = "regression-v1: Linear(3,7),Tanh,Linear(7,1); FP32"
TASK_DEFINITION = (
    "regression-v1: three FP32 features; scalar FP32 target; no normalization"
)
TRAINING_DEFINITION = (
    "regression-v1: Huber(delta=1); RMSprop(lr=.005,alpha=.99,eps=1e-8,"
    "momentum=0,centered=False,weight_decay=0); cyclic batches<=8"
)


class LocalRegression:
    """Example application; all model/data/optimizer decisions live here."""

    def __init__(self, path: Path) -> None:
        self.data_identity = file_sha256(path)
        with np.load(path, allow_pickle=False) as data:
            self.inputs, self.targets = self._arrays(data["inputs"], data["targets"])
            if "evaluation_inputs" in data and "evaluation_targets" in data:
                self.evaluation_data = self._arrays(
                    data["evaluation_inputs"], data["evaluation_targets"]
                )
            elif "evaluation_inputs" in data or "evaluation_targets" in data:
                raise ValueError("provide both held-out inputs and targets")
            else:
                self.evaluation_data = None
        # Check the input did not change while being loaded.
        if file_sha256(path) != self.data_identity:
            raise ValueError("local data changed during preparation")
        self.model = nn.Sequential(nn.Linear(3, 7), nn.Tanh(), nn.Linear(7, 1)).float()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.005)
        self.cursor = 0
        self.trainer = PyTorchTrainer(
            model=self.model,
            train_step=self.step,
            save_state=self.save_state,
            load_state=self.load_state,
            state_identity=";".join(
                (
                    MODEL_DEFINITION,
                    TASK_DEFINITION,
                    TRAINING_DEFINITION,
                    self.data_identity,
                )
            ),
            evaluate=self.evaluate if self.evaluation_data is not None else None,
        )

    @staticmethod
    def _arrays(
        inputs: np.ndarray, targets: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.dtype != np.float32 or targets.dtype != np.float32:
            raise ValueError("regression inputs and targets must be FP32")
        if (
            inputs.ndim != 2
            or inputs.shape[1] != 3
            or targets.shape != (len(inputs), 1)
            or not len(inputs)
        ):
            raise ValueError("expected nonempty inputs [N,3] and targets [N,1]")
        if not np.isfinite(inputs).all() or not np.isfinite(targets).all():
            raise ValueError("local regression data must be finite")
        return torch.tensor(inputs), torch.tensor(targets)

    def step(self, model: nn.Module) -> float:
        model.train()
        count = min(8, len(self.inputs))
        indices = (torch.arange(count) + self.cursor) % len(self.inputs)
        self.cursor = (self.cursor + count) % len(self.inputs)
        self.optimizer.zero_grad(set_to_none=True)
        prediction = model(self.inputs[indices])
        loss = torch.nn.functional.huber_loss(prediction, self.targets[indices])
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        self.optimizer.step()  # pyright: ignore[reportUnknownMemberType]
        return float(loss.detach())

    def evaluate(self, model: nn.Module) -> dict[str, float]:
        assert self.evaluation_data is not None
        inputs, targets = self.evaluation_data
        error = model(inputs) - targets
        return {
            "mae": float(error.abs().mean()),
            "rmse": float(error.square().mean().sqrt()),
        }

    def save_state(self) -> dict[str, np.ndarray]:
        """RMSprop-specific state is owned by this example, not Dromeus."""
        state = {"cursor": np.array([self.cursor], dtype=np.int64)}
        for name, parameter in self.model.named_parameters():
            values = self.optimizer.state.get(parameter, {})
            for key in ("step", "square_avg"):
                if key in values:
                    state[f"{name}.{key}"] = (
                        cast(torch.Tensor, values[key]).detach().cpu().numpy().copy()
                    )
        return state

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        cursor = state.get("cursor")
        if (
            cursor is None
            or cursor.dtype != np.int64
            or cursor.shape != (1,)
            or not 0 <= cursor[0] < len(self.inputs)
        ):
            raise ValueError("invalid local batch cursor")
        optimizer_state: dict[int, dict[str, torch.Tensor]] = {}
        expected = {"cursor"}
        for index, (name, parameter) in enumerate(self.model.named_parameters()):
            keys = {f"{name}.step", f"{name}.square_avg"}
            if keys & state.keys():
                if not keys <= state.keys():
                    raise ValueError("incomplete RMSprop checkpoint")
                step, average = state[f"{name}.step"], state[f"{name}.square_avg"]
                if (
                    step.shape != ()
                    or step.dtype != np.float32
                    or step < 0
                    or average.shape != tuple(parameter.shape)
                    or average.dtype != np.float32
                    or (average < 0).any()
                ):
                    raise ValueError("invalid RMSprop tensor state")
                optimizer_state[index] = {
                    "step": torch.tensor(step),
                    "square_avg": torch.tensor(average),
                }
                expected.update(keys)
        if state.keys() != expected:
            raise ValueError("unexpected application checkpoint fields")
        payload = self.optimizer.state_dict()
        payload["state"] = optimizer_state
        self.optimizer.load_state_dict(payload)
        self.cursor = int(cursor[0])


def prepare_training(draft: DraftRunSpec) -> PreparedApplication:
    local = LocalRegression(Path(os.environ["DROMEUS_LOCAL_DATA"]))
    return prepare_application(
        draft=draft,
        trainer=local.trainer,
        model_definition=MODEL_DEFINITION,
        task_definition=TASK_DEFINITION,
        training_definition=TRAINING_DEFINITION,
    )
