"""Deterministic Adam classifier on the core callback trainer, for algorithm tests."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import nn

from dromeus.training.trainer import PyTorchTrainer

_INPUTS = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]], dtype=torch.float32)
_TARGETS = torch.tensor([0, 0, 1, 1], dtype=torch.long)
_BATCH_SIZE = 2
_ADAM_KEYS = ("step", "exp_avg", "exp_avg_sq")


class _AdamApplication:
    """Sequential batches; every value that affects later steps is checkpointed."""

    def __init__(self, model: nn.Linear) -> None:
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8
        )
        self.cursor = 0

    def step(self, model: nn.Module) -> float:
        indices = torch.arange(self.cursor, self.cursor + _BATCH_SIZE) % len(_INPUTS)
        self.cursor = (self.cursor + _BATCH_SIZE) % len(_INPUTS)
        self.optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(_INPUTS[indices]), _TARGETS[indices])
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        self.optimizer.step()  # pyright: ignore[reportUnknownMemberType]
        return float(loss.detach())

    def save_state(self) -> dict[str, np.ndarray]:
        state = {"cursor": np.array([self.cursor], dtype=np.int64)}
        for name, parameter in self.model.named_parameters():
            values = self.optimizer.state.get(parameter, {})
            for key in _ADAM_KEYS:
                if key in values:
                    state[f"{name}.{key}"] = (
                        cast(torch.Tensor, values[key]).detach().cpu().numpy().copy()
                    )
        return state

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        optimizer_state: dict[int, dict[str, torch.Tensor]] = {}
        for index, (name, _) in enumerate(self.model.named_parameters()):
            if f"{name}.step" in state:
                optimizer_state[index] = {
                    key: torch.tensor(state[f"{name}.{key}"]) for key in _ADAM_KEYS
                }
        payload = self.optimizer.state_dict()
        payload["state"] = optimizer_state
        self.optimizer.load_state_dict(payload)
        self.cursor = int(state["cursor"][0])


def build_adam_trainer(*, initial: tuple[float, float]) -> PyTorchTrainer:
    """Linear(1, 2) classifier whose two weights start at ``initial``."""
    model = nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor(initial, dtype=torch.float32).view(2, 1))
    application = _AdamApplication(model)
    return PyTorchTrainer(
        model=model,
        train_step=application.step,
        save_state=application.save_state,
        load_state=application.load_state,
        state_identity="test-noloco-linear-adam",
    )
