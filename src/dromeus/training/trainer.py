"""Adapt application-owned PyTorch training to Dromeus tensor exchange.

Callbacks own batching, optimizer, loss, scheduling, and random generators. State
callbacks must save/restore all of that state as numeric arrays, including a
loader position and RNG states when relevant. They never receive peer data.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import nn

from dromeus.manifests.canonical import load_safetensors
from dromeus.manifests.models import TensorSchema
from dromeus.training.base import EvaluationResult
from dromeus.training.model_state import (
    floating_model_state,
    model_state_alias_groups,
    tensor_schema_for_model,
)
from dromeus.training.state import InitialCheckpoint, create_initial_checkpoint

_from_numpy = cast(Callable[[np.ndarray], torch.Tensor], torch.from_numpy)  # pyright: ignore[reportUnknownMemberType]

TrainStep = Callable[[nn.Module], float | None]
Evaluate = Callable[[nn.Module], Mapping[str, float]]
SaveState = Callable[[], dict[str, np.ndarray]]
LoadState = Callable[[dict[str, np.ndarray]], None]


def _copy_state(state: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, value in state.items():
        if not name or value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError("checkpoint state needs named finite numeric arrays")
        result[name] = value.copy(order="C")
    return result


class PyTorchTrainer:
    """A custom local step plus complete application state, with no model registry.

    Exchange currently supports non-scalar FP32 floating model state. Integer
    buffers stay local and are checkpointed. Callbacks must not replace the model
    or mutate its schema. Each instance is used serially by one node's runtime.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        train_step: TrainStep,
        save_state: SaveState,
        load_state: LoadState,
        state_identity: str,
        evaluate: Evaluate | None = None,
    ) -> None:
        if not state_identity.strip():
            raise ValueError(
                "state identity must bind local data and training configuration"
            )
        self._model = model
        self._train_step = train_step
        self._evaluate = evaluate
        self._save_state = save_state
        self._load_state = load_state
        self._identity = np.frombuffer(
            hashlib.sha256(state_identity.encode()).digest(), dtype=np.uint8
        ).copy()
        self._contract_identity: str | None = None
        values = floating_model_state(model)
        if not values:
            raise ValueError("model must expose floating exchangeable state")
        if any(
            value.dtype != torch.float32 or value.ndim == 0 for value in values.values()
        ):
            raise ValueError(
                "exchange requires non-scalar FP32 state; use shape [1] for scalars"
            )
        self._schema = tensor_schema_for_model(model)
        self._aliases = model_state_alias_groups(model)
        self._steps = 0
        self._local_loss: float | None = None
        # Reject unusable state before the node starts AXL.
        self.checkpoint_tensors()

    @property
    def tensor_schema(self) -> TensorSchema:
        return self._schema

    @property
    def completed_steps(self) -> int:
        return self._steps

    def bind_contract(self, identity: str) -> None:
        """Bind durable state to shared application semantics before formation."""
        if self._contract_identity is not None:
            if identity != self._contract_identity:
                raise ValueError("trainer is already bound to another application")
            return
        if self._steps:
            raise ValueError("bind the application contract before training")
        self._identity = np.frombuffer(
            hashlib.sha256(self._identity.tobytes() + identity.encode()).digest(),
            dtype=np.uint8,
        ).copy()
        self._contract_identity = identity

    @property
    def local_loss(self) -> float | None:
        return self._local_loss

    def weights(self) -> dict[str, np.ndarray]:
        self._validate_alias_layout()
        if tensor_schema_for_model(self._model) != self._schema:
            raise ValueError("application changed its model tensor schema")
        return _copy_state(
            {
                name: value.detach().cpu().numpy()
                for name, value in floating_model_state(self._model).items()
            }
        )

    def load_weights(self, weights: dict[str, np.ndarray]) -> None:
        current = self.weights()
        candidate = self._validate_like(weights, current)
        self._validate_alias_values(candidate)
        with torch.no_grad():
            for name, target in floating_model_state(self._model).items():
                target.copy_(_from_numpy(candidate[name]).to(target.device))

    def train_local_steps(self, step_count: int) -> None:
        if step_count <= 0:
            raise ValueError("local step count must be positive")
        for _ in range(step_count):
            loss = self._train_step(self._model)
            if loss is not None and not math.isfinite(loss):
                raise ValueError("application returned non-finite local loss")
            self._local_loss = None if loss is None else float(loss)
            self._steps += 1
        self.weights()  # Detect non-finite or structurally changed model state.

    def evaluate(self) -> EvaluationResult | None:
        if self._evaluate is None:
            return None
        modes = [(module, module.training) for module in self._model.modules()]
        try:
            self._model.eval()
            with torch.no_grad():
                return EvaluationResult(self._evaluate(self._model))
        finally:
            for module, training in modes:
                module.training = training

    def create_initial_checkpoint(
        self, path: Path, *, model_definition: str
    ) -> InitialCheckpoint:
        self.weights()
        return create_initial_checkpoint(
            path, model=self._model, model_definition=model_definition
        )

    def load_checkpoint(self, path: Path) -> None:
        """Load group initial weights, retaining local application state."""
        self.load_weights(load_safetensors(str(path)))

    def checkpoint_tensors(self) -> dict[str, np.ndarray]:
        self._validate_alias_layout()
        state = {
            f"model.{name}": value.detach().cpu().numpy().copy()
            for name, value in self._model.state_dict().items()
        }
        state.update(
            {
                f"application.{name}": value
                for name, value in _copy_state(self._save_state()).items()
            }
        )
        state.update(
            {
                "meta.version": np.array([1], dtype=np.int64),
                "meta.identity": self._identity.copy(),
                "meta.steps": np.array([self._steps], dtype=np.int64),
                "meta.loss": np.array(
                    [] if self._local_loss is None else [self._local_loss],
                    dtype=np.float64,
                ),
                "meta.modes": np.array(
                    [module.training for module in self._model.modules()],
                    dtype=np.bool_,
                ),
            }
        )
        return _copy_state(state)

    def load_checkpoint_tensors(self, state: dict[str, np.ndarray]) -> None:
        """Validate model/meta before restoring callbacks; roll back on failure.

        Application load_state must also accept its own save_state output. If an
        application violates that contract, rollback failure is fatal to the run.
        """
        candidate = _copy_state(state)
        old = self.checkpoint_tensors()
        model_keys = {name for name in old if name.startswith("model.")}
        meta_keys = {name for name in old if name.startswith("meta.")}
        if {
            name for name in candidate if not name.startswith("application.")
        } != model_keys | meta_keys:
            raise ValueError("checkpoint model or metadata keys mismatch")
        self._validate_like(
            {k: candidate[k] for k in model_keys}, {k: old[k] for k in model_keys}
        )
        self._validate_alias_values(
            {name.removeprefix("model."): candidate[name] for name in model_keys}
        )
        for key in meta_keys - {"meta.loss"}:
            self._validate_like({key: candidate[key]}, {key: old[key]})
        if not np.array_equal(
            candidate["meta.version"], old["meta.version"]
        ) or not np.array_equal(candidate["meta.identity"], self._identity):
            raise ValueError("checkpoint version or local state identity mismatch")
        if candidate["meta.steps"][0] < 0:
            raise ValueError("checkpoint step count must be non-negative")
        loss = candidate["meta.loss"]
        if loss.dtype != np.float64 or loss.shape not in ((0,), (1,)):
            raise ValueError("invalid checkpoint loss")
        try:
            self._restore(candidate)
        except Exception:
            self._restore(old)
            raise

    def _restore(self, state: dict[str, np.ndarray]) -> None:
        self._load_state(
            {
                name.removeprefix("application."): value.copy()
                for name, value in state.items()
                if name.startswith("application.")
            }
        )
        with torch.no_grad():
            for name, value in self._model.state_dict().items():
                value.copy_(_from_numpy(state[f"model.{name}"]).to(value.device))
        for module, mode in zip(
            self._model.modules(), state["meta.modes"], strict=True
        ):
            module.training = bool(mode)
        self._steps = int(state["meta.steps"][0])
        loss = state["meta.loss"]
        self._local_loss = float(loss[0]) if loss.size else None

    def _validate_alias_layout(self) -> None:
        if model_state_alias_groups(self._model) != self._aliases:
            raise ValueError("application changed its model alias layout")

    def _validate_alias_values(self, values: Mapping[str, np.ndarray]) -> None:
        for group in self._aliases:
            if (
                group[0] not in values
            ):  # Integer buffers do not participate in exchange.
                continue
            if any(
                not np.array_equal(values[group[0]], values[name]) for name in group[1:]
            ):
                raise ValueError(f"conflicting values for tied model aliases: {group}")

    @staticmethod
    def _validate_like(
        state: Mapping[str, np.ndarray], expected: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        if state.keys() != expected.keys():
            raise ValueError("model state keys mismatch")
        candidate = _copy_state(state)
        if any(
            candidate[name].shape != value.shape or candidate[name].dtype != value.dtype
            for name, value in expected.items()
        ):
            raise ValueError("model state shape or dtype mismatch")
        return candidate
