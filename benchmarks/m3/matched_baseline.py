"""Ordinary training matched to the owned DP adapter's Poisson semantics.

This public-data benchmark has no clipping, noise, accounting or privacy claim.
The application supplies its optimizer factory and a shared semantic definition.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from typing import Any, Literal, cast

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dromeus.application import (
    PreparedApplication,
    definition_hash,
    prepare_application,
)
from dromeus.manifests.canonical import canonical_json
from dromeus.manifests.models import DraftRunSpec
from dromeus.training.privacy import opacus_module
from dromeus.training.private_state import decode_state, encode_state
from dromeus.training.trainer import PyTorchTrainer


class BaselineRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    batch_size: int = Field(gt=0)
    logical_steps: int = Field(gt=0)
    loss: Literal["mse", "cross_entropy"]
    schedule: Literal["constant", "linear_decay"] = "constant"
    optimizer_definition: str = Field(min_length=1)
    evaluation_interval_rounds: int = Field(default=1, gt=0)

    @property
    def training_definition(self) -> str:
        return json.dumps(
            {
                "arm": "non_private_unclipped",
                "version": 1,
                "sampling": "opacus1.6-poisson-one-new-iterator-per-logical-step",
                "normalization": "loss-sum/int(N*resolved-q)",
                "empty_draw": "zero-gradient-optimizer-step",
                **self.model_dump(mode="json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )


class _BaselineTrainer(PyTorchTrainer):
    def load_checkpoint_tensors(self, state: dict[str, np.ndarray]) -> None:
        owned = decode_state(
            {
                k.removeprefix("application."): v
                for k, v in state.items()
                if k.startswith("application.")
            }
        )
        if (
            "meta.steps" not in state
            or state["meta.steps"].shape != (1,)
            or state["meta.steps"][0] != owned["steps"]
        ):
            raise ValueError("baseline checkpoint counters disagree")
        super().load_checkpoint_tensors(state)


def matched_draft(
    reference: DraftRunSpec, recipe: BaselineRecipe, *, run_id: str
) -> DraftRunSpec:
    """Preserve the reference NoLoCo/task settings; bind the baseline recipe."""
    if (
        reference.algorithm_id != "noloco"
        or reference.round_count * reference.local_steps != recipe.logical_steps
    ):
        raise ValueError("matched baseline requires the same NoLoCo horizon")
    if run_id == reference.run_id:
        raise ValueError("comparison arms need distinct run identities")
    spec = reference.private_training
    if spec is not None and (
        spec.batch_size != recipe.batch_size
        or spec.loss != recipe.loss
        or spec.schedule != recipe.schedule
        or canonical_json(spec.optimizer).decode() != recipe.optimizer_definition
    ):
        raise ValueError("baseline recipe differs from private training reference")
    values = reference.model_dump(mode="python")
    values.update(
        run_id=run_id,
        privacy=None,
        private_training=None,
        application_training={
            "policy_id": "matched-poisson-baseline-v1",
            "definition_hash": definition_hash(recipe.training_definition),
        },
    )
    return DraftRunSpec.model_validate(values)


class MatchedBaseline:
    """Deterministic per-example model, state-owned optimizer and Poisson RNG.

    Trusted factories/evaluators must not consume hidden global randomness.
    Any optimizer randomness must be retained by its numeric state_dict contract.
    """

    def __init__(
        self,
        *,
        model_factory: Callable[[], nn.Module],
        optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
        inputs: torch.Tensor,
        targets: torch.Tensor,
        recipe: BaselineRecipe,
        sampling_seed: int,
        local_data_identity: str,
        max_physical_batch_size: int,
        evaluate: Callable[[nn.Module], Mapping[str, float]] | None = None,
    ) -> None:
        if (
            inputs.ndim < 2
            or inputs.dtype != torch.float32
            or not len(inputs)
            or len(inputs) != len(targets)
            or not torch.isfinite(inputs).all()
            or not torch.isfinite(targets).all()
            or not local_data_identity.strip()
            or max_physical_batch_size <= 0
        ):
            raise ValueError("fixed finite FP32 inputs and baseline identity required")
        if recipe.loss == "mse" and (
            targets.dtype != torch.float32 or targets.ndim < 2
        ):
            raise ValueError("MSE needs FP32 feature targets")
        if recipe.loss == "cross_entropy" and (
            targets.dtype != torch.int64 or targets.ndim != 1
        ):
            raise ValueError("cross entropy needs integer class targets")
        self.recipe = recipe
        self.model = model_factory()
        stochastic = (
            nn.Dropout,
            nn.Dropout1d,
            nn.Dropout2d,
            nn.Dropout3d,
            nn.AlphaDropout,
            nn.FeatureAlphaDropout,
        )
        if any(isinstance(m, stochastic) for m in self.model.modules()):
            raise ValueError(
                "baseline requires deterministic per-example model behavior"
            )
        self.optimizer = optimizer_factory(self.model)
        params = [p for group in self.optimizer.param_groups for p in group["params"]]
        expected = list(self.model.parameters())
        if len(params) != len(expected) or {id(p) for p in params} != {
            id(p) for p in expected
        }:
            raise ValueError("optimizer must own every model parameter exactly once")
        self._initial_rates = tuple(float(g["lr"]) for g in self.optimizer.param_groups)
        if any(not math.isfinite(r) or r <= 0 for r in self._initial_rates):
            raise ValueError("positive finite optimizer learning rates required")
        self._rng = torch.Generator().manual_seed(sampling_seed)
        ordinary = DataLoader(
            TensorDataset(inputs.detach().clone(), targets.detach().clone()),
            batch_size=recipe.batch_size,
            num_workers=0,
            generator=self._rng,
        )
        self._loader: Any = opacus_module(
            "opacus.data_loader"
        ).DPDataLoader.from_data_loader(ordinary)
        self.sample_rate = float(self._loader.sample_rate)
        self.expected_batch_size = int(len(inputs) * self.sample_rate)
        if self.expected_batch_size < 1:
            raise ValueError("invalid expected batch size")
        self._physical_limit = max_physical_batch_size
        self._steps = 0
        self._physical_steps = 0
        self._examples = 0
        self._empty_draws = 0
        identity = hashlib.sha256(
            (
                recipe.training_definition
                + local_data_identity
                + str(
                    (
                        sampling_seed,
                        len(inputs),
                        self.sample_rate,
                        max_physical_batch_size,
                        self._initial_rates,
                    )
                )
            ).encode()
        ).hexdigest()
        self.trainer = _BaselineTrainer(
            model=self.model,
            train_step=self._step,
            save_state=self._save,
            load_state=self._load,
            state_identity=identity,
            evaluate=evaluate,
        )

    @property
    def statistics(self) -> dict[str, object]:
        return {
            "arm": "non_private_unclipped",
            "sampling": "poisson",
            "sample_rate": self.sample_rate,
            "expected_batch_size": self.expected_batch_size,
            "logical_steps": self._steps,
            "physical_steps": self._physical_steps,
            "sampled_examples": self._examples,
            "empty_draws": self._empty_draws,
            "epsilon": None,
        }

    def _step(self, model: nn.Module) -> float:
        if self._steps >= self.recipe.logical_steps:
            raise ValueError("baseline horizon exhausted")
        inputs, targets = next(iter(self._loader))
        model.train()
        self.optimizer.zero_grad(set_to_none=True)
        device = next(model.parameters()).device
        count = len(inputs)
        total_loss = 0.0
        # Use the same balanced physical splits as BatchSplittingSampler. Every
        # chunk contributes to one expected-batch-normalized logical gradient.
        chunks = max(1, math.ceil(count / self._physical_limit))
        for x, y in zip(
            torch.tensor_split(inputs, chunks),
            torch.tensor_split(targets, chunks),
            strict=True,
        ):
            prediction = model(x.to(device))
            y = y.to(device)
            if not len(x):
                loss_sum = prediction.sum()
            elif self.recipe.loss == "mse":
                loss_sum = (
                    (prediction - y).square().reshape(len(x), -1).mean(dim=1).sum()
                )
            else:
                loss_sum = nn.functional.cross_entropy(prediction, y, reduction="sum")
            loss = loss_sum / self.expected_batch_size
            cast(Any, loss).backward()
            total_loss += float(loss.detach())
            self._physical_steps += 1
        self.optimizer.step()  # pyright: ignore[reportUnknownMemberType]
        self._steps += 1
        self._examples += count
        self._empty_draws += int(count == 0)
        if self.recipe.schedule == "linear_decay":
            for group, initial in zip(
                self.optimizer.param_groups, self._initial_rates, strict=True
            ):
                group["lr"] = initial * (1 - self._steps / self.recipe.logical_steps)
        return total_loss

    def _save(self) -> dict[str, np.ndarray]:
        return encode_state(
            {
                "version": 1,
                "steps": self._steps,
                "physical_steps": self._physical_steps,
                "examples": self._examples,
                "empty_draws": self._empty_draws,
                "rng": self._rng.get_state(),
                "optimizer": self.optimizer.state_dict(),
            }
        )

    def _load(self, arrays: dict[str, np.ndarray]) -> None:
        state = decode_state(arrays)
        if (
            set(state)
            != {
                "version",
                "steps",
                "physical_steps",
                "examples",
                "empty_draws",
                "rng",
                "optimizer",
            }
            or state["version"] != 1
        ):
            raise ValueError("baseline checkpoint schema mismatch")
        steps = state["steps"]
        if (
            type(steps) is not int
            or not 0 <= steps <= self.recipe.logical_steps
            or any(
                type(state[k]) is not int or state[k] < 0
                for k in ("physical_steps", "examples", "empty_draws")
            )
            or state["physical_steps"] < steps
            or state["empty_draws"] > steps
        ):
            raise ValueError("invalid baseline checkpoint counters")
        reference = self.optimizer.state_dict()["param_groups"]
        groups = state["optimizer"]["param_groups"]
        if len(groups) != len(reference):
            raise ValueError("baseline optimizer group mismatch")
        for group, current, initial in zip(
            groups, reference, self._initial_rates, strict=True
        ):
            rate = (
                initial * (1 - steps / self.recipe.logical_steps)
                if self.recipe.schedule == "linear_decay"
                else initial
            )
            if (
                group.keys() != current.keys()
                or group["lr"] != rate
                or any(group[k] != current[k] for k in current if k != "lr")
            ):
                raise ValueError("baseline optimizer or schedule mismatch")
        rng = state["rng"]
        if (
            not isinstance(rng, torch.Tensor)
            or rng.dtype != torch.uint8
            or rng.shape != self._rng.get_state().shape
        ):
            raise ValueError("baseline RNG state mismatch")
        probe = torch.Generator()
        probe.set_state(rng)
        self.optimizer.load_state_dict(state["optimizer"])
        self._rng.set_state(rng)
        self._steps = steps
        self._physical_steps = state["physical_steps"]
        self._examples = state["examples"]
        self._empty_draws = state["empty_draws"]

    def prepare(
        self,
        draft: DraftRunSpec,
        *,
        model_definition: str,
        task_definition: str,
        evaluation_interval: int,
    ) -> PreparedApplication:
        if draft.privacy is not None or draft.private_training is not None:
            raise ValueError("baseline requires a non-private draft")
        if evaluation_interval != self.recipe.evaluation_interval_rounds:
            raise ValueError("baseline evaluation cadence mismatch")
        if draft.round_count * draft.local_steps != self.recipe.logical_steps:
            raise ValueError("baseline draft horizon mismatch")
        return prepare_application(
            draft=draft,
            trainer=self.trainer,
            model_definition=model_definition,
            task_definition=task_definition,
            training_definition=self.recipe.training_definition,
            evaluation_interval=evaluation_interval,
        )
