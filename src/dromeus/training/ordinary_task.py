"""Owned ordinary execution of the same finite-record application task."""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler

from dromeus.manifests.canonical import canonical_hash, canonical_json
from dromeus.manifests.models import FactoryOptimizerSpec, TaskTrainingSpec
from dromeus.training.private_data import dataset_size, make_task_loader
from dromeus.training.private_model import (
    ModelStateGuard,
    describe_private_model,
    registered_model_state,
    tensor_digest,
)
from dromeus.training.private_state import decode_state, encode_state
from dromeus.training.private_trainer import build_private_optimizer
from dromeus.training.task import TrainingTask, validate_record_losses
from dromeus.training.trainer import PyTorchTrainer


class _ShuffleBatches(Sampler[list[int]]):
    """Retain explicit epoch permutation/cursor instead of an opaque iterator."""

    def __init__(self, count: int, batch: int, generator: torch.Generator) -> None:
        self.count, self.batch, self.generator = count, batch, generator
        self.order = torch.empty(0, dtype=torch.int64)
        self.position = 0

    def __len__(self) -> int:
        return math.ceil(self.count / self.batch)

    def __iter__(self) -> Iterator[list[int]]:
        if not len(self.order) or self.position == self.count:
            self.order = torch.randperm(self.count, generator=self.generator)
            self.position = 0
        while self.position < self.count:
            end = min(self.position + self.batch, self.count)
            indices = cast(
                list[int], cast(Any, self.order[self.position : end]).tolist()
            )
            self.position = end
            yield indices


class OrdinaryTaskTrainer(PyTorchTrainer):
    """The task interface with ordinary gradients and explicit sampling policy.

    Shuffle execution uses only Torch. Poisson execution deliberately requires
    the optional privacy dependency to match the private experiment sampler.
    """

    def __init__(
        self,
        *,
        task: TrainingTask,
        specification: TaskTrainingSpec,
        local_data_identity: str,
        max_logical_steps: int,
        research_seed: int,
        max_physical_batch_size: int,
    ) -> None:
        if (
            not local_data_identity.strip()
            or max_logical_steps <= 0
            or max_physical_batch_size <= 0
            or specification.optimizer.torch_version != torch.__version__
        ):
            raise ValueError(
                "ordinary task requires valid identity, limits and Torch version"
            )
        task.check_definition(specification)
        if (
            task.public_evaluate is not None
            and specification.evaluation_provenance != "public_fixed_holdout"
        ):
            raise ValueError("ordinary task evaluation provenance is undeclared")
        self.task, self.specification = task, specification
        self._specification_hash = canonical_hash(specification)
        self._horizon, self._physical_limit = max_logical_steps, max_physical_batch_size
        self._logical_steps = self._physical_steps = 0
        self._failed = False
        self._population = dataset_size(task.dataset)
        self._raw_model = task.model_factory()
        report = describe_private_model(self._raw_model)
        if (
            report.trainable_parameter_names != specification.trainable_parameter_names
            or report.parameter_aliases != specification.parameter_aliases
            or report.public_buffer_hashes != specification.public_buffer_hashes
            or report.frozen_parameter_hashes != specification.frozen_parameter_hashes
        ):
            raise ValueError("ordinary model differs from shared task specification")
        self._guard = ModelStateGuard(self._raw_model, report)
        optimizer_spec = specification.optimizer
        if isinstance(optimizer_spec, FactoryOptimizerSpec):
            if task.optimizer_registration is None:
                raise ValueError("factory optimizer requires its local registration")
            self._optimizer = task.optimizer_registration.build(
                self._raw_model, optimizer_spec
            )
        else:
            if task.optimizer_registration is not None:
                raise ValueError(
                    "optimizer registration requires factory specification"
                )
            self._optimizer = build_private_optimizer(self._raw_model, optimizer_spec)
        self._parameters = tuple(
            p for p in self._raw_model.parameters() if p.requires_grad
        )
        self._initial_groups = copy.deepcopy(
            self._optimizer.state_dict()["param_groups"]
        )
        self._sampling_rng = torch.Generator().manual_seed(research_seed)
        self._model_rng: torch.Tensor = (
            torch.Generator().manual_seed(research_seed + 1).get_state()
        )
        device = next(self._raw_model.parameters()).device
        self._cuda_rng: torch.Tensor | None = (
            torch.Generator(device=device).manual_seed(research_seed + 2).get_state()
            if device.type == "cuda"
            else None
        )
        self._shuffle: _ShuffleBatches | None = None
        if specification.sampling == "poisson":
            self._loader = make_task_loader(
                task.dataset,
                task.batch_adapter,
                batch_size=specification.batch_size,
                generator=self._sampling_rng,
            )
            self._sample_rate = self._loader.sample_rate
        else:
            self._shuffle = _ShuffleBatches(
                self._population, specification.batch_size, self._sampling_rng
            )
            self._loader = DataLoader(  # pyright: ignore[reportUnknownMemberType]
                task.dataset,
                batch_sampler=self._shuffle,
                collate_fn=task.batch_adapter.collate,
                generator=self._sampling_rng,
                num_workers=0,
            )
            self._sample_rate = None
        identity = hashlib.sha256(
            canonical_json(specification)
            + local_data_identity.encode()
            + str(
                (self._population, max_logical_steps, max_physical_batch_size)
            ).encode()
        ).hexdigest()
        super().__init__(
            model=self._raw_model,
            train_step=self._owned_step,
            save_state=self._save_state,
            load_state=self._load_state,
            state_identity=identity,
            evaluate=task.public_evaluate,
            exchange_names=specification.trainable_parameter_names,
        )

    @property
    def sample_rate(self) -> float | None:
        return self._sample_rate

    @property
    def physical_steps(self) -> int:
        return self._physical_steps

    def weights(self) -> dict[str, np.ndarray]:
        self._check()
        return super().weights()

    def checkpoint_tensors(self) -> dict[str, np.ndarray]:
        self._check()
        return super().checkpoint_tensors()

    def _check(self) -> None:
        if self._failed:
            raise ValueError("ordinary trainer failed")
        if canonical_hash(self.specification) != self._specification_hash:
            raise ValueError("ordinary task specification changed")
        if dataset_size(self.task.dataset) != self._population:
            raise ValueError("ordinary dataset population changed")
        self._guard.check()
        actual = tuple(
            p for group in self._optimizer.param_groups for p in group["params"]
        )
        if len(actual) != len(self._parameters) or {id(p) for p in actual} != {
            id(p) for p in self._parameters
        }:
            raise ValueError("ordinary optimizer parameter ownership changed")
        self._validate_optimizer(self._optimizer.state_dict(), self._logical_steps)

    def train_local_steps(self, step_count: int) -> None:
        self._check()
        if step_count <= 0 or self._logical_steps + step_count > self._horizon:
            raise ValueError(
                "ordinary training horizon exhausted or invalid step count"
            )
        try:
            super().train_local_steps(step_count)
        except Exception:
            self._failed = True
            raise

    def _owned_step(self, _: nn.Module) -> float:
        self._check()
        device = next(self._raw_model.parameters()).device
        devices = [device.index or 0] if device.type == "cuda" else []
        with cast(Any, torch.random).fork_rng(devices=devices):
            torch.set_rng_state(self._model_rng)
            if self._cuda_rng is not None:
                torch.cuda.set_rng_state(self._cuda_rng, device)
            self._raw_model.train()
            batch = next(iter(self._loader))
            count = self.task.batch_adapter.size(batch)
            normalizer = (
                int(self._population * self._sample_rate)
                if self._sample_rate is not None
                else max(count, 1)
            )
            self._optimizer.zero_grad(set_to_none=True)
            for parameter in self._parameters:
                parameter.grad = torch.zeros_like(parameter)
            total_loss = 0.0
            for start in range(0, count, self._physical_limit):
                physical = self.task.batch_adapter.slice(
                    batch, start, min(count, start + self._physical_limit)
                )
                physical = self.task.batch_adapter.to_device(physical, device)
                values = validate_record_losses(
                    self.task.loss_per_record(self._raw_model, physical),
                    self.task.batch_adapter.size(physical),
                )
                total_loss += float(values.detach().sum())
                cast(Any, values.sum() / normalizer).backward()
                self._physical_steps += 1
            if count == 0:
                self._physical_steps += 1
            self._guard.check()
            self._optimizer.step()
            self._logical_steps += 1
            if self.specification.schedule == "linear_decay":
                for group, initial in zip(
                    self._optimizer.param_groups, self._initial_groups, strict=True
                ):
                    group["lr"] = initial["lr"] * (
                        1 - self._logical_steps / self._horizon
                    )
            self._optimizer.zero_grad(set_to_none=True)
            self._guard.check()
            self._model_rng = torch.get_rng_state().clone()
            if devices:
                self._cuda_rng = torch.cuda.get_rng_state(device).clone()
        return total_loss / max(count, 1)

    def _save_state(self) -> dict[str, np.ndarray]:
        return encode_state(
            {
                "version": 1,
                "specification": self._specification_hash,
                "optimizer": self._optimizer.state_dict(),
                "logical_steps": self._logical_steps,
                "physical_steps": self._physical_steps,
                "sampling_rng": self._sampling_rng.get_state(),
                "model_rng": self._model_rng,
                "cuda_rng": self._cuda_rng,
                "shuffle_order": self._shuffle.order if self._shuffle else None,
                "shuffle_position": self._shuffle.position if self._shuffle else None,
            }
        )

    def _validate_optimizer(self, state: dict[str, Any], steps: int) -> None:
        spec = self.specification.optimizer
        if isinstance(spec, FactoryOptimizerSpec):
            registration = self.task.optimizer_registration
            assert registration is not None
            registration.validate_checkpoint(
                self._optimizer,
                self._raw_model,
                spec,
                state,
                logical_steps=steps,
                horizon=self._horizon,
                schedule=self.specification.schedule,
            )
            return
        if set(state) != {"state", "param_groups"}:
            raise ValueError("ordinary optimizer checkpoint fields mismatch")
        groups = state["param_groups"]
        if len(groups) != len(self._initial_groups):
            raise ValueError("ordinary optimizer checkpoint groups mismatch")
        for group, initial in zip(groups, self._initial_groups, strict=True):
            expected = dict(initial)
            if self.specification.schedule == "linear_decay":
                expected["lr"] *= 1 - steps / self._horizon
            if group != expected:
                raise ValueError("ordinary optimizer checkpoint options mismatch")
        fields: set[str] = (
            ({"momentum_buffer"} if spec.momentum else set())
            if spec.name == "sgd"
            else {"step", "exp_avg", "exp_avg_sq"}
        )
        ids: set[int] = set(range(len(self._parameters))) if steps and fields else set()
        if set(state["state"]) != ids:
            raise ValueError("ordinary optimizer checkpoint coverage mismatch")
        for index in ids:
            values = state["state"][index]
            if set(values) != fields:
                raise ValueError("ordinary optimizer checkpoint state mismatch")
            for name, tensor in values.items():
                shape = () if name == "step" else self._parameters[index].shape
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.shape != shape
                    or tensor.dtype != torch.float32
                ):
                    raise ValueError("ordinary optimizer checkpoint tensor mismatch")
                if name == "step" and float(tensor) != steps:
                    raise ValueError("ordinary optimizer checkpoint step mismatch")

    def _load_state(self, arrays: dict[str, np.ndarray]) -> None:
        state = decode_state(arrays)
        expected = set(decode_state(self._save_state()))
        if (
            not isinstance(state, dict)
            or set(cast(dict[str, Any], state)) != expected
            or state["version"] != 1
        ):
            raise ValueError("ordinary task checkpoint schema mismatch")
        state = cast(dict[str, Any], state)
        if state["specification"] != self._specification_hash:
            raise ValueError("ordinary task checkpoint specification mismatch")
        steps, physical = state["logical_steps"], state["physical_steps"]
        if (
            type(steps) is not int
            or not 0 <= steps <= self._horizon
            or type(physical) is not int
            or physical < steps
        ):
            raise ValueError("ordinary task checkpoint counters mismatch")
        self._validate_optimizer(state["optimizer"], steps)
        for key, current in (
            ("sampling_rng", self._sampling_rng.get_state()),
            ("model_rng", self._model_rng),
            ("cuda_rng", self._cuda_rng),
        ):
            saved = state[key]
            if current is None:
                if saved is not None:
                    raise ValueError("ordinary task checkpoint RNG device mismatch")
            elif (
                not isinstance(saved, torch.Tensor)
                or saved.dtype != torch.uint8
                or saved.shape != current.shape
            ):
                raise ValueError("ordinary task checkpoint RNG mismatch")
        probe = torch.Generator()
        probe.set_state(state["sampling_rng"])
        probe.set_state(state["model_rng"])
        if self._cuda_rng is not None:
            cuda_probe = torch.Generator(device=self._parameters[0].device)
            cuda_probe.set_state(state["cuda_rng"])
        if self._shuffle is not None:
            order, position = state["shuffle_order"], state["shuffle_position"]
            if (
                not isinstance(order, torch.Tensor)
                or order.dtype != torch.int64
                or order.ndim != 1
                or type(position) is not int
            ):
                raise ValueError("ordinary shuffle checkpoint mismatch")
            if (not len(order) and position != 0) or (
                len(order)
                and (
                    len(order) != self._population
                    or not 0 <= position <= len(order)
                    or not torch.equal(
                        order.sort().values, torch.arange(self._population)
                    )
                )
            ):
                raise ValueError("ordinary shuffle permutation mismatch")
        elif (
            state["shuffle_order"] is not None or state["shuffle_position"] is not None
        ):
            raise ValueError("ordinary sampler checkpoint mismatch")
        self._optimizer.load_state_dict(state["optimizer"])
        self._sampling_rng.set_state(state["sampling_rng"])
        self._model_rng, self._cuda_rng = state["model_rng"], state["cuda_rng"]
        if self._shuffle is not None:
            self._shuffle.order = cast(torch.Tensor, state["shuffle_order"])
            self._shuffle.position = cast(int, state["shuffle_position"])
        self._logical_steps, self._physical_steps = steps, physical

    def load_checkpoint_tensors(self, state: dict[str, np.ndarray]) -> None:
        self._check()
        arrays = {
            k.removeprefix("application."): v
            for k, v in state.items()
            if k.startswith("application.")
        }
        decoded = decode_state(arrays)
        if "meta.steps" not in state or int(state["meta.steps"][0]) != decoded.get(
            "logical_steps"
        ):
            raise ValueError("ordinary checkpoint counters disagree")
        immutable = {
            **self.specification.public_buffer_hashes,
            **self.specification.frozen_parameter_hashes,
        }
        persistent = registered_model_state(self._raw_model)
        for name, digest in immutable.items():
            if name not in persistent:
                continue
            values = state.get(f"model.{name}")
            if values is None or tensor_digest(torch.from_numpy(values)) != digest:  # pyright: ignore[reportUnknownMemberType]
                raise ValueError("ordinary checkpoint changed immutable public state")
        super().load_checkpoint_tensors(state)
        self._guard.check()
