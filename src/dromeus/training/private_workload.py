"""Internal workload adapters for the two supported private-task generations.

Workloads prepare gradients and enforce their model/task contracts. The trainer
owns Opacus, optimizer invocation, accounting, reservations and logical steps.
Legacy tensors keep their original sampler, loss reduction and state inventory.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dromeus.manifests.models import (
    FactoryOptimizerSpec,
    PrivateTrainingSpec,
    TaskTrainingSpec,
)
from dromeus.training.model_state import model_state_alias_groups
from dromeus.training.optimizer_factory import PrivateOptimizerFactory
from dromeus.training.privacy import opacus_module
from dromeus.training.private_data import make_task_loader
from dromeus.training.private_model import (
    ModelStateGuard,
    ensure_grad_samples,
    registered_model_state,
    tensor_digest,
    validate_private_model,
)
from dromeus.training.task import TrainingTask, validate_record_losses

type PublicEvaluation = Callable[[nn.Module], Mapping[str, float]]
type WeightSnapshot = Callable[[], dict[str, np.ndarray]]


class PrivateWorkloadStep(Protocol):
    """Workload checks scoped to one draw and all its physical batches."""

    def prepare_gradients(self, private_model: nn.Module, batch: Any) -> None: ...

    def after_optimizer_step(self) -> None: ...


class PrivateWorkload(Protocol):
    model: nn.Module
    evaluate: PublicEvaluation | None
    optimizer_registration: PrivateOptimizerFactory | None
    exchange_names: tuple[str, ...] | None
    poisson_sampling: bool

    def make_loader(self, generator: torch.Generator | None) -> DataLoader[Any]: ...

    def configure_private_model(self, private_model: Any) -> None: ...

    def check(self) -> None: ...

    def begin_step(self) -> PrivateWorkloadStep: ...

    def export_tensors(self, weights: WeightSnapshot) -> dict[str, np.ndarray]: ...

    def validate_checkpoint(self, state: dict[str, np.ndarray]) -> None: ...


def prepare_private_workload(
    *,
    specification: PrivateTrainingSpec | TaskTrainingSpec,
    task: TrainingTask | None,
    model_factory: Callable[[], nn.Module] | None,
    inputs: torch.Tensor | None,
    targets: torch.Tensor | None,
    evaluate: PublicEvaluation | None,
    optimizer_registration: PrivateOptimizerFactory | None,
) -> PrivateWorkload:
    """Select compatibility once, without rewriting specification identities."""
    if isinstance(specification, TaskTrainingSpec):
        if (
            task is None
            or model_factory is not None
            or inputs is not None
            or targets is not None
        ):
            raise ValueError("generic private preparation requires exactly one task")
        if specification.sampling != "poisson":
            raise ValueError("private tasks require Poisson sampling")
        task.check_definition(specification)
        if (
            task.public_evaluate is not None
            and specification.evaluation_provenance != "public_fixed_holdout"
        ):
            raise ValueError("evaluation requires public fixed holdout provenance")
        return _ApplicationWorkload(task, specification)
    if task is not None:
        raise ValueError("generic task requires version 2 training specification")
    if model_factory is None or inputs is None or targets is None:
        raise ValueError("legacy private preparation requires tensor inputs and model")
    return _LegacyTensorWorkload(
        model_factory, inputs, targets, specification, evaluate, optimizer_registration
    )


class _LegacyTensorWorkload:
    """Keep the v1 tensor recipe intact, including its restrictive model profile."""

    exchange_names: tuple[str, ...] | None = None
    poisson_sampling = True

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        inputs: torch.Tensor,
        targets: torch.Tensor,
        specification: PrivateTrainingSpec,
        evaluate: PublicEvaluation | None,
        optimizer_registration: PrivateOptimizerFactory | None,
    ) -> None:
        if (
            inputs.ndim < 2
            or inputs.dtype != torch.float32
            or len(inputs) != len(targets)
            or not len(inputs)
        ):
            raise ValueError("fixed FP32 tensor dataset required")
        if not torch.isfinite(inputs).all() or not torch.isfinite(targets).all():
            raise ValueError("invalid local data")
        if specification.loss == "mse" and (
            targets.dtype != torch.float32 or targets.ndim < 2
        ):
            raise ValueError("MSE requires FP32 targets with batch and feature axes")
        if specification.loss == "cross_entropy" and (
            targets.dtype != torch.int64 or targets.ndim != 1
        ):
            raise ValueError("cross entropy requires integer class targets")
        self.specification = specification
        self.evaluate = evaluate
        self.optimizer_registration = optimizer_registration
        self.model = model_factory()
        self._validate_model()
        # Clone at loader preparation, after optimizer construction as in v1.
        self._data: tuple[torch.Tensor, torch.Tensor] | None = (inputs, targets)

    def _validate_model(self) -> None:
        model = self.model
        if list(model.buffers()) or model_state_alias_groups(model):
            raise ValueError("private model buffers and aliases are unsupported")
        approved = {
            nn.Sequential,
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.GroupNorm,
            nn.ReLU,
            nn.Tanh,
            nn.Sigmoid,
            nn.Flatten,
            nn.Identity,
            nn.MaxPool2d,
            nn.AvgPool2d,
            nn.AdaptiveAvgPool2d,
        }
        for module in model.modules():
            if (
                type(module) not in approved
                or "forward" in module.__dict__
                or getattr(module, "_forward_hooks")
                or getattr(module, "_forward_pre_hooks")
            ):
                raise ValueError("unsupported private model leaf module")
        named = tuple(model.named_parameters())
        spec = self.specification.optimizer
        if (
            (set(name for name, _ in named) != set(spec.parameter_names))
            if isinstance(spec, FactoryOptimizerSpec)
            else tuple(name for name, _ in named) != spec.parameter_names
        ):
            raise ValueError("private parameter names or order mismatch")
        if any(
            not p.requires_grad or p.dtype != torch.float32 or p.is_sparse
            for _, p in named
        ):
            raise ValueError(
                "private profile requires all parameters trainable dense FP32"
            )
        opacus_module("opacus.validators").ModuleValidator.validate(model, strict=True)

    def make_loader(self, generator: torch.Generator | None) -> DataLoader[Any]:
        assert self._data is not None
        inputs, targets = self._data
        loader = DataLoader(
            TensorDataset(inputs.detach().clone(), targets.detach().clone()),
            batch_size=self.specification.batch_size,
            num_workers=0,
            generator=generator,
        )
        self._data = None  # Do not retain the caller's tensors alongside the copy.
        return loader

    def configure_private_model(self, private_model: Any) -> None:
        # Opacus already forbids accumulation when it converts this loader.
        pass

    def check(self) -> None:
        # The v1 profile relies on the trainer's common execution checks.
        pass

    def begin_step(self) -> PrivateWorkloadStep:
        return _LegacyTensorStep(self.model, self.specification)

    def export_tensors(self, weights: WeightSnapshot) -> dict[str, np.ndarray]:
        return weights()

    def validate_checkpoint(self, state: dict[str, np.ndarray]) -> None:
        # The v1 model profile has no frozen parameters or public buffers.
        pass


class _LegacyTensorStep:
    def __init__(self, model: nn.Module, specification: PrivateTrainingSpec) -> None:
        self._model, self._specification = model, specification

    def prepare_gradients(self, private_model: nn.Module, batch: Any) -> None:
        device = next(self._model.parameters()).device
        inputs, targets = batch
        prediction = private_model(inputs.to(device))
        targets = targets.to(device)
        if self._specification.loss == "mse":
            loss = (
                (prediction - targets).square().reshape(len(inputs), -1).mean()
                if len(inputs)
                else prediction.sum()
            )
        else:
            loss = (
                nn.functional.cross_entropy(prediction, targets, reduction="mean")
                if len(inputs)
                else prediction.sum()
            )
        cast(Any, loss).backward()

    def after_optimizer_step(self) -> None:
        pass


class _ApplicationWorkload:
    """Generic task semantics and state checks, without a tensor-task fallback."""

    poisson_sampling = False  # TaskDataLoader already owns its Poisson sampler.

    def __init__(self, task: TrainingTask, specification: TaskTrainingSpec) -> None:
        self._task, self._specification = task, specification
        self.evaluate = task.public_evaluate
        self.optimizer_registration = task.optimizer_registration
        self.exchange_names: tuple[str, ...] | None = (
            specification.trainable_parameter_names
        )
        self.model = task.model_factory()
        report = validate_private_model(
            self.model,
            task.loss_per_record,
            task.batch_adapter,
            task.batch_adapter.example,
        )
        for name in (
            "trainable_parameter_names",
            "parameter_aliases",
            "public_buffer_hashes",
            "frozen_parameter_hashes",
            "gradient_backend",
        ):
            if getattr(report, name) != getattr(specification, name):
                raise ValueError(f"model {name} differs from shared task specification")
        self._guard = ModelStateGuard(self.model, report)

    def make_loader(self, generator: torch.Generator | None) -> DataLoader[Any]:
        return make_task_loader(
            self._task.dataset,
            self._task.batch_adapter,
            batch_size=self._specification.batch_size,
            generator=generator,
        )

    def configure_private_model(self, private_model: Any) -> None:
        private_model.forbid_grad_accumulation()

    def check(self) -> None:
        self._guard.check()
        self._task.check_definition(self._specification)

    def begin_step(self) -> PrivateWorkloadStep:
        return _ApplicationStep(self.model, self._task, self._guard)

    def export_tensors(self, weights: WeightSnapshot) -> dict[str, np.ndarray]:
        return {
            name: value.detach().cpu().numpy().copy()
            for name, value in registered_model_state(self.model).items()
        }

    def validate_checkpoint(self, state: dict[str, np.ndarray]) -> None:
        immutable = {
            **self._specification.public_buffer_hashes,
            **self._specification.frozen_parameter_hashes,
        }
        persistent = registered_model_state(self.model)
        for name, expected in immutable.items():
            if name not in persistent:
                continue
            value = state.get(f"model.{name}")
            if value is None or tensor_digest(torch.tensor(value.copy())) != expected:
                raise ValueError("checkpoint changed immutable public model state")


class _ApplicationStep:
    """Capture guards before sampling and release them with the logical call."""

    def __init__(
        self, model: nn.Module, task: TrainingTask, guard: ModelStateGuard
    ) -> None:
        self._model, self._task, self._guard = model, task, guard
        self._device = next(model.parameters()).device
        self._cpu_rng = torch.get_rng_state()
        self._cuda_rng = (
            torch.cuda.get_rng_state(self._device)
            if self._device.type == "cuda"
            else None
        )
        self._expected_parameters = self._parameters()

    def _parameters(self) -> dict[str, torch.Tensor]:
        return {
            name: p.detach().clone()
            for name, p in self._model.named_parameters()
            if p.requires_grad
        }

    def _parameters_changed(self) -> bool:
        return any(
            not torch.equal(p.detach(), self._expected_parameters[name])
            for name, p in self._model.named_parameters()
            if p.requires_grad
        )

    def prepare_gradients(self, private_model: nn.Module, batch: Any) -> None:
        if self._parameters_changed():
            raise ValueError("task data callback mutated model parameters")
        device = next(self._model.parameters()).device
        batch = self._task.batch_adapter.to_device(batch, device)
        count = self._task.batch_adapter.size(batch)
        if count:
            losses = self._task.loss_per_record(private_model, batch)
            validate_record_losses(losses, count)
            cast(Any, losses.mean()).backward()
            if self._parameters_changed():
                raise ValueError("task forward/backward mutated model parameters")
        ensure_grad_samples(self._model, count)
        self._guard.check()

    def after_optimizer_step(self) -> None:
        self._expected_parameters = self._parameters()
        if not torch.equal(self._cpu_rng, torch.get_rng_state()) or (
            self._cuda_rng is not None
            and not torch.equal(self._cuda_rng, torch.cuda.get_rng_state(self._device))
        ):
            raise ValueError("task data/model callbacks consumed undeclared RNG")
