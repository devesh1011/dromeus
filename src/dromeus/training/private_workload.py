"""Internal workload adapter for private application tasks.

The workload prepares gradients and enforces the model/task contract. The trainer
owns Opacus, optimizer invocation, accounting, reservations and logical steps.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dromeus.manifests.models import (
    TaskTrainingSpec,
)
from dromeus.training.optimizer_factory import PrivateOptimizerFactory
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
    *, specification: TaskTrainingSpec, task: TrainingTask
) -> PrivateWorkload:
    """Bind one application task to the group's private training specification."""
    if specification.sampling != "poisson":
        raise ValueError("private tasks require Poisson sampling")
    task.check_definition(specification)
    if (
        task.public_evaluate is not None
        and specification.evaluation_provenance != "public_fixed_holdout"
    ):
        raise ValueError("evaluation requires public fixed holdout provenance")
    return _ApplicationWorkload(task, specification)


class _ApplicationWorkload:
    """Generic task semantics and state checks, without a tensor-task fallback."""

    poisson_sampling = False  # TaskDataLoader already owns its Poisson sampler.

    def __init__(self, task: TrainingTask, specification: TaskTrainingSpec) -> None:
        self._task, self._specification = task, specification
        self._native_private_model: Any = None
        self._native_sampler_token: Any = None
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
            gradient_backend=specification.gradient_backend,
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
        if self._specification.gradient_backend == "opacus_native_b1":
            from dromeus.training.native_gradients import install_native_b1_samplers

            self._native_sampler_token = install_native_b1_samplers(private_model)
            self._native_private_model = private_model

    def check(self) -> None:
        self._guard.check()
        self._task.check_definition(self._specification)
        if self._native_private_model is not None:
            from dromeus.training.native_gradients import check_native_b1_samplers

            check_native_b1_samplers(
                self._native_private_model, self._native_sampler_token
            )

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
