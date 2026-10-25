"""Owned Opacus trainer for fixed local tensor datasets and public evaluation.

Application factories and model code are trusted. Dense FP32 optimizers use
legacy presets or a group-bound trusted factory contract. Raw losses never leave
this adapter, including research mode.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dromeus.manifests.canonical import canonical_hash, canonical_json, save_safetensors
from dromeus.manifests.models import (
    FactoryOptimizerSpec,
    PrivacyPolicy,
    PrivacyReservation,
    PrivateOptimizerSpec,
    PrivateTrainingSpec,
    TaskTrainingSpec,
)
from dromeus.training.base import EvaluationResult
from dromeus.training.model_state import model_state_alias_groups
from dromeus.training.optimizer_factory import (
    PrivateOptimizerFactory,
    SanitizedOptimizer,
    validate_builtin_optimizer_checkpoint,
)

# Preserve the original builder import for application and benchmark callers.
from dromeus.training.optimizer_factory import (
    build_private_optimizer as build_private_optimizer,
)
from dromeus.training.privacy import (
    ReservationStore,
    opacus_module,
    privacy_epsilon,
    reserve_horizon,
)
from dromeus.training.private_data import make_task_loader
from dromeus.training.private_model import (
    ModelStateGuard,
    ensure_grad_samples,
    registered_model_state,
    tensor_digest,
    validate_private_model,
)
from dromeus.training.private_state import decode_state, encode_state
from dromeus.training.task import TrainingTask, validate_record_losses
from dromeus.training.trainer import PyTorchTrainer


def private_training_definition(
    specification: PrivateTrainingSpec | TaskTrainingSpec,
) -> str:
    return canonical_json(specification).decode()


class PrivateTrainer(PyTorchTrainer):
    """Retain private model/optimizer/loader and count complete logical mechanisms."""

    def __init__(
        self,
        *,
        model_factory: Callable[[], nn.Module] | None = None,
        inputs: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        task: TrainingTask | None = None,
        policy: PrivacyPolicy,
        specification: PrivateTrainingSpec | TaskTrainingSpec,
        run_id: str,
        local_data_identity: str,
        reservation_store: ReservationStore,
        max_physical_batch_size: int,
        research_seed: int | None = None,
        evaluate: Callable[[nn.Module], Mapping[str, float]] | None = None,
        optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer] | None = None,
        optimizer_registration: PrivateOptimizerFactory | None = None,
    ) -> None:
        if specification.optimizer.torch_version != torch.__version__:
            raise ValueError("private optimizer Torch version mismatch")
        if max_physical_batch_size <= 0 or not local_data_identity.strip():
            raise ValueError(
                "physical batch limit and local data identity are required"
            )
        secure = policy.randomness_profile == "secure"
        if secure and research_seed is not None:
            raise ValueError("secure mode cannot accept a research seed")
        if not secure and research_seed is None:
            raise ValueError("public research mode requires an explicit local seed")
        if (
            evaluate is not None
            and specification.evaluation_provenance != "public_fixed_holdout"
        ):
            raise ValueError("evaluation requires public fixed holdout provenance")
        self._task = task
        self._model_guard: ModelStateGuard | None = None
        if isinstance(specification, TaskTrainingSpec):
            if (
                task is None
                or model_factory is not None
                or inputs is not None
                or targets is not None
            ):
                raise ValueError(
                    "generic private preparation requires exactly one task"
                )
            if specification.sampling != "poisson":
                raise ValueError("private tasks require Poisson sampling")
            task.check_definition(specification)
            model_factory = task.model_factory
            optimizer_registration = task.optimizer_registration
            evaluate = task.public_evaluate
            if (
                evaluate is not None
                and specification.evaluation_provenance != "public_fixed_holdout"
            ):
                raise ValueError("evaluation requires public fixed holdout provenance")
        elif task is not None:
            raise ValueError("generic task requires version 2 training specification")
        else:
            if model_factory is None or inputs is None or targets is None:
                raise ValueError(
                    "legacy private preparation requires tensor inputs and model"
                )
            self._validate_tensor_data(inputs, targets, specification)
        self.policy = policy
        self.specification = specification
        self._specification_hash = canonical_hash(specification)
        self._physical_limit = max_physical_batch_size
        self._failed = False
        self._physical_steps = 0
        self._logical_steps = 0
        self._reservation_store = reservation_store
        assert model_factory is not None
        self._raw_model = model_factory()
        self._validate_model()
        spec = specification.optimizer
        self._optimizer_registration = optimizer_registration
        if isinstance(spec, FactoryOptimizerSpec):
            if optimizer_registration is None or optimizer_factory is not None:
                raise ValueError(
                    "factory optimizer requires exactly one local registration"
                )
            self._base_optimizer = optimizer_registration.build(self._raw_model, spec)
        else:
            if optimizer_registration is not None:
                raise ValueError(
                    "optimizer registration requires a factory specification"
                )
            self._base_optimizer = (
                optimizer_factory(self._raw_model)
                if optimizer_factory
                else build_private_optimizer(self._raw_model, spec)
            )
        self._validate_optimizer()
        self._initial_optimizer_groups: list[dict[str, Any]] = (
            copy.deepcopy(self._base_optimizer.state_dict()["param_groups"])
            if isinstance(spec, PrivateOptimizerSpec)
            else []
        )
        self._initial_learning_rates = tuple(
            float(g["lr"]) for g in self._base_optimizer.param_groups
        )
        self._sampling_generator = (
            None if secure else torch.Generator().manual_seed(int(research_seed or 0))
        )
        device = next(self._raw_model.parameters()).device
        self._noise_generator = (
            None
            if secure
            else torch.Generator(device=device).manual_seed(int(research_seed or 0) + 1)
        )
        if task is None:
            assert inputs is not None and targets is not None
            loader = DataLoader(
                TensorDataset(inputs.detach().clone(), targets.detach().clone()),
                batch_size=specification.batch_size,
                num_workers=0,
                generator=self._sampling_generator,
            )
        else:
            loader = make_task_loader(
                task.dataset,
                task.batch_adapter,
                batch_size=specification.batch_size,
                generator=self._sampling_generator,
            )
        self._population_size = len(cast(Any, loader.dataset))
        try:
            self._engine = opacus_module().PrivacyEngine(
                accountant="prv", secure_mode=secure
            )
            self._private_model, self._optimizer, self._loader = (
                self._engine.make_private(
                    module=self._raw_model,
                    optimizer=SanitizedOptimizer(self._base_optimizer)
                    if optimizer_registration
                    else self._base_optimizer,
                    data_loader=loader,
                    noise_multiplier=policy.noise_multiplier,
                    max_grad_norm=policy.max_grad_norm,
                    poisson_sampling=task is None,
                    clipping="flat",
                    grad_sample_mode="hooks",
                    noise_generator=self._noise_generator,
                )
            )
            if task is not None:
                self._private_model.forbid_grad_accumulation()
        except Exception:
            raise ValueError(
                "private preparation failed; verify model and secure RNG installation"
            ) from None
        self._accountant_hook = self._optimizer.step_hook
        self._noise_stream = self._optimizer.generator
        self._sampling_stream = self._loader.batch_sampler.generator
        self._sample_rate = 1 / len(self._loader)
        self._reservation = PrivacyReservation(
            run_id=run_id,
            policy_hash=canonical_hash(policy),
            sample_rate=self._sample_rate,
            noise_multiplier=policy.noise_multiplier,
            logical_steps=policy.max_logical_steps,
        )
        reserve_horizon(reservation_store, self._reservation, policy)
        identity = hashlib.sha256(
            canonical_json(policy)
            + canonical_json(specification)
            + local_data_identity.encode()
            + str(
                (self._population_size, self._sample_rate, max_physical_batch_size)
            ).encode()
        ).hexdigest()
        super().__init__(
            model=self._raw_model,
            train_step=self._owned_step,
            save_state=self._save_private_state,
            load_state=self._load_private_state,
            state_identity=identity,
            evaluate=evaluate,
            exchange_names=specification.trainable_parameter_names
            if isinstance(specification, TaskTrainingSpec)
            else None,
        )

    @staticmethod
    def _validate_tensor_data(
        inputs: torch.Tensor, targets: torch.Tensor, specification: PrivateTrainingSpec
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

    def _validate_model(self) -> None:
        model = self._raw_model
        if isinstance(self.specification, TaskTrainingSpec):
            assert self._task is not None
            report = validate_private_model(
                model,
                self._task.loss_per_record,
                self._task.batch_adapter,
                self._task.batch_adapter.example,
            )
            for name in (
                "trainable_parameter_names",
                "parameter_aliases",
                "public_buffer_hashes",
                "frozen_parameter_hashes",
                "gradient_backend",
            ):
                if getattr(report, name) != getattr(self.specification, name):
                    raise ValueError(
                        f"model {name} differs from shared task specification"
                    )
            self._model_guard = ModelStateGuard(model, report)
            return
        # Buffers and aliases may carry unaudited statistics or alter clipping.
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

    def _validate_optimizer(self) -> None:
        spec = self.specification.optimizer
        actual = self._base_optimizer
        if isinstance(spec, FactoryOptimizerSpec):
            assert self._optimizer_registration is not None
            self._optimizer_registration.check(
                actual,
                self._raw_model,
                spec,
                logical_steps=0,
                horizon=self.policy.max_logical_steps,
                schedule=self.specification.schedule,
            )
            self._parameter_ids = tuple(id(p) for p in self._raw_model.parameters())
            return
        expected = build_private_optimizer(self._raw_model, spec)
        if type(actual) is not type(expected) or len(actual.param_groups) != 1:
            raise ValueError("unsupported private optimizer class or groups")
        group = actual.param_groups[0]
        expected_group = expected.param_groups[0]
        if set(group) != set(expected_group) or any(
            group[key] != expected_group[key]
            for key in expected_group
            if key != "params"
        ):
            raise ValueError(
                "private optimizer options differ from shared specification"
            )
        if [id(p) for p in group["params"]] != [
            id(p) for p in self._raw_model.parameters() if p.requires_grad
        ]:
            raise ValueError(
                "private optimizer must cover the complete ordered parameter group"
            )
        self._parameter_ids = tuple(id(p) for p in self._raw_model.parameters())
        self._optimizer_options = {
            k: v for k, v in group.items() if k not in {"params", "lr"}
        }

    def _check_execution(self) -> None:
        self._reservation_store.check(self._reservation)
        if self._failed:
            raise ValueError("private trainer failed; start a new accounted run")
        if canonical_hash(self.specification) != self._specification_hash:
            raise ValueError("private optimizer specification changed")
        group = self._base_optimizer.param_groups
        spec = self.specification.optimizer
        if isinstance(spec, FactoryOptimizerSpec):
            assert self._optimizer_registration is not None
            self._optimizer_registration.check(
                self._base_optimizer,
                self._raw_model,
                spec,
                logical_steps=self._logical_steps,
                horizon=self.policy.max_logical_steps,
                schedule=self.specification.schedule,
            )
        elif len(group) != 1 or tuple(id(p) for p in group[0]["params"]) != tuple(
            id(p) for p in self._raw_model.parameters() if p.requires_grad
        ):
            raise ValueError("private optimizer parameter groups changed")
        if (
            isinstance(spec, PrivateOptimizerSpec)
            and {k: v for k, v in group[0].items() if k not in {"params", "lr"}}
            != self._optimizer_options
        ):
            raise ValueError("private optimizer options changed")
        for current, initial in zip(group, self._initial_learning_rates, strict=True):
            expected_lr = initial
            if self.specification.schedule == "linear_decay":
                expected_lr *= 1 - self._logical_steps / self.policy.max_logical_steps
            if current["lr"] != expected_lr:
                raise ValueError("private optimizer learning rate changed")
        if (
            self._optimizer.noise_multiplier != self.policy.noise_multiplier
            or self._optimizer.max_grad_norm != self.policy.max_grad_norm
        ):
            raise ValueError("private mechanism changed")
        if (
            1 / len(self._loader) != self._sample_rate
            or len(self._loader.dataset) != self._population_size
            or self._optimizer.expected_batch_size
            != int(len(self._loader.dataset) * self._sample_rate)
        ):
            raise ValueError("private sampling changed")
        if (
            self._optimizer.step_hook is not self._accountant_hook
            or self._optimizer.generator is not self._noise_stream
            or self._loader.batch_sampler.generator is not self._sampling_stream
            or self._loader.batch_sampler.sample_rate != self._sample_rate
            or self._loader.batch_sampler.num_samples != len(self._loader.dataset)
            or self._optimizer.secure_mode
            != (self.policy.randomness_profile == "secure")
            or self._optimizer.loss_reduction != "mean"
        ):
            raise ValueError("private randomness, sampling or accountant hook changed")
        if tuple(id(p) for p in self._raw_model.parameters()) != self._parameter_ids:
            raise ValueError("private model parameters changed")
        if self._model_guard is not None:
            self._model_guard.check()
            assert self._task is not None and isinstance(
                self.specification, TaskTrainingSpec
            )
            self._task.check_definition(self.specification)
        if any(
            s != self.policy.noise_multiplier or q != self._sample_rate
            for s, q, _ in self.accounting_history
        ):
            raise ValueError("accountant mechanism changed")
        if self._steps != self._logical_steps:
            raise ValueError("trainer and logical step count disagree")
        if sum(n for _, _, n in self.accounting_history) != self._logical_steps:
            raise ValueError("accountant and logical step count disagree")

    @property
    def accounting_history(self) -> tuple[tuple[float, float, int], ...]:
        return tuple(
            (float(s), float(q), int(n)) for s, q, n in self._engine.accountant.history
        )

    @property
    def physical_steps(self) -> int:
        return self._physical_steps

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    def accounting(self) -> dict[str, float | int | str]:
        return {
            "privacy_epsilon": privacy_epsilon(self.accounting_history, self.policy),
            "rdp_epsilon": privacy_epsilon(
                self.accounting_history, self.policy, accountant="rdp"
            ),
            "delta": self.policy.delta,
            "logical_steps": self._logical_steps,
            "reserved_steps": self.policy.max_logical_steps,
            "randomness_profile": self.policy.randomness_profile,
        }

    def train_local_steps(self, step_count: int) -> None:
        self._check_execution()
        if (
            step_count <= 0
            or self._logical_steps + step_count > self.policy.max_logical_steps
        ):
            raise ValueError("private horizon exhausted or invalid step count")
        try:
            super().train_local_steps(step_count)
        except Exception:
            self._failed = True
            raise ValueError("private local training failed") from None

    def _owned_step(self, _: nn.Module) -> None:
        self._check_execution()
        if self._logical_steps >= self.policy.max_logical_steps:
            raise ValueError("private horizon exhausted")
        self._private_model.train()
        manager = opacus_module("opacus.utils.batch_memory_manager").BatchMemoryManager
        # Each logical call consumes one independent Poisson draw. No partially
        # consumed loader iterator survives a boundary or research checkpoint.
        before = self._logical_steps
        cpu_rng = torch.get_rng_state() if self._task is not None else None
        device = next(self._raw_model.parameters()).device
        cuda_rng = (
            torch.cuda.get_rng_state(device)
            if self._task is not None and device.type == "cuda"
            else None
        )
        expected_parameters = (
            {
                name: p.detach().clone()
                for name, p in self._raw_model.named_parameters()
                if p.requires_grad
            }
            if self._task is not None
            else {}
        )
        with manager(
            data_loader=self._loader,
            max_physical_batch_size=self._physical_limit,
            optimizer=self._optimizer,
        ) as physical_loader:
            for batch in physical_loader:
                self._optimizer.zero_grad()
                device = next(self._raw_model.parameters()).device
                if self._task is not None:
                    if any(
                        not torch.equal(p.detach(), expected_parameters[name])
                        for name, p in self._raw_model.named_parameters()
                        if p.requires_grad
                    ):
                        raise ValueError("task data callback mutated model parameters")
                    batch = self._task.batch_adapter.to_device(batch, device)
                    count = self._task.batch_adapter.size(batch)
                    if count:
                        losses = self._task.loss_per_record(self._private_model, batch)
                        validate_record_losses(losses, count)
                        cast(Any, losses.mean()).backward()
                        if any(
                            not torch.equal(p.detach(), expected_parameters[name])
                            for name, p in self._raw_model.named_parameters()
                            if p.requires_grad
                        ):
                            raise ValueError(
                                "task forward/backward mutated model parameters"
                            )
                    ensure_grad_samples(self._raw_model, count)
                    assert self._model_guard is not None
                    self._model_guard.check()
                else:
                    assert isinstance(self.specification, PrivateTrainingSpec)
                    inputs, targets = batch
                    prediction = self._private_model(inputs.to(device))
                    targets = targets.to(device)
                    if self.specification.loss == "mse":
                        loss = (
                            (prediction - targets)
                            .square()
                            .reshape(len(inputs), -1)
                            .mean()
                            if len(inputs)
                            else prediction.sum()
                        )
                    else:
                        loss = (
                            nn.functional.cross_entropy(
                                prediction, targets, reduction="mean"
                            )
                            if len(inputs)
                            else prediction.sum()
                        )
                    cast(Any, loss).backward()
                self._optimizer.step()
                if self._task is not None:
                    expected_parameters = {
                        name: p.detach().clone()
                        for name, p in self._raw_model.named_parameters()
                        if p.requires_grad
                    }
                if (
                    cpu_rng is not None
                    and not torch.equal(cpu_rng, torch.get_rng_state())
                ) or (
                    cuda_rng is not None
                    and not torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
                ):
                    raise ValueError(
                        "task data/model callbacks consumed undeclared RNG"
                    )
                self._physical_steps += 1
                completed = sum(n for _, _, n in self.accounting_history)
                if completed == self._logical_steps + 1:
                    self._logical_steps = completed
                    if self.specification.schedule == "linear_decay":
                        for group, initial in zip(
                            self._base_optimizer.param_groups,
                            self._initial_learning_rates,
                            strict=True,
                        ):
                            group["lr"] = initial * (
                                1 - completed / self.policy.max_logical_steps
                            )
                    self._optimizer.zero_grad()
                    break
                if completed != self._logical_steps:
                    raise ValueError("invalid logical mechanism boundary")
        if self._logical_steps != before + 1:
            raise ValueError("private logical step did not complete")
        return None

    def weights(self) -> dict[str, np.ndarray]:
        self._check_execution()
        return super().weights()

    def evaluate(self) -> EvaluationResult | None:
        self._check_execution()
        try:
            return super().evaluate()
        except Exception:
            raise ValueError("public evaluation failed") from None

    def export_model(self, path: Path) -> None:
        """Release only sanitized model tensors; no training state or metadata."""
        if self._task is not None:
            self._check_execution()
            save_safetensors(
                {
                    name: value.detach().cpu().numpy().copy()
                    for name, value in registered_model_state(self._raw_model).items()
                },
                str(path),
            )
            return
        save_safetensors(self.weights(), str(path))

    def _save_private_state(self) -> dict[str, np.ndarray]:
        state: dict[str, Any] = {
            "version": 1,
            "specification": canonical_hash(self.specification),
            "policy": canonical_hash(self.policy),
            "sample_rate": self._sample_rate,
            "optimizer": self._base_optimizer.state_dict(),
            "accountant": list(self.accounting_history),
            "logical_steps": self._logical_steps,
            "physical_steps": self._physical_steps,
            "sampling_rng": self._sampling_generator.get_state()
            if self._sampling_generator is not None
            else None,
            "noise_rng": self._noise_generator.get_state()
            if self._noise_generator is not None
            else None,
        }
        return encode_state(state)

    def _load_private_state(self, arrays: dict[str, np.ndarray]) -> None:
        state = decode_state(arrays)
        self._validate_restore(state)
        # Restore through Opacus so a factory delegate rebinds replacement groups.
        self._optimizer.load_state_dict(state["optimizer"])
        self._engine.accountant.history = list(state["accountant"])
        self._logical_steps = state["logical_steps"]
        self._physical_steps = state["physical_steps"]
        assert (
            self._sampling_generator is not None and self._noise_generator is not None
        )
        self._sampling_generator.set_state(state["sampling_rng"])
        self._noise_generator.set_state(state["noise_rng"])

    def _validate_restore(self, state: Any) -> None:
        expected = {
            "version",
            "specification",
            "policy",
            "sample_rate",
            "optimizer",
            "accountant",
            "logical_steps",
            "physical_steps",
            "sampling_rng",
            "noise_rng",
        }
        if (
            not isinstance(state, dict)
            or set(cast(dict[str, Any], state)) != expected
            or state["version"] != 1
        ):
            raise ValueError("research checkpoint schema mismatch")
        if (
            state["specification"] != canonical_hash(self.specification)
            or state["policy"] != canonical_hash(self.policy)
            or state["sample_rate"] != self._sample_rate
        ):
            raise ValueError(
                "research checkpoint mechanism or optimizer identity mismatch"
            )
        state = cast(dict[str, Any], state)
        steps: Any = state["logical_steps"]
        if (
            type(steps) is not int
            or not 0 <= steps <= self.policy.max_logical_steps
            or type(state["physical_steps"]) is not int
            or state["physical_steps"] < steps
        ):
            raise ValueError("invalid research checkpoint counters")
        history = state["accountant"]
        if (
            any(
                s != self.policy.noise_multiplier
                or q != self._sample_rate
                or type(n) is not int
                or n <= 0
                for s, q, n in history
            )
            or sum(n for _, _, n in history) != steps
        ):
            raise ValueError("research checkpoint accountant history mismatch")
        spec = self.specification.optimizer
        if isinstance(spec, FactoryOptimizerSpec):
            assert self._optimizer_registration is not None
            self._optimizer_registration.validate_checkpoint(
                self._base_optimizer,
                self._raw_model,
                spec,
                state["optimizer"],
                logical_steps=steps,
                horizon=self.policy.max_logical_steps,
                schedule=self.specification.schedule,
            )
        else:
            validate_builtin_optimizer_checkpoint(
                state["optimizer"],
                spec=spec,
                parameters=tuple(
                    p for p in self._raw_model.parameters() if p.requires_grad
                ),
                initial_groups=self._initial_optimizer_groups,
                logical_steps=steps,
                horizon=self.policy.max_logical_steps,
                schedule=self.specification.schedule,
            )
        for key, generator in (
            ("sampling_rng", self._sampling_generator),
            ("noise_rng", self._noise_generator),
        ):
            if (
                generator is None
                or not isinstance(state[key], torch.Tensor)
                or state[key].dtype != torch.uint8
                or state[key].shape != generator.get_state().shape
            ):
                raise ValueError("research checkpoint RNG mismatch")
            probe = torch.Generator(device=generator.device)
            probe.set_state(state[key])

    def load_checkpoint_tensors(self, state: dict[str, np.ndarray]) -> None:
        self._check_execution()
        if self.policy.randomness_profile == "secure":
            raise ValueError("secure private runs are fresh-start only; resume refused")
        payload = {
            k.removeprefix("application."): v
            for k, v in state.items()
            if k.startswith("application.")
        }
        decoded = decode_state(payload)
        self._validate_restore(decoded)
        if (
            "meta.steps" not in state
            or int(state["meta.steps"][0]) != decoded["logical_steps"]
        ):
            raise ValueError("checkpoint counters disagree")
        if "meta.loss" not in state or state["meta.loss"].size:
            raise ValueError("private checkpoint contains raw loss")
        if isinstance(self.specification, TaskTrainingSpec):
            immutable = {
                **self.specification.public_buffer_hashes,
                **self.specification.frozen_parameter_hashes,
            }
            persistent = registered_model_state(self._raw_model)
            for name, expected in immutable.items():
                if name not in persistent:
                    continue
                value = state.get(f"model.{name}")
                if (
                    value is None
                    or tensor_digest(torch.tensor(value.copy())) != expected
                ):
                    raise ValueError("checkpoint changed immutable public model state")
        super().load_checkpoint_tensors(state)
