"""Owned Opacus trainer for fixed local tensor datasets and public evaluation.

Application factories and model code are trusted. Only the allowlisted dense FP32
profile is supported. Raw losses never leave this adapter, including research mode.
"""

from __future__ import annotations

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
    PrivacyPolicy,
    PrivacyReservation,
    PrivateOptimizerSpec,
    PrivateTrainingSpec,
)
from dromeus.training.base import EvaluationResult
from dromeus.training.model_state import model_state_alias_groups
from dromeus.training.privacy import (
    ReservationStore,
    opacus_module,
    privacy_epsilon,
    reserve_horizon,
)
from dromeus.training.private_state import decode_state, encode_state
from dromeus.training.trainer import PyTorchTrainer


def private_training_definition(specification: PrivateTrainingSpec) -> str:
    return canonical_json(specification).decode()


def build_private_optimizer(
    model: nn.Module, spec: PrivateOptimizerSpec
) -> torch.optim.Optimizer:
    parameters = list(model.parameters())
    common: dict[str, Any] = dict(
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
        foreach=False,
        fused=False,
        differentiable=False,
        maximize=False,
    )
    if spec.name == "sgd":
        return torch.optim.SGD(
            parameters,
            momentum=spec.momentum,
            dampening=spec.dampening,
            nesterov=spec.nesterov,
            **common,
        )
    factory = torch.optim.Adam if spec.name == "adam" else torch.optim.AdamW
    return factory(
        parameters,
        betas=spec.betas,
        eps=spec.numerical_epsilon,
        amsgrad=False,
        capturable=False,
        **common,
    )


class PrivateTrainer(PyTorchTrainer):
    """Retain private model/optimizer/loader and count complete logical mechanisms."""

    def __init__(
        self,
        *,
        model_factory: Callable[[], nn.Module],
        inputs: torch.Tensor,
        targets: torch.Tensor,
        policy: PrivacyPolicy,
        specification: PrivateTrainingSpec,
        run_id: str,
        local_data_identity: str,
        reservation_store: ReservationStore,
        max_physical_batch_size: int,
        research_seed: int | None = None,
        evaluate: Callable[[nn.Module], Mapping[str, float]] | None = None,
        optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer] | None = None,
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
        self.policy = policy
        self.specification = specification
        self._physical_limit = max_physical_batch_size
        self._failed = False
        self._physical_steps = 0
        self._logical_steps = 0
        self._reservation_store = reservation_store
        self._raw_model = model_factory()
        self._validate_model()
        spec = specification.optimizer
        self._base_optimizer = (
            optimizer_factory(self._raw_model)
            if optimizer_factory
            else build_private_optimizer(self._raw_model, spec)
        )
        self._validate_optimizer()
        self._sampling_generator = (
            None if secure else torch.Generator().manual_seed(int(research_seed or 0))
        )
        device = next(self._raw_model.parameters()).device
        self._noise_generator = (
            None
            if secure
            else torch.Generator(device=device).manual_seed(int(research_seed or 0) + 1)
        )
        loader = DataLoader(
            TensorDataset(inputs.detach().clone(), targets.detach().clone()),
            batch_size=specification.batch_size,
            num_workers=0,
            generator=self._sampling_generator,
        )
        try:
            self._engine = opacus_module().PrivacyEngine(
                accountant="prv", secure_mode=secure
            )
            self._private_model, self._optimizer, self._loader = (
                self._engine.make_private(
                    module=self._raw_model,
                    optimizer=self._base_optimizer,
                    data_loader=loader,
                    noise_multiplier=policy.noise_multiplier,
                    max_grad_norm=policy.max_grad_norm,
                    poisson_sampling=True,
                    clipping="flat",
                    grad_sample_mode="hooks",
                    noise_generator=self._noise_generator,
                )
            )
        except Exception:
            raise ValueError(
                "private preparation failed; verify model and secure RNG installation"
            ) from None
        self._accountant_hook = self._optimizer.step_hook
        self._noise_stream = self._optimizer.generator
        self._sampling_stream = self._loader.batch_sampler.generator
        self._sample_rate = float(self._loader.sample_rate)
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
            + str((len(inputs), self._sample_rate, max_physical_batch_size)).encode()
        ).hexdigest()
        super().__init__(
            model=self._raw_model,
            train_step=self._owned_step,
            save_state=self._save_private_state,
            load_state=self._load_private_state,
            state_identity=identity,
            evaluate=evaluate,
        )

    def _validate_model(self) -> None:
        model = self._raw_model
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
        if (
            tuple(name for name, _ in named)
            != self.specification.optimizer.parameter_names
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
            id(p) for p in self._raw_model.parameters()
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
        group = self._base_optimizer.param_groups
        if (
            len(group) != 1
            or tuple(id(p) for p in group[0]["params"]) != self._parameter_ids
        ):
            raise ValueError("private optimizer parameter groups changed")
        if {
            k: v for k, v in group[0].items() if k not in {"params", "lr"}
        } != self._optimizer_options:
            raise ValueError("private optimizer options changed")
        expected_lr = self.specification.optimizer.learning_rate
        if self.specification.schedule == "linear_decay":
            expected_lr *= 1 - self._logical_steps / self.policy.max_logical_steps
        if group[0]["lr"] != expected_lr:
            raise ValueError("private optimizer learning rate changed")
        if (
            self._optimizer.noise_multiplier != self.policy.noise_multiplier
            or self._optimizer.max_grad_norm != self.policy.max_grad_norm
        ):
            raise ValueError("private mechanism changed")
        if (
            self._loader.sample_rate != self._sample_rate
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
        with manager(
            data_loader=self._loader,
            max_physical_batch_size=self._physical_limit,
            optimizer=self._optimizer,
        ) as physical_loader:
            for inputs, targets in physical_loader:
                self._optimizer.zero_grad()
                device = next(self._raw_model.parameters()).device
                prediction = self._private_model(inputs.to(device))
                targets = targets.to(device)
                if self.specification.loss == "mse":
                    loss = (
                        (prediction - targets).square().reshape(len(inputs), -1).mean()
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
                self._physical_steps += 1
                completed = sum(n for _, _, n in self.accounting_history)
                if completed == self._logical_steps + 1:
                    self._logical_steps = completed
                    if self.specification.schedule == "linear_decay":
                        self._base_optimizer.param_groups[0]["lr"] = (
                            self.specification.optimizer.learning_rate
                            * (1 - completed / self.policy.max_logical_steps)
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
        self._base_optimizer.load_state_dict(state["optimizer"])
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
        optimizer = state["optimizer"]
        if (
            set(optimizer) != {"state", "param_groups"}
            or len(optimizer["param_groups"]) != 1
        ):
            raise ValueError("invalid optimizer checkpoint")
        group = optimizer["param_groups"][0]
        reference = self._base_optimizer.state_dict()["param_groups"][0]
        if group.keys() != reference.keys() or any(
            group[k] != reference[k] for k in reference if k != "lr"
        ):
            raise ValueError("optimizer checkpoint group mismatch")
        learning_rate = self.specification.optimizer.learning_rate
        if self.specification.schedule == "linear_decay":
            learning_rate *= 1 - steps / self.policy.max_logical_steps
        if group["lr"] != learning_rate:
            raise ValueError("checkpoint scheduler mismatch")
        spec = self.specification.optimizer
        keys: set[str] = (
            ({"momentum_buffer"} if spec.momentum else set())
            if spec.name == "sgd"
            else {"step", "exp_avg", "exp_avg_sq"}
        )
        ids: set[int] = (
            set(range(len(self._parameter_ids))) if steps and keys else set()
        )
        if set(optimizer["state"]) != ids:
            raise ValueError("optimizer checkpoint state coverage mismatch")
        for index, parameter in enumerate(self._raw_model.parameters()):
            values = optimizer["state"].get(index, {})
            if set(values) != keys and index in ids:
                raise ValueError("optimizer checkpoint state fields mismatch")
            for key, tensor in values.items():
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.dtype != torch.float32
                    or tensor.shape != (() if key == "step" else parameter.shape)
                ):
                    raise ValueError("optimizer checkpoint tensor mismatch")
                if key == "step" and float(tensor) != steps:
                    raise ValueError("optimizer checkpoint step mismatch")
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
        super().load_checkpoint_tensors(state)
