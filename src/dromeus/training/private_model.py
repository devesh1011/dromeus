"""Model capabilities and public-fixture checks for owned private execution.

Application code is trusted. These checks validate a concrete gradient backend
and immutable state contract, not arbitrary Python or every future input branch.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import torch
from torch import nn

from dromeus.training.model_state import model_state_alias_groups
from dromeus.training.privacy import opacus_module


class ModelBatchAdapter(Protocol):
    def size(self, batch: Any) -> int: ...

    def slice(self, batch: Any, start: int, end: int) -> Any: ...

    def to_device(self, batch: Any, device: torch.device) -> Any: ...


@dataclass(frozen=True)
class ModelCompatibility:
    """The public state and unique parameter semantics of one prepared model."""

    trainable_parameter_names: tuple[str, ...]
    parameter_aliases: tuple[tuple[str, ...], ...]
    public_buffer_hashes: Mapping[str, str]
    frozen_parameter_hashes: Mapping[str, str]
    gradient_backend: Literal["opacus_hooks_v1"] = "opacus_hooks_v1"


def tensor_digest(value: torch.Tensor) -> str:
    """Hash public tensor values without binding their physical device."""
    tensor = value.detach().cpu().contiguous()
    metadata = json.dumps([str(tensor.dtype), list(tensor.shape)]).encode()
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(metadata + b"\x00" + raw).hexdigest()


def registered_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return load_state_dict-compatible registered tensors, without callbacks.

    Nonpersistent buffers stay local but are still hash-bound by compatibility.
    Parameter aliases remain named so standard model loading preserves tying.
    """
    state: dict[str, torch.Tensor] = dict(
        model.named_parameters(remove_duplicate=False)
    )
    for path, module in cast(
        Iterable[tuple[str, nn.Module]], model.named_modules(remove_duplicate=False)
    ):
        omitted = cast(set[str], getattr(module, "_non_persistent_buffers_set"))
        for name, value in module.named_buffers(recurse=False, remove_duplicate=False):
            if name not in omitted:
                state[f"{path}.{name}" if path else name] = value
    return state


def describe_private_model(model: nn.Module) -> ModelCompatibility:
    """Inspect structure without running the model or reading a local dataset."""
    registered = registered_model_state(model)
    state: Mapping[str, Any] = model.state_dict()
    if set(state) != set(registered):
        raise ValueError(
            "private serialization must contain exactly registered parameters "
            "and buffers"
        )
    for name, actual in state.items():
        expected = registered[name]
        if (
            not isinstance(actual, torch.Tensor)
            or actual.shape != expected.shape
            or actual.dtype != expected.dtype
            or actual.device != expected.device
            or not torch.equal(actual, expected.detach())
        ):
            raise ValueError(
                f"{name}: serialized state differs from its registered tensor"
            )
    if len({p.device for p in model.parameters()}) != 1:
        raise ValueError("private model parameters require one device")
    names: list[str] = []
    frozen: dict[str, str] = {}
    for name, parameter in model.named_parameters():
        if (
            parameter.dtype != torch.float32
            or parameter.is_sparse
            or parameter.ndim == 0
            or not bool(torch.isfinite(parameter).all())
        ):
            raise ValueError(f"{name}: private parameters require finite dense FP32")
        if parameter.requires_grad:
            names.append(name)
        else:
            frozen[name] = tensor_digest(parameter)
    if not names:
        raise ValueError("private model requires trainable parameters")
    aliases = model_state_alias_groups(model)
    parameter_objects = {
        name: id(parameter)
        for name, parameter in model.named_parameters(remove_duplicate=False)
    }
    for group in aliases:
        parameter_names = [name for name in group if name in parameter_objects]
        if parameter_names and (
            len(parameter_names) != len(group)
            or len({parameter_objects[name] for name in parameter_names}) != 1
        ):
            raise ValueError("parameter aliases must share one Parameter object")
    return ModelCompatibility(
        trainable_parameter_names=tuple(names),
        parameter_aliases=aliases,
        public_buffer_hashes={
            name: tensor_digest(value)
            for name, value in model.named_buffers(remove_duplicate=False)
        },
        frozen_parameter_hashes=frozen,
    )


def _backend_structure(model: nn.Module) -> None:
    samplers = opacus_module("opacus.grad_sample").GradSampleModule.GRAD_SAMPLERS
    for path, module in cast(Iterable[tuple[str, nn.Module]], model.named_modules()):
        label = path or "<root>"
        if isinstance(
            module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
        ):
            raise ValueError(f"{label}: batch-dependent normalization is unsupported")
        if (
            isinstance(
                module,
                (
                    nn.Dropout,
                    nn.Dropout1d,
                    nn.Dropout2d,
                    nn.Dropout3d,
                    nn.AlphaDropout,
                    nn.FeatureAlphaDropout,
                ),
            )
            and module.p
        ):
            raise ValueError(f"{label}: stochastic layer requires a model RNG profile")
        if isinstance(module, nn.RReLU):
            raise ValueError(f"{label}: stochastic layer requires a model RNG profile")
        if isinstance(module, nn.Embedding) and (
            module.sparse or module.scale_grad_by_freq or module.max_norm is not None
        ):
            raise ValueError(f"{label}: embedding gradient/state options unsupported")
        direct = any(p.requires_grad for p in module.parameters(recurse=False))
        if direct and type(module) not in samplers:
            raise ValueError(f"{label}: no registered Opacus gradient sampler")
        if direct and any(True for _ in module.buffers()):
            raise ValueError(
                f"{label}: Opacus hooks cannot handle trainable-layer buffers"
            )
        if (
            getattr(module, "_forward_hooks")
            or getattr(module, "_forward_pre_hooks")
            or getattr(module, "_backward_hooks")
        ):
            raise ValueError(
                f"{label}: pre-existing model hooks need an explicit profile"
            )
    opacus_module("opacus.validators").ModuleValidator.validate(model, strict=True)


def ensure_grad_samples(model: nn.Module, batch_size: int) -> None:
    """Require one full gradient per record; unused parameters contribute zeros.

    Invoke after backward and before the Opacus optimizer. Missing hook results
    with an ordinary gradient or an incomplete backward cannot be guessed.
    """
    if batch_size < 0:
        raise ValueError("invalid record count")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if getattr(parameter, "_forward_counter", 0):
            raise ValueError(f"{name}: unresolved per-record backward coverage")
        sample = getattr(parameter, "grad_sample", None)
        if sample is None:
            if parameter.grad is not None:
                raise ValueError(f"{name}: missing per-record gradient sampler result")
            sample = parameter.new_zeros((batch_size, *parameter.shape))
            cast(Any, parameter).grad_sample = sample
        if (
            not isinstance(sample, torch.Tensor)
            or sample.shape != (batch_size, *parameter.shape)
            or not bool(torch.isfinite(sample).all())
        ):
            raise ValueError(f"{name}: invalid per-record gradient shape or values")
        if (
            batch_size
            and parameter.grad is not None
            and not torch.allclose(sample.mean(0), parameter.grad, rtol=1e-4, atol=1e-6)
        ):
            raise ValueError(
                f"{name}: per-record gradients do not cover aggregate gradient"
            )


def _losses(
    loss: Callable[[nn.Module, Any], torch.Tensor],
    model: nn.Module,
    batch: Any,
    count: int,
) -> torch.Tensor:
    result: Any = loss(model, batch)
    if (
        not isinstance(result, torch.Tensor)
        or not result.is_floating_point()
        or result.shape != (count,)
        or not bool(torch.isfinite(result).all())
        or not result.requires_grad
    ):
        raise ValueError(
            "loss callback must return finite differentiable record losses [B]"
        )
    return result


def _state_hashes(model: nn.Module) -> dict[str, str]:
    return {name: tensor_digest(value) for name, value in model.state_dict().items()}


def validate_private_model(
    model: nn.Module,
    loss_per_record: Callable[[nn.Module, Any], torch.Tensor],
    batch_adapter: ModelBatchAdapter,
    public_batch: Any,
) -> ModelCompatibility:
    """Check supported hooks against single-record autograd on PUBLIC fixtures.

    The caller must supply synthetic/public representative inputs. No actual
    training records are sampled. All probes use disposable model copies.
    """
    report = describe_private_model(model)
    probe = copy.deepcopy(model).train()
    _backend_structure(probe)
    device = next(probe.parameters()).device
    batch = batch_adapter.to_device(copy.deepcopy(public_batch), device)
    count = batch_adapter.size(batch)
    if count < 2:
        raise ValueError("public compatibility fixture requires at least two records")
    devices = [device.index or 0] if device.type == "cuda" else []
    with cast(Any, torch.random).fork_rng(devices=devices):
        rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(device).clone() if devices else None
        original = _state_hashes(probe)
        references: dict[str, list[torch.Tensor]] = {
            name: [] for name in report.trainable_parameter_names
        }
        named = [(n, p) for n, p in probe.named_parameters() if p.requires_grad]
        record_values: list[torch.Tensor] = []
        for index in range(count):
            record = batch_adapter.slice(batch, index, index + 1)
            value = _losses(loss_per_record, probe, record, 1)
            record_values.append(value.detach())
            gradients = cast(
                tuple[torch.Tensor | None, ...],
                torch.autograd.grad(
                    value.sum(), [p for _, p in named], allow_unused=True
                ),
            )
            for (name, parameter), gradient in zip(named, gradients, strict=True):
                references[name].append(
                    torch.zeros_like(parameter)
                    if gradient is None
                    else gradient.detach()
                )
        if _state_hashes(probe) != original:
            raise ValueError("model/loss mutated public model state during forward")
        hooked = copy.deepcopy(model).train()
        wrapper: Any = opacus_module("opacus.grad_sample").GradSampleModule(
            hooked, batch_first=True, loss_reduction="mean", strict=True
        )
        try:
            values = _losses(loss_per_record, wrapper, batch, count)
            if not torch.allclose(
                values.detach(), torch.cat(record_values), rtol=1e-4, atol=1e-6
            ):
                raise ValueError(
                    "loss depends on other records or changes with batch size"
                )
            cast(Any, values.mean()).backward()
            ensure_grad_samples(hooked, count)
            for name, parameter in hooked.named_parameters():
                if not parameter.requires_grad:
                    continue
                observed = cast(torch.Tensor, getattr(parameter, "grad_sample"))
                expected = torch.stack(references[name])
                if not torch.allclose(observed, expected, rtol=1e-4, atol=1e-6):
                    raise ValueError(
                        f"{name}: per-record gradient differs from autograd"
                    )
            if _state_hashes(hooked) != original:
                raise ValueError("model/loss mutated public model state during forward")
        finally:
            wrapper.remove_hooks()
        if not torch.equal(rng, torch.get_rng_state()) or (
            cuda_rng is not None
            and not torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
        ):
            raise ValueError(
                "model/loss consumed global randomness without an RNG profile"
            )
    return report


class ModelStateGuard:
    """Preserve declared immutable public state and parameter objects during a run."""

    def __init__(self, model: nn.Module, report: ModelCompatibility) -> None:
        if describe_private_model(model) != report:
            raise ValueError("model differs from declared compatibility report")
        self._model = model
        self._report = copy.deepcopy(report)
        self._parameters = tuple(
            (name, id(p), p.requires_grad, tuple(p.shape), p.dtype, p.device)
            for name, p in model.named_parameters()
        )
        self._buffers = tuple((name, id(b)) for name, b in model.named_buffers())

    def check(self) -> None:
        actual = tuple(
            (name, id(p), p.requires_grad, tuple(p.shape), p.dtype, p.device)
            for name, p in self._model.named_parameters()
        )
        if actual != self._parameters:
            raise ValueError(
                "private model parameter identity/schema/trainability changed"
            )
        if (
            tuple((name, id(b)) for name, b in self._model.named_buffers())
            != self._buffers
        ):
            raise ValueError("private model buffer identity changed")
        if describe_private_model(self._model) != self._report:
            raise ValueError(
                "private model aliases, public buffers or frozen parameters changed"
            )
