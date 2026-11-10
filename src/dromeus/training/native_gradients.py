"""Opt-in native FP32 per-record reductions for a public physical cap of one.

Selection belongs to the caller's frozen public configuration. This module does
not select a backend from observed record counts or alter coverage/DP guards.
Optional runtime dependencies load only when the backend is explicitly used.
"""

from __future__ import annotations

import importlib
import math
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from torch import nn

NATIVE_B1_BACKEND = "opacus_native_b1"
Sampler = Callable[[Any, list[Any], Any], dict[Any, Any]]


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _runtime() -> tuple[Any, Any]:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        raise ValueError("native B1 requires pinned Torch 2.13.0") from None
    _require(
        torch.__version__ in {"2.13.0", "2.13.0+cpu", "2.13.0+cu130"},
        "native B1 requires Torch 2.13.0 CPU or cu130",
    )
    from dromeus.training.privacy import opacus_module

    base = opacus_module("opacus.grad_sample").GradSampleModule
    return torch, base


def _leaf_signature(module: Any, torch: Any) -> tuple[Any, ...]:
    nn = torch.nn
    kind = cast(type[Any], type(module))
    _require(
        kind in (nn.Linear, nn.LayerNorm, nn.Embedding),
        "native B1 supports exact Linear/LayerNorm/Embedding leaves only",
    )
    expected: Any = getattr(kind, "forward")
    namespace = {
        nn.Linear: "torch.nn.modules.linear",
        nn.LayerNorm: "torch.nn.modules.normalization",
        nn.Embedding: "torch.nn.modules.sparse",
    }[kind]
    _require(
        "forward" not in module.__dict__
        and getattr(module.forward, "__func__", None) is expected
        and expected.__module__ == namespace
        and expected.__qualname__ == kind.__name__ + ".forward"
        and getattr(module, "_compiled_call_impl", None) is None,
        "native B1 forbids foreign leaf forward overrides",
    )
    parameters = tuple(module.named_parameters(recurse=False))
    _require(
        bool(parameters)
        and all(p.dtype == torch.float32 for _, p in parameters)
        and not tuple(module.buffers(recurse=False)),
        "native B1 requires FP32 leaves without trainable-layer buffers",
    )
    if kind is nn.Linear:
        options = (module.in_features, module.out_features, module.bias is not None)
    elif kind is nn.LayerNorm:
        _require(
            math.isfinite(module.eps) and module.eps > 0,
            "native B1 LayerNorm epsilon must be finite and positive",
        )
        options = (
            tuple(module.normalized_shape),
            module.eps,
            module.elementwise_affine,
            module.bias is not None,
        )
    else:
        _require(
            module.max_norm is None
            and not module.sparse
            and not module.scale_grad_by_freq,
            "native B1 forbids mutating/sparse/frequency Embedding options",
        )
        options = (
            module.num_embeddings,
            module.embedding_dim,
            module.padding_idx,
            module.norm_type,
        )
    return (
        kind,
        id(module),
        expected,
        options,
        tuple(
            (n, id(p), tuple(p.shape), p.dtype, p.device, p.requires_grad)
            for n, p in parameters
        ),
    )


def _model_signature(model: Any, torch: Any) -> tuple[Any, ...]:
    _require(isinstance(model, torch.nn.Module), "native B1 requires a model module")
    leaves: list[tuple[Any, ...]] = []
    for path, module in model.named_modules():
        if any(p.requires_grad for p in module.parameters(recurse=False)):
            leaves.append((path, _leaf_signature(module, torch)))
    _require(bool(leaves), "native B1 requires trainable supported leaves")
    return tuple(leaves)


def validate_native_b1_model(raw_model: nn.Module) -> None:
    """Inspect native leaves without forward execution or record/weight copies."""
    torch, _ = _runtime()
    _model_signature(raw_model, torch)


def _native_b1_sampler(
    layer: Any, activations: list[Any], backprops: Any
) -> dict[Any, Any]:
    torch, _ = _runtime()
    _leaf_signature(layer, torch)
    _require(
        len(activations) == 1
        and isinstance(backprops, torch.Tensor)
        and backprops.ndim >= 1
        and backprops.shape[0] == 1
        and backprops.dtype == torch.float32,
        "native B1 sampler requires exactly one nonempty physical record",
    )
    x = activations[0]
    _require(
        isinstance(x, torch.Tensor)
        and x.ndim >= 1
        and x.shape[0] == 1
        and x.device == backprops.device,
        "native B1 activation batch/device differs",
    )
    x = x.detach()
    if type(layer) is torch.nn.Linear:
        _require(x.dtype == torch.float32, "native B1 Linear activation must be FP32")
        dy = backprops.detach()[0].reshape(-1, layer.out_features)
        values = x[0].reshape(-1, layer.in_features)
        result: dict[Any, Any] = {}
        if layer.weight.requires_grad:
            result[layer.weight] = dy.T.mm(values).unsqueeze(0)
        if layer.bias is not None and layer.bias.requires_grad:
            result[layer.bias] = dy.sum(0).unsqueeze(0)
        return result
    parameters = tuple(
        (name, p)
        for name, p in layer.named_parameters(recurse=False)
        if p.requires_grad
    )
    with torch.enable_grad():
        # These views share immutable weight storage, but own their autograd
        # metadata. Functional calls trigger no module hooks or original .grad.
        views = {
            name: p.detach().requires_grad_(p.requires_grad)
            for name, p in layer.named_parameters(recurse=False)
        }
        if type(layer) is torch.nn.LayerNorm:
            _require(
                x.dtype == torch.float32, "native B1 LayerNorm activation must be FP32"
            )
            output = torch.nn.functional.layer_norm(
                x,
                layer.normalized_shape,
                views.get("weight"),
                views.get("bias"),
                layer.eps,
            )
        else:
            _require(
                x.dtype in (torch.int32, torch.int64),
                "native B1 Embedding indices must be integral",
            )
            output = torch.nn.functional.embedding(
                x,
                views["weight"],
                layer.padding_idx,
                None,
                layer.norm_type,
                False,
                False,
            )
        gradients = torch.autograd.grad(
            output,
            tuple(views[name] for name, _ in parameters),
            grad_outputs=backprops.detach(),
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
    return {
        parameter: gradient.detach().unsqueeze(0)
        for (_, parameter), gradient in zip(parameters, gradients, strict=True)
    }


@dataclass(frozen=True, slots=True, eq=False)
class NativeB1SamplerToken:
    """Immutable, tensor-free identity certificate for one wrapper instance."""

    wrapper_ref: weakref.ReferenceType[Any] = field(repr=False)
    model_ref: weakref.ReferenceType[Any] = field(repr=False)
    mapping: Mapping[Any, Sampler] = field(repr=False)
    global_mapping: Any = field(repr=False)
    global_snapshot: Mapping[Any, Any] = field(repr=False)
    model_signature: tuple[Any, ...] = field(repr=False)


def _wrapper_model(wrapper: Any, torch: Any, base: Any) -> Any:
    _require(
        type(wrapper) is base
        and wrapper.batch_first is True
        and wrapper.loss_reduction == "mean"
        and not wrapper.force_functorch,
        "native B1 requires the standard batch-first mean Opacus wrapper",
    )
    model = getattr(wrapper, "_module", None)
    _require(isinstance(model, torch.nn.Module), "native B1 wrapper model missing")
    return model


def install_native_b1_samplers(wrapper: Any) -> NativeB1SamplerToken:
    """Install once after caller selected this backend with public physical cap1."""
    torch, base = _runtime()
    model = _wrapper_model(wrapper, torch, base)
    signature = _model_signature(model, torch)
    _require(
        "GRAD_SAMPLERS" not in wrapper.__dict__,
        "native B1 refuses an existing instance sampler override",
    )
    global_mapping = base.GRAD_SAMPLERS
    mapping: Mapping[Any, Sampler] = MappingProxyType(
        {
            leaf: _native_b1_sampler
            for leaf in (torch.nn.Linear, torch.nn.LayerNorm, torch.nn.Embedding)
        }
    )
    token = NativeB1SamplerToken(
        weakref.ref(wrapper),
        weakref.ref(model),
        mapping,
        global_mapping,
        MappingProxyType(dict(global_mapping)),
        signature,
    )
    wrapper.GRAD_SAMPLERS = mapping
    check_native_b1_samplers(wrapper, token)
    return token


def check_native_b1_samplers(wrapper: Any, token: NativeB1SamplerToken) -> None:
    """Fail closed on wrapper, leaf, callback, or shared registry replacement."""
    _require(type(token) is NativeB1SamplerToken, "native B1 identity token required")
    torch, base = _runtime()
    model = _wrapper_model(wrapper, torch, base)
    _require(
        token.wrapper_ref() is wrapper
        and token.model_ref() is model
        and wrapper.__dict__.get("GRAD_SAMPLERS") is token.mapping
        and set(token.mapping)
        == {torch.nn.Linear, torch.nn.LayerNorm, torch.nn.Embedding}
        and all(fn is _native_b1_sampler for fn in token.mapping.values()),
        "native B1 wrapper/sampler identity changed",
    )
    _require(
        base.GRAD_SAMPLERS is token.global_mapping
        and set(base.GRAD_SAMPLERS) == set(token.global_snapshot)
        and all(base.GRAD_SAMPLERS[k] is v for k, v in token.global_snapshot.items()),
        "native B1 global sampler registry changed",
    )
    _require(
        _model_signature(model, torch) == token.model_signature,
        "native B1 model/leaf configuration changed",
    )
