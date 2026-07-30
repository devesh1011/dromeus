"""PyTorch model construction and exchangeable-state schema."""

from __future__ import annotations

import hashlib
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from dromeus.manifests.models import RESNET32_MODEL_ID, TensorSchema
from dromeus.manifests.models import Tensor as TensorSpec

MODEL_ID = RESNET32_MODEL_ID


def model_definition(*, input_channels: int = 3, num_classes: int = 10) -> str:
    """Return the architecture identity for one ResNet-32 configuration."""
    return (
        f"resnet32:input-channels={input_channels}:conv3x3-16:bn:"
        "basic-blocks=5,5,5:channels=16,32,64:option-a-shortcut:gap:"
        f"linear={num_classes}"
    )


MODEL_DEFINITION = model_definition()
MODEL_DEFINITION_HASH = hashlib.sha256(MODEL_DEFINITION.encode()).hexdigest()
_TORCH_TO_SCHEMA_DTYPE: dict[
    torch.dtype,
    Literal["float16", "float32", "float64"],
] = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
}


class _BasicBlock(nn.Module):
    """Residual block used internally by ResNet-32."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self._stride = stride
        self._channel_padding = out_channels - in_channels

    def forward(self, images: Tensor) -> Tensor:
        residual = images
        output = F.relu(self.bn1(self.conv1(images)), inplace=False)
        output = self.bn2(self.conv2(output))
        if self._stride != 1:
            residual = residual[:, :, :: self._stride, :: self._stride]
        if self._channel_padding:
            left = self._channel_padding // 2
            residual = F.pad(
                residual,
                (0, 0, 0, 0, left, self._channel_padding - left),
            )
        return F.relu(output + residual, inplace=False)


class ResNet32(nn.Module):
    """ResNet-32 with configurable input channels and output classes."""

    def __init__(self, *, input_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
        )
        stages: list[nn.Module] = []
        in_channels = 16
        for stage_index, out_channels in enumerate((16, 32, 64)):
            blocks: list[nn.Module] = []
            for block_index in range(5):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                blocks.append(_BasicBlock(in_channels, out_channels, stride=stride))
                in_channels = out_channels
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.classifier = nn.Linear(64, num_classes)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, images: Tensor) -> Tensor:
        features = self.stages(self.stem(images))
        features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        return self.classifier(features)


def build_model(
    *,
    seed: int,
    input_channels: int = 3,
    num_classes: int = 10,
) -> ResNet32:
    """Construct a reproducibly initialized model without changing caller RNG."""
    with torch.random.fork_rng(devices=[]):  # pyright: ignore[reportUnknownMemberType]
        torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
        return ResNet32(
            input_channels=input_channels,
            num_classes=num_classes,
        )


def floating_model_state(model: nn.Module) -> dict[str, Tensor]:
    """Return exchangeable state, excluding integer BatchNorm counters."""
    return {
        name: value
        for name, value in model.state_dict().items()
        if value.is_floating_point()
    }


def tensor_schema_for_model(model: nn.Module | None = None) -> TensorSchema:
    """Derive the wire schema from parameters and floating-point model buffers."""
    target = model or build_model(seed=0)
    tensors: list[TensorSpec] = []
    for name, value in floating_model_state(target).items():
        dtype = _TORCH_TO_SCHEMA_DTYPE.get(value.dtype)
        if dtype is None:
            raise ValueError(f"unsupported model state dtype: {value.dtype}")
        tensors.append(
            TensorSpec(
                name=name,
                dtype=dtype,
                shape=tuple(value.shape),
            )
        )
    return TensorSchema(tensors=tuple(tensors))


__all__ = [
    "MODEL_DEFINITION",
    "MODEL_DEFINITION_HASH",
    "MODEL_ID",
    "ResNet32",
    "build_model",
    "floating_model_state",
    "model_definition",
    "tensor_schema_for_model",
]
