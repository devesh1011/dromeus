"""Frozen CIFAR-10 ResNet-18 with affine GroupNorm."""

from __future__ import annotations

import hashlib

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from dromeus.manifests.models import RESNET18_GROUPNORM_MODEL_ID, TensorSchema
from dromeus.training.model_state import floating_model_state
from dromeus.training.model_state import tensor_schema_for_model as _tensor_schema

MODEL_ID = RESNET18_GROUPNORM_MODEL_ID
_TORCHVISION_COMMIT = "0fba2e84fe255a2fcd81bd0b10c74d7fca99a89f"


def model_definition(*, input_channels: int = 3, num_classes: int = 10) -> str:
    """Return the complete architecture identity for one model configuration."""
    return (
        f"resnet18-groupnorm-cifar10-v1:torchvision={_TORCHVISION_COMMIT}:"
        f"input-channels={input_channels}:"
        "stem=conv3x3-64-stride1-padding1:no-maxpool:"
        "basic-blocks=2,2,2,2:channels=64,128,256,512:"
        "downsample=conv1x1:groupnorm=32-affine:gap:"
        f"linear={num_classes}"
    )


MODEL_DEFINITION = model_definition()
MODEL_DEFINITION_HASH = hashlib.sha256(MODEL_DEFINITION.encode()).hexdigest()
PARAMETER_COUNT = 11_173_962
RAW_FP32_PARAMETER_BYTES = 44_695_848
TENSOR_SCHEMA_HASH = "2c0b3aacfde1526af9458f8804895d442ad32be5cddd982b1d27185c07480706"


def _conv3x3(in_channels: int, out_channels: int, *, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def _conv1x1(in_channels: int, out_channels: int, *, stride: int) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class _BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride=stride)
        self.norm1 = nn.GroupNorm(32, out_channels, affine=True)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels, affine=True)
        self.downsample = (
            nn.Sequential(
                _conv1x1(in_channels, out_channels, stride=stride),
                nn.GroupNorm(32, out_channels, affine=True),
            )
            if stride != 1 or in_channels != out_channels
            else None
        )

    def forward(self, images: Tensor) -> Tensor:
        residual = images
        output = F.relu(self.norm1(self.conv1(images)), inplace=False)
        output = self.norm2(self.conv2(output))
        if self.downsample is not None:
            residual = self.downsample(images)
        return F.relu(output + residual, inplace=False)


class ResNet18GroupNorm(nn.Module):
    """TorchVision ResNet-18 topology with the frozen CIFAR/GN adaptation."""

    def __init__(self, *, input_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        self._in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(32, 64, affine=True),
            nn.ReLU(inplace=False),
        )
        self.layer1 = self._make_layer(64, block_count=2, stride=1)
        self.layer2 = self._make_layer(128, block_count=2, stride=2)
        self.layer3 = self._make_layer(256, block_count=2, stride=2)
        self.layer4 = self._make_layer(512, block_count=2, stride=2)
        self.classifier = nn.Linear(512, num_classes)
        self._initialize()

    def _make_layer(
        self,
        out_channels: int,
        *,
        block_count: int,
        stride: int,
    ) -> nn.Sequential:
        blocks: list[nn.Module] = [
            _BasicBlock(self._in_channels, out_channels, stride=stride)
        ]
        self._in_channels = out_channels
        blocks.extend(
            _BasicBlock(self._in_channels, out_channels, stride=1)
            for _ in range(1, block_count)
        )
        return nn.Sequential(*blocks)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, images: Tensor) -> Tensor:
        features = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(images)))))
        features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        return self.classifier(features)


def build_model(
    *,
    seed: int,
    input_channels: int = 3,
    num_classes: int = 10,
) -> ResNet18GroupNorm:
    """Construct a reproducibly initialized model without changing caller RNG."""
    with torch.random.fork_rng(devices=[]):  # pyright: ignore[reportUnknownMemberType]
        torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
        return ResNet18GroupNorm(
            input_channels=input_channels,
            num_classes=num_classes,
        )


def tensor_schema_for_model(model: nn.Module | None = None) -> TensorSchema:
    return _tensor_schema(model or build_model(seed=0))


__all__ = [
    "MODEL_DEFINITION",
    "MODEL_DEFINITION_HASH",
    "MODEL_ID",
    "PARAMETER_COUNT",
    "RAW_FP32_PARAMETER_BYTES",
    "ResNet18GroupNorm",
    "TENSOR_SCHEMA_HASH",
    "build_model",
    "floating_model_state",
    "model_definition",
    "tensor_schema_for_model",
]
