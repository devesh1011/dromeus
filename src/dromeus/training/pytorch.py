"""Deterministic CIFAR-10 data and PyTorch training primitives."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast
from urllib.request import urlopen

import numpy as np
import torch
from safetensors.torch import (
    load_file as _load_file,  # pyright: ignore[reportUnknownVariableType]
)
from safetensors.torch import (
    save_file as _save_file,  # pyright: ignore[reportUnknownVariableType]
)
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from dromeus.manifests.models import Tensor as TensorSpec
from dromeus.manifests.models import TensorSchema

IMAGE_SHAPE = (3, 32, 32)
CLASS_COUNT = 10
MODEL_DEFINITION = "cifar-cnn-v1:conv3x3-16:gn4:conv3x3-32:gn4:gap:linear"
MODEL_DEFINITION_HASH = hashlib.sha256(MODEL_DEFINITION.encode()).hexdigest()
_TORCH_TO_SCHEMA_DTYPE: dict[torch.dtype, Literal["float16", "float32", "float64"]] = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
}
_load_checkpoint = cast(Callable[..., dict[str, Tensor]], _load_file)
_save_checkpoint = cast(Callable[..., None], _save_file)


class CIFARDataError(ValueError):
    """CIFAR data does not satisfy the local contract."""


@dataclass(frozen=True, slots=True)
class CIFAR10Data:
    """Validated channel-first CIFAR-10 arrays with uint8 or float pixels."""

    images: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        images = np.asarray(self.images)
        labels = np.asarray(self.labels)
        if images.ndim != 4 or images.shape[1:] != IMAGE_SHAPE:
            raise CIFARDataError("images must have shape (N, 3, 32, 32)")
        if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
            raise CIFARDataError("labels must be one-dimensional and match images")
        if images.shape[0] == 0:
            raise CIFARDataError("dataset must contain at least one image")
        if images.dtype not in (np.dtype(np.uint8), np.dtype(np.float32)):
            raise CIFARDataError("images must be uint8 or float32")
        if labels.dtype not in (np.dtype(np.int64), np.dtype(np.int32)):
            raise CIFARDataError("labels must be int32 or int64")
        if not np.isfinite(images).all() or not np.isfinite(labels).all():
            raise CIFARDataError("dataset contains non-finite values")
        if images.dtype == np.dtype(np.float32) and (
            float(images.min()) < 0.0 or float(images.max()) > 1.0
        ):
            raise CIFARDataError("float images must be normalized to [0, 1]")
        if np.any(labels < 0) or np.any(labels >= CLASS_COUNT):
            raise CIFARDataError("labels must be in [0, 10)")
        images = np.ascontiguousarray(images)
        labels = np.ascontiguousarray(labels, dtype=np.int64)
        images.setflags(write=False)
        labels.setflags(write=False)
        object.__setattr__(self, "images", images)
        object.__setattr__(self, "labels", labels)

    @classmethod
    def synthetic(cls, *, sample_count: int, seed: int) -> CIFAR10Data:
        """Create deterministic uint8 data for smoke tests and local pilots."""
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        generator = np.random.default_rng(seed)
        images = generator.integers(
            0,
            256,
            size=(sample_count, *IMAGE_SHAPE),
            dtype=np.uint8,
        )
        labels = generator.integers(0, CLASS_COUNT, size=sample_count, dtype=np.int64)
        return cls(images=images, labels=labels)

    @classmethod
    def from_npz(cls, path: Path) -> CIFAR10Data:
        """Load validated arrays from an NPZ with ``images`` and ``labels`` keys."""
        try:
            with np.load(path, allow_pickle=False) as archive:
                images = archive["images"]
                labels = archive["labels"]
        except (OSError, KeyError, ValueError) as error:
            raise CIFARDataError(f"cannot load CIFAR NPZ: {path}") from error
        return cls(images=images, labels=labels)

    @classmethod
    def download_npz(
        cls,
        *,
        url: str,
        destination: Path,
        sha256: str,
    ) -> CIFAR10Data:
        """Download an immutable NPZ into place after verifying its SHA-256."""
        if len(sha256) != 64 or any(
            character not in "0123456789abcdef" for character in sha256
        ):
            raise ValueError("sha256 must be a lowercase hexadecimal digest")
        temporary = destination.with_suffix(destination.suffix + ".part")
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        try:
            with urlopen(url, timeout=30) as response, temporary.open("wb") as handle:
                while block := response.read(1024 * 1024):
                    handle.write(block)
                    digest.update(block)
            if digest.hexdigest() != sha256:
                raise CIFARDataError("downloaded CIFAR archive checksum mismatch")
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return cls.from_npz(destination)

    def split_iid(
        self,
        *,
        participant_count: int,
        seed: int,
    ) -> tuple[CIFAR10Data, ...]:
        """Return equal, disjoint, deterministic IID partitions."""
        if participant_count <= 0 or self.images.shape[0] % participant_count:
            raise ValueError("sample count must divide participant count exactly")
        generator = np.random.default_rng(seed)
        order = generator.permutation(self.images.shape[0])
        partitions: list[CIFAR10Data] = []
        for indices in np.array_split(order, participant_count):
            partitions.append(
                CIFAR10Data(images=self.images[indices], labels=self.labels[indices])
            )
        return tuple(partitions)

    def tensors(self) -> tuple[Tensor, Tensor]:
        """Return normalized tensors suitable for a PyTorch data loader."""
        images = self.images.astype(np.float32, copy=False)
        if self.images.dtype == np.dtype(np.uint8):
            images = images / 255.0
        return (
            torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
                cast(Any, np.ascontiguousarray(images).copy())
            ),
            torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
                cast(Any, np.ascontiguousarray(self.labels, dtype=np.int64).copy())
            ),
        )


class CIFARGroupNormCNN(nn.Module):
    """Small versioned CIFAR CNN with no batch-dependent state."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, CLASS_COUNT)

    def forward(self, images: Tensor) -> Tensor:
        features = self.features(images).flatten(1)
        return self.classifier(features)


def build_model(*, seed: int) -> CIFARGroupNormCNN:
    """Construct a reproducibly initialized model without changing caller RNG."""
    with torch.random.fork_rng(devices=[]):  # pyright: ignore[reportUnknownMemberType]
        torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
        return CIFARGroupNormCNN()


def tensor_schema_for_model(model: nn.Module | None = None) -> TensorSchema:
    """Derive the wire tensor schema from model parameters."""
    target = model or CIFARGroupNormCNN()
    specs: list[TensorSpec] = []
    for name, parameter in target.named_parameters():
        dtype = _TORCH_TO_SCHEMA_DTYPE.get(parameter.dtype)
        if dtype is None:
            raise CIFARDataError(
                f"unsupported model parameter dtype: {parameter.dtype}"
            )
        specs.append(TensorSpec(name=name, dtype=dtype, shape=tuple(parameter.shape)))
    return TensorSchema(tensors=tuple(specs))


def create_initial_checkpoint(path: Path, *, seed: int) -> None:
    """Write the canonical initial model checkpoint for an initiator."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(seed=seed)
    _save_checkpoint(
        {name: value.detach().cpu() for name, value in model.named_parameters()},
        str(path),
        metadata={"model_definition": MODEL_DEFINITION},
    )


def checkpoint_hash(path: Path) -> str:
    """Return the SHA-256 digest used by formation manifests."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class CIFAR10Trainer:
    """Own a local CIFAR model, SGD optimizer, loader, and checkpoint seam."""

    def __init__(
        self,
        *,
        train_data: CIFAR10Data,
        test_data: CIFAR10Data | None = None,
        seed: int = 0,
        batch_size: int = 32,
        learning_rate: float = 0.1,
        device: str = "cpu",
        augment: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if learning_rate <= 0 or not np.isfinite(learning_rate):
            raise ValueError("learning_rate must be positive and finite")
        self._device = torch.device(device)
        self._model = build_model(seed=seed).to(self._device)
        self._optimizer = torch.optim.SGD(
            self._model.parameters(), lr=learning_rate, momentum=0.0
        )
        self._augment = augment
        self._augmentation_generator = torch.Generator(device="cpu")
        self._augmentation_generator.manual_seed(seed + 1)
        self._loader_generator = torch.Generator(device="cpu")
        self._loader_generator.manual_seed(seed + 2)
        self._batch_size = batch_size
        self._train_data = train_data
        self._test_data = test_data or train_data
        self._train_loader = self._make_loader(train_data, shuffle=True)
        self._train_iterator = iter(self._train_loader)
        self._tensor_schema = tensor_schema_for_model(self._model)

    @property
    def tensor_schema(self) -> TensorSchema:
        return self._tensor_schema

    @staticmethod
    def tensor_schema_for_model() -> TensorSchema:
        return tensor_schema_for_model()

    def weights(self) -> dict[str, np.ndarray]:
        return {
            name: parameter.detach().cpu().numpy().copy()
            for name, parameter in self._model.named_parameters()
        }

    def load_weights(self, weights: dict[str, np.ndarray]) -> None:
        expected = {
            name: parameter for name, parameter in self._model.named_parameters()
        }
        if set(weights) != set(expected):
            raise ValueError("weight names do not match model")
        with torch.no_grad():
            for name, parameter in expected.items():
                value = np.asarray(weights[name])
                if value.shape != tuple(parameter.shape) or value.dtype != np.float32:
                    raise ValueError(f"weight {name} does not match model")
                parameter.copy_(
                    torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
                        cast(Any, np.ascontiguousarray(value))
                    ).to(self._device)
                )

    def train_local_steps(self, step_count: int) -> None:
        if step_count < 0:
            raise ValueError("step_count must be non-negative")
        self._model.train()
        for _ in range(step_count):
            images, labels = self._next_batch()
            self._optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(self._model(images), labels)
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            self._optimizer.step()  # pyright: ignore[reportUnknownMemberType]

    def stochastic_gradients(self) -> dict[str, np.ndarray]:
        """Compute one minibatch gradient without mutating model weights."""
        self._model.train()
        images, labels = self._next_batch()
        self._optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(self._model(images), labels)
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        gradients: dict[str, np.ndarray] = {}
        for name, parameter in self._model.named_parameters():
            if parameter.grad is None:
                raise RuntimeError(f"missing gradient for {name}")
            gradients[name] = parameter.grad.detach().cpu().numpy().copy()
        return gradients

    def evaluate(self, data: CIFAR10Data | None = None) -> tuple[float, float]:
        """Return mean cross-entropy and accuracy without changing train state."""
        target = data or self._test_data
        loader = self._make_loader(target, shuffle=False)
        was_training = self._model.training
        self._model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                logits = self._model(images)
                batch_size = labels.shape[0]
                total_loss += (
                    float(nn.functional.cross_entropy(logits, labels).item())
                    * batch_size
                )
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                total += batch_size
        self._model.train(was_training)
        return total_loss / total, correct / total

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        _save_checkpoint(
            {
                name: value.detach().cpu()
                for name, value in self._model.named_parameters()
            },
            str(path),
            metadata={"model_definition": MODEL_DEFINITION},
        )

    def load_checkpoint(self, path: Path) -> None:
        values = _load_checkpoint(str(path), device=str(self._device))
        self.load_weights(
            {
                name: value.detach().cpu().numpy().astype(np.float32, copy=False)
                for name, value in values.items()
            }
        )

    @staticmethod
    def checkpoint_hash(path: Path) -> str:
        return checkpoint_hash(path)

    def _make_loader(self, data: CIFAR10Data, *, shuffle: bool) -> DataLoader[Any]:
        images, labels = data.tensors()
        dataset = TensorDataset(images, labels)
        generator = self._loader_generator if shuffle else None
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=shuffle,
            generator=generator,
            drop_last=False,
            num_workers=0,
        )

    def _next_batch(self) -> tuple[Tensor, Tensor]:
        try:
            images, labels = next(self._train_iterator)
        except StopIteration:
            self._train_iterator = iter(self._train_loader)
            images, labels = next(self._train_iterator)
        images = images.to(self._device)
        labels = labels.to(self._device)
        if self._augment:
            flips = (
                torch.rand(images.shape[0], generator=self._augmentation_generator)
                < 0.5
            )
            if bool(flips.any()):
                images = images.clone()
                images[flips] = torch.flip(images[flips], dims=(3,))
        return images, labels


__all__ = [
    "CIFAR10Data",
    "CIFAR10Trainer",
    "CIFARDataError",
    "CIFARGroupNormCNN",
    "MODEL_DEFINITION",
    "MODEL_DEFINITION_HASH",
    "build_model",
    "checkpoint_hash",
    "create_initial_checkpoint",
    "tensor_schema_for_model",
]
