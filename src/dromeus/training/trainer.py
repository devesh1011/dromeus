"""Generic PyTorch classification trainer."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from safetensors.torch import (
    load_file as _load_file,  # pyright: ignore[reportUnknownVariableType]
)
from safetensors.torch import (
    save_file as _save_file,  # pyright: ignore[reportUnknownVariableType]
)
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from dromeus.manifests.canonical import file_sha256
from dromeus.manifests.models import TensorSchema
from dromeus.training.resnet32 import floating_model_state, tensor_schema_for_model

BatchTransform = Callable[[Tensor, bool, torch.Generator], Tensor]
_TRAINING_STATE_PREFIX = "__dromeus_training__."
_COMPLETED_STEPS = f"{_TRAINING_STATE_PREFIX}completed_steps"
_BATCHES_CONSUMED = f"{_TRAINING_STATE_PREFIX}batches_consumed"
_AUGMENTATION_RNG = f"{_TRAINING_STATE_PREFIX}augmentation_rng"
_LOADER_EPOCH_RNG = f"{_TRAINING_STATE_PREFIX}loader_epoch_rng"
_MOMENTUM_PREFIX = f"{_TRAINING_STATE_PREFIX}momentum."
_load_checkpoint = cast(Callable[..., dict[str, Tensor]], _load_file)
_save_checkpoint = cast(Callable[..., None], _save_file)


@dataclass(frozen=True, slots=True)
class InitialCheckpoint:
    """Canonical checkpoint handoff data for initiator formation."""

    path: Path
    tensor_schema: TensorSchema
    sha256: str


def derive_benchmark_seed(benchmark_seed: int, purpose: str) -> int:
    """Derive one stable RNG seed for a named benchmark concern."""
    digest = hashlib.sha256(f"{benchmark_seed}:{purpose}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def create_initial_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    model_definition: str,
) -> InitialCheckpoint:
    """Write and describe the canonical checkpoint before formation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_checkpoint(
        {
            name: value.detach().cpu()
            for name, value in floating_model_state(model).items()
        },
        str(path),
        metadata={"model_definition": model_definition},
    )
    return InitialCheckpoint(
        path=path,
        tensor_schema=tensor_schema_for_model(model),
        sha256=checkpoint_hash(path),
    )


def checkpoint_hash(path: Path) -> str:
    """Return a checkpoint's SHA-256 digest."""
    return file_sha256(path)


class PyTorchTrainer:
    """Own a classification model, SGD optimizer, loader, and checkpoint seam."""

    def __init__(
        self,
        *,
        model: nn.Module,
        model_definition: str,
        train_data: Dataset[tuple[Tensor, int]],
        test_data: Dataset[tuple[Tensor, int]] | None = None,
        seed: int = 0,
        batch_size: int = 32,
        learning_rate: float = 0.1,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        learning_rate_milestones: tuple[int, ...] = (),
        learning_rate_gamma: float = 0.1,
        device: str = "cpu",
        augment: bool = True,
        batch_transform: BatchTransform | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if learning_rate <= 0 or not np.isfinite(learning_rate):
            raise ValueError("learning_rate must be positive and finite")
        if not 0 <= momentum < 1 or not np.isfinite(momentum):
            raise ValueError("momentum must be finite in [0, 1)")
        if weight_decay < 0 or not np.isfinite(weight_decay):
            raise ValueError("weight_decay must be finite and non-negative")
        if any(milestone <= 0 for milestone in learning_rate_milestones) or any(
            right <= left
            for left, right in zip(
                learning_rate_milestones,
                learning_rate_milestones[1:],
                strict=False,
            )
        ):
            raise ValueError("learning-rate milestones must be strictly increasing")
        if not 0 < learning_rate_gamma < 1 or not np.isfinite(
            learning_rate_gamma
        ):
            raise ValueError("learning_rate_gamma must be finite in (0, 1)")
        self._device = torch.device(device)
        self._model_definition = model_definition
        self._model = model.to(self._device)
        self._optimizer = torch.optim.SGD(
            self._model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        self._base_learning_rate = learning_rate
        self._learning_rate_milestones = learning_rate_milestones
        self._learning_rate_gamma = learning_rate_gamma
        self._completed_steps = 0
        self._augment = augment
        self._batch_transform = batch_transform
        self._augmentation_generator = torch.Generator(device="cpu")
        self._augmentation_generator.manual_seed(seed + 1)
        self._loader_generator = torch.Generator(device="cpu")
        self._loader_generator.manual_seed(seed + 2)
        self._batch_size = batch_size
        self._test_data = test_data or train_data
        self._train_loader = self._make_loader(train_data, shuffle=True)
        self._batches_consumed = 0
        self._epoch_generator_state = self._loader_generator.get_state().clone()
        self._train_iterator = iter(self._train_loader)
        self._tensor_schema = tensor_schema_for_model(self._model)
        self._last_local_loss: float | None = None

    @property
    def tensor_schema(self) -> TensorSchema:
        return self._tensor_schema

    @property
    def last_local_loss(self) -> float | None:
        return self._last_local_loss

    @property
    def learning_rate(self) -> float:
        return float(self._optimizer.param_groups[0]["lr"])

    def weights(self) -> dict[str, np.ndarray]:
        return {
            name: value.detach().cpu().numpy().copy()
            for name, value in floating_model_state(self._model).items()
        }

    def load_weights(self, weights: dict[str, np.ndarray]) -> None:
        expected = floating_model_state(self._model)
        if set(weights) != set(expected):
            raise ValueError("weight names do not match model")
        with torch.no_grad():
            for name, target in expected.items():
                value = np.asarray(weights[name])
                expected_dtype = target.detach().cpu().numpy().dtype
                if value.shape != tuple(target.shape) or value.dtype != expected_dtype:
                    raise ValueError(f"weight {name} does not match model")
                target.copy_(
                    torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
                        cast(Any, np.ascontiguousarray(value))
                    ).to(self._device)
                )

    def checkpoint_tensors(self) -> dict[str, np.ndarray]:
        """Return model, optimizer, schedule, and stochastic-loader state."""
        state = self.weights()
        state[_COMPLETED_STEPS] = np.array([self._completed_steps], dtype=np.int64)
        state[_BATCHES_CONSUMED] = np.array(
            [self._batches_consumed],
            dtype=np.int64,
        )
        state[_AUGMENTATION_RNG] = (
            self._augmentation_generator.get_state().cpu().numpy().copy()
        )
        state[_LOADER_EPOCH_RNG] = self._epoch_generator_state.cpu().numpy().copy()
        for name, parameter in self._model.named_parameters():
            momentum = self._optimizer.state.get(parameter, {}).get(
                "momentum_buffer"
            )
            if isinstance(momentum, Tensor):
                state[f"{_MOMENTUM_PREFIX}{name}"] = (
                    momentum.detach().cpu().numpy().copy()
                )
        return state

    def load_checkpoint_tensors(self, state: dict[str, np.ndarray]) -> None:
        """Restore a complete tensor checkpoint produced by `checkpoint_tensors`."""
        required = {
            _COMPLETED_STEPS,
            _BATCHES_CONSUMED,
            _AUGMENTATION_RNG,
            _LOADER_EPOCH_RNG,
        }
        if not required.issubset(state):
            raise ValueError("training checkpoint metadata is incomplete")
        model_names = set(floating_model_state(self._model))
        if not model_names.issubset(state):
            raise ValueError("training checkpoint model state is incomplete")
        self.load_weights({name: state[name] for name in model_names})
        completed_steps = _single_non_negative_int(state[_COMPLETED_STEPS])
        batches_consumed = _single_non_negative_int(state[_BATCHES_CONSUMED])
        if batches_consumed > len(self._train_loader):
            raise ValueError("training checkpoint batch position is invalid")

        self._optimizer.state.clear()
        for name, parameter in self._model.named_parameters():
            key = f"{_MOMENTUM_PREFIX}{name}"
            if key not in state:
                continue
            value = np.asarray(state[key])
            expected_dtype = parameter.detach().cpu().numpy().dtype
            if value.dtype != expected_dtype or value.shape != tuple(parameter.shape):
                raise ValueError(f"momentum state {name} does not match model")
            self._optimizer.state[parameter]["momentum_buffer"] = (
                torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
                    cast(Any, np.ascontiguousarray(value))
                ).to(self._device)
            )

        augmentation_state = _rng_state_tensor(
            state[_AUGMENTATION_RNG],
            name="augmentation",
        )
        loader_epoch_state = _rng_state_tensor(
            state[_LOADER_EPOCH_RNG],
            name="loader",
        )
        self._augmentation_generator.set_state(augmentation_state)
        self._loader_generator.set_state(loader_epoch_state)
        self._epoch_generator_state = loader_epoch_state.clone()
        self._train_iterator = iter(self._train_loader)
        for _ in range(batches_consumed):
            try:
                next(self._train_iterator)
            except StopIteration as error:
                raise ValueError(
                    "training checkpoint batch position is invalid"
                ) from error
        self._batches_consumed = batches_consumed
        self._completed_steps = completed_steps
        self._apply_learning_rate()

    def train_local_steps(self, step_count: int) -> None:
        if step_count < 0:
            raise ValueError("step_count must be non-negative")
        self._model.train()
        for _ in range(step_count):
            self._apply_learning_rate()
            images, labels = self._next_batch()
            self._optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(self._model(images), labels)
            self._last_local_loss = float(loss.detach().cpu().item())
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            self._optimizer.step()  # pyright: ignore[reportUnknownMemberType]
            self._completed_steps += 1

    def evaluate(
        self,
        data: Dataset[tuple[Tensor, int]] | None = None,
    ) -> tuple[float, float]:
        """Return mean cross-entropy and accuracy without changing train state."""
        loader = self._make_loader(data or self._test_data, shuffle=False)
        was_training = self._model.training
        self._model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images = self._prepare_batch(images.to(self._device), augment=False)
                labels = labels.to(self._device)
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
                for name, value in floating_model_state(self._model).items()
            },
            str(path),
            metadata={"model_definition": self._model_definition},
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

    def _make_loader(
        self,
        data: Dataset[tuple[Tensor, int]],
        *,
        shuffle: bool,
    ) -> DataLoader[Any]:
        return DataLoader(
            data,
            batch_size=self._batch_size,
            shuffle=shuffle,
            generator=self._loader_generator if shuffle else None,
            drop_last=False,
            num_workers=0,
        )

    def _next_batch(self) -> tuple[Tensor, Tensor]:
        try:
            images, labels = next(self._train_iterator)
        except StopIteration:
            self._epoch_generator_state = self._loader_generator.get_state().clone()
            self._train_iterator = iter(self._train_loader)
            self._batches_consumed = 0
            images, labels = next(self._train_iterator)
        self._batches_consumed += 1
        return (
            self._prepare_batch(images.to(self._device), augment=self._augment),
            labels.to(self._device),
        )

    def _prepare_batch(self, images: Tensor, *, augment: bool) -> Tensor:
        if self._batch_transform is None:
            return images
        return self._batch_transform(
            images,
            augment,
            self._augmentation_generator,
        )

    def _apply_learning_rate(self) -> None:
        decay_count = sum(
            self._completed_steps >= milestone
            for milestone in self._learning_rate_milestones
        )
        learning_rate = self._base_learning_rate * (
            self._learning_rate_gamma**decay_count
        )
        for group in self._optimizer.param_groups:
            group["lr"] = learning_rate


def _single_non_negative_int(value: np.ndarray) -> int:
    array = np.asarray(value)
    if array.dtype != np.int64 or array.shape != (1,) or int(array[0]) < 0:
        raise ValueError("training checkpoint counter is invalid")
    return int(array[0])


def _rng_state_tensor(value: np.ndarray, *, name: str) -> Tensor:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 1:
        raise ValueError(f"{name} RNG state is invalid")
    return torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
        cast(Any, np.ascontiguousarray(array))
    )


__all__ = [
    "BatchTransform",
    "InitialCheckpoint",
    "PyTorchTrainer",
    "checkpoint_hash",
    "create_initial_checkpoint",
    "derive_benchmark_seed",
]
