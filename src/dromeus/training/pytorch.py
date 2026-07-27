"""Deterministic CIFAR-10 data and PyTorch training primitives."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    Dataset as HuggingFaceDataset,  # pyright: ignore[reportMissingTypeStubs]
)
from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    load_dataset,  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
)
from safetensors.torch import (
    load_file as _load_file,  # pyright: ignore[reportUnknownVariableType]
)
from safetensors.torch import (
    save_file as _save_file,  # pyright: ignore[reportUnknownVariableType]
)
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import (  # pyright: ignore[reportMissingTypeStubs]
    CIFAR10 as TorchvisionCIFAR10,  # pyright: ignore[reportMissingTypeStubs]
)
from torchvision.transforms import ToTensor  # pyright: ignore[reportMissingTypeStubs]

from dromeus.manifests.models import DraftRunSpec, SealedManifest, TensorSchema
from dromeus.manifests.models import Tensor as TensorSpec

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
class IIDPartitionProvenance:
    """Identity of one deterministic IID split member."""

    seed: int
    participant_count: int
    partition_index: int
    source_sample_count: int
    indices_sha256: str


@dataclass(frozen=True, slots=True)
class CIFARDataProvenance:
    """Canonical CIFAR source and split identity."""

    source: str
    split: Literal["train", "test"]


@dataclass(frozen=True, slots=True)
class InitialCheckpoint:
    """Canonical checkpoint handoff data for initiator formation."""

    path: Path
    tensor_schema: TensorSchema
    sha256: str


@dataclass(frozen=True, slots=True)
class CIFAR10Data(Dataset[tuple[Tensor, int]]):
    """Thin view over a standard CIFAR-10 dataset."""

    _dataset: Dataset[tuple[Tensor, int]]
    _indices: tuple[int, ...] | None = None
    _partition_provenance: IIDPartitionProvenance | None = None
    _source_provenance: CIFARDataProvenance | None = None

    @classmethod
    def from_torchvision(
        cls,
        *,
        root: Path,
        train: bool = True,
        download: bool = False,
    ) -> CIFAR10Data:
        """Open the standard CIFAR-10 dataset with tensor conversion."""
        try:
            dataset = TorchvisionCIFAR10(
                root=str(root),
                train=train,
                transform=ToTensor(),
                download=download,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise CIFARDataError(f"cannot open CIFAR-10 dataset at {root}") from error
        return cls(
            cast(Dataset[tuple[Tensor, int]], dataset),
            _source_provenance=CIFARDataProvenance(
                source="torchvision-cifar10",
                split="train" if train else "test",
            ),
        )

    @classmethod
    def from_huggingface(
        cls,
        *,
        train: bool = True,
        cache_dir: Path | None = None,
    ) -> CIFAR10Data:
        """Open CIFAR-10 from its Hugging Face dataset source."""
        try:
            datasets = load_dataset(
                "uoft-cs/cifar10",
                cache_dir=str(cache_dir) if cache_dir is not None else None,
            )
            dataset = cast(Any, datasets)["train" if train else "test"]
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            raise CIFARDataError("cannot open Hugging Face CIFAR-10 dataset") from error
        return cls(
            _HuggingFaceCIFAR10(cast(HuggingFaceDataset, dataset)),
            _source_provenance=CIFARDataProvenance(
                source="huggingface-uoft-cs-cifar10",
                split="train" if train else "test",
            ),
        )

    def split_iid(
        self,
        *,
        participant_count: int,
        seed: int,
    ) -> tuple[CIFAR10Data, ...]:
        """Return equal, disjoint, deterministic IID partitions."""
        if participant_count <= 0 or len(self) % participant_count:
            raise ValueError("sample count must divide participant count exactly")
        source_sample_count = len(self)
        index_groups = iid_partition_indices(
            source_sample_count=source_sample_count,
            participant_count=participant_count,
            seed=seed,
        )
        partitions: list[CIFAR10Data] = []
        for partition_index, indices in enumerate(index_groups):
            partitions.append(
                CIFAR10Data(
                    self._dataset,
                    indices,
                    IIDPartitionProvenance(
                        seed=seed,
                        participant_count=participant_count,
                        partition_index=partition_index,
                        source_sample_count=source_sample_count,
                        indices_sha256=_indices_hash(indices),
                    ),
                    self._source_provenance,
                )
            )
        return tuple(partitions)

    @property
    def partition_provenance(self) -> IIDPartitionProvenance | None:
        return self._partition_provenance

    @property
    def source_provenance(self) -> CIFARDataProvenance | None:
        return self._source_provenance

    def matches_source(
        self,
        *,
        source: str,
        split: Literal["train", "test"],
    ) -> bool:
        """Return whether this view came from the declared trusted loader."""
        provenance = self._source_provenance
        if (
            provenance is None
            or provenance.source != source
            or provenance.split != split
        ):
            return False
        if source == "torchvision-cifar10":
            return isinstance(self._dataset, TorchvisionCIFAR10) and bool(
                self._dataset.train
            ) == (split == "train")
        return True

    def __len__(self) -> int:
        return (
            len(self._indices)
            if self._indices is not None
            else len(cast(Sized, self._dataset))
        )

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        dataset_index = self._indices[index] if self._indices is not None else index
        image, label = self._dataset[dataset_index]
        return image, int(label)


def iid_partition_indices(
    *,
    source_sample_count: int,
    participant_count: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    """Return the canonical deterministic IID index groups."""
    if source_sample_count <= 0 or participant_count <= 0:
        raise ValueError("sample and participant counts must be positive")
    if source_sample_count % participant_count:
        raise ValueError("sample count must divide participant count exactly")
    order = np.random.default_rng(seed).permutation(source_sample_count)
    return tuple(
        tuple(int(index) for index in indices)
        for indices in np.array_split(order, participant_count)
    )


def iid_partition_index_hashes(
    *,
    source_sample_count: int,
    participant_count: int,
    seed: int,
) -> tuple[str, ...]:
    """Return stable identities for the canonical IID index groups."""
    return tuple(
        _indices_hash(indices)
        for indices in iid_partition_indices(
            source_sample_count=source_sample_count,
            participant_count=participant_count,
            seed=seed,
        )
    )


def _indices_hash(indices: tuple[int, ...]) -> str:
    values = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


class _HuggingFaceCIFAR10(Dataset[tuple[Tensor, int]]):
    def __init__(self, dataset: HuggingFaceDataset) -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        row = cast(dict[str, Any], self._dataset[index])
        image = ToTensor()(row["img"])
        if tuple(image.shape) != IMAGE_SHAPE:
            raise CIFARDataError(
                f"expected CIFAR-10 image shape, got {tuple(image.shape)}"
            )
        return image, int(row["label"])


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


def create_initial_checkpoint(path: Path, *, seed: int) -> InitialCheckpoint:
    """Write and describe the canonical checkpoint before formation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(seed=seed)
    _save_checkpoint(
        {name: value.detach().cpu() for name, value in model.named_parameters()},
        str(path),
        metadata={"model_definition": MODEL_DEFINITION},
    )
    return InitialCheckpoint(
        path=path,
        tensor_schema=tensor_schema_for_model(model),
        sha256=checkpoint_hash(path),
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
        self._last_local_loss: float | None = None

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

    @property
    def last_local_loss(self) -> float | None:
        return self._last_local_loss

    def train_local_steps(self, step_count: int) -> None:
        if step_count < 0:
            raise ValueError("step_count must be non-negative")
        self._model.train()
        for _ in range(step_count):
            images, labels = self._next_batch()
            self._optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(self._model(images), labels)
            self._last_local_loss = float(loss.detach().cpu().item())
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            self._optimizer.step()  # pyright: ignore[reportUnknownMemberType]

    def stochastic_gradients(self) -> dict[str, np.ndarray]:
        """Compute one minibatch gradient without mutating model weights."""
        self._model.train()
        images, labels = self._next_batch()
        self._optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(self._model(images), labels)
        self._last_local_loss = float(loss.detach().cpu().item())
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
        generator = self._loader_generator if shuffle else None
        return DataLoader(
            data,
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


@dataclass(frozen=True, slots=True)
class PreparedCIFARTraining:
    """Local CIFAR data retained behind training-owned construction methods."""

    _partitions: tuple[CIFAR10Data, ...]
    _test_data: CIFAR10Data
    benchmark_seed: int

    def create_initial_checkpoint(self, path: Path) -> InitialCheckpoint:
        return create_initial_checkpoint(path, seed=self.benchmark_seed)

    def create_trainer(
        self,
        *,
        manifest: SealedManifest,
        local_public_key: str,
    ) -> CIFAR10Trainer:
        node_indices = {
            participant.public_key: participant.node_index
            for participant in manifest.participants
        }
        node_index = node_indices[local_public_key]
        partition_index = manifest.dataset.node_index_partitions[node_index]
        return CIFAR10Trainer(
            train_data=self._partitions[partition_index],
            test_data=self._test_data,
            seed=self.benchmark_seed + node_index,
            batch_size=32,
            learning_rate=manifest.learning_rate,
            device="cpu",
            augment=True,
        )


def prepare_cifar_training(
    *,
    draft: DraftRunSpec,
    cifar_root: Path,
    benchmark_seed: int,
) -> PreparedCIFARTraining:
    """Load and validate local CIFAR data before membership becomes ready."""
    train_data = CIFAR10Data.from_torchvision(
        root=cifar_root,
        train=True,
        download=False,
    )
    test_data = CIFAR10Data.from_torchvision(
        root=cifar_root,
        train=False,
        download=False,
    )
    if len(train_data) != draft.dataset.sample_count:
        raise ValueError("local CIFAR-10 sample count does not match draft")
    partitions = train_data.split_iid(
        participant_count=4,
        seed=draft.dataset.iid_partition_seed,
    )
    if tuple(len(partition) for partition in partitions) != (
        draft.dataset.partition_sample_counts
    ):
        raise ValueError("local CIFAR-10 partitions do not match draft")
    return PreparedCIFARTraining(
        _partitions=partitions,
        _test_data=test_data,
        benchmark_seed=benchmark_seed,
    )


__all__ = [
    "CIFAR10Data",
    "CIFAR10Trainer",
    "CIFARDataProvenance",
    "CIFARDataError",
    "CIFARGroupNormCNN",
    "InitialCheckpoint",
    "IIDPartitionProvenance",
    "MODEL_DEFINITION",
    "MODEL_DEFINITION_HASH",
    "PreparedCIFARTraining",
    "build_model",
    "checkpoint_hash",
    "create_initial_checkpoint",
    "iid_partition_index_hashes",
    "iid_partition_indices",
    "prepare_cifar_training",
    "tensor_schema_for_model",
]
