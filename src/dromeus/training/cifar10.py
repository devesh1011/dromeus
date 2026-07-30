"""Hugging Face CIFAR-10 training recipe."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import partial
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
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

from dromeus.manifests.models import DraftRunSpec, SealedManifest
from dromeus.training.data import ClassificationData, DataProvenance
from dromeus.training.resnet32 import MODEL_DEFINITION, build_model
from dromeus.training.trainer import (
    InitialCheckpoint,
    PyTorchTrainer,
    derive_benchmark_seed,
)
from dromeus.training.trainer import (
    create_initial_checkpoint as _create_initial_checkpoint,
)

DATA_SOURCE = "huggingface-uoft-cs-cifar10"
DATASET_REPOSITORY = "uoft-cs/cifar10"
DATASET_REVISION = "0b2714987fa478483af9968de7c934580d0bb9a2"
DATASET_VERSION = f"huggingface-{DATASET_REVISION}"
IMAGE_SHAPE = (3, 32, 32)
CLASS_COUNT = 10
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
PREPROCESSING_DEFINITION = (
    "pil-rgb-to-chw-float32-div255;"
    "seeded-reflect-crop:padding=4;"
    "seeded-horizontal-flip:p=0.5;"
    "channel-normalization:mean=0.4914,0.4822,0.4465:"
    "std=0.2470,0.2435,0.2616"
)
PREPROCESSING_HASH = hashlib.sha256(PREPROCESSING_DEFINITION.encode()).hexdigest()


class CIFAR10DataError(ValueError):
    """CIFAR-10 data does not satisfy the recipe contract."""


class _HuggingFaceCIFAR10(Dataset[tuple[Tensor, int]]):
    def __init__(self, dataset: HuggingFaceDataset) -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        row = cast(dict[str, Any], self._dataset[index])
        pixels = np.array(row["img"].convert("RGB"), dtype=np.uint8, copy=True)
        if pixels.shape != (IMAGE_SHAPE[1], IMAGE_SHAPE[2], IMAGE_SHAPE[0]):
            raise CIFAR10DataError(
                f"expected CIFAR-10 image shape, got {pixels.shape}"
            )
        image = (
            torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
                cast(Any, np.ascontiguousarray(pixels))
            )
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            .div_(255)
        )
        return image, int(row["label"])


def load_cifar10(
    *,
    train: bool = True,
    cache_dir: Path | None = None,
) -> ClassificationData:
    """Load the pinned Hugging Face CIFAR-10 split."""
    split: Literal["train", "test"] = "train" if train else "test"
    try:
        dataset = load_dataset(
            DATASET_REPOSITORY,
            split=split,
            revision=DATASET_REVISION,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise CIFAR10DataError("cannot open Hugging Face CIFAR-10 dataset") from error
    return ClassificationData(
        _HuggingFaceCIFAR10(dataset),
        _source_provenance=DataProvenance(source=DATA_SOURCE, split=split),
    )


def create_initial_checkpoint(path: Path, *, seed: int) -> InitialCheckpoint:
    """Create the recipe's deterministic ResNet-32 checkpoint."""
    return _create_initial_checkpoint(
        path,
        model=build_model(seed=seed),
        model_definition=MODEL_DEFINITION,
    )


def create_trainer(
    *,
    train_data: ClassificationData,
    test_data: ClassificationData | None = None,
    seed: int = 0,
    batch_size: int = 128,
    learning_rate: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    learning_rate_milestones: tuple[int, ...] = (8_000, 12_000),
    learning_rate_gamma: float = 0.1,
    device: str = "cpu",
    augment: bool = True,
    crop_padding: int = 4,
    normalize: bool = True,
) -> PyTorchTrainer:
    """Construct the trainer configured by the CIFAR-10 recipe."""
    if crop_padding < 0:
        raise ValueError("crop_padding must be non-negative")
    return PyTorchTrainer(
        model=build_model(seed=seed),
        model_definition=MODEL_DEFINITION,
        train_data=train_data,
        test_data=test_data,
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        learning_rate_milestones=learning_rate_milestones,
        learning_rate_gamma=learning_rate_gamma,
        device=device,
        augment=augment,
        batch_transform=partial(
            _prepare_batch,
            crop_padding=crop_padding,
            normalize=normalize,
        ),
    )


def _prepare_batch(
    images: Tensor,
    augment: bool,
    generator: torch.Generator,
    *,
    crop_padding: int,
    normalize: bool,
) -> Tensor:
    if augment and crop_padding:
        padded = F.pad(images, (crop_padding,) * 4, mode="reflect")
        maximum = crop_padding * 2 + 1
        rows = torch.randint(maximum, (images.shape[0],), generator=generator).to(
            images.device
        )
        columns = torch.randint(
            maximum,
            (images.shape[0],),
            generator=generator,
        ).to(images.device)
        batch_indices = torch.arange(images.shape[0], device=images.device).view(
            -1,
            1,
            1,
        )
        row_indices = rows.view(-1, 1, 1) + torch.arange(
            IMAGE_SHAPE[1],
            device=images.device,
        ).view(1, -1, 1)
        column_indices = columns.view(-1, 1, 1) + torch.arange(
            IMAGE_SHAPE[2],
            device=images.device,
        ).view(1, 1, -1)
        images = padded.permute(0, 2, 3, 1)[
            batch_indices,
            row_indices,
            column_indices,
        ].permute(0, 3, 1, 2)
    if augment:
        flips = (
            torch.rand(images.shape[0], generator=generator) < 0.5
        ).to(images.device)
        if bool(flips.any()):
            images = images.clone()
            images[flips] = torch.flip(images[flips], dims=(3,))
    if normalize:
        mean = images.new_tensor(MEAN).view(1, 3, 1, 1)
        std = images.new_tensor(STD).view(1, 3, 1, 1)
        images = (images - mean) / std
    return images


@dataclass(frozen=True, slots=True)
class PreparedCIFAR10Training:
    """Validated local CIFAR-10 partitions and common test data."""

    _partitions: tuple[ClassificationData, ...]
    _test_data: ClassificationData
    initialization_seed: int
    trainer_seed: int

    def create_initial_checkpoint(self, path: Path) -> InitialCheckpoint:
        return create_initial_checkpoint(path, seed=self.initialization_seed)

    def create_trainer(
        self,
        *,
        manifest: SealedManifest,
        local_public_key: str,
    ) -> PyTorchTrainer:
        node_indices = {
            participant.public_key: participant.node_index
            for participant in manifest.participants
        }
        node_index = node_indices[local_public_key]
        partition_index = manifest.dataset.node_index_partitions[node_index]
        policy = manifest.training
        if policy is None:
            raise ValueError("CIFAR-10 recipe requires an active training policy")
        return create_trainer(
            train_data=self._partitions[partition_index],
            test_data=self._test_data,
            seed=self.trainer_seed + node_index,
            batch_size=policy.batch_size,
            learning_rate=manifest.learning_rate,
            momentum=policy.momentum,
            weight_decay=policy.weight_decay,
            learning_rate_milestones=policy.learning_rate_milestones,
            learning_rate_gamma=policy.learning_rate_gamma,
            device="cpu",
            augment=True,
            crop_padding=policy.crop_padding,
            normalize=policy.normalize,
        )


def prepare_training(
    *,
    draft: DraftRunSpec,
    cache_dir: Path,
    benchmark_seed: int,
) -> PreparedCIFAR10Training:
    """Load and validate local data before membership becomes ready."""
    train_data = load_cifar10(cache_dir=cache_dir, train=True)
    test_data = load_cifar10(cache_dir=cache_dir, train=False)
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
    return PreparedCIFAR10Training(
        _partitions=partitions,
        _test_data=test_data,
        initialization_seed=derive_benchmark_seed(
            benchmark_seed,
            "model-initialization",
        ),
        trainer_seed=derive_benchmark_seed(benchmark_seed, "local-training"),
    )


__all__ = [
    "CLASS_COUNT",
    "CIFAR10DataError",
    "DATASET_REPOSITORY",
    "DATASET_REVISION",
    "DATASET_VERSION",
    "DATA_SOURCE",
    "IMAGE_SHAPE",
    "PREPROCESSING_DEFINITION",
    "PREPROCESSING_HASH",
    "PreparedCIFAR10Training",
    "create_initial_checkpoint",
    "create_trainer",
    "load_cifar10",
    "prepare_training",
]
