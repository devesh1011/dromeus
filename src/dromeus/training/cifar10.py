"""Hugging Face CIFAR-10 training recipe."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
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

from dromeus.manifests.canonical import canonical_hash
from dromeus.manifests.models import (
    NOLOCO_ALGORITHM_ID,
    DraftRunSpec,
    SealedManifest,
)
from dromeus.training.data import ClassificationData, DataProvenance
from dromeus.training.models import resolve_model
from dromeus.training.resnet32 import MODEL_DEFINITION_HASH as RESNET32_DEFINITION_HASH
from dromeus.training.resnet32 import MODEL_ID as RESNET32_MODEL_ID
from dromeus.training.trainer import (
    InitialCheckpoint,
    PyTorchTrainer,
    TrainerSettings,
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


def create_initial_checkpoint(
    path: Path,
    *,
    seed: int,
    model_id: str = RESNET32_MODEL_ID,
    model_definition_hash: str | None = None,
) -> InitialCheckpoint:
    """Create one deterministic checkpoint from a built-in model recipe."""
    recipe = resolve_model(model_id, definition_hash=model_definition_hash)
    return _create_initial_checkpoint(
        path,
        model=recipe.build(seed=seed),
        model_definition=recipe.definition,
    )


def _default_trainer_settings() -> TrainerSettings:
    return TrainerSettings(
        batch_size=128,
        momentum=0.9,
        weight_decay=1e-4,
        learning_rate_milestones=(8_000, 12_000),
    )


@dataclass(frozen=True, slots=True)
class CIFAR10TrainerSettings:
    """Validated generic and CIFAR-specific trainer construction settings."""

    trainer: TrainerSettings = field(default_factory=_default_trainer_settings)
    model_id: str = RESNET32_MODEL_ID
    model_definition_hash: str | None = None
    crop_padding: int = 4
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.crop_padding < 0:
            raise ValueError("crop_padding must be non-negative")


def create_trainer(
    *,
    train_data: ClassificationData,
    test_data: ClassificationData | None = None,
    settings: CIFAR10TrainerSettings | None = None,
) -> PyTorchTrainer:
    """Construct the trainer configured by the CIFAR-10 recipe."""
    settings = settings or CIFAR10TrainerSettings()
    _configure_device(settings.trainer.device)
    recipe = resolve_model(
        settings.model_id,
        definition_hash=settings.model_definition_hash,
    )
    return PyTorchTrainer(
        model=recipe.build(seed=settings.trainer.seed),
        model_definition=recipe.definition,
        train_data=train_data,
        test_data=test_data,
        settings=settings.trainer,
        batch_transform=partial(
            _prepare_batch,
            crop_padding=settings.crop_padding,
            normalize=settings.normalize,
        ),
    )


def _configure_device(device: str) -> None:
    target = torch.device(device)
    if target.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise ValueError("CUDA training requested but CUDA is unavailable")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


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
    model_id: str = RESNET32_MODEL_ID
    model_definition_hash: str = RESNET32_DEFINITION_HASH
    device: str = "cpu"

    def create_initial_checkpoint(self, path: Path) -> InitialCheckpoint:
        return create_initial_checkpoint(
            path,
            seed=self.initialization_seed,
            model_id=self.model_id,
            model_definition_hash=self.model_definition_hash,
        )

    def create_trainer(
        self,
        *,
        manifest: SealedManifest,
        local_public_key: str,
    ) -> PyTorchTrainer:
        if manifest.model_id != self.model_id:
            raise ValueError("formed manifest model does not match prepared training")
        node_indices = {
            participant.public_key: participant.node_index
            for participant in manifest.participants
        }
        node_index = node_indices.get(local_public_key)
        if node_index is None:
            raise ValueError("local public key is not a sealed participant")
        recipe = resolve_model(
            self.model_id,
            definition_hash=self.model_definition_hash,
        )
        if manifest.model_definition_hash != recipe.definition_hash:
            raise ValueError("formed manifest model does not match prepared training")
        if canonical_hash(manifest.tensor_schema) != recipe.tensor_schema_hash:
            raise ValueError(
                "formed manifest tensor schema does not match prepared training"
            )
        partition_index = manifest.dataset.node_index_partitions[node_index]
        policy = manifest.training
        if policy is None:
            raise ValueError("CIFAR-10 recipe requires an active training policy")
        noloco = manifest.algorithm_id == NOLOCO_ALGORITHM_ID
        config = manifest.algorithm_config
        if noloco and config is None:
            raise ValueError("NoLoCo trainer configuration is missing")
        adam = config.adam if config is not None else None
        return create_trainer(
            train_data=self._partitions[partition_index],
            test_data=self._test_data,
            settings=CIFAR10TrainerSettings(
                trainer=TrainerSettings(
                    seed=self.trainer_seed + node_index,
                    batch_size=policy.batch_size,
                    learning_rate=(
                        adam.learning_rate
                        if adam is not None
                        else manifest.learning_rate
                    ),
                    optimizer="adam" if noloco else "sgd",
                    momentum=0.0 if noloco else policy.momentum,
                    weight_decay=0.0 if noloco else policy.weight_decay,
                    adam_beta1=adam.beta1 if adam is not None else 0.9,
                    adam_beta2=adam.beta2 if adam is not None else 0.999,
                    adam_epsilon=adam.epsilon if adam is not None else 1e-8,
                    gradient_clip_norm=(
                        adam.gradient_clip_norm if adam is not None else None
                    ),
                    learning_rate_milestones=policy.learning_rate_milestones,
                    learning_rate_gamma=policy.learning_rate_gamma,
                    learning_rate_schedule=policy.learning_rate_schedule,
                    device=self.device,
                    augment=True,
                ),
                model_id=self.model_id,
                model_definition_hash=self.model_definition_hash,
                crop_padding=policy.crop_padding,
                normalize=policy.normalize,
            ),
        )


def prepare_training(
    *,
    draft: DraftRunSpec,
    cache_dir: Path,
    benchmark_seed: int,
    device: str = "cpu",
) -> PreparedCIFAR10Training:
    """Load and validate local data before membership becomes ready."""
    resolve_model(draft.model_id, definition_hash=draft.model_definition_hash)
    train_data = load_cifar10(cache_dir=cache_dir, train=True)
    test_data = load_cifar10(cache_dir=cache_dir, train=False)
    if len(train_data) != draft.dataset.sample_count:
        raise ValueError("local CIFAR-10 sample count does not match draft")
    partitions = train_data.split_iid(
        participant_count=draft.dataset.participant_count,
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
        model_id=draft.model_id,
        model_definition_hash=draft.model_definition_hash,
        device=device,
    )


__all__ = [
    "CLASS_COUNT",
    "CIFAR10DataError",
    "CIFAR10TrainerSettings",
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
