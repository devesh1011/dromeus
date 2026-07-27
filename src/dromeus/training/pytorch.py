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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import (  # pyright: ignore[reportMissingTypeStubs]
    CIFAR10 as TorchvisionCIFAR10,  # pyright: ignore[reportMissingTypeStubs]
)
from torchvision.transforms import ToTensor  # pyright: ignore[reportMissingTypeStubs]

from dromeus.manifests.models import (
    CIFAR_CNN_MODEL_ID,
    CIFAR_RESNET32_MODEL_ID,
    DraftRunSpec,
    SealedManifest,
    TensorSchema,
)
from dromeus.manifests.models import Tensor as TensorSpec

IMAGE_SHAPE = (3, 32, 32)
CLASS_COUNT = 10
CIFAR10_ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR10_DATASET_VERSION = f"torchvision-python-{CIFAR10_ARCHIVE_MD5}"
PREPROCESSING_DEFINITION = (
    "torchvision.ToTensor;seeded-horizontal-flip:p=0.5;normalization=none"
)
PREPROCESSING_HASH = hashlib.sha256(PREPROCESSING_DEFINITION.encode()).hexdigest()
MODEL_DEFINITION = "cifar-cnn-v1:conv3x3-16:gn4:conv3x3-32:gn4:gap:linear"
MODEL_DEFINITION_HASH = hashlib.sha256(MODEL_DEFINITION.encode()).hexdigest()
CIFAR_RESNET32_PREPROCESSING_DEFINITION = (
    "torchvision.ToTensor;seeded-reflect-crop:padding=4;"
    "seeded-horizontal-flip:p=0.5;"
    "channel-normalization:mean=0.4914,0.4822,0.4465:"
    "std=0.2470,0.2435,0.2616"
)
CIFAR_RESNET32_PREPROCESSING_HASH = hashlib.sha256(
    CIFAR_RESNET32_PREPROCESSING_DEFINITION.encode()
).hexdigest()
CIFAR_RESNET32_MODEL_DEFINITION = (
    "cifar-resnet32-v2:conv3x3-16:bn:"
    "basic-blocks=5,5,5:channels=16,32,64:option-a-shortcut:gap:linear"
)
CIFAR_RESNET32_MODEL_DEFINITION_HASH = hashlib.sha256(
    CIFAR_RESNET32_MODEL_DEFINITION.encode()
).hexdigest()
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
_MODEL_DEFINITIONS = {
    CIFAR_CNN_MODEL_ID: MODEL_DEFINITION,
    CIFAR_RESNET32_MODEL_ID: CIFAR_RESNET32_MODEL_DEFINITION,
}
_TRAINING_STATE_PREFIX = "__dromeus_training__."
_COMPLETED_STEPS = f"{_TRAINING_STATE_PREFIX}completed_steps"
_BATCHES_CONSUMED = f"{_TRAINING_STATE_PREFIX}batches_consumed"
_AUGMENTATION_RNG = f"{_TRAINING_STATE_PREFIX}augmentation_rng"
_LOADER_EPOCH_RNG = f"{_TRAINING_STATE_PREFIX}loader_epoch_rng"
_MOMENTUM_PREFIX = f"{_TRAINING_STATE_PREFIX}momentum."
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


class CIFARBasicBlock(nn.Module):
    """Original CIFAR ResNet basic block with parameter-free option-A shortcut."""

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
            right = self._channel_padding - left
            residual = F.pad(residual, (0, 0, 0, 0, left, right))
        return F.relu(output + residual, inplace=False)


class CIFARResNet32(nn.Module):
    """ResNet-32 for 32x32 CIFAR inputs (6n+2 with n=5)."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
        )
        stages: list[nn.Module] = []
        in_channels = 16
        for stage_index, out_channels in enumerate((16, 32, 64)):
            blocks: list[nn.Module] = []
            for block_index in range(5):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                blocks.append(
                    CIFARBasicBlock(in_channels, out_channels, stride=stride)
                )
                in_channels = out_channels
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.classifier = nn.Linear(64, CLASS_COUNT)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, images: Tensor) -> Tensor:
        features = self.stages(self.stem(images))
        features = F.avg_pool2d(features, features.shape[-1]).flatten(1)
        return self.classifier(features)


def build_model(
    *, seed: int, model_id: str = CIFAR_CNN_MODEL_ID
) -> nn.Module:
    """Construct a reproducibly initialized model without changing caller RNG."""
    with torch.random.fork_rng(devices=[]):  # pyright: ignore[reportUnknownMemberType]
        torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
        if model_id == CIFAR_CNN_MODEL_ID:
            return CIFARGroupNormCNN()
        if model_id == CIFAR_RESNET32_MODEL_ID:
            return CIFARResNet32()
        raise ValueError(f"unsupported CIFAR model_id: {model_id}")


def derive_benchmark_seed(benchmark_seed: int, purpose: str) -> int:
    """Derive one stable RNG seed for a named benchmark concern."""
    digest = hashlib.sha256(f"{benchmark_seed}:{purpose}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _model_definition(model_id: str) -> str:
    try:
        return _MODEL_DEFINITIONS[model_id]
    except KeyError as error:
        raise ValueError(f"unsupported CIFAR model_id: {model_id}") from error


def _floating_model_state(model: nn.Module) -> dict[str, Tensor]:
    """Return exchangeable state, excluding integer BatchNorm counters."""
    return {
        name: value
        for name, value in model.state_dict().items()
        if value.is_floating_point()
    }


def tensor_schema_for_model(model: nn.Module | None = None) -> TensorSchema:
    """Derive the wire schema from parameters and floating-point model buffers."""
    target = model or CIFARGroupNormCNN()
    specs: list[TensorSpec] = []
    for name, value in _floating_model_state(target).items():
        dtype = _TORCH_TO_SCHEMA_DTYPE.get(value.dtype)
        if dtype is None:
            raise CIFARDataError(f"unsupported model state dtype: {value.dtype}")
        specs.append(TensorSpec(name=name, dtype=dtype, shape=tuple(value.shape)))
    return TensorSchema(tensors=tuple(specs))


def create_initial_checkpoint(
    path: Path,
    *,
    seed: int,
    model_id: str = CIFAR_CNN_MODEL_ID,
) -> InitialCheckpoint:
    """Write and describe the canonical checkpoint before formation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(seed=seed, model_id=model_id)
    _save_checkpoint(
        {
            name: value.detach().cpu()
            for name, value in _floating_model_state(model).items()
        },
        str(path),
        metadata={"model_definition": _model_definition(model_id)},
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
        model_id: str = CIFAR_CNN_MODEL_ID,
        batch_size: int = 32,
        learning_rate: float = 0.1,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        learning_rate_milestones: tuple[int, ...] = (),
        learning_rate_gamma: float = 0.1,
        device: str = "cpu",
        augment: bool = True,
        crop_padding: int = 0,
        normalize: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if learning_rate <= 0 or not np.isfinite(learning_rate):
            raise ValueError("learning_rate must be positive and finite")
        if not 0 <= momentum < 1 or not np.isfinite(momentum):
            raise ValueError("momentum must be finite in [0, 1)")
        if weight_decay < 0 or not np.isfinite(weight_decay):
            raise ValueError("weight_decay must be finite and non-negative")
        if (
            any(milestone <= 0 for milestone in learning_rate_milestones)
            or any(
                right <= left
                for left, right in zip(
                    learning_rate_milestones,
                    learning_rate_milestones[1:],
                    strict=False,
                )
            )
        ):
            raise ValueError("learning-rate milestones must be strictly increasing")
        if not 0 < learning_rate_gamma < 1 or not np.isfinite(learning_rate_gamma):
            raise ValueError("learning_rate_gamma must be finite in (0, 1)")
        if crop_padding < 0:
            raise ValueError("crop_padding must be non-negative")
        self._device = torch.device(device)
        self._model_id = model_id
        self._model = build_model(seed=seed, model_id=model_id).to(self._device)
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
        self._crop_padding = crop_padding
        self._normalize = normalize
        self._augmentation_generator = torch.Generator(device="cpu")
        self._augmentation_generator.manual_seed(seed + 1)
        self._loader_generator = torch.Generator(device="cpu")
        self._loader_generator.manual_seed(seed + 2)
        self._batch_size = batch_size
        self._train_data = train_data
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

    @staticmethod
    def tensor_schema_for_model(
        model_id: str = CIFAR_CNN_MODEL_ID,
    ) -> TensorSchema:
        return tensor_schema_for_model(build_model(seed=0, model_id=model_id))

    def weights(self) -> dict[str, np.ndarray]:
        return {
            name: value.detach().cpu().numpy().copy()
            for name, value in _floating_model_state(self._model).items()
        }

    def load_weights(self, weights: dict[str, np.ndarray]) -> None:
        expected = _floating_model_state(self._model)
        if set(weights) != set(expected):
            raise ValueError("weight names do not match model")
        with torch.no_grad():
            for name, target in expected.items():
                value = np.asarray(weights[name])
                if value.shape != tuple(target.shape) or value.dtype != np.float32:
                    raise ValueError(f"weight {name} does not match model")
                target.copy_(
                    torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
                        cast(Any, np.ascontiguousarray(value))
                    ).to(self._device)
                )

    def checkpoint_tensors(self) -> dict[str, np.ndarray]:
        """Return model, optimizer, schedule, and stochastic-loader state."""
        state = self.weights()
        state[_COMPLETED_STEPS] = np.array(
            [self._completed_steps], dtype=np.int64
        )
        state[_BATCHES_CONSUMED] = np.array(
            [self._batches_consumed], dtype=np.int64
        )
        state[_AUGMENTATION_RNG] = (
            self._augmentation_generator.get_state().cpu().numpy().copy()
        )
        state[_LOADER_EPOCH_RNG] = (
            self._epoch_generator_state.cpu().numpy().copy()
        )
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
        model_names = set(_floating_model_state(self._model))
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
            if value.dtype != np.float32 or value.shape != tuple(parameter.shape):
                raise ValueError(f"momentum state {name} does not match model")
            self._optimizer.state[parameter]["momentum_buffer"] = (
                torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
                    cast(Any, np.ascontiguousarray(value))
                ).to(self._device)
            )

        augmentation_state = _rng_state_tensor(
            state[_AUGMENTATION_RNG], name="augmentation"
        )
        loader_epoch_state = _rng_state_tensor(
            state[_LOADER_EPOCH_RNG], name="loader"
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

    @property
    def last_local_loss(self) -> float | None:
        return self._last_local_loss

    @property
    def learning_rate(self) -> float:
        return float(self._optimizer.param_groups[0]["lr"])

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
                images = self._prepare_images(images.to(self._device), augment=False)
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
                for name, value in _floating_model_state(self._model).items()
            },
            str(path),
            metadata={"model_definition": _model_definition(self._model_id)},
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
            self._epoch_generator_state = self._loader_generator.get_state().clone()
            self._train_iterator = iter(self._train_loader)
            self._batches_consumed = 0
            images, labels = next(self._train_iterator)
        self._batches_consumed += 1
        images = self._prepare_images(images.to(self._device), augment=self._augment)
        labels = labels.to(self._device)
        return images, labels

    def _prepare_images(self, images: Tensor, *, augment: bool) -> Tensor:
        if augment and self._crop_padding:
            padded = F.pad(
                images,
                (self._crop_padding,) * 4,
                mode="reflect",
            )
            maximum = self._crop_padding * 2 + 1
            rows = torch.randint(
                maximum,
                (images.shape[0],),
                generator=self._augmentation_generator,
            )
            columns = torch.randint(
                maximum,
                (images.shape[0],),
                generator=self._augmentation_generator,
            )
            rows = rows.to(images.device)
            columns = columns.to(images.device)
            batch_indices = torch.arange(
                images.shape[0], device=images.device
            ).view(-1, 1, 1)
            row_indices = rows.view(-1, 1, 1) + torch.arange(
                IMAGE_SHAPE[1], device=images.device
            ).view(1, -1, 1)
            column_indices = columns.view(-1, 1, 1) + torch.arange(
                IMAGE_SHAPE[2], device=images.device
            ).view(1, 1, -1)
            images = padded.permute(0, 2, 3, 1)[
                batch_indices,
                row_indices,
                column_indices,
            ].permute(0, 3, 1, 2)
        if augment:
            flips = (
                torch.rand(images.shape[0], generator=self._augmentation_generator)
                < 0.5
            ).to(images.device)
            if bool(flips.any()):
                images = images.clone()
                images[flips] = torch.flip(images[flips], dims=(3,))
        if self._normalize:
            mean = images.new_tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
            std = images.new_tensor(CIFAR10_STD).view(1, 3, 1, 1)
            images = (images - mean) / std
        return images

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


@dataclass(frozen=True, slots=True)
class PreparedCIFARTraining:
    """Local CIFAR data retained behind training-owned construction methods."""

    _partitions: tuple[CIFAR10Data, ...]
    _test_data: CIFAR10Data
    initialization_seed: int
    trainer_seed: int

    def create_initial_checkpoint(
        self,
        path: Path,
        *,
        model_id: str = CIFAR_CNN_MODEL_ID,
    ) -> InitialCheckpoint:
        return create_initial_checkpoint(
            path,
            seed=self.initialization_seed,
            model_id=model_id,
        )

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
        policy = manifest.training
        return CIFAR10Trainer(
            train_data=self._partitions[partition_index],
            test_data=self._test_data,
            seed=self.trainer_seed + node_index,
            model_id=manifest.model_id,
            batch_size=policy.batch_size if policy is not None else 32,
            learning_rate=manifest.learning_rate,
            momentum=policy.momentum if policy is not None else 0.0,
            weight_decay=policy.weight_decay if policy is not None else 0.0,
            learning_rate_milestones=(
                policy.learning_rate_milestones if policy is not None else ()
            ),
            learning_rate_gamma=(
                policy.learning_rate_gamma if policy is not None else 0.1
            ),
            device="cpu",
            augment=True,
            crop_padding=policy.crop_padding if policy is not None else 0,
            normalize=policy.normalize if policy is not None else False,
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
        initialization_seed=derive_benchmark_seed(
            benchmark_seed,
            "model-initialization",
        ),
        trainer_seed=derive_benchmark_seed(benchmark_seed, "local-training"),
    )


__all__ = [
    "CIFAR10_ARCHIVE_MD5",
    "CIFAR10_DATASET_VERSION",
    "CIFAR10Data",
    "CIFAR10Trainer",
    "CIFARBasicBlock",
    "CIFARDataProvenance",
    "CIFARDataError",
    "CIFARGroupNormCNN",
    "CIFARResNet32",
    "CIFAR_CNN_MODEL_ID",
    "CIFAR_RESNET32_MODEL_DEFINITION",
    "CIFAR_RESNET32_MODEL_DEFINITION_HASH",
    "CIFAR_RESNET32_MODEL_ID",
    "CIFAR_RESNET32_PREPROCESSING_DEFINITION",
    "CIFAR_RESNET32_PREPROCESSING_HASH",
    "InitialCheckpoint",
    "IIDPartitionProvenance",
    "MODEL_DEFINITION",
    "MODEL_DEFINITION_HASH",
    "PREPROCESSING_DEFINITION",
    "PREPROCESSING_HASH",
    "PreparedCIFARTraining",
    "build_model",
    "checkpoint_hash",
    "create_initial_checkpoint",
    "derive_benchmark_seed",
    "iid_partition_index_hashes",
    "iid_partition_indices",
    "prepare_cifar_training",
    "tensor_schema_for_model",
]
