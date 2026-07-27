"""Centralized FedAvg benchmark reference."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import TypeAdapter, ValidationError

from dromeus.manifests.models import (
    DatasetContract,
    EnvironmentFingerprint,
    Identifier,
    SealedManifest,
    Sha256,
)
from dromeus.training.pytorch import (
    CIFAR10Data,
    CIFAR10Trainer,
    checkpoint_hash,
    iid_partition_index_hashes,
)


@dataclass(frozen=True, slots=True)
class FedAvgConfig:
    """Frozen local-training settings shared with a D-PSGD run."""

    local_steps: int
    round_count: int
    learning_rate: float
    model_id: Identifier
    model_definition_hash: Sha256
    dataset: DatasetContract
    environment: EnvironmentFingerprint
    data_source: str
    test_sample_count: int
    trainer_seed: int = 17
    batch_size: int = 32
    device: str = "cpu"
    augment: bool = True

    @classmethod
    def from_manifest(
        cls,
        manifest: SealedManifest,
        *,
        trainer_seed: int = 17,
        batch_size: int = 32,
        device: str = "cpu",
        augment: bool = True,
    ) -> FedAvgConfig:
        """Reuse the sealed run's local optimizer and round settings."""
        return cls(
            local_steps=manifest.local_steps,
            round_count=manifest.round_count,
            learning_rate=manifest.learning_rate,
            model_id=manifest.model_id,
            model_definition_hash=manifest.model_definition_hash,
            dataset=manifest.dataset,
            environment=manifest.environment,
            data_source="torchvision-cifar10",
            test_sample_count=10_000,
            trainer_seed=trainer_seed,
            batch_size=batch_size,
            device=device,
            augment=augment,
        )

    def __post_init__(self) -> None:
        if self.local_steps <= 0 or self.round_count <= 0:
            raise ValueError("local_steps and round_count must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.test_sample_count <= 0:
            raise ValueError("test_sample_count must be positive")
        if not self.data_source:
            raise ValueError("data_source must not be empty")
        if self.learning_rate <= 0 or not math.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be positive and finite")
        if self.model_definition_hash != self.environment.model_definition_hash:
            raise ValueError("model definition hash does not match environment")

    def training_signature(self) -> tuple[object, ...]:
        """Return settings that must remain equal across benchmark methods."""
        return (
            self.local_steps,
            self.round_count,
            self.learning_rate,
            self.model_id,
            self.model_definition_hash,
            json.dumps(self.dataset.model_dump(mode="json"), sort_keys=True),
            json.dumps(self.environment.model_dump(mode="json"), sort_keys=True),
            self.data_source,
            self.test_sample_count,
            self.batch_size,
            self.device,
            self.augment,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "local_steps": self.local_steps,
            "round_count": self.round_count,
            "learning_rate": self.learning_rate,
            "model_id": self.model_id,
            "model_definition_hash": self.model_definition_hash,
            "dataset": self.dataset.model_dump(mode="json"),
            "environment": self.environment.model_dump(mode="json"),
            "data_source": self.data_source,
            "test_sample_count": self.test_sample_count,
            "trainer_seed": self.trainer_seed,
            "batch_size": self.batch_size,
            "device": self.device,
            "augment": self.augment,
        }


@dataclass(frozen=True, slots=True)
class FedAvgRound:
    """One central averaging round and its common-test evaluation."""

    round_id: int
    local_losses: tuple[float, ...]
    loss: float
    accuracy: float

    def as_dict(self) -> dict[str, object]:
        return {
            "round_id": self.round_id,
            "local_losses": list(self.local_losses),
            "loss": self.loss,
            "accuracy": self.accuracy,
        }


@dataclass(frozen=True, slots=True)
class FedAvgResult:
    """Deterministic centralized reference output."""

    rounds: tuple[FedAvgRound, ...]
    config: FedAvgConfig
    initial_checkpoint_hash: Sha256

    @property
    def final_loss(self) -> float:
        return self.rounds[-1].loss

    @property
    def final_accuracy(self) -> float:
        return self.rounds[-1].accuracy

    def as_dict(self) -> dict[str, object]:
        return {
            "rounds": [round_result.as_dict() for round_result in self.rounds],
            "final_loss": self.final_loss,
            "final_accuracy": self.final_accuracy,
            "config": self.config.as_dict(),
            "initial_checkpoint_hash": self.initial_checkpoint_hash,
        }

    def write(self, path: Path) -> None:
        """Write the canonical raw result used by aggregate reporting."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_dict(), allow_nan=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )


def average_weights(
    models: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Average equal-sized floating-point model weights centrally."""
    if not models:
        raise ValueError("at least one model is required")
    names = tuple(sorted(models[0]))
    if not names:
        raise ValueError("models must contain at least one tensor")
    if any(set(model) != set(names) for model in models):
        raise ValueError("model tensor names do not match")
    averaged: dict[str, np.ndarray] = {}
    for name in names:
        values = [np.asarray(model[name]) for model in models]
        reference = values[0]
        if not np.issubdtype(reference.dtype, np.floating):
            raise ValueError(f"tensor {name} must be floating point")
        if any(
            value.shape != reference.shape
            or value.dtype != reference.dtype
            or not np.isfinite(value).all()
            for value in values
        ):
            raise ValueError(f"tensor {name} schemas or values do not match")
        averaged[name] = np.mean(np.stack(values), axis=0, dtype=np.float64).astype(
            reference.dtype
        )
    return averaged


def run_fedavg(
    *,
    partitions: Sequence[CIFAR10Data],
    test_data: CIFAR10Data,
    initial_checkpoint: Path,
    config: FedAvgConfig,
) -> FedAvgResult:
    """Run four local blocks followed by central equal-weight averaging."""
    if len(partitions) != 4:
        raise ValueError("FedAvg reference requires exactly four partitions")
    if any(len(partition) == 0 for partition in partitions):
        raise ValueError("FedAvg partitions must not be empty")
    if not test_data.matches_source(source=config.data_source, split="test"):
        raise ValueError("FedAvg test data source does not match frozen config")
    if len(test_data) != config.test_sample_count:
        raise ValueError("FedAvg test data does not match frozen config")
    source_sample_count = sum(len(partition) for partition in partitions)
    if source_sample_count != config.dataset.sample_count:
        raise ValueError("FedAvg sample count does not match frozen config")
    if tuple(len(partition) for partition in partitions) != (
        config.dataset.partition_sample_counts
    ):
        raise ValueError("FedAvg partition sizes do not match frozen config")
    expected_hashes = iid_partition_index_hashes(
        source_sample_count=source_sample_count,
        participant_count=4,
        seed=config.dataset.iid_partition_seed,
    )
    for partition_index, partition in enumerate(partitions):
        provenance = partition.partition_provenance
        if (
            provenance is None
            or provenance.seed != config.dataset.iid_partition_seed
            or provenance.participant_count != 4
            or provenance.partition_index != partition_index
            or provenance.source_sample_count != source_sample_count
            or provenance.indices_sha256 != expected_hashes[partition_index]
            or not partition.matches_source(
                source=config.data_source,
                split="train",
            )
        ):
            raise ValueError("FedAvg partition provenance does not match config")

    trainers = tuple(
        CIFAR10Trainer(
            train_data=partition,
            test_data=test_data,
            seed=config.trainer_seed + index,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            device=config.device,
            augment=config.augment,
        )
        for index, partition in enumerate(partitions)
    )
    for trainer in trainers:
        trainer.load_checkpoint(initial_checkpoint)

    round_results: list[FedAvgRound] = []
    for round_id in range(config.round_count):
        for trainer in trainers:
            trainer.train_local_steps(config.local_steps)
        averaged = average_weights([trainer.weights() for trainer in trainers])
        for trainer in trainers:
            trainer.load_weights(averaged)
        loss, accuracy = trainers[0].evaluate()
        round_results.append(
            FedAvgRound(
                round_id=round_id,
                local_losses=tuple(
                    float(trainer.last_local_loss or 0.0) for trainer in trainers
                ),
                loss=float(loss),
                accuracy=float(accuracy),
            )
        )
    return FedAvgResult(
        rounds=tuple(round_results),
        config=config,
        initial_checkpoint_hash=checkpoint_hash(initial_checkpoint),
    )


@dataclass(frozen=True, slots=True)
class FedAvgSeedInput:
    """Inputs for one frozen-seed centralized reference run."""

    seed: int
    partitions: tuple[CIFAR10Data, ...]
    test_data: CIFAR10Data
    initial_checkpoint: Path
    config: FedAvgConfig


def run_fedavg_seeds(
    inputs: Sequence[FedAvgSeedInput],
) -> tuple[tuple[int, FedAvgResult], ...]:
    """Run exactly three FedAvg references with one shared configuration."""
    if len(inputs) != 3 or len({item.seed for item in inputs}) != 3:
        raise ValueError("exactly three distinct FedAvg seeds are required")
    if any(item.seed != item.config.trainer_seed for item in inputs):
        raise ValueError("FedAvg seed must match its trainer seed")
    signatures = {item.config.training_signature() for item in inputs}
    if len(signatures) != 1:
        raise ValueError("FedAvg seed configurations do not match")
    return tuple(
        (
            item.seed,
            run_fedavg(
                partitions=item.partitions,
                test_data=item.test_data,
                initial_checkpoint=item.initial_checkpoint,
                config=item.config,
            ),
        )
        for item in sorted(inputs, key=lambda value: value.seed)
    )


def load_fedavg_result(path: Path) -> FedAvgResult:
    """Load and validate one canonical raw FedAvg artifact."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return TypeAdapter(FedAvgResult).validate_python(value)
    except (OSError, ValueError, TypeError, ValidationError) as error:
        raise ValueError(f"invalid FedAvg result: {path}") from error


__all__ = [
    "FedAvgConfig",
    "FedAvgResult",
    "FedAvgRound",
    "FedAvgSeedInput",
    "average_weights",
    "load_fedavg_result",
    "run_fedavg",
    "run_fedavg_seeds",
]
