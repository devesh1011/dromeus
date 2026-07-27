"""Centralized FedAvg benchmark reference."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dromeus.manifests.models import SealedManifest
from dromeus.training.pytorch import CIFAR10Data, CIFAR10Trainer


@dataclass(frozen=True, slots=True)
class FedAvgConfig:
    """Frozen local-training settings shared with a D-PSGD run."""

    local_steps: int
    round_count: int
    learning_rate: float
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
        if self.learning_rate <= 0 or not math.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be positive and finite")


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
        }


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
        averaged[name] = np.mean(
            np.stack(values), axis=0, dtype=np.float64
        ).astype(reference.dtype)
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
    return FedAvgResult(rounds=tuple(round_results))


__all__ = [
    "FedAvgConfig",
    "FedAvgResult",
    "FedAvgRound",
    "average_weights",
    "run_fedavg",
]
