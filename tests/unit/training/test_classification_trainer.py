from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from dromeus.adapters.classification.torch_trainer import (
    PyTorchTrainer,
    TrainerSettings,
)
from dromeus.manifests.models import WarmupCosineSchedule


def test_trainer_settings_reject_invalid_optimizer_configuration() -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        TrainerSettings(batch_size=0)

    with pytest.raises(ValueError, match="warmup-cosine and milestone"):
        TrainerSettings(
            learning_rate_milestones=(10,),
            learning_rate_schedule=WarmupCosineSchedule(
                schedule_id="linear-warmup-cosine-v1",
                total_inner_steps=10,
                warmup_inner_steps=1,
                start_learning_rate=0.01,
                peak_learning_rate=0.1,
                final_learning_rate=0.01,
            ),
        )


def test_trainer_accepts_a_generic_model_and_classification_dataset() -> None:
    images = torch.randn(8, 1, 4, 4)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    data = TensorDataset(images, labels)
    model = nn.Sequential(nn.Flatten(), nn.Linear(16, 3))
    trainer = PyTorchTrainer(
        model=model,
        model_definition="test-linear-classifier",
        train_data=data,  # pyright: ignore[reportArgumentType]
        test_data=data,  # pyright: ignore[reportArgumentType]
        settings=TrainerSettings(
            batch_size=4,
            learning_rate=0.01,
            augment=False,
        ),
    )
    before = trainer.weights()

    trainer.train_local_steps(1)
    loss, accuracy = trainer.evaluate()

    assert any(
        not np.array_equal(before[name], value)
        for name, value in trainer.weights().items()
    )
    assert np.isfinite(loss)
    assert 0 <= accuracy <= 1


def test_trainer_runs_clipped_adam_and_restores_moments() -> None:
    images = torch.full((4, 1), 1_000.0)
    labels = torch.zeros(4, dtype=torch.long)
    data = TensorDataset(images, labels)
    model = nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    trainer = PyTorchTrainer(
        model=model,
        model_definition="test-linear-adam",
        train_data=data,  # pyright: ignore[reportArgumentType]
        settings=TrainerSettings(
            batch_size=4,
            learning_rate=0.001,
            optimizer="adam",
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            gradient_clip_norm=1.0,
            augment=False,
        ),
    )

    trainer.train_local_steps(1)
    state = trainer.checkpoint_tensors()
    first_moment = state["__dromeus_training__.adam.exp_avg.weight"]

    assert 0 < float(np.max(np.abs(first_moment))) <= 0.1
    assert "__dromeus_training__.adam.exp_avg_sq.weight" in state
    assert "__dromeus_training__.adam.step.weight" in state

    restored_model = nn.Linear(1, 2, bias=False)
    restored = PyTorchTrainer(
        model=restored_model,
        model_definition="test-linear-adam",
        train_data=data,  # pyright: ignore[reportArgumentType]
        settings=TrainerSettings(
            batch_size=4,
            learning_rate=0.001,
            optimizer="adam",
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            gradient_clip_norm=1.0,
            augment=False,
        ),
    )
    restored.load_checkpoint_tensors(state)

    trainer.train_local_steps(1)
    restored.train_local_steps(1)

    assert all(
        np.array_equal(value, restored.weights()[name])
        for name, value in trainer.weights().items()
    )
