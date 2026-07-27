from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from dromeus.training.trainer import PyTorchTrainer


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
        batch_size=4,
        learning_rate=0.01,
        augment=False,
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
