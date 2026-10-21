"""Generic task behavior through the public trainer/application interfaces."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from dromeus.manifests.canonical import load_safetensors
from dromeus.training.trainer import PyTorchTrainer


class SharedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(2, 2, bias=False)
        self.right = nn.Linear(2, 2, bias=False)
        self.right.weight = self.left.weight
        self.frozen = nn.Parameter(torch.ones(2), requires_grad=False)
        self.register_buffer("public_constant", torch.ones(2))


def test_declared_exchange_preserves_ties_and_excludes_frozen_state(
    tmp_path: Path,
) -> None:
    model = SharedModel()
    trainer = PyTorchTrainer(
        model=model,
        train_step=lambda _: None,
        save_state=lambda: {},
        load_state=lambda _: None,
        state_identity="public-fixture",
        exchange_names=("left.weight",),
    )
    assert set(trainer.weights()) == {"left.weight"}
    initial = trainer.create_initial_checkpoint(
        tmp_path / "initial.safetensors", model_definition="shared-public-model"
    )
    assert {tensor.name for tensor in initial.tensor_schema.tensors} == {"left.weight"}
    assert set(load_safetensors(str(initial.path))) == {"left.weight"}
    trainer.load_weights({"left.weight": np.full((2, 2), 3, dtype=np.float32)})
    assert model.right.weight is model.left.weight
    torch.testing.assert_close(model.right.weight, torch.full((2, 2), 3.0))
    torch.testing.assert_close(model.frozen, torch.ones(2))
    torch.testing.assert_close(model.public_constant, torch.ones(2))
    checkpoint = trainer.checkpoint_tensors()
    assert "model.right.weight" in checkpoint
    trainer.load_checkpoint_tensors(checkpoint)


def test_generic_private_task_supports_token_tree_and_frozen_tied_model(
    tmp_path: Path,
) -> None:
    from typing import Any

    from torch.utils.data import Dataset

    from dromeus.application import describe_task
    from dromeus.manifests.models import PrivacyPolicy, PrivateOptimizerSpec
    from dromeus.persistence.privacy_ledger import PrivacyLedger
    from dromeus.training.private_trainer import PrivateTrainer
    from dromeus.training.task import TensorBatchAdapter, TrainingTask

    class Tokens(Dataset[dict[str, torch.Tensor]]):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return {"tokens": torch.tensor([index % 3, (index + 1) % 3])}

    class LanguageModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(3, 2)
            self.output = nn.Linear(2, 3, bias=False)
            self.output.weight = self.embedding.weight
            self.frozen = nn.Parameter(torch.ones(2), requires_grad=False)
            self.register_buffer("positions", torch.arange(2))

        def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, None]:
            return self.output(self.embedding(tokens) + self.frozen), None

    def factory() -> nn.Module:
        with torch.random.fork_rng():  # pyright: ignore[reportUnknownMemberType]
            torch.manual_seed(8)  # pyright: ignore[reportUnknownMemberType]
            return LanguageModel()

    def loss(model: nn.Module, batch: Any) -> torch.Tensor:
        logits, _ = model(batch["tokens"])
        return logits.square().mean(dim=(1, 2))

    task = TrainingTask(
        model_factory=factory,
        dataset=Tokens(),
        batch_adapter=TensorBatchAdapter(
            example={"tokens": torch.tensor([[0, 1], [1, 2]])}, definition="tokens-v1"
        ),
        loss_per_record=loss,
        model_definition="tied-language-fixture-v1",
        task_definition="finite-token-records-v1",
        loss_definition="mean-logit-square-v1",
        implementation_hash="a" * 64,
        record_unit="token-sequence",
    )
    spec = describe_task(
        task,
        optimizer=PrivateOptimizerSpec(
            name="sgd",
            torch_version=torch.__version__,
            parameter_names=("embedding.weight",),
            learning_rate=0.01,
        ),
        batch_size=4,
    )
    trainer = PrivateTrainer(
        task=task,
        policy=PrivacyPolicy(
            noise_multiplier=1,
            max_grad_norm=1,
            delta=1e-5,
            max_logical_steps=2,
            randomness_profile="reproducible_public_benchmark",
        ),
        specification=spec,
        run_id="generic-text",
        local_data_identity="tokens-fixture",
        reservation_store=PrivacyLedger.create(tmp_path / "ledger", lineage="tokens"),
        max_physical_batch_size=1,
        research_seed=11,
    )
    before = trainer.weights()
    assert set(before) == {"embedding.weight"}
    trainer.train_local_steps(1)
    assert not np.array_equal(
        before["embedding.weight"], trainer.weights()["embedding.weight"]
    )
    state = trainer.checkpoint_tensors()
    np.testing.assert_array_equal(
        state["model.embedding.weight"], state["model.output.weight"]
    )
    np.testing.assert_array_equal(state["model.frozen"], np.ones(2, dtype=np.float32))
    assert trainer.accounting_history == ((1.0, 1.0, 1),)
    assert trainer.local_loss is None
