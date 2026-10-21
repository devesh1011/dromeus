"""Generic task records preserve their dtype, structure and sampling identity."""

from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch.utils.data import Dataset, IterableDataset

from dromeus.training.private_data import make_task_loader
from dromeus.training.task import TensorBatchAdapter


def fixture() -> dict[str, Any]:
    return {
        "tokens": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "extra": (torch.ones(2, 3, dtype=torch.bool), [torch.zeros(2, 1)]),
    }


def test_tensor_batch_retains_tokens_masks_and_container_shape() -> None:
    adapter = TensorBatchAdapter(example=fixture(), definition="token-batch-v1")
    batch = adapter.collate(
        [
            {
                "tokens": torch.tensor([9, 8, 7]),
                "extra": (torch.tensor([True, False, True]), [torch.tensor([2.0])]),
            },
            {
                "tokens": torch.tensor([6, 5, 4]),
                "extra": (torch.tensor([False, True, True]), [torch.tensor([3.0])]),
            },
            {
                "tokens": torch.tensor([3, 2, 1]),
                "extra": (torch.tensor([True, True, False]), [torch.tensor([4.0])]),
            },
        ]
    )
    assert adapter.size(batch) == 3
    assert batch["tokens"].dtype == torch.int64
    assert batch["extra"][0].dtype == torch.bool
    assert isinstance(batch["extra"], tuple)
    assert isinstance(batch["extra"][1], list)
    sliced = adapter.slice(batch, 1, 3)
    assert sliced["tokens"].tolist() == [[6, 5, 4], [3, 2, 1]]
    assert adapter.size(adapter.empty()) == 0
    assert adapter.empty()["tokens"].shape == (0, 3)
    assert adapter.size(adapter.to_device(batch, "cpu")) == 3


class Records(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, count: int = 10) -> None:
        self.count = count
        self.reads = 0

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        self.reads += 1
        return {"tokens": torch.tensor([index, index + 1])}


def test_loader_preparation_never_reads_private_records() -> None:
    pytest.importorskip("opacus")
    records = Records()
    adapter = TensorBatchAdapter(
        example={"tokens": torch.tensor([[0, 1], [1, 2]])},
        definition="tokens-v1",
    )
    loader = make_task_loader(
        records, adapter, batch_size=4, generator=torch.Generator().manual_seed(17)
    )
    assert records.reads == 0
    assert loader.sample_rate == 1 / 3
    assert len(loader) == 3
    assert loader.population_size == 10
    batch = next(iter(loader))
    assert records.reads == adapter.size(batch)
    assert records.reads > 0


def test_batch_definition_binds_dtype_and_shape_but_not_public_fixture_values() -> None:
    first = TensorBatchAdapter(example=torch.zeros(2, 3), definition="tokens-v1")
    second = TensorBatchAdapter(example=torch.ones(5, 3), definition="tokens-v1")
    integers = TensorBatchAdapter(
        example=torch.ones(2, 3, dtype=torch.int64), definition="tokens-v1"
    )
    wider = TensorBatchAdapter(example=torch.zeros(2, 4), definition="tokens-v1")
    assert first.definition_hash == second.definition_hash
    assert first.definition_hash != integers.definition_hash
    assert first.definition_hash != wider.definition_hash


def test_empty_draw_does_not_read_or_resample_a_record() -> None:
    pytest.importorskip("opacus")
    records = Records()
    adapter = TensorBatchAdapter(
        example={"tokens": torch.tensor([[0, 1], [1, 2]])},
        definition="tokens-v1",
    )
    loader = make_task_loader(
        records, adapter, batch_size=1, generator=torch.Generator().manual_seed(7)
    )
    batch = next(iter(loader))
    assert adapter.size(batch) == 0
    assert records.reads == 0
    assert loader.sample_rate == 0.1


def test_population_mutation_rejects_before_another_record_read() -> None:
    pytest.importorskip("opacus")
    records = Records()
    adapter = TensorBatchAdapter(
        example={"tokens": torch.tensor([[0, 1], [1, 2]])},
        definition="tokens-v1",
    )
    loader = make_task_loader(records, adapter, batch_size=4, generator=None)
    records.count = 11
    with pytest.raises(ValueError, match="population changed"):
        next(iter(loader))
    assert records.reads == 0


@pytest.mark.parametrize(
    "bad",
    [torch.zeros(2, 4), torch.zeros(2, 3, dtype=torch.int64), torch.tensor(1.0)],
)
def test_batch_schema_rejects_incompatible_tensors(bad: torch.Tensor) -> None:
    adapter = TensorBatchAdapter(example=torch.zeros(2, 3), definition="float-v1")
    with pytest.raises(ValueError, match="shape|dtype|record axis"):
        adapter.size(bad)


def test_batch_schema_rejects_misaligned_records_and_collator_filtering() -> None:
    with pytest.raises(ValueError, match="aligned records"):
        TensorBatchAdapter(
            example={"tokens": torch.zeros(2, 3), "targets": torch.zeros(3, 3)},
            definition="bad-fixture",
        )
    adapter = TensorBatchAdapter(
        example=torch.zeros(2, 3),
        definition="filter-v1",
        collate_fn=lambda records: torch.stack(list(records[:-1])),
    )
    with pytest.raises(ValueError, match="record count"):
        adapter.collate([torch.zeros(3), torch.ones(3)])


def test_record_loss_rejects_scalar_nonfinite_and_detached_values() -> None:
    from dromeus.training.task import validate_record_losses

    good = torch.tensor([0.0, 2.0], requires_grad=True)
    assert validate_record_losses(good, 2) is good
    for bad in (
        good.mean(),
        torch.tensor([float("nan"), 1.0], requires_grad=True),
        good.detach(),
    ):
        with pytest.raises(ValueError, match="per-record"):
            validate_record_losses(bad, 2)
    assert validate_record_losses(torch.empty(0), 0).numel() == 0


class Unbounded(IterableDataset[torch.Tensor]):
    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:
            yield torch.ones(3)


def test_unknown_population_fails_before_loader_construction() -> None:
    adapter = TensorBatchAdapter(example=torch.zeros(2, 3), definition="float-v1")
    with pytest.raises(ValueError, match="finite indexed"):
        make_task_loader(Unbounded(), adapter, batch_size=4, generator=None)


def test_public_fixture_is_copied_from_the_application() -> None:
    source = torch.tensor([[1, 2], [3, 4]])
    adapter = TensorBatchAdapter(example=source, definition="tokens-v1")
    source.fill_(99)
    assert adapter.example.tolist() == [[1, 2], [3, 4]]
    assert adapter.empty().dtype == torch.int64


def test_task_definition_matches_shared_contract_and_rejects_loss_change() -> None:
    import hashlib

    from dromeus.manifests.models import PrivateOptimizerSpec, TaskTrainingSpec
    from dromeus.training.task import TrainingTask

    adapter = TensorBatchAdapter(
        example={"tokens": torch.tensor([[0, 1], [1, 2]])}, definition="tokens-v1"
    )
    task = TrainingTask(
        model_factory=lambda: torch.nn.Linear(2, 1),
        dataset=Records(),
        batch_adapter=adapter,
        loss_per_record=lambda model, batch: model(batch["tokens"].float()).sum(1),
        model_definition="linear-v1",
        task_definition="next-token-v1",
        loss_definition="sum-v1",
        implementation_hash="1" * 64,
    )
    spec = TaskTrainingSpec(
        optimizer=PrivateOptimizerSpec(
            name="sgd",
            torch_version=torch.__version__,
            parameter_names=("weight", "bias"),
            learning_rate=0.01,
        ),
        batch_size=2,
        task_definition_hash=hashlib.sha256(b"next-token-v1").hexdigest(),
        implementation_hash="1" * 64,
        batch_definition_hash=adapter.definition_hash,
        loss_definition_hash=hashlib.sha256(b"sum-v1").hexdigest(),
        record_unit="example",
        trainable_parameter_names=("weight", "bias"),
    )
    task.check_definition(spec)
    with pytest.raises(ValueError, match="loss"):
        task.check_definition(
            spec.model_copy(update={"loss_definition_hash": "2" * 64})
        )
