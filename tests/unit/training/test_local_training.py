from __future__ import annotations

import hashlib
import json
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from support.sample_manifest import manifest_data
from torch import Tensor, nn
from torch.utils.data import Dataset

from dromeus.adapters.classification.data import LocalClassificationData
from dromeus.adapters.classification.torch_trainer import PyTorchTrainer
from dromeus.adapters.classification.trainer import (
    LocalClassificationTrainer,
    PreparedLocalTraining,
    prepare_local_training,
)
from dromeus.manifests.canonical import load_safetensors, save_safetensors
from dromeus.manifests.models import (
    ClassificationTaskContract,
    DraftRunSpec,
    SealedManifest,
    TensorSchema,
)
from dromeus.membership.formation import seal_manifest
from dromeus.training.base import CheckpointTrainer, WeightTrainer

_DEFINITION = "custom test classifier; float32; input=2; output=2"
_DEFINITION_HASH = hashlib.sha256(_DEFINITION.encode()).hexdigest()
_PREPROCESSING_HASH = "a" * 64
_FINGERPRINT = "__dromeus_local__.data_fingerprint"


@pytest.fixture(autouse=True)
def single_torch_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


class _Data(Dataset[object]):
    def __init__(
        self, count: int = 7, *, offset: float = 0.0, label: int | None = None
    ):
        self.inputs = torch.arange(count * 2, dtype=torch.float32).reshape(count, 2)
        self.inputs = self.inputs / 10.0 + offset
        self.labels = [index % 2 if label is None else label for index in range(count)]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> object:
        return self.inputs[index], self.labels[index]


def _data(
    train: Dataset[object] | None = None, *, evaluation: bool = True
) -> LocalClassificationData:
    return LocalClassificationData(
        train_data=_Data() if train is None else train,
        evaluation_data=_Data(3, offset=5.0) if evaluation else None,
        label_names=("cat", "truck"),
        preprocessing_hash=_PREPROCESSING_HASH,
    )


def _model(*, dropout: bool = False) -> nn.Module:
    if dropout:
        return nn.Sequential(
            nn.Linear(2, 8), nn.ReLU(), nn.Dropout(0.5), nn.Linear(8, 2)
        )
    return nn.Linear(2, 2)


def _draft(*, adam: bool = False) -> DraftRunSpec:
    data = manifest_data()
    for name in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del data[name]
    data.update(
        manifest_version=4,
        expected_participant_count=4,
        model_id="custom-classifier-v1",
        model_definition_hash=_DEFINITION_HASH,
        dataset=ClassificationTaskContract(
            dataset_id="local-classification-v1",
            input_shape=(2,),
            input_dtype="float32",
            label_names=("cat", "truck"),
            preprocessing_hash=_PREPROCESSING_HASH,
        ),
    )
    data["environment"]["model_definition_hash"] = _DEFINITION_HASH
    data["training"].update(
        crop_padding=0,
        normalize=False,
        batch_size=4,
        momentum=0.8,
        weight_decay=0.01,
        learning_rate_milestones=[3, 7],
    )
    if adam:
        data.update(
            optimizer="adam",
            algorithm_id="noloco",
            local_steps=50,
            learning_rate=0.001,
            round_count=2,
            algorithm_config={
                "alpha": 0.5,
                "beta": 0.7,
                "gamma": 0.7,
                "inner_steps": 50,
                "adam": {
                    "learning_rate": 0.001,
                    "beta1": 0.8,
                    "beta2": 0.95,
                    "epsilon": 1e-7,
                    "gradient_clip_norm": 1.0,
                },
            },
            artifact_codecs=(
                {"artifact_name": "outer_gradient", "codec_id": "identity-v1"},
                {"artifact_name": "slow_weights", "codec_id": "identity-v1"},
            ),
        )
        data["transport"].update(chunk_size_bytes=1024, window_size=1)
        data["training"].update(
            learning_rate_milestones=[],
            learning_rate_schedule={
                "schedule_id": "linear-warmup-cosine-v1",
                "total_inner_steps": 100,
                "warmup_inner_steps": 5,
                "start_learning_rate": 0.0001,
                "peak_learning_rate": 0.001,
                "final_learning_rate": 0.0001,
            },
        )
    return DraftRunSpec.model_validate(data)


def _prepare(
    *,
    draft: DraftRunSpec | None = None,
    data: LocalClassificationData | None = None,
    model: nn.Module | None = None,
) -> PreparedLocalTraining:
    return prepare_local_training(
        draft=_draft() if draft is None else draft,
        data=_data() if data is None else data,
        model=_model() if model is None else model,
        model_definition=_DEFINITION,
        seed=17,
    )


def _seal(
    prepared: PreparedLocalTraining, root: Path, draft: DraftRunSpec
) -> tuple[SealedManifest, Path]:
    initial = prepared.create_initial_checkpoint(root / "initial.safetensors")
    return (
        seal_manifest(
            draft=draft,
            participant_keys={f"peer-{index}" for index in range(4)},
            tensor_schema=initial.tensor_schema,
            initial_checkpoint_hash=initial.sha256,
        ),
        initial.path,
    )


def _trainer(
    prepared: PreparedLocalTraining, root: Path, draft: DraftRunSpec
) -> LocalClassificationTrainer:
    manifest, path = _seal(prepared, root, draft)
    trainer = prepared.create_trainer(manifest=manifest, local_public_key="peer-0")
    trainer.load_checkpoint(path)
    return trainer


def _same_state(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> None:
    assert first.keys() == second.keys()
    for name in first:
        assert np.array_equal(first[name], second[name]), name


def test_custom_model_and_data_train_without_mutating_supplied_or_prepared_model(
    tmp_path: Path,
) -> None:
    model = _model()
    model.eval()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    draft = _draft()
    prepared = _prepare(draft=draft, model=model)
    first = _trainer(prepared, tmp_path, draft)
    second = _trainer(prepared, tmp_path / "second", draft)
    baseline = second.weights()
    interface: WeightTrainer = first

    assert isinstance(first, CheckpointTrainer)
    first.train_local_steps(4)
    assert interface.local_loss is not None
    assert interface.evaluate() is not None
    assert first.settings.seed == 17
    assert not first.settings.augment
    assert first.settings.weight_decay == 0.01
    assert first.settings.momentum == 0.8
    assert first.learning_rate == pytest.approx(0.01)
    assert prepared.tensor_schema == first.tensor_schema
    assert not model.training
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])
    _same_state(second.weights(), baseline)
    assert any(
        not np.array_equal(first.weights()[key], baseline[key]) for key in baseline
    )


@pytest.mark.parametrize("count", [1, 3, 13])
def test_unequal_tiny_missing_class_data_is_valid_and_has_no_eval_fallback(
    count: int, tmp_path: Path
) -> None:
    draft = _draft()
    prepared = _prepare(
        draft=draft, data=_data(_Data(count, label=1), evaluation=False)
    )
    trainer = _trainer(prepared, tmp_path, draft)

    assert prepared.data_metadata.train_sample_count == count
    assert prepared.data_metadata.training_label_counts == (0, count)
    assert prepared.data_metadata.evaluation_sample_count is None
    assert trainer.evaluate() is None
    trainer.train_local_steps(3)
    assert trainer.local_loss is not None
    assert trainer.evaluate() is None


def test_adam_projection_preserves_sealed_schedule_and_hyperparameters(
    tmp_path: Path,
) -> None:
    draft = _draft(adam=True)
    prepared = _prepare(draft=draft)
    manifest, _ = _seal(prepared, tmp_path, draft)
    trainer = prepared.create_trainer(manifest=manifest, local_public_key="peer-2")

    assert trainer.settings.seed == 19
    assert trainer.settings.optimizer == "adam"
    assert trainer.settings.learning_rate == 0.001
    assert trainer.settings.adam_beta1 == 0.8
    assert trainer.settings.adam_beta2 == 0.95
    assert trainer.settings.adam_epsilon == 1e-7
    assert trainer.settings.gradient_clip_norm == 1.0
    assert trainer.settings.weight_decay == 0.01
    assert draft.training is not None
    assert (
        trainer.settings.learning_rate_schedule == draft.training.learning_rate_schedule
    )
    trainer.train_local_steps(50)
    assert trainer.local_loss is not None


@pytest.mark.parametrize("adam", [False, True])
def test_durable_checkpoint_continuation_is_bit_identical_with_dropout(
    adam: bool, tmp_path: Path
) -> None:
    draft = _draft(adam=adam)
    prepared = _prepare(draft=draft, model=_model(dropout=True))
    first = _trainer(prepared, tmp_path, draft)
    second = _trainer(prepared, tmp_path / "second", draft)
    global_rng = torch.get_rng_state().clone()
    first.train_local_steps(3)
    assert torch.equal(torch.get_rng_state(), global_rng)
    checkpoint = tmp_path / "durable.safetensors"
    save_safetensors(first.checkpoint_tensors(), str(checkpoint))
    second.load_checkpoint_tensors(load_safetensors(str(checkpoint)))
    first.train_local_steps(5)
    second.train_local_steps(5)

    _same_state(first.checkpoint_tensors(), second.checkpoint_tensors())
    assert first.learning_rate == second.learning_rate
    assert torch.equal(torch.get_rng_state(), global_rng)


@pytest.mark.parametrize("change", ["missing", "foreign", "actual-data"])
def test_checkpoint_rejects_changed_or_missing_data_fingerprint_before_mutation(
    change: str, tmp_path: Path
) -> None:
    draft = _draft()
    records = _Data()
    prepared = _prepare(draft=draft, data=_data(records))
    trainer = _trainer(prepared, tmp_path, draft)
    trainer.train_local_steps(2)
    checkpoint = trainer.checkpoint_tensors()
    initial_weights = {name: value + 100 for name, value in trainer.weights().items()}
    checkpoint.update(initial_weights)
    before = trainer.checkpoint_tensors()
    if change == "missing":
        del checkpoint[_FINGERPRINT]
    elif change == "foreign":
        checkpoint[_FINGERPRINT] = np.zeros(32, dtype=np.uint8)
    else:
        records.inputs[0, 0] += 1

    with pytest.raises(ValueError, match="fingerprint|dataset changed"):
        trainer.load_checkpoint_tensors(checkpoint)
    _same_state(trainer.checkpoint_tensors(), before)


def test_preparation_rejects_wrong_model_definition() -> None:
    with pytest.raises(ValueError, match="definition hash"):
        prepare_local_training(
            draft=_draft(),
            data=_data(),
            model=_model(),
            model_definition="wrong",
            seed=17,
        )


def test_preparation_requires_the_explicit_v4_contract() -> None:
    with pytest.raises(ValueError, match="manifest v4"):
        _prepare(draft=_draft().model_copy(update={"manifest_version": 3}))


def test_preparation_rejects_unsupported_device_rng() -> None:
    with pytest.raises(ValueError, match="CPU or CUDA"):
        prepare_local_training(
            draft=_draft(),
            data=_data(),
            model=_model(),
            model_definition=_DEFINITION,
            seed=17,
            device="mps",
        )


@pytest.mark.parametrize("kind", ["float64", "nonfinite", "integer-buffer", "output"])
def test_preparation_rejects_incompatible_model(kind: str) -> None:
    model = _model()
    if kind == "float64":
        model.double()
    elif kind == "nonfinite":
        with torch.no_grad():
            next(model.parameters()).fill_(float("nan"))
    elif kind == "integer-buffer":
        model.register_buffer("counter", torch.zeros(1, dtype=torch.int64))
    else:
        model = nn.Linear(2, 3)
    with pytest.raises(ValueError, match="FP32|logits"):
        _prepare(model=model)


@pytest.mark.parametrize("field", ["crop_padding", "normalize"])
def test_preprocessed_local_data_rejects_cifar_augmentation_policy(field: str) -> None:
    draft = _draft()
    assert draft.training is not None
    policy = draft.training.model_copy(
        update={field: 1 if field == "crop_padding" else True}
    )
    with pytest.raises(ValueError, match="preprocessed"):
        _prepare(draft=draft.model_copy(update={"training": policy}))


@pytest.mark.parametrize(
    "change", ["draft-hash", "sealed-fields", "task", "membership", "schema"]
)
def test_trainer_construction_rejects_incompatible_sealed_inputs(
    change: str, tmp_path: Path
) -> None:
    draft = _draft()
    prepared = _prepare(draft=draft)
    manifest, _ = _seal(prepared, tmp_path, draft)
    public_key = "peer-0"
    if change == "draft-hash":
        manifest = manifest.model_copy(update={"draft_hash": "b" * 64})
    elif change == "sealed-fields":
        manifest = manifest.model_copy(update={"learning_rate": 0.9})
    elif change == "task":
        manifest = manifest.model_copy(
            update={
                "dataset": manifest.dataset.model_copy(
                    update={"label_names": ("truck", "cat")}
                )
            }
        )
    elif change == "membership":
        public_key = "unsealed"
    else:
        manifest = manifest.model_copy(
            update={
                "tensor_schema": TensorSchema(
                    tensors=(manifest.tensor_schema.tensors[0],)
                )
            }
        )

    with pytest.raises(ValueError, match="sealed|prepared|participant"):
        prepared.create_trainer(manifest=manifest, local_public_key=public_key)


def test_prepared_draft_validation_rejects_another_run() -> None:
    prepared = _prepare()
    with pytest.raises(ValueError, match="draft does not match"):
        prepared.validate_draft(_draft().model_copy(update={"run_id": "another-run"}))


def test_changing_records_after_preparation_prevents_trainer_creation(
    tmp_path: Path,
) -> None:
    draft = _draft()
    records = _Data()
    prepared = _prepare(draft=draft, data=_data(records))
    manifest, _ = _seal(prepared, tmp_path, draft)
    records.labels[0] = 1
    with pytest.raises(ValueError, match="changed after preparation"):
        prepared.create_trainer(manifest=manifest, local_public_key="peer-0")


def test_independent_nodes_keep_distinct_private_fingerprints(tmp_path: Path) -> None:
    draft = _draft()
    common_model = _model()
    left = _prepare(draft=draft, model=common_model, data=_data(_Data(1, label=0)))
    right = _prepare(draft=draft, model=common_model, data=_data(_Data(11, label=1)))
    manifest, initial = _seal(left, tmp_path, draft)
    first = left.create_trainer(manifest=manifest, local_public_key="peer-0")
    second = right.create_trainer(manifest=manifest, local_public_key="peer-1")
    first.load_checkpoint(initial)
    second.load_checkpoint(initial)
    _same_state(first.weights(), second.weights())
    first.train_local_steps(3)
    second.train_local_steps(3)
    assert left.draft_hash == right.draft_hash
    assert left.data_metadata.fingerprint != right.data_metadata.fingerprint
    before = deepcopy(second.checkpoint_tensors())
    with pytest.raises(ValueError, match="fingerprint"):
        second.load_checkpoint_tensors(first.checkpoint_tensors())
    _same_state(second.checkpoint_tensors(), before)


def test_concurrent_trainers_keep_stochastic_model_rng_independent(
    tmp_path: Path,
) -> None:
    draft = _draft()
    prepared = _prepare(draft=draft, model=_model(dropout=True))
    first = _trainer(prepared, tmp_path / "first", draft)
    second = _trainer(prepared, tmp_path / "second", draft)
    reference = _trainer(prepared, tmp_path / "reference", draft)
    global_rng = torch.get_rng_state().clone()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(trainer.train_local_steps, 5) for trainer in (first, second)
        ]
        for future in futures:
            future.result()
    reference.train_local_steps(5)

    _same_state(first.checkpoint_tensors(), reference.checkpoint_tensors())
    _same_state(second.checkpoint_tensors(), reference.checkpoint_tensors())
    assert torch.equal(torch.get_rng_state(), global_rng)


@pytest.mark.parametrize(
    "change",
    [
        "batch_size",
        "weight_decay",
        "momentum",
        "milestones",
        "seed",
        "model_id",
        "requires_grad",
    ],
)
def test_restore_context_rejects_valid_but_different_training_construction(
    change: str, tmp_path: Path
) -> None:
    draft = _draft()
    model = _model()
    source = _trainer(_prepare(draft=draft, model=model), tmp_path / "source", draft)
    source.train_local_steps(3)
    target_data = draft.model_dump(mode="python")
    if change in {"batch_size", "weight_decay", "momentum"}:
        target_data["training"][change] = {
            "batch_size": 2,
            "weight_decay": 0.02,
            "momentum": 0.7,
        }[change]
    elif change == "milestones":
        target_data["training"]["learning_rate_milestones"] = [4, 8]
    elif change == "model_id":
        target_data["model_id"] = "another-model-identity"
    target_draft = DraftRunSpec.model_validate(target_data)
    target_model = deepcopy(model)
    if change == "requires_grad":
        next(target_model.parameters()).requires_grad_(False)
    prepared = prepare_local_training(
        draft=target_draft,
        data=_data(),
        model=target_model,
        model_definition=_DEFINITION,
        seed=18 if change == "seed" else 17,
    )
    target = _trainer(prepared, tmp_path / "target", target_draft)
    control = _trainer(prepared, tmp_path / "control", target_draft)
    for trainer in (target, control):
        trainer.train_local_steps(2)
    before = target.checkpoint_tensors()
    before_loss = target.local_loss
    with pytest.raises(ValueError, match="restore context"):
        target.load_checkpoint_tensors(source.checkpoint_tensors())
    _same_state(target.checkpoint_tensors(), before)
    assert target.local_loss == before_loss
    for trainer in (target, control):
        trainer.train_local_steps(2)
    _same_state(target.checkpoint_tensors(), control.checkpoint_tensors())


def test_explicit_restore_can_move_to_another_run_with_same_task_model_policy_data(
    tmp_path: Path,
) -> None:
    draft = _draft(adam=True)
    other_draft = DraftRunSpec.model_validate(
        {**draft.model_dump(mode="python"), "run_id": "new-offline-run"}
    )
    model = _model(dropout=True)
    source = _trainer(_prepare(draft=draft, model=model), tmp_path / "source", draft)
    restored = _trainer(
        _prepare(draft=other_draft, model=model), tmp_path / "target", other_draft
    )
    source.train_local_steps(3)
    restored.load_checkpoint_tensors(source.checkpoint_tensors())
    source.train_local_steps(3)
    restored.train_local_steps(3)
    _same_state(source.checkpoint_tensors(), restored.checkpoint_tensors())


@pytest.mark.parametrize("adam", [False, True])
@pytest.mark.parametrize(
    "corruption",
    [
        "nan-weight",
        "weight-shape",
        "weight-dtype",
        "missing-weight",
        "nan-optimizer",
        "optimizer-shape",
        "missing-optimizer",
        "unknown-key",
        "negative-counter",
        "counter-dtype",
        "batch-position",
        "model-rng",
        "loader-rng",
        "augmentation-rng",
        "missing-context",
        "wrong-context",
        "optimizer-inventory",
    ],
)
def test_malformed_restore_preserves_live_weights_optimizer_and_rng(
    adam: bool, corruption: str, tmp_path: Path
) -> None:
    draft = _draft(adam=adam)
    prepared = _prepare(draft=draft, model=_model(dropout=True))
    trainer = _trainer(prepared, tmp_path / "live", draft)
    control = _trainer(prepared, tmp_path / "control", draft)
    for node in (trainer, control):
        node.train_local_steps(3)
    before = trainer.checkpoint_tensors()
    before_loss = trainer.local_loss
    state = deepcopy(before)
    # Valid changed weights would expose an early partial install on later failures.
    model_key = next(iter(trainer.weights()))
    state[model_key] += 100
    optimizer_key = next(
        key
        for key in state
        if key.startswith("__dromeus_training__.adam.exp_avg.")
        or key.startswith("__dromeus_training__.momentum.")
    )
    if corruption == "nan-weight":
        state[model_key].fill(np.nan)
    elif corruption == "weight-shape":
        state[model_key] = np.zeros((1,), dtype=np.float32)
    elif corruption == "weight-dtype":
        state[model_key] = state[model_key].astype(np.float64)
    elif corruption == "missing-weight":
        del state[model_key]
    elif corruption == "nan-optimizer":
        state[optimizer_key].fill(np.nan)
    elif corruption == "optimizer-shape":
        state[optimizer_key] = np.zeros((1,), dtype=np.float32)
    elif corruption == "missing-optimizer":
        del state[optimizer_key]
    elif corruption == "unknown-key":
        state["unexpected"] = np.zeros(1, dtype=np.float32)
    elif corruption == "negative-counter":
        state["__dromeus_training__.completed_steps"] = np.array([-1], dtype=np.int64)
    elif corruption == "counter-dtype":
        state["__dromeus_training__.completed_steps"] = np.array([3], dtype=np.float32)
    elif corruption == "batch-position":
        state["__dromeus_training__.batches_consumed"] = np.array([2], dtype=np.int64)
    elif corruption in {"model-rng", "loader-rng", "augmentation-rng"}:
        key = {
            "model-rng": "__dromeus_local__.torch_cpu_rng",
            "loader-rng": "__dromeus_training__.loader_epoch_rng",
            "augmentation-rng": "__dromeus_training__.augmentation_rng",
        }[corruption]
        state[key] = np.zeros(1, dtype=np.uint8)
    elif corruption == "missing-context":
        del state["__dromeus_local__.restore_context_v1"]
    elif corruption == "wrong-context":
        state["__dromeus_local__.restore_context_v1"] = np.zeros(32, dtype=np.uint8)
    else:
        state["__dromeus_local__.optimizer_keys_v1"] = np.frombuffer(
            b"{}", dtype=np.uint8
        )
    with pytest.raises(ValueError):
        trainer.load_checkpoint_tensors(state)
    _same_state(trainer.checkpoint_tensors(), before)
    assert trainer.local_loss == before_loss
    for node in (trainer, control):
        node.train_local_steps(2)
    _same_state(trainer.checkpoint_tensors(), control.checkpoint_tensors())


@pytest.mark.parametrize(
    "corruption",
    [
        "negative-variance",
        "fractional-step",
        "negative-step",
        "step-ahead",
        "incomplete-triple",
        "past-schedule",
    ],
)
def test_adam_restore_rejects_invalid_state_values(
    corruption: str, tmp_path: Path
) -> None:
    draft = _draft(adam=True)
    trainer = _trainer(_prepare(draft=draft), tmp_path, draft)
    trainer.train_local_steps(3)
    before = trainer.checkpoint_tensors()
    state = deepcopy(before)
    key = next(
        name
        for name in state
        if name.startswith("__dromeus_training__.adam.exp_avg_sq.")
    )
    step_key = next(
        name for name in state if name.startswith("__dromeus_training__.adam.step.")
    )
    if corruption == "negative-variance":
        state[key].fill(-1)
    elif corruption in {"fractional-step", "negative-step", "step-ahead"}:
        state[step_key][0] = {
            "fractional-step": 1.5,
            "negative-step": -1,
            "step-ahead": 4,
        }[corruption]
    elif corruption == "past-schedule":
        state["__dromeus_training__.completed_steps"][0] = 101
    else:
        del state[key]
        # Even an internally updated inventory cannot excuse half an Adam triple.
        names = json.loads(state["__dromeus_local__.optimizer_keys_v1"].tobytes())
        names.remove(key)
        state["__dromeus_local__.optimizer_keys_v1"] = np.frombuffer(
            json.dumps(names).encode(), dtype=np.uint8
        )
    with pytest.raises(ValueError):
        trainer.load_checkpoint_tensors(state)
    _same_state(trainer.checkpoint_tensors(), before)


def test_candidate_backend_failure_cannot_partly_replace_the_live_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    draft = _draft(adam=True)
    prepared = _prepare(draft=draft, model=_model(dropout=True))
    trainer = _trainer(prepared, tmp_path / "live", draft)
    control = _trainer(prepared, tmp_path / "control", draft)
    for node in (trainer, control):
        node.train_local_steps(3)
    before = trainer.checkpoint_tensors()
    state = deepcopy(before)
    for key in trainer.weights():
        state[key] += 100
    original = PyTorchTrainer.load_checkpoint_tensors

    def fail_after_candidate_mutation(
        self: PyTorchTrainer, tensors: dict[str, np.ndarray]
    ) -> None:
        original(self, tensors)
        raise RuntimeError("injected late backend restore failure")

    monkeypatch.setattr(
        PyTorchTrainer, "load_checkpoint_tensors", fail_after_candidate_mutation
    )
    with pytest.raises(ValueError, match="cannot be restored"):
        trainer.load_checkpoint_tensors(state)
    _same_state(trainer.checkpoint_tensors(), before)
    monkeypatch.setattr(PyTorchTrainer, "load_checkpoint_tensors", original)
    for node in (trainer, control):
        node.train_local_steps(2)
    _same_state(trainer.checkpoint_tensors(), control.checkpoint_tensors())


class _ScaleModel(nn.Module):
    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(shape, dtype=torch.float32))
        self.classifier = nn.Linear(2, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(inputs) * self.scale


@pytest.mark.parametrize("adam", [False, True])
def test_single_value_vector_parameter_and_optimizer_state_round_trip(
    adam: bool, tmp_path: Path
) -> None:
    draft = _draft(adam=adam)
    prepared = _prepare(draft=draft, model=_ScaleModel((1,)))
    first = _trainer(prepared, tmp_path / "first", draft)
    second = _trainer(prepared, tmp_path / "second", draft)
    first.train_local_steps(3)
    path = tmp_path / "scalar.safetensors"
    save_safetensors(first.checkpoint_tensors(), str(path))
    second.load_checkpoint_tensors(load_safetensors(str(path)))
    first.train_local_steps(2)
    second.train_local_steps(2)
    _same_state(first.checkpoint_tensors(), second.checkpoint_tensors())


def test_scalar_model_state_rejects_before_ready() -> None:
    with pytest.raises(ValueError, match="non-scalar"):
        _prepare(model=_ScaleModel(()))
