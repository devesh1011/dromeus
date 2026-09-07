from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from PIL import Image
from support.sample_manifest import manifest_data
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset

import benchmarks.workloads.cifar10.dataset as cifar10_recipe
from benchmarks.workloads.cifar10.dataset import (
    DATASET_REPOSITORY,
    DATASET_REVISION,
    DATASET_VERSION,
    PREPROCESSING_DEFINITION,
    PREPROCESSING_HASH,
    CIFAR10DataError,
    CIFAR10TrainerSettings,
    PreparedCIFAR10Training,
    create_initial_checkpoint,
    create_trainer,
    load_cifar10,
)
from benchmarks.workloads.cifar10.partitioning import (
    ClassificationData,
    IIDPartitionProvenance,
)
from benchmarks.workloads.cifar10.resnet18_groupnorm import (
    MODEL_DEFINITION_HASH as RESNET18_DEFINITION_HASH,
)
from benchmarks.workloads.cifar10.resnet18_groupnorm import (
    MODEL_ID as RESNET18_MODEL_ID,
)
from benchmarks.workloads.cifar10.resnet18_groupnorm import (
    tensor_schema_for_model as resnet18_tensor_schema,
)
from benchmarks.workloads.cifar10.resnet32 import (
    MODEL_DEFINITION_HASH as RESNET32_DEFINITION_HASH,
)
from benchmarks.workloads.cifar10.resnet32 import MODEL_ID as RESNET32_MODEL_ID
from benchmarks.workloads.cifar10.resnet32 import (
    tensor_schema_for_model as resnet32_tensor_schema,
)
from dromeus.adapters.classification.torch_trainer import (
    TrainerSettings,
    derive_benchmark_seed,
)
from dromeus.manifests.canonical import canonical_hash
from dromeus.manifests.models import (
    DraftRunSpec,
    SealedManifest,
    WarmupCosineSchedule,
)
from dromeus.persistence.archive import RunArchive
from dromeus.persistence.run_store import RunStore


def _cifar_settings(**changes: Any) -> CIFAR10TrainerSettings:
    defaults = CIFAR10TrainerSettings()
    return replace(defaults, trainer=replace(defaults.trainer, **changes))


def _small_classification_data() -> ClassificationData:
    return ClassificationData(
        cast(
            Dataset[tuple[Tensor, int]],
            TensorDataset(
                torch.zeros((8, 3, 32, 32), dtype=torch.float32),
                torch.arange(8, dtype=torch.long),
            ),
        )
    )


def test_cifar_contract_constants_are_canonical() -> None:
    assert DATASET_VERSION == f"huggingface-{DATASET_REVISION}"
    assert PREPROCESSING_HASH == hashlib.sha256(
        PREPROCESSING_DEFINITION.encode()
    ).hexdigest()


def test_benchmark_seed_concerns_are_stable_and_separate() -> None:
    values = {
        derive_benchmark_seed(17, purpose)
        for purpose in (
            "model-initialization",
            "local-training",
            "consensus-sketch",
        )
    }

    assert len(values) == 3
    assert derive_benchmark_seed(17, "local-training") == derive_benchmark_seed(
        17, "local-training"
    )


def test_cifar_trainer_settings_preserve_recipe_defaults() -> None:
    settings = CIFAR10TrainerSettings()

    assert settings.trainer.batch_size == 128
    assert settings.trainer.momentum == 0.9
    assert settings.trainer.weight_decay == 1e-4
    assert settings.trainer.learning_rate_milestones == (8_000, 12_000)
    assert settings.crop_padding == 4
    assert settings.normalize is True


def test_prepared_training_maps_noloco_adam_settings() -> None:
    data = manifest_data()
    data.update(
        {
            "algorithm_id": "noloco",
            "model_id": RESNET18_MODEL_ID,
            "model_definition_hash": RESNET18_DEFINITION_HASH,
            "tensor_schema": resnet18_tensor_schema().model_dump(mode="python"),
            "environment": {
                **data["environment"],
                "model_definition_hash": RESNET18_DEFINITION_HASH,
            },
            "optimizer": "adam",
            "learning_rate": 0.001,
            "training": {
                **data["training"],
                "learning_rate_milestones": [],
                "learning_rate_schedule": {
                    "schedule_id": "linear-warmup-cosine-v1",
                    "total_inner_steps": 5_000,
                    "warmup_inner_steps": 500,
                    "start_learning_rate": 0.0001,
                    "peak_learning_rate": 0.001,
                    "final_learning_rate": 0.0001,
                },
            },
            "algorithm_config": {
                "alpha": 0.5,
                "beta": 0.7,
                "gamma": 0.7,
                "inner_steps": 50,
                "adam": {
                    "learning_rate": 0.001,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "epsilon": 1e-8,
                    "gradient_clip_norm": 1.0,
                },
            },
            "artifact_codecs": [
                {"artifact_name": "outer_gradient", "codec_id": "identity-v1"},
                {"artifact_name": "slow_weights", "codec_id": "identity-v1"},
            ],
            "transport": {
                **data["transport"],
                "chunk_size_bytes": 1024,
                "window_size": 1,
            },
        }
    )
    draft_data = data.copy()
    for field in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del draft_data[field]
    data["draft_hash"] = canonical_hash(DraftRunSpec.model_validate(draft_data))
    manifest = SealedManifest.model_validate(data)
    small_data = _small_classification_data()
    prepared = PreparedCIFAR10Training(
        _partitions=(small_data,) * 4,
        _test_data=small_data,
        initialization_seed=1,
        trainer_seed=2,
        model_id=RESNET18_MODEL_ID,
        model_definition_hash=RESNET18_DEFINITION_HASH,
    )

    trainer = prepared.create_trainer(
        manifest=manifest,
        local_public_key="peer-0",
    )

    assert trainer.tensor_schema == manifest.tensor_schema
    assert trainer.settings.optimizer == "adam"
    assert trainer.settings.adam_beta1 == 0.9
    assert trainer.settings.adam_beta2 == 0.999
    assert trainer.settings.adam_epsilon == 1e-8
    assert trainer.settings.gradient_clip_norm == 1.0
    assert trainer.learning_rate == pytest.approx(0.0001)
    assert trainer.last_local_loss is None


def test_prepared_training_maps_dpsgd_sgd_settings() -> None:
    data = manifest_data()
    data.update(
        {
            "model_id": RESNET32_MODEL_ID,
            "model_definition_hash": RESNET32_DEFINITION_HASH,
            "tensor_schema": resnet32_tensor_schema().model_dump(mode="python"),
            "environment": {
                **data["environment"],
                "model_definition_hash": RESNET32_DEFINITION_HASH,
            },
        }
    )
    draft_data = data.copy()
    for field in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del draft_data[field]
    data["draft_hash"] = canonical_hash(DraftRunSpec.model_validate(draft_data))
    manifest = SealedManifest.model_validate(data)
    small_data = _small_classification_data()
    prepared = PreparedCIFAR10Training(
        _partitions=(small_data,) * 4,
        _test_data=small_data,
        initialization_seed=1,
        trainer_seed=2,
        model_id=RESNET32_MODEL_ID,
        model_definition_hash=RESNET32_DEFINITION_HASH,
    )

    trainer = prepared.create_trainer(
        manifest=manifest,
        local_public_key="peer-0",
    )

    assert trainer.tensor_schema == manifest.tensor_schema
    assert trainer.settings == TrainerSettings(
        seed=2,
        batch_size=128,
        learning_rate=0.1,
        optimizer="sgd",
        momentum=0.9,
        weight_decay=1e-4,
        learning_rate_milestones=(8_000, 12_000),
        learning_rate_gamma=0.1,
        device="cpu",
        augment=True,
    )


def test_prepared_training_rejects_formed_model_mismatch() -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    placeholder = cast(ClassificationData, object())
    prepared = PreparedCIFAR10Training(
        _partitions=(placeholder,) * 4,
        _test_data=placeholder,
        initialization_seed=1,
        trainer_seed=2,
        model_id=RESNET18_MODEL_ID,
        model_definition_hash=RESNET18_DEFINITION_HASH,
    )

    with pytest.raises(
        ValueError,
        match="formed manifest model does not match prepared training",
    ):
        prepared.create_trainer(manifest=manifest, local_public_key="peer-0")


def test_prepared_training_rejects_unsealed_local_identity() -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    placeholder = cast(ClassificationData, object())
    prepared = PreparedCIFAR10Training(
        _partitions=(placeholder,) * 4,
        _test_data=placeholder,
        initialization_seed=1,
        trainer_seed=2,
        model_id=manifest.model_id,
        model_definition_hash=manifest.model_definition_hash,
    )

    with pytest.raises(
        ValueError,
        match="local public key is not a sealed participant",
    ):
        prepared.create_trainer(manifest=manifest, local_public_key="unknown-peer")


@pytest.fixture(scope="session")
def cifar10_data() -> ClassificationData:
    cache_dir = Path(
        os.environ.get(
            "DROMEUS_CIFAR_CACHE",
            Path.home() / ".cache" / "dromeus" / "cifar10",
        )
    )
    if not cache_dir.exists():
        pytest.skip("real CIFAR-10 data unavailable in DROMEUS_CIFAR_CACHE")
    try:
        return load_cifar10(
            cache_dir=cache_dir,
            train=False,
        )
    except CIFAR10DataError:
        pytest.skip(
            "real CIFAR-10 data unavailable; set DROMEUS_CIFAR_CACHE to writable cache"
        )


def test_huggingface_loader_returns_real_cifar10(
    cifar10_data: ClassificationData,
) -> None:
    image, label = cifar10_data[0]

    assert len(cifar10_data) == 10_000
    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert 0 <= label < 10


def test_huggingface_loader_pins_source_revision_and_decodes_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeDataset:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, object]:
            assert index == 0
            return {"img": Image.new("RGB", (32, 32), "red"), "label": 3}

    def fake_load_dataset(repository: str, **kwargs: object) -> FakeDataset:
        calls.append({"repository": repository, **kwargs})
        return FakeDataset()

    monkeypatch.setattr(cifar10_recipe, "load_dataset", fake_load_dataset)

    data = load_cifar10(cache_dir=tmp_path, train=False)
    image, label = data[0]

    assert calls == [
        {
            "repository": DATASET_REPOSITORY,
            "split": "test",
            "revision": DATASET_REVISION,
            "cache_dir": str(tmp_path),
        }
    ]
    assert data.matches_source(
        source="huggingface-uoft-cs-cifar10",
        split="test",
    )
    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert label == 3


@pytest.mark.parametrize("participant_count", [4, 8, 16])
def test_iid_partitions_are_reproducible_and_disjoint(
    participant_count: int,
    cifar10_data: ClassificationData,
) -> None:
    first = cifar10_data.split_iid(
        participant_count=participant_count,
        seed=11,
    )
    second = cifar10_data.split_iid(
        participant_count=participant_count,
        seed=11,
    )

    assert [[part[index][1] for index in range(len(part))] for part in first] == [
        [part[index][1] for index in range(len(part))] for part in second
    ]
    assert all(part[0][0].shape == (3, 32, 32) for part in first)
    assert len({part[index][1] for part in first for index in range(len(part))}) > 1
    provenance = [part.partition_provenance for part in first]
    assert all(isinstance(value, IIDPartitionProvenance) for value in provenance)
    assert [
        (value.seed, value.participant_count, value.partition_index)
        for value in provenance
        if value
    ] == [(11, participant_count, index) for index in range(participant_count)]
    assert all(
        value and value.source_sample_count == len(cifar10_data) for value in provenance
    )
    assert (
        len({value.indices_sha256 for value in provenance if value})
        == participant_count
    )


def test_checkpoint_is_deterministic_and_matches_trainer_schema(
    tmp_path: Path,
    cifar10_data: ClassificationData,
) -> None:
    first_path = tmp_path / "first.safetensors"
    second_path = tmp_path / "second.safetensors"

    first = create_initial_checkpoint(first_path, seed=17)
    second = create_initial_checkpoint(second_path, seed=17)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first.path == first_path
    assert first.tensor_schema == second.tensor_schema
    assert first.sha256 == second.sha256

    trainer = create_trainer(
        train_data=cifar10_data,
        settings=_cifar_settings(seed=17, batch_size=4),
    )
    trainer.load_checkpoint(first_path)
    assert trainer.tensor_schema == first.tensor_schema
    assert trainer.checkpoint_hash(first_path) == trainer.checkpoint_hash(second_path)


def test_initial_checkpoint_returns_formation_handoff(tmp_path: Path) -> None:
    prepared = create_initial_checkpoint(tmp_path / "checkpoint.safetensors", seed=17)

    assert prepared.path.is_file()
    assert prepared.tensor_schema.tensors
    assert (
        prepared.sha256
        == create_initial_checkpoint(
            tmp_path / "checkpoint-2.safetensors", seed=17
        ).sha256
    )


def test_trainer_runs_sgd_and_evaluates(
    cifar10_data: ClassificationData,
) -> None:
    trainer = create_trainer(
        train_data=cifar10_data,
        settings=_cifar_settings(seed=3, batch_size=4, learning_rate=0.05),
    )
    before = trainer.weights()

    trainer.train_local_steps(2)
    loss, accuracy = trainer.evaluate(cifar10_data)

    assert any(
        not np.array_equal(before[name], value)
        for name, value in trainer.weights().items()
    )
    assert np.isfinite(loss)
    assert 0.0 <= accuracy <= 1.0


def test_resnet_trainer_uses_momentum_schedule_and_full_float_state(
    cifar10_data: ClassificationData,
    tmp_path: Path,
) -> None:
    trainer = create_trainer(
        train_data=cifar10_data,
        settings=_cifar_settings(
            seed=3,
            batch_size=4,
            learning_rate=0.1,
            momentum=0.9,
            weight_decay=1e-4,
            learning_rate_milestones=(1,),
            learning_rate_gamma=0.1,
        ),
    )
    before = trainer.weights()

    trainer.train_local_steps(2)
    checkpoint_state = trainer.checkpoint_tensors()
    store = RunStore(tmp_path / "run")
    store.initialize(SealedManifest.model_validate(manifest_data()))
    store.persist_commit(
        committed_round=0,
        algorithm_state=checkpoint_state,
        state_checksum="a" * 64,
        schedule={"round_id": 0, "peer": "peer-1"},
    )
    archive = RunArchive.open(tmp_path / "run")
    assert archive.algorithm_state is not None
    persisted_state = archive.algorithm_state.load_tensors()

    assert trainer.learning_rate == pytest.approx(0.01)
    assert int(checkpoint_state["__dromeus_training__.completed_steps"][0]) == 2
    assert any(
        name.startswith("__dromeus_training__.momentum.")
        for name in checkpoint_state
    )
    assert any(name.endswith("running_mean") for name in before)
    assert any(
        name.endswith("running_mean")
        and not np.array_equal(before[name], trainer.weights()[name])
        for name in before
    )

    restored = create_trainer(
        train_data=cifar10_data,
        settings=_cifar_settings(
            seed=99,
            batch_size=4,
            learning_rate=0.1,
            momentum=0.9,
            weight_decay=1e-4,
            learning_rate_milestones=(1,),
            learning_rate_gamma=0.1,
        ),
    )
    restored.load_checkpoint_tensors(persisted_state)

    assert restored.learning_rate == pytest.approx(0.01)
    assert all(
        np.array_equal(value, restored.weights()[name])
        for name, value in trainer.weights().items()
    )

    trainer.train_local_steps(1)
    restored.train_local_steps(1)

    assert all(
        np.array_equal(value, restored.weights()[name])
        for name, value in trainer.weights().items()
    )


def test_warmup_cosine_trainer_resumes_at_the_exact_next_step(
    cifar10_data: ClassificationData,
) -> None:
    schedule = WarmupCosineSchedule(
        schedule_id="linear-warmup-cosine-v1",
        total_inner_steps=4,
        warmup_inner_steps=2,
        start_learning_rate=0.001,
        peak_learning_rate=0.01,
        final_learning_rate=0.001,
    )
    trainer = create_trainer(
        train_data=cifar10_data,
        settings=_cifar_settings(
            seed=3,
            batch_size=4,
            learning_rate=0.01,
            optimizer="adam",
            learning_rate_milestones=(),
            learning_rate_schedule=schedule,
        ),
    )

    assert trainer.learning_rate == pytest.approx(0.001)
    trainer.train_local_steps(1)
    assert trainer.learning_rate == pytest.approx(0.01)
    state = trainer.checkpoint_tensors()

    restored = create_trainer(
        train_data=cifar10_data,
        settings=_cifar_settings(
            seed=99,
            batch_size=4,
            learning_rate=0.01,
            optimizer="adam",
            learning_rate_milestones=(),
            learning_rate_schedule=schedule,
        ),
    )
    restored.load_checkpoint_tensors(state)
    assert restored.learning_rate == pytest.approx(0.01)

    trainer.train_local_steps(3)
    restored.train_local_steps(3)

    assert trainer.learning_rate == pytest.approx(0.001)
    assert all(
        np.array_equal(value, restored.weights()[name])
        for name, value in trainer.weights().items()
    )
    with pytest.raises(ValueError, match="exceeds learning-rate schedule"):
        trainer.train_local_steps(1)
