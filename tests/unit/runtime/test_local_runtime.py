from __future__ import annotations

import asyncio
import hashlib
from contextvars import ContextVar
from pathlib import Path

import numpy as np
import pytest
import torch
from support.in_memory_transport import InMemoryNetwork, InMemoryTransport
from support.sample_manifest import manifest_data
from torch import nn
from torch.utils.data import Dataset

from benchmarks.workloads.cifar10 import dataset as cifar10
from dromeus import runtime as runtime_module
from dromeus.adapters.classification.data import (
    LocalClassificationData,
    validate_local_data,
)
from dromeus.adapters.classification.runtime import (
    PreparedLocalTraining,
    prepare_local_training,
)
from dromeus.algorithms.noloco import NoLoCoAlgorithm
from dromeus.manifests.canonical import canonical_json
from dromeus.manifests.models import ClassificationTaskContract, DraftRunSpec
from dromeus.membership.formation import FormationResult, create_invitation
from dromeus.persistence.archive import RunArchive
from dromeus.persistence.run_store import RunStore
from dromeus.runtime import (
    FailureConfig,
    InitiatorFormation,
    NodeRunResult,
    NodeRuntime,
    NodeState,
    ParticipantFormation,
    TrainingConfig,
)

_MODEL_DEFINITION = "local-runtime-linear-2x2-v1"
_DATA_OWNER: ContextVar[int | None] = ContextVar("local_test_data_owner", default=None)


def _task() -> ClassificationTaskContract:
    return ClassificationTaskContract(
        dataset_id="local-classification-v1",
        input_shape=(2,),
        input_dtype="float32",
        label_names=("negative", "positive"),
        preprocessing_hash="a" * 64,
    )


def _draft(compressed: bool) -> DraftRunSpec:
    data = manifest_data()
    for field in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del data[field]
    model_hash = hashlib.sha256(_MODEL_DEFINITION.encode()).hexdigest()
    data.update(
        manifest_version=4,
        expected_participant_count=4,
        algorithm_id="noloco",
        model_id="local-runtime-classifier",
        model_definition_hash=model_hash,
        dataset=_task(),
        local_steps=50,
        round_count=2,
        optimizer="adam",
        learning_rate=0.005,
        algorithm_config={
            "alpha": 0.5,
            "beta": 0.7,
            "gamma": 0.7,
            "inner_steps": 50,
            "adam": {
                "learning_rate": 0.005,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
                "gradient_clip_norm": 1.0,
            },
        },
        artifact_codecs=(
            [
                {
                    "artifact_name": "outer_gradient",
                    "codec_id": "topk-bitmap-int8-v2",
                    "top_k_fraction": 0.5,
                    "lossy_allowed": True,
                },
                {
                    "artifact_name": "slow_weights",
                    "codec_id": "dense-int8-v1",
                    "lossy_allowed": True,
                },
            ]
            if compressed
            else [
                {"artifact_name": name, "codec_id": "identity-v1"}
                for name in ("outer_gradient", "slow_weights")
            ]
        ),
        training={
            "batch_size": 4,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "learning_rate_milestones": [],
            "learning_rate_gamma": 0.1,
            "crop_padding": 0,
            "normalize": False,
            "final_consensus_rounds": 0,
        },
        transport={
            "max_payload_bytes": 8 * 1024 * 1024,
            "max_retries": 3,
            "retry_timeout_seconds": 1.0,
            "chunk_size_bytes": 128,
            "window_size": 2,
        },
        consensus_sketch={"size": 4096, "seed": 9},
    )
    data["environment"]["model_definition_hash"] = model_hash
    return DraftRunSpec.model_validate(data)


class _PrivateDataset(Dataset[object]):
    """A worker cannot read another worker's dataset, even in this process."""

    def __init__(self, owner: int, count: int, *, held_out: bool = False) -> None:
        self.owner = owner
        self.count = count
        self.held_out = held_out
        self.reads = 0

    def __len__(self) -> int:
        assert _DATA_OWNER.get() == self.owner, "another node accessed private data"
        return self.count

    def __getitem__(self, index: int) -> object:
        assert _DATA_OWNER.get() == self.owner, "another node accessed private data"
        if not 0 <= index < self.count:
            raise IndexError(index)
        self.reads += 1
        label = self.owner % 2
        sign = 1.0 if label else -1.0
        offset = 0.2 if self.held_out else 0.0
        inputs = torch.tensor(
            [sign * (1.0 + index / 100 + offset), (self.owner + 1) / 10],
            dtype=torch.float32,
        )
        return inputs, torch.tensor(label)


@pytest.mark.parametrize("compressed", (False, True), ids=("identity", "bitmap-int8"))
def test_four_runtime_nodes_train_only_own_unequal_missing_class_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, compressed: bool
) -> None:
    def forbidden_cifar(*args: object, **kwargs: object) -> None:
        raise AssertionError("independent local data must not prepare CIFAR partitions")

    monkeypatch.setattr(cifar10, "prepare_training", forbidden_cifar)
    assert not hasattr(runtime_module, "prepare_training")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        asyncio.run(_run_four_nodes(tmp_path, compressed))
    finally:
        torch.set_num_threads(previous_threads)


async def _run_four_nodes(tmp_path: Path, compressed: bool) -> None:
    draft = _draft(compressed)
    task = _task()
    network = InMemoryNetwork()
    transports = [
        InMemoryTransport(network=network, public_key=f"peer-{index}")
        for index in range(4)
    ]
    prepared: list[PreparedLocalTraining] = []
    fingerprints: list[str] = []
    datasets: list[_PrivateDataset] = []
    for index, count in enumerate((1, 3, 5, 7)):
        token = _DATA_OWNER.set(index)
        try:
            datasets.append(_PrivateDataset(index, count))
            data = LocalClassificationData(
                train_data=datasets[-1],
                evaluation_data=_PrivateDataset(index, 2, held_out=True),
                label_names=task.label_names,
                preprocessing_hash=task.preprocessing_hash,
                source_id=f"private-node-{index}",
            )
            fingerprints.append(validate_local_data(data, task).fingerprint)
            model = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                model.weight.fill_(float(index))
            prepared.append(
                prepare_local_training(
                    draft=draft,
                    data=data,
                    model=model,
                    model_definition=_MODEL_DEFINITION,
                    seed=19,
                )
            )
        finally:
            _DATA_OWNER.reset(token)
    initial = prepared[0].create_initial_checkpoint(tmp_path / "initial.safetensors")
    invitation = create_invitation(
        draft=draft,
        initiator_public_key="peer-0",
        bootstrap_uri="axl://local-runtime",
    )
    nodes = [
        NodeRuntime(
            transport=transport,
            draft=draft,
            environment=draft.environment,
            dataset=draft.dataset,
            artifact_root=tmp_path / f"node-{index}" / "formation",
            failure=FailureConfig.for_run_root(tmp_path / f"node-{index}"),
            local_tensor_schema=prepared[index].tensor_schema,
        )
        for index, transport in enumerate(transports)
    ]

    async def run(index: int) -> NodeRunResult:
        token = _DATA_OWNER.set(index)
        try:

            def build(result: FormationResult) -> TrainingConfig:
                return prepared[index].build_config(
                    result=result,
                    local_public_key=f"peer-{index}",
                    run_root=tmp_path / f"node-{index}",
                )

            return await nodes[index].run_to_completion(
                formation=(
                    InitiatorFormation(
                        bootstrap_uri=invitation.bootstrap_uri,
                        checkpoint_path=initial.path,
                        tensor_schema=initial.tensor_schema,
                    )
                    if index == 0
                    else ParticipantFormation(invitation=invitation)
                ),
                training_factory=build,
            )
        finally:
            _DATA_OWNER.reset(token)

    results = await asyncio.wait_for(
        asyncio.gather(*(run(index) for index in range(4))), timeout=30.0
    )
    assert len({result.formation.manifest_hash for result in results}) == 1
    assert len(set(fingerprints)) == 4
    assert all(node.state is NodeState.STOPPED for node in nodes)
    for index, result in enumerate(results):
        assert len(result.commits) == 2
        assert result.formation.manifest.initial_checkpoint_hash == initial.sha256
        state = RunStore(tmp_path / f"node-{index}" / "run-store").load_state()
        assert state.committed_round == 1
        assert state.terminal is not None and state.terminal.result == "complete"
        assert state.prepared_commit is None
        assert state.algorithm_state is not None
        archive = RunArchive.open(tmp_path / f"node-{index}" / "run-store")
        assert archive.algorithm_state is not None
        checkpoint = archive.algorithm_state.load_tensors()
        fingerprint = checkpoint["noloco.v1.trainer.__dromeus_local__.data_fingerprint"]
        assert fingerprint.dtype == np.uint8
        assert fingerprint.tobytes().hex() == fingerprints[index]
        assert int(checkpoint["noloco.v1.completed_outer_steps"][0]) == 2
        assert all(np.isfinite(value).all() for value in checkpoint.values())
        token = _DATA_OWNER.set(index)
        try:
            restored = prepared[index].build_config(
                result=result.formation,
                local_public_key=f"peer-{index}",
                run_root=tmp_path / f"restored-{index}",
            ).algorithm
            assert isinstance(restored, NoLoCoAlgorithm)
            restored.load_checkpoint_tensors(checkpoint)
            restored_state = restored.checkpoint_tensors()
            assert set(restored_state) == set(checkpoint)
            assert all(
                np.array_equal(restored_state[name], value)
                for name, value in checkpoint.items()
            )
        finally:
            _DATA_OWNER.reset(token)
        assert datasets[index].reads > datasets[index].count * 3
        shared_bytes = canonical_json(result.formation.manifest)
        assert result.formation.manifest.dataset == task
        for private_value in (
            "private-node-",
            "train_path",
            "fingerprint",
            "sample_count",
        ):
            assert private_value.encode() not in shared_bytes
