from __future__ import annotations

import asyncio
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from support.application_fixture import application_draft, local_file
from support.in_memory_transport import InMemoryNetwork, InMemoryTransport
from support.sample_manifest import manifest_data
from torch import nn

from dromeus import node as node_module
from dromeus.algorithms.noloco import NoLoCoAlgorithm
from dromeus.application import (
    PreparedApplication,
    prepare_application,
)
from dromeus.manifests.canonical import canonical_hash, canonical_json
from dromeus.manifests.models import DraftRunSpec, SealedManifest
from dromeus.membership.formation import (
    FormationResult,
    ReadyValidationError,
    seal_manifest,
    validate_ready,
)
from dromeus.node import NodeConfig, NodeRole
from dromeus.persistence.archive import RunArchive
from dromeus.runtime import build_algorithm
from dromeus.telemetry.events import JsonlEventSink
from dromeus.telemetry.evidence import EvidenceLog, TaskRoundMetricsEvidence
from dromeus.telemetry.metrics import JsonlMetricsPublisher, RoundTiming
from dromeus.training.base import EvaluationResult
from dromeus.training.trainer import PyTorchTrainer
from dromeus.transport.axl import AXLBridgeConfig
from examples.custom_training import (
    MODEL_DEFINITION,
    TASK_DEFINITION,
    TRAINING_DEFINITION,
    LocalRegression,
)


def prepared(draft: DraftRunSpec, local: LocalRegression) -> PreparedApplication:
    return prepare_application(
        draft=draft,
        trainer=local.trainer,
        model_definition=MODEL_DEFINITION,
        task_definition=TASK_DEFINITION,
        training_definition=TRAINING_DEFINITION,
    )


@pytest.mark.parametrize(
    "algorithm,compressed,omit_algorithm",
    [
        ("dpsgd", False, False),
        ("noloco", False, False),
        ("noloco", True, False),
        ("noloco", True, True),
    ],
)
def test_custom_regression_through_public_node_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    algorithm: str,
    compressed: bool,
    omit_algorithm: bool,
) -> None:
    network = InMemoryNetwork()

    class LocalTransport(InMemoryTransport):
        async def topology(self) -> dict[str, object]:
            return {"test": True}

    transports = [
        LocalTransport(network=network, public_key=f"peer-{rank}") for rank in range(4)
    ]

    def transport(config: AXLBridgeConfig) -> LocalTransport:
        return transports[int(config.base_url.rsplit(":", 1)[1]) - 9100]

    monkeypatch.setattr(node_module, "AXLTransport", transport)
    draft = application_draft(algorithm, compressed)
    draft_path = tmp_path / "draft.json"
    if omit_algorithm:
        draft_path.write_text(draft.model_dump_json(exclude={"algorithm_id"}))
    else:
        draft_path.write_bytes(canonical_json(draft))
    locals_ = [
        LocalRegression(local_file(tmp_path / f"data-{rank}.npz", rank))
        for rank in range(4)
    ]
    initial = [
        {key: value.copy() for key, value in local.trainer.weights().items()}
        for local in locals_
    ]

    def factory_for(rank: int) -> Callable[[DraftRunSpec], PreparedApplication]:
        return lambda actual: prepared(actual, locals_[rank])

    async def run() -> None:
        await asyncio.wait_for(
            asyncio.gather(
                *(
                    node_module.run_node(
                        NodeConfig(
                            role=NodeRole.INITIATOR
                            if rank == 0
                            else NodeRole.PARTICIPANT,
                            draft_path=draft_path,
                            axl_bridge_url=f"http://127.0.0.1:{9100 + rank}",
                            run_root=tmp_path / f"node-{rank}",
                            invitation_path=tmp_path / "invitation.json",
                            bootstrap_uri="axl://custom-regression",
                        ),
                        prepare_training=factory_for(rank),
                    )
                    for rank in range(4)
                )
            ),
            timeout=30,
        )

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        asyncio.run(run())
    finally:
        torch.set_num_threads(old_threads)
    assert len({local.data_identity for local in locals_}) == 4
    for rank, local in enumerate(locals_):
        assert local.trainer.completed_steps == 9
        assert any(
            not np.array_equal(initial[rank][name], value)
            for name, value in local.trainer.weights().items()
        )
        root = tmp_path / f"node-{rank}"
        archive = RunArchive.open(root / "run-store")
        assert archive.manifest.algorithm_id == algorithm
        assert archive.state.terminal is not None
        assert archive.state.terminal.result == "complete"
        log = EvidenceLog.open(
            root / "logs/dromeus.jsonl",
            run_id=draft.run_id,
            manifest_hash=archive.manifest_hash,
        )
        metrics = [
            record
            for record in log.records
            if isinstance(record, TaskRoundMetricsEvidence)
        ]
        assert len(metrics) == 3
        assert all(
            set(record.evaluation_metrics) == {"mae", "rmse"} for record in metrics
        )
        assert all(record.evaluation_accuracy is None for record in metrics)
        assert archive.algorithm_state is not None
        checkpoint = archive.algorithm_state.load_tensors()
        if algorithm == "noloco":
            assert int(checkpoint["noloco.v1.completed_outer_steps"][0]) == 3
        restored = LocalRegression(tmp_path / f"data-{rank}.npz")
        prepared(draft, restored)
        if algorithm == "noloco":
            restored_algorithm = build_algorithm(
                manifest=archive.manifest, trainer=restored.trainer
            )
            assert isinstance(restored_algorithm, NoLoCoAlgorithm)
            restored_algorithm.load_checkpoint_tensors(checkpoint)
        else:
            restored.trainer.load_checkpoint_tensors(checkpoint)
        for name, value in local.trainer.checkpoint_tensors().items():
            np.testing.assert_array_equal(
                value, restored.trainer.checkpoint_tensors()[name]
            )
        local.trainer.train_local_steps(1)
        restored.trainer.train_local_steps(1)
        for name, value in local.trainer.weights().items():
            np.testing.assert_array_equal(value, restored.trainer.weights()[name])


def test_core_imports_without_benchmark_packages() -> None:
    code = """
import importlib.abc, sys
class DenyBenchmarks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'benchmarks', 'datasets', 'PIL'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, DenyBenchmarks())
import dromeus.node, dromeus.application, dromeus.training.trainer
assert not any('resnet' in name.lower() or 'cifar10' in name for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


@pytest.mark.parametrize(
    "definition", ["model_definition", "task_definition", "training_definition"]
)
def test_shared_definition_mismatch_rejected(tmp_path: Path, definition: str) -> None:
    local = LocalRegression(local_file(tmp_path / "data.npz"))
    definitions = dict(
        model_definition=MODEL_DEFINITION,
        task_definition=TASK_DEFINITION,
        training_definition=TRAINING_DEFINITION,
    )
    definitions[definition] = "wrong"
    with pytest.raises(ValueError, match="definition"):
        prepare_application(
            draft=application_draft(),
            trainer=local.trainer,
            model_definition=definitions["model_definition"],
            task_definition=definitions["task_definition"],
            training_definition=definitions["training_definition"],
        )


def test_signed_loss_and_named_metrics_are_retained(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    trainer = PyTorchTrainer(
        model=model,
        train_step=lambda _: -2.0,
        save_state=dict,
        load_state=lambda _: None,
        state_identity="stateless-negative-objective",
        evaluate=lambda _: {"signed_score": -0.5},
    )
    trainer.train_local_steps(1)
    assert trainer.local_loss == -2.0
    assert trainer.evaluate() == EvaluationResult({"signed_score": -0.5})
    sink = JsonlEventSink(tmp_path / "metrics.jsonl")
    publisher = JsonlMetricsPublisher(
        sink=sink, run_id="custom", manifest_hash="a" * 64, node_id="peer-0"
    )

    async def publish() -> None:
        await publisher.start()
        assert publisher.submit(
            RoundTiming(
                round_id=0,
                peer_id="peer-1",
                local_compute_seconds=0.0,
                peer_wait_seconds=0.0,
                transfer_seconds=0.0,
                mixing_seconds=0.0,
                evaluation_seconds=0.0,
                local_loss=-2.0,
                evaluation_metrics={"signed_score": -0.5},
            )
        )
        await publisher.stop()

    asyncio.run(publish())
    log = EvidenceLog.open(
        tmp_path / "metrics.jsonl", run_id="custom", manifest_hash="a" * 64
    )
    record = log.records[0]
    assert isinstance(record, TaskRoundMetricsEvidence)
    assert record.local_loss == -2.0 and record.evaluation_metrics == {
        "signed_score": -0.5
    }


def test_checkpoint_failure_rolls_back_and_rejects_other_local_data(
    tmp_path: Path,
) -> None:
    local = LocalRegression(local_file(tmp_path / "data.npz"))
    local.trainer.train_local_steps(3)
    old = local.trainer.checkpoint_tensors()
    bad = {key: value.copy() for key, value in old.items()}
    bad["application.cursor"] = np.array([-1], dtype=np.int64)
    with pytest.raises(ValueError, match="cursor"):
        local.trainer.load_checkpoint_tensors(bad)
    for key, value in old.items():
        np.testing.assert_array_equal(value, local.trainer.checkpoint_tensors()[key])
    other = LocalRegression(local_file(tmp_path / "other.npz", rank=1))
    with pytest.raises(ValueError, match="identity"):
        other.trainer.load_checkpoint_tensors(old)


def test_partial_application_restore_is_rolled_back_and_nan_rejected() -> None:
    model = nn.Linear(2, 1)
    current = np.array([4.0], dtype=np.float32)
    loads: list[float] = []

    def save() -> dict[str, np.ndarray]:
        return {"value": current.copy()}

    def restore(state: dict[str, np.ndarray]) -> None:
        nonlocal current
        current = state["value"].copy()
        loads.append(float(current[0]))
        if current[0] < 0:
            raise ValueError("application rejected state after a partial change")

    trainer = PyTorchTrainer(
        model=model,
        train_step=lambda _: None,
        save_state=save,
        load_state=restore,
        state_identity="rollback-test",
    )
    assert trainer.evaluate() is None
    old = trainer.checkpoint_tensors()
    bad = {key: value.copy() for key, value in old.items()}
    bad["application.value"][0] = -1.0
    with pytest.raises(ValueError, match="partial change"):
        trainer.load_checkpoint_tensors(bad)
    assert loads == [-1.0, 4.0]
    for key, value in old.items():
        np.testing.assert_array_equal(value, trainer.checkpoint_tensors()[key])
    bad["model.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        trainer.load_checkpoint_tensors(bad)
    assert loads == [-1.0, 4.0]


def test_integer_buffers_are_local_and_checkpointed() -> None:
    model = nn.Linear(2, 1)
    model.register_buffer("counter", torch.tensor(7, dtype=torch.int64))
    trainer = PyTorchTrainer(
        model=model,
        train_step=lambda _: None,
        save_state=dict,
        load_state=lambda _: None,
        state_identity="integer-buffer-test",
    )
    assert "counter" not in trainer.weights()
    checkpoint = trainer.checkpoint_tensors()
    model.get_buffer("counter").fill_(8)
    trainer.load_weights(trainer.weights())
    assert int(model.get_buffer("counter")) == 8
    trainer.load_checkpoint_tensors(checkpoint)
    assert int(model.get_buffer("counter")) == 7
    assert checkpoint["model.counter"].shape == ()


@pytest.mark.parametrize("field", ["training", "optimizer", "learning_rate"])
def test_application_manifest_rejects_recipe_fields(field: str) -> None:
    value = application_draft().model_dump(mode="python")
    value[field] = {
        "training": manifest_data()["training"],
        "optimizer": "adam",
        "learning_rate": 0.1,
    }[field]
    with pytest.raises(ValueError, match="application"):
        DraftRunSpec.model_validate(value)


def test_initial_checkpoint_copies_noncontiguous_and_shared_storage(
    tmp_path: Path,
) -> None:
    from dromeus.manifests.canonical import load_safetensors

    model = nn.Module()
    parameter = nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3).T)
    model.register_parameter("weight", parameter)
    model.register_parameter("shared", parameter)
    trainer = PyTorchTrainer(
        model=model,
        train_step=lambda _: None,
        save_state=dict,
        load_state=lambda _: None,
        state_identity="shared-storage",
    )
    initial = trainer.create_initial_checkpoint(
        tmp_path / "initial.safetensors", model_definition="shared-v1"
    )
    saved = load_safetensors(str(initial.path))
    for key, value in trainer.weights().items():
        np.testing.assert_array_equal(saved[key], value)


def _formed_application(
    draft: DraftRunSpec, local: LocalRegression, root: Path
) -> FormationResult:
    initial = local.trainer.create_initial_checkpoint(
        root / "initial.safetensors", model_definition=MODEL_DEFINITION
    )
    manifest = seal_manifest(
        draft=draft,
        participant_keys={f"peer-{rank}" for rank in range(4)},
        initial_checkpoint_hash=initial.sha256,
        tensor_schema=initial.tensor_schema,
    )
    return FormationResult(manifest, canonical_hash(manifest), initial.path)


@pytest.mark.parametrize("change", ["beta", "round_count", "algorithm"])
def test_actual_sealed_policy_is_checked_before_ready_and_build(
    tmp_path: Path, change: str
) -> None:
    draft = application_draft()
    local = LocalRegression(local_file(tmp_path / "data.npz"))
    workload = prepared(draft, local)
    result = _formed_application(draft, local, tmp_path)
    value = result.manifest.model_dump(mode="python")
    if change == "beta":
        value["algorithm_config"]["beta"] = 0.9
    elif change == "round_count":
        value["round_count"] += 1
    else:
        value.update(algorithm_id="dpsgd", algorithm_config=None, artifact_codecs=None)
    changed = SealedManifest.model_validate(value)
    with pytest.raises(ReadyValidationError, match="draft"):
        validate_ready(
            manifest=changed,
            local_public_key="peer-0",
            environment=draft.environment,
            dataset=draft.dataset,
            checkpoint_hash=changed.initial_checkpoint_hash,
            local_tensor_schema=local.trainer.tensor_schema,
        )
    with pytest.raises(ValueError, match="draft"):
        workload.build_config(
            result=FormationResult(
                changed, canonical_hash(changed), result.checkpoint_path
            ),
            local_public_key="peer-0",
            run_root=tmp_path / "run",
        )


@pytest.mark.parametrize(
    "change", ["alpha", "beta", "gamma", "inner_steps", "codec", "fraction"]
)
def test_application_checkpoint_rejects_different_outer_policy(
    tmp_path: Path, change: str
) -> None:
    path = local_file(tmp_path / "data.npz")
    draft = application_draft(compressed=True)
    local = LocalRegression(path)
    prepared(draft, local)
    result = _formed_application(draft, local, tmp_path)
    original = build_algorithm(manifest=result.manifest, trainer=local.trainer)
    checkpoint = original.checkpoint_tensors()
    value = draft.model_dump(mode="python")
    if change in ("alpha", "beta", "gamma"):
        value["algorithm_config"][change] += 0.1
    elif change == "inner_steps":
        value["local_steps"] += 1
        value["algorithm_config"]["inner_steps"] += 1
    elif change == "codec":
        value["artifact_codecs"] = application_draft().artifact_codecs
    else:
        value["artifact_codecs"][0]["top_k_fraction"] = 0.25
    changed = DraftRunSpec.model_validate(value)
    fresh = LocalRegression(path)
    prepared(changed, fresh)
    other = _formed_application(changed, fresh, tmp_path / "other")
    algorithm = build_algorithm(manifest=other.manifest, trainer=fresh.trainer)
    assert isinstance(algorithm, NoLoCoAlgorithm)
    before = algorithm.checkpoint_tensors()
    with pytest.raises(ValueError, match="identity|context"):
        algorithm.load_checkpoint_tensors(checkpoint)
    for name, tensor in before.items():
        np.testing.assert_array_equal(tensor, algorithm.checkpoint_tensors()[name])


@pytest.mark.parametrize("mode", ["checkpoint", "weights"])
def test_conflicting_tied_state_rejects_before_any_mutation(mode: str) -> None:
    model = nn.Module()
    model.register_parameter("first", nn.Parameter(torch.ones(2, 2)))
    model.register_parameter("second", model.get_parameter("first"))
    loads: list[bool] = []
    trainer = PyTorchTrainer(
        model=model,
        train_step=lambda _: None,
        save_state=dict,
        load_state=lambda _: loads.append(True),
        state_identity="tied-model",
    )
    old = trainer.checkpoint_tensors()
    values = trainer.weights() if mode == "weights" else trainer.checkpoint_tensors()
    prefix = "" if mode == "weights" else "model."
    values[prefix + "first"].fill(3)
    values[prefix + "second"].fill(4)
    with pytest.raises(ValueError, match="alias|tied"):
        if mode == "weights":
            trainer.load_weights(values)
        else:
            trainer.load_checkpoint_tensors(values)
    assert loads == []
    for name, value in old.items():
        np.testing.assert_array_equal(value, trainer.checkpoint_tensors()[name])
    values[prefix + "second"].fill(3)
    if mode == "weights":
        trainer.load_weights(values)
    else:
        trainer.load_checkpoint_tensors(values)
    assert all(np.all(value == 3) for value in trainer.weights().values())


@pytest.mark.parametrize("separate_storage", [False, True])
def test_partial_storage_overlap_rejects_during_preparation(
    separate_storage: bool,
) -> None:
    model = nn.Module()
    if separate_storage:
        backing = np.ones(6, dtype=np.float32)
        first = torch.from_numpy(backing[:4])  # pyright: ignore[reportUnknownMemberType]
        second = torch.from_numpy(backing[2:])  # pyright: ignore[reportUnknownMemberType]
    else:
        storage = torch.ones(6)
        first, second = storage[:4], storage[2:]
    model.register_parameter("first", nn.Parameter(first))
    model.register_parameter("second", nn.Parameter(second))
    with pytest.raises(ValueError, match="overlap"):
        PyTorchTrainer(
            model=model,
            train_step=lambda _: None,
            save_state=dict,
            load_state=lambda _: None,
            state_identity="overlapping-model",
        )


def test_disjoint_model_views_share_storage_safely() -> None:
    storage = torch.ones(6)
    model = nn.Module()
    model.register_parameter("first", nn.Parameter(storage[:3]))
    model.register_parameter("second", nn.Parameter(storage[3:]))
    trainer = PyTorchTrainer(
        model=model,
        train_step=lambda _: None,
        save_state=dict,
        load_state=lambda _: None,
        state_identity="disjoint-model",
    )
    values = trainer.weights()
    values["first"].fill(2)
    values["second"].fill(3)
    trainer.load_weights(values)
    checkpoint = trainer.checkpoint_tensors()
    trainer.load_checkpoint_tensors(checkpoint)
    for name, expected in values.items():
        np.testing.assert_array_equal(expected, trainer.weights()[name])


def test_older_application_context_is_rejected_and_same_policy_new_run_is_valid(
    tmp_path: Path,
) -> None:
    draft = application_draft()
    path = local_file(tmp_path / "data.npz")
    old = LocalRegression(path)
    assert draft.application_training is not None
    old.trainer.bind_contract(
        ":".join(
            (
                draft.model_definition_hash,
                canonical_hash(draft.dataset),
                canonical_hash(draft.application_training),
            )
        )
    )
    checkpoint = old.trainer.checkpoint_tensors()
    current = LocalRegression(path)
    prepared(draft, current)
    with pytest.raises(ValueError, match="identity"):
        current.trainer.load_checkpoint_tensors(checkpoint)
    current.trainer.train_local_steps(2)
    checkpoint = current.trainer.checkpoint_tensors()
    other_run = draft.model_copy(update={"run_id": "explicit-offline-continuation"})
    fresh = LocalRegression(path)
    prepared(other_run, fresh)
    fresh.trainer.load_checkpoint_tensors(checkpoint)
    current.trainer.train_local_steps(1)
    fresh.trainer.train_local_steps(1)
    for name, value in current.trainer.checkpoint_tensors().items():
        np.testing.assert_array_equal(value, fresh.trainer.checkpoint_tensors()[name])
