from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import NoReturn

import pytest
import torch
from support.sample_manifest import manifest_data
from torch import nn
from torch.utils.data import TensorDataset

from dromeus import node as node_module
from dromeus.adapters.classification.data import LocalClassificationData
from dromeus.adapters.classification.runtime import (
    PreparedLocalTraining,
    prepare_local_training,
)
from dromeus.manifests.canonical import canonical_json
from dromeus.manifests.models import ClassificationTaskContract, DraftRunSpec
from dromeus.node import NodeConfig, NodeRole

_MODEL_DEFINITION = "local-node-linear-2x2-v1"


def _draft(*, local: bool = True) -> DraftRunSpec:
    data = manifest_data()
    for field in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del data[field]
    if local:
        definition_hash = hashlib.sha256(_MODEL_DEFINITION.encode()).hexdigest()
        data.update(
            manifest_version=4,
            expected_participant_count=4,
            model_id="local-node-classifier",
            model_definition_hash=definition_hash,
            dataset=ClassificationTaskContract(
                dataset_id="local-classification-v1",
                input_shape=(2,),
                input_dtype="float32",
                label_names=("negative", "positive"),
                preprocessing_hash="a" * 64,
            ),
            local_steps=1,
            round_count=2,
        )
        data["environment"]["model_definition_hash"] = definition_hash
        data["training"].update(crop_padding=0, normalize=False)
    return DraftRunSpec.model_validate(data)


def _config(tmp_path: Path, draft: DraftRunSpec) -> NodeConfig:
    draft_path = tmp_path / "draft.json"
    draft_path.write_bytes(canonical_json(draft))
    return NodeConfig(
        role=NodeRole.INITIATOR,
        draft_path=draft_path,
        axl_bridge_url="http://127.0.0.1:9002",
        run_root=tmp_path / "run",
        invitation_path=tmp_path / "invitation.json",
        bootstrap_uri="axl://local-node-test",
    )


def _prepare(draft: DraftRunSpec, *, fault: str | None = None) -> PreparedLocalTraining:
    assert isinstance(draft.dataset, ClassificationTaskContract)
    labels = draft.dataset.label_names
    preprocessing = draft.dataset.preprocessing_hash
    inputs = torch.tensor([[1.0, 2.0]])
    definition = _MODEL_DEFINITION
    model = nn.Linear(2, 2, bias=False)
    if fault == "label-meaning":
        labels = tuple(reversed(labels))
    elif fault == "preprocessing":
        preprocessing = "b" * 64
    elif fault == "input-shape":
        inputs = torch.ones((1, 3))
    elif fault == "input-nonfinite":
        inputs[0, 0] = float("nan")
    elif fault == "model-input":
        model = nn.Linear(3, 2, bias=False)
    elif fault == "model-output":
        model = nn.Linear(2, 3, bias=False)
    elif fault == "model-definition":
        definition = "wrong-model-definition"
    return prepare_local_training(
        draft=draft,
        data=LocalClassificationData(
            train_data=TensorDataset(inputs, torch.tensor([0])),
            evaluation_data=None,
            label_names=labels,
            preprocessing_hash=preprocessing,
        ),
        model=model,
        model_definition=definition,
        seed=17,
    )


class _ReachedTransport(Exception):
    pass


def test_custom_local_factory_completes_real_preflight_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    draft = _draft()
    received: list[DraftRunSpec] = []

    def factory(actual: DraftRunSpec) -> PreparedLocalTraining:
        received.append(actual)
        return _prepare(actual)

    def transport(_config: object) -> NoReturn:
        assert received == [draft]
        raise _ReachedTransport

    assert not hasattr(node_module, "prepare_cifar_training")
    monkeypatch.setattr(node_module, "AXLTransport", transport)
    with pytest.raises(_ReachedTransport):
        asyncio.run(
            node_module.run_node(_config(tmp_path, draft), prepare_training=factory)
        )
    assert not (tmp_path / "invitation.json").exists()


@pytest.mark.parametrize(
    "fault",
    (
        "label-meaning",
        "preprocessing",
        "input-shape",
        "input-nonfinite",
        "model-input",
        "model-output",
        "model-definition",
    ),
)
def test_local_task_data_and_model_failures_never_start_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    def forbidden_transport(_config: object) -> NoReturn:
        raise AssertionError("invalid local preflight started AXL transport")

    def factory(draft: DraftRunSpec) -> PreparedLocalTraining:
        return _prepare(draft, fault=fault)

    assert not hasattr(node_module, "prepare_cifar_training")
    monkeypatch.setattr(node_module, "AXLTransport", forbidden_transport)
    with pytest.raises(ValueError):
        asyncio.run(
            node_module.run_node(_config(tmp_path, _draft()), prepare_training=factory)
        )
    assert not (tmp_path / "invitation.json").exists()


def test_factory_prepared_for_another_draft_fails_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    draft = _draft()
    prepared = _prepare(draft.model_copy(update={"run_id": "another-run"}))

    def factory(_draft: DraftRunSpec) -> PreparedLocalTraining:
        return prepared

    def forbidden_transport(_config: object) -> NoReturn:
        raise AssertionError("mismatched prepared draft started AXL transport")

    monkeypatch.setattr(node_module, "AXLTransport", forbidden_transport)
    with pytest.raises(ValueError, match="draft does not match"):
        asyncio.run(
            node_module.run_node(_config(tmp_path, draft), prepare_training=factory)
        )


def test_node_config_has_no_workload_specific_fields() -> None:
    assert (
        not {"local_data_path", "dataset_cache", "training_device", "benchmark_seed"}
        & NodeConfig.model_fields.keys()
    )
