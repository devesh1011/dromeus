from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError
from support.sample_manifest import manifest_data

from dromeus import node as node_module
from dromeus.manifests.models import DraftRunSpec, SealedManifest
from dromeus.membership.formation import FormationResult, create_invitation
from dromeus.node import NodeRole, load_node_config, main
from dromeus.runtime import ParticipantFormation, TrainingConfig
from dromeus.training.state import InitialCheckpoint


def test_load_node_config_validates_frozen_initiator_inputs(tmp_path: Path) -> None:
    config_path = tmp_path / "node.yaml"
    config_path.write_text(
        "\n".join(
            (
                "role: initiator",
                f"draft_path: {tmp_path / 'draft.yaml'}",
                "axl_bridge_url: http://127.0.0.1:9002",
                f"run_root: {tmp_path / 'run'}",
                f"invitation_path: {tmp_path / 'invitation.json'}",
                "bootstrap_uri: tls://bootstrap.example:9000",
            )
        ),
        encoding="utf-8",
    )

    config = load_node_config(config_path)

    assert config.role is NodeRole.INITIATOR
    assert config.axl_bridge_url == "http://127.0.0.1:9002"
    assert config.run_root == tmp_path / "run"

    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "\nextra: rejected\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="extra"):
        load_node_config(config_path)


def test_load_node_config_rejects_non_loopback_axl_bridge(tmp_path: Path) -> None:
    config_path = tmp_path / "node.yaml"
    config_path.write_text(
        "\n".join(
            (
                "role: participant",
                f"draft_path: {tmp_path / 'draft.yaml'}",
                "axl_bridge_url: http://worker.example:9002",
                f"run_root: {tmp_path / 'run'}",
                f"invitation_path: {tmp_path / 'invitation.json'}",
                "bootstrap_uri: tls://bootstrap.example:9000",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="loopback"):
        load_node_config(config_path)


def test_run_node_uses_deep_runtime_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    draft_data = manifest.model_dump(mode="python")
    for field in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del draft_data[field]
    draft = DraftRunSpec.model_validate(draft_data)
    draft_path = tmp_path / "draft.yaml"
    draft_path.write_text(draft.model_dump_json(), encoding="utf-8")
    invitation_path = tmp_path / "invitation.json"
    invitation = create_invitation(
        draft=draft,
        initiator_public_key="peer-0",
        bootstrap_uri="tls://bootstrap.example:9000",
    )
    invitation_path.write_text(invitation.model_dump_json(), encoding="utf-8")
    captured: dict[str, object] = {}

    class FakeTransport:
        def __init__(self, config: object) -> None:
            captured["transport_config"] = config

        async def local_public_key(self) -> str:
            return "peer-1"

    class FakeRuntime:
        def __init__(self, **kwargs: object) -> None:
            captured["runtime_init"] = kwargs

        async def run_to_completion(self, **kwargs: object) -> None:
            captured["lifecycle"] = kwargs

    base_training = cast(TrainingConfig, object())
    decorated_training = cast(TrainingConfig, object())

    class FakePreparedTraining:
        tensor_schema = None

        def validate_draft(self, draft: DraftRunSpec) -> None:
            assert draft.run_id == manifest.run_id

        def create_initial_checkpoint(self, path: Path) -> InitialCheckpoint:
            raise AssertionError("participant does not create initial checkpoint")

        def build_config(self, **kwargs: object) -> TrainingConfig:
            captured["training_build"] = kwargs
            return base_training

    def fake_prepare_training(actual: DraftRunSpec) -> FakePreparedTraining:
        captured["training_prepare"] = actual
        return FakePreparedTraining()

    def decorate_training(
        result: FormationResult,
        local_public_key: str,
        training: TrainingConfig,
    ) -> TrainingConfig:
        captured["training_decorator"] = {
            "result": result,
            "local_public_key": local_public_key,
            "training": training,
        }
        return decorated_training

    monkeypatch.setattr(node_module, "AXLTransport", FakeTransport)
    monkeypatch.setattr(node_module, "NodeRuntime", FakeRuntime)
    config = node_module.NodeConfig(
        role=NodeRole.PARTICIPANT,
        draft_path=draft_path,
        axl_bridge_url="http://127.0.0.1:9002",
        run_root=tmp_path / "run",
        invitation_path=invitation_path,
        bootstrap_uri="tls://bootstrap.example:9000",
    )

    asyncio.run(
        node_module.run_node(
            config,
            training_decorator=decorate_training,
            prepare_training=fake_prepare_training,
        )
    )

    lifecycle = cast(dict[str, object], captured["lifecycle"])
    assert captured["training_prepare"] == draft
    assert isinstance(lifecycle["formation"], ParticipantFormation)
    assert set(lifecycle) == {
        "formation",
        "training_factory",
        "manifest_expectation",
        "ready_hook",
        "completion_hook",
    }
    training_factory = lifecycle["training_factory"]
    assert callable(training_factory)
    formation_result = FormationResult(
        manifest=manifest,
        manifest_hash="a" * 64,
        checkpoint_path=tmp_path / "checkpoint.safetensors",
    )
    assert training_factory(formation_result) is decorated_training
    decorator_call = cast(dict[str, object], captured["training_decorator"])
    assert decorator_call == {
        "result": formation_result,
        "local_public_key": "peer-1",
        "training": base_training,
    }


def test_main_executes_config_instead_of_exiting_after_start_event(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "node.yaml"
    config_path.write_text(
        "\n".join(
            (
                "role: initiator",
                f"draft_path: {tmp_path / 'missing-draft.yaml'}",
                "axl_bridge_url: http://127.0.0.1:9002",
                f"run_root: {tmp_path / 'run'}",
                f"invitation_path: {tmp_path / 'invitation.json'}",
                "bootstrap_uri: tls://bootstrap.example:9000",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        main(
            [
                "--config",
                str(config_path),
                "--factory",
                "dromeus.application:prepare_application",
            ]
        )
