from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pytest
from support.sample_manifest import manifest_data

from dromeus import node as node_module
from dromeus.application import installed_versions
from dromeus.errors import ConfigError
from dromeus.manifests.models import DraftRunSpec, SealedManifest
from dromeus.membership.formation import FormationResult, create_invitation
from dromeus.node import NodeRole, load_node_config
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
    with pytest.raises(ConfigError, match="extra"):
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

    with pytest.raises(ConfigError, match="loopback"):
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
    # run_node refuses drafts made for another Dromeus or PyTorch version.
    dromeus_version, pytorch_version = installed_versions()
    draft_data["environment"] = {
        **draft_data["environment"],
        "dromeus_version": dromeus_version,
        "pytorch_version": pytorch_version,
    }
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
            self.event_sink = kwargs["event_sink"]

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


def test_enrollment_window_is_an_initiator_setting(tmp_path: Path) -> None:
    def config_text(role: str) -> str:
        return "\n".join(
            (
                f"role: {role}",
                f"draft_path: {tmp_path / 'draft.yaml'}",
                "axl_bridge_url: http://127.0.0.1:9002",
                f"run_root: {tmp_path / 'run'}",
                f"invitation_path: {tmp_path / 'invitation.json'}",
                "bootstrap_uri: tls://bootstrap.example:9000",
                "enrollment_window_seconds: 600",
            )
        )

    config_path = tmp_path / "node.yaml"
    config_path.write_text(config_text("initiator"), encoding="utf-8")
    assert load_node_config(config_path).enrollment_window_seconds == 600

    config_path.write_text(config_text("participant"), encoding="utf-8")
    with pytest.raises(ConfigError, match="initiator only"):
        load_node_config(config_path)


def test_initiator_writes_and_enforces_its_enrollment_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import UTC, datetime, timedelta

    from dromeus.manifests.models import Invitation
    from dromeus.runtime import InitiatorFormation

    manifest = SealedManifest.model_validate(manifest_data())
    draft_data = manifest.model_dump(mode="python")
    for field in ("draft_hash", "participants", "initial_checkpoint_hash", "tensor_schema"):
        del draft_data[field]
    dromeus_version, pytorch_version = installed_versions()
    draft_data["environment"] = {
        **draft_data["environment"],
        "dromeus_version": dromeus_version,
        "pytorch_version": pytorch_version,
    }
    draft_path = tmp_path / "draft.yaml"
    draft_path.write_text(
        DraftRunSpec.model_validate(draft_data).model_dump_json(), encoding="utf-8"
    )
    captured: dict[str, object] = {}

    class FakeTransport:
        def __init__(self, config: object) -> None:
            pass

        async def local_public_key(self) -> str:
            return "peer-0"

    class FakeRuntime:
        def __init__(self, **kwargs: object) -> None:
            self.event_sink = kwargs["event_sink"]

        async def run_to_completion(self, **kwargs: object) -> None:
            captured["formation"] = kwargs["formation"]

    class FakePreparedTraining:
        tensor_schema = manifest.tensor_schema

        def validate_draft(self, draft: DraftRunSpec) -> None:
            pass

        def create_initial_checkpoint(self, path: Path) -> InitialCheckpoint:
            return InitialCheckpoint(
                path=path, tensor_schema=manifest.tensor_schema, sha256="0" * 64
            )

        def build_config(self, **kwargs: object) -> TrainingConfig:
            raise AssertionError("formation is faked; training is never built")

    monkeypatch.setattr(node_module, "AXLTransport", FakeTransport)
    monkeypatch.setattr(node_module, "NodeRuntime", FakeRuntime)
    config = node_module.NodeConfig(
        role=NodeRole.INITIATOR,
        draft_path=draft_path,
        axl_bridge_url="http://127.0.0.1:9002",
        run_root=tmp_path / "run",
        invitation_path=tmp_path / "invitation.json",
        bootstrap_uri="tls://bootstrap.example:9000",
        enrollment_window_seconds=600,
    )
    before = datetime.now(UTC)

    asyncio.run(
        node_module.run_node(
            config, prepare_training=lambda draft: FakePreparedTraining()
        )
    )

    invitation = Invitation.model_validate_json(
        (tmp_path / "invitation.json").read_bytes()
    )
    formation = captured["formation"]
    assert isinstance(formation, InitiatorFormation)
    assert invitation.enrollment_expires_at == formation.enrollment_expires_at
    assert invitation.enrollment_expires_at is not None
    assert (
        before + timedelta(seconds=600)
        <= invitation.enrollment_expires_at
        <= datetime.now(UTC) + timedelta(seconds=600)
    )


def test_participant_waits_for_a_completely_copied_invitation(tmp_path: Path) -> None:
    """Copying the invitation (scp, cp) creates the file before its bytes arrive."""
    from support.sample_manifest import manifest_data

    from dromeus.manifests.canonical import canonical_json
    from dromeus.manifests.models import DraftRunSpec
    from dromeus.membership.formation import create_invitation
    from dromeus.node import _read_invitation  # pyright: ignore[reportPrivateUsage]

    values = manifest_data()
    for field in ("participants", "draft_hash", "initial_checkpoint_hash", "tensor_schema"):
        del values[field]
    invitation = create_invitation(
        draft=DraftRunSpec.model_validate(values),
        initiator_public_key="peer-0",
        bootstrap_uri="axl://bootstrap",
    )
    path = tmp_path / "invitation.json"
    path.write_bytes(b"")  # created, not yet written
    payload = canonical_json(invitation)

    async def deliver_later() -> None:
        await asyncio.sleep(0.3)
        path.write_bytes(payload[: len(payload) // 2])  # partly written
        await asyncio.sleep(0.3)
        path.write_bytes(payload)

    async def scenario() -> object:
        writer = asyncio.create_task(deliver_later())
        received = await _read_invitation(path, timeout_seconds=5)
        await writer
        return received

    assert asyncio.run(scenario()) == invitation


def test_unreadable_invitation_times_out_as_formation_error(tmp_path: Path) -> None:
    from dromeus.errors import FormationError
    from dromeus.node import _read_invitation  # pyright: ignore[reportPrivateUsage]

    path = tmp_path / "invitation.json"
    path.write_text("{not json")

    with pytest.raises(FormationError, match="invitation"):
        asyncio.run(_read_invitation(path, timeout_seconds=0.5))
