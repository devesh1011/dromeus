from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from dromeus.node import NodeRole, load_node_config, main


def test_load_node_config_validates_frozen_initiator_inputs(tmp_path: Path) -> None:
    config_path = tmp_path / "node.yaml"
    config_path.write_text(
        "\n".join(
            (
                "role: initiator",
                f"draft_path: {tmp_path / 'draft.yaml'}",
                "axl_bridge_url: http://127.0.0.1:9002",
                f"run_root: {tmp_path / 'run'}",
                f"dataset_cache: {tmp_path / 'cifar'}",
                f"invitation_path: {tmp_path / 'invitation.json'}",
                "bootstrap_uri: tls://bootstrap.example:9000",
                "benchmark_seed: 17",
            )
        ),
        encoding="utf-8",
    )

    config = load_node_config(config_path)

    assert config.role is NodeRole.INITIATOR
    assert config.benchmark_seed == 17
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
                f"dataset_cache: {tmp_path / 'cifar'}",
                f"invitation_path: {tmp_path / 'invitation.json'}",
                "bootstrap_uri: tls://bootstrap.example:9000",
                "benchmark_seed: 17",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="loopback"):
        load_node_config(config_path)


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
                f"dataset_cache: {tmp_path / 'cifar'}",
                f"invitation_path: {tmp_path / 'invitation.json'}",
                "bootstrap_uri: tls://bootstrap.example:9000",
                "benchmark_seed: 17",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        main(["--config", str(config_path)])
