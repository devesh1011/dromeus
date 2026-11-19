"""`dromeus doctor` against a real AXL bridge with live peer links."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
import yaml
from examples.custom_training import task
from support.local_axl import local_axl_cluster

from dromeus import make_draft
from dromeus.diagnostics import Status, diagnose

pytestmark = pytest.mark.skipif(
    not os.environ.get("DROMEUS_AXL_BINARY"),
    reason="set DROMEUS_AXL_BINARY to run the real AXL doctor test",
)


def test_doctor_passes_on_a_connected_axl_node(tmp_path: Path) -> None:
    binary = Path(os.environ["DROMEUS_AXL_BINARY"])
    draft = tmp_path / "draft.yaml"
    make_draft(
        task(),
        draft,
        run_id="doctor-axl",
        participants=4,
        rounds=2,
        local_steps=1,
        axl_version="local-test",
    )

    async def scenario() -> list[tuple[str, Status, str]]:
        async with local_axl_cluster(binary=binary, log_root=tmp_path / "axl") as nodes:
            config = tmp_path / "node.yaml"
            config.write_text(
                yaml.safe_dump(
                    {
                        "role": "initiator",
                        "draft_path": str(draft),
                        "axl_bridge_url": nodes[0].bridge_url,
                        "run_root": str(tmp_path / "run"),
                        "invitation_path": str(tmp_path / "invitation.json"),
                        "bootstrap_uri": "axl://doctor",
                    }
                )
            )
            findings = await diagnose(
                config, task=task(), clock_synchronized=lambda: True
            )
            return [(f.check, f.status, f.detail) for f in findings]

    findings = asyncio.run(scenario())

    assert [(check, status) for check, status, _ in findings] == [
        ("config", Status.OK),
        ("draft", Status.OK),
        ("task", Status.OK),
        ("axl bridge", Status.OK),
        ("node key", Status.OK),
        ("peers", Status.OK),
        ("clock", Status.OK),
    ]
    assert dict((check, detail) for check, _, detail in findings)["peers"] == (
        "connected to 3 peer(s)"
    )
