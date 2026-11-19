"""`dromeus doctor` checks: each finding, its status, and the fix it names."""

from __future__ import annotations

import asyncio
import dataclasses
import subprocess
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from dromeus import diagnostics
from dromeus.application import installed_versions, load_draft, make_draft
from dromeus.diagnostics import Finding, Status, diagnose, system_clock_synchronized
from dromeus.manifests.canonical import canonical_json
from dromeus.membership.formation import create_invitation
from dromeus.training.task import TrainingTask
from dromeus.transport.interface import TransportError
from examples.custom_training import task as example_task

BOOTSTRAP = "tls://bootstrap.example:9000"
BRIDGE_URL = "http://127.0.0.1:9002"
KEY = "f" * 64
HEALTHY_TOPOLOGY: dict[str, object] = {
    "our_public_key": KEY,
    "peers": [
        {"public_key": "a" * 64, "up": True},
        {"public_key": "a" * 64, "up": True},  # AXL lists one link per direction
        {"public_key": "b" * 64, "up": False},
    ],
}
PARTICIPANT_CHECKS = [
    "config",
    "draft",
    "axl build",
    "axl bridge",
    "node key",
    "peers",
    "invitation",
    "clock",
]
NO_DRAFT_CHECKS = ["config", "draft", "axl bridge", "node key", "peers", "clock"]


def _clock_ok() -> bool | None:
    return True


@pytest.fixture(autouse=True)
def bundled_axl(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bundled AXL is the build the test drafts pin."""
    monkeypatch.setattr(diagnostics, "bundled_axl_version", lambda: "1.0.0")


def _write_draft(path: Path, *, rounds: int = 2) -> None:
    make_draft(
        example_task(),
        path,
        run_id="diag-001",
        participants=4,
        rounds=rounds,
        local_steps=2,
        axl_version="1.0.0",
    )


def _write_config(tmp_path: Path, *, role: str = "participant") -> Path:
    config_path = tmp_path / "node.yaml"
    config_path.write_text(
        "\n".join(
            (
                f"role: {role}",
                f"draft_path: {tmp_path / 'draft.yaml'}",
                f"axl_bridge_url: {BRIDGE_URL}",
                f"run_root: {tmp_path / 'run'}",
                f"invitation_path: {tmp_path / 'invitation.json'}",
                f"bootstrap_uri: {BOOTSTRAP}",
            )
        ),
        encoding="utf-8",
    )
    return config_path


def _invitation_bytes(
    tmp_path: Path,
    *,
    draft_name: str = "draft.yaml",
    expires_in: timedelta | None = timedelta(hours=1),
    bootstrap_uri: str = BOOTSTRAP,
) -> bytes:
    expires_at = None if expires_in is None else datetime.now(UTC) + expires_in
    invitation = create_invitation(
        draft=load_draft(tmp_path / draft_name),
        initiator_public_key="peer-0",
        bootstrap_uri=bootstrap_uri,
        enrollment_expires_at=expires_at,
    )
    return canonical_json(invitation)


def _ready(
    tmp_path: Path,
    *,
    role: str = "participant",
    expires_in: timedelta | None = timedelta(hours=1),
) -> Path:
    """A participant with a draft, a valid invitation and a matching config."""
    _write_draft(tmp_path / "draft.yaml")
    (tmp_path / "invitation.json").write_bytes(
        _invitation_bytes(tmp_path, expires_in=expires_in)
    )
    return _write_config(tmp_path, role=role)


def _bridge(
    monkeypatch: pytest.MonkeyPatch,
    outcome: Mapping[str, object] | Exception,
) -> None:
    """Replace the AXL client: it answers with `outcome` or raises it."""

    class FakeAXLTransport:
        def __init__(self, config: object) -> None:
            self.config = config

        async def topology(self) -> dict[str, object]:
            if isinstance(outcome, Exception):
                raise outcome
            return dict(outcome)

    monkeypatch.setattr(diagnostics, "AXLTransport", FakeAXLTransport)


def _diagnose(
    config_path: Path,
    *,
    task: TrainingTask | None = None,
    clock: Callable[[], bool | None] = _clock_ok,
) -> list[Finding]:
    return asyncio.run(diagnose(config_path, task=task, clock_synchronized=clock))


def _checks(findings: list[Finding]) -> list[str]:
    return [finding.check for finding in findings]


def _finding(findings: list[Finding], check: str) -> Finding:
    matches = [finding for finding in findings if finding.check == check]
    assert len(matches) == 1, f"expected one {check!r} finding in {_checks(findings)}"
    return matches[0]


def _problem(findings: list[Finding], check: str) -> Finding:
    """The finding for `check`: it must not be OK, and it must name a fix."""
    finding = _finding(findings, check)
    assert finding.status is not Status.OK, finding
    assert finding.fix, finding
    return finding


def test_invalid_config_reports_only_a_config_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _ready(tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "role: participant", "role: boss"
        ),
        encoding="utf-8",
    )
    _bridge(monkeypatch, HEALTHY_TOPOLOGY)

    findings = _diagnose(config_path)

    assert _checks(findings) == ["config"]
    assert findings[0].status is Status.FAIL
    assert "role" in findings[0].detail
    assert findings[0].fix


def test_healthy_participant_passes_every_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _ready(tmp_path)
    _bridge(monkeypatch, HEALTHY_TOPOLOGY)

    findings = _diagnose(config_path)

    assert _checks(findings) == PARTICIPANT_CHECKS
    assert all(finding.status is Status.OK for finding in findings)
    assert all(finding.fix is None for finding in findings)
    assert (
        "run diag-001: 4 participants, 2 rounds" in _finding(findings, "draft").detail
    )
    assert _finding(findings, "peers").detail == "connected to 1 peer(s)"


def test_initiator_skips_the_invitation_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _ready(tmp_path, role="initiator")
    _bridge(monkeypatch, HEALTHY_TOPOLOGY)

    findings = _diagnose(config_path)

    assert _checks(findings) == [
        check for check in PARTICIPANT_CHECKS if check != "invitation"
    ]
    assert all(finding.status is Status.OK for finding in findings)


def test_draft_made_from_the_given_task_passes_the_task_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _ready(tmp_path)
    _bridge(monkeypatch, HEALTHY_TOPOLOGY)

    findings = _diagnose(config_path, task=example_task())

    assert _checks(findings)[:3] == ["config", "draft", "task"]
    assert _finding(findings, "task").status is Status.OK


def test_task_with_other_settings_fails_the_task_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _ready(tmp_path)
    _bridge(monkeypatch, HEALTHY_TOPOLOGY)
    other_task = dataclasses.replace(example_task(), batch_size=4)

    findings = _diagnose(config_path, task=other_task)

    finding = _problem(findings, "task")
    assert "batch_size" in finding.detail
    assert finding.fix is not None and "make_draft" in finding.fix


def test_draft_for_another_dromeus_version_fails_the_draft_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _ready(tmp_path)
    dromeus_version, _ = installed_versions()
    draft_path = tmp_path / "draft.yaml"
    text = draft_path.read_text(encoding="utf-8")
    assert f"dromeus_version: {dromeus_version}" in text
    draft_path.write_text(
        text.replace(f"dromeus_version: {dromeus_version}", "dromeus_version: 0.0.1"),
        encoding="utf-8",
    )
    _bridge(monkeypatch, HEALTHY_TOPOLOGY)

    findings = _diagnose(config_path, task=example_task())

    assert "Dromeus 0.0.1" in _problem(findings, "draft").detail
    assert _checks(findings) == NO_DRAFT_CHECKS


def test_missing_draft_skips_the_checks_that_need_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _ready(tmp_path)
    (tmp_path / "draft.yaml").unlink()
    _bridge(monkeypatch, HEALTHY_TOPOLOGY)

    findings = _diagnose(config_path, task=example_task())

    _problem(findings, "draft")
    assert _checks(findings) == NO_DRAFT_CHECKS


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(TransportError("failed to query AXL topology"), id="unreachable"),
        pytest.param(ValueError("Expecting value"), id="not-json"),
    ],
)
def test_unreachable_bridge_fails_and_skips_node_key_and_peers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    config_path = _ready(tmp_path)
    _bridge(monkeypatch, error)

    findings = _diagnose(config_path)

    assert BRIDGE_URL in _problem(findings, "axl bridge").detail
    assert _checks(findings) == [
        "config",
        "draft",
        "axl build",
        "axl bridge",
        "invitation",
        "clock",
    ]


def test_bridge_without_a_public_key_fails_the_node_key_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _ready(tmp_path)
    _bridge(monkeypatch, {"peers": [{"public_key": "a" * 64, "up": True}]})

    findings = _diagnose(config_path)

    _problem(findings, "node key")
    assert _finding(findings, "peers").status is Status.OK


@pytest.mark.parametrize(
    "topology",
    [
        pytest.param({"our_public_key": KEY, "peers": []}, id="no-peers"),
        pytest.param(
            {"our_public_key": KEY, "peers": [{"up": False}, {"up": None}]},
            id="all-down",
        ),
        pytest.param(
            {"our_public_key": KEY, "peers": [{"up": True}]}, id="up-without-key"
        ),
        pytest.param({"our_public_key": KEY}, id="no-peer-list"),
    ],
)
def test_zero_connected_peers_fails_the_peers_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    topology: Mapping[str, object],
) -> None:
    config_path = _ready(tmp_path)
    _bridge(monkeypatch, topology)

    finding = _problem(_diagnose(config_path), "peers")

    assert finding.detail.startswith("no peer link up")


def test_missing_invitation_is_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _ready(tmp_path)
    (tmp_path / "invitation.json").unlink()
    _bridge(monkeypatch, HEALTHY_TOPOLOGY)

    findings = _diagnose(config_path)

    finding = _problem(findings, "invitation")
    assert finding.status is Status.WARN
    assert "not delivered yet" in finding.detail


def test_incomplete_invitation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _ready(tmp_path)
    payload = (tmp_path / "invitation.json").read_bytes()
    (tmp_path / "invitation.json").write_bytes(payload[: len(payload) // 2])
    _bridge(monkeypatch, HEALTHY_TOPOLOGY)

    finding = _problem(_diagnose(config_path), "invitation")

    assert "incomplete or invalid" in finding.detail


def test_expired_invitation_fails_and_shows_both_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _ready(tmp_path, expires_in=timedelta(minutes=-5))
    _bridge(monkeypatch, HEALTHY_TOPOLOGY)

    finding = _problem(_diagnose(config_path), "invitation")

    assert "expired at" in finding.detail
    assert "this machine's UTC time" in finding.detail


def test_invitation_made_from_another_draft_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _ready(tmp_path)
    _write_draft(tmp_path / "other.yaml", rounds=3)
    (tmp_path / "invitation.json").write_bytes(
        _invitation_bytes(tmp_path, draft_name="other.yaml")
    )
    _bridge(monkeypatch, HEALTHY_TOPOLOGY)

    finding = _problem(_diagnose(config_path), "invitation")

    assert "different draft" in finding.detail


def test_invitation_for_another_bootstrap_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _ready(tmp_path)
    (tmp_path / "invitation.json").write_bytes(
        _invitation_bytes(tmp_path, bootstrap_uri="tls://other.example:9000")
    )
    _bridge(monkeypatch, HEALTHY_TOPOLOGY)

    finding = _problem(_diagnose(config_path), "invitation")

    assert "other.example" in finding.detail


@pytest.mark.parametrize(
    ("answer", "status"),
    [(False, Status.FAIL), (None, Status.WARN)],
    ids=["not-synchronized", "unknown"],
)
def test_unsynchronized_or_unknown_clock_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    answer: bool | None,
    status: Status,
) -> None:
    config_path = _ready(tmp_path)
    _bridge(monkeypatch, HEALTHY_TOPOLOGY)

    findings = _diagnose(config_path, clock=lambda: answer)

    assert _problem(findings, "clock").status is status


def _no_timedatectl(name: str) -> str | None:
    return None


def _timedatectl(name: str) -> str | None:
    return "/usr/bin/timedatectl"


def test_system_clock_is_unknown_without_timedatectl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics.sys, "platform", "linux")
    monkeypatch.setattr(diagnostics.shutil, "which", _no_timedatectl)

    assert system_clock_synchronized() is None


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [("yes\n", True), ("no\n", False), ("\n", None)],
)
def test_system_clock_reads_timedatectl_on_linux(
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    expected: bool | None,
) -> None:
    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert args == ["timedatectl", "show", "-p", "NTPSynchronized", "--value"]
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(diagnostics.sys, "platform", "linux")
    monkeypatch.setattr(diagnostics.shutil, "which", _timedatectl)
    monkeypatch.setattr(diagnostics.subprocess, "run", fake_run)

    assert system_clock_synchronized() is expected


def test_system_clock_is_unknown_when_timedatectl_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timed_out(
        args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(args, 5)

    monkeypatch.setattr(diagnostics.sys, "platform", "linux")
    monkeypatch.setattr(diagnostics.shutil, "which", _timedatectl)
    monkeypatch.setattr(diagnostics.subprocess, "run", timed_out)

    assert system_clock_synchronized() is None


@pytest.mark.parametrize(
    ("bundled", "expected"),
    [
        pytest.param("2.0.0", "bundled is 2.0.0", id="other-build"),
        pytest.param(None, "no AXL is bundled here", id="none-bundled"),
    ],
)
def test_draft_pinning_another_axl_build_warns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bundled: str | None,
    expected: str,
) -> None:
    config_path = _ready(tmp_path)
    _bridge(monkeypatch, HEALTHY_TOPOLOGY)
    monkeypatch.setattr(diagnostics, "bundled_axl_version", lambda: bundled)

    findings = _diagnose(config_path)

    finding = _problem(findings, "axl build")
    assert finding.status is Status.WARN
    assert finding.detail == f"draft pins AXL 1.0.0; {expected}"
    assert "1.0.0" in (finding.fix or "")
