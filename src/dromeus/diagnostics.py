"""Check whether this machine is ready to run a node, without training.

`dromeus doctor` prints the findings from `diagnose`. Each check looks at one
thing (the node config, the draft, the AXL build and bridge, peer links, the
invitation, the clock) and records what it saw. A finding that is not OK
carries one line telling the user what to do. Nothing here joins a group or
trains a model.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import cast

import yaml
from pydantic import ValidationError

from dromeus.application import (
    bundled_axl_version,
    check_environment,
    load_draft,
    require_task_matches,
)
from dromeus.errors import ConfigError, DraftError, DromeusError, TaskError, one_line
from dromeus.manifests.canonical import canonical_hash
from dromeus.manifests.models import DraftRunSpec, Invitation
from dromeus.node import NodeConfig, NodeRole, load_node_config
from dromeus.training.task import TrainingTask
from dromeus.transport.axl import AXLBridgeConfig, AXLTransport
from dromeus.transport.interface import TransportError


class Status(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class Finding:
    """What one check observed, and what to do about it when it is not OK."""

    check: str
    status: Status
    detail: str
    fix: str | None = None

    def __post_init__(self) -> None:
        if self.status is not Status.OK and not self.fix:
            raise ValueError(f"{self.check} finding is not OK but has no fix")


CONFIG_FIX = (
    "Correct the node config fields named in the error: role, draft_path, "
    "axl_bridge_url, run_root, invitation_path, bootstrap_uri."
)
DRAFT_FIX = (
    "Point draft_path at the group's draft file, and install the Dromeus and "
    "PyTorch versions it names."
)
TASK_FIX = (
    "The draft was made from a different task definition; regenerate it with "
    "make_draft from this task, or use the group's task code."
)
BRIDGE_FIX = (
    "Start the AXL node on this machine and make sure its API port matches "
    "axl_bridge_url in the node config."
)
NODE_KEY_FIX = (
    "Restart the AXL node on this machine; the bridge did not report its public key."
)
PEERS_FIX = (
    "Check the Peers list and bootstrap address in the AXL config, and that this "
    "machine can reach them outbound."
)
INVITATION_MISSING_FIX = (
    "Copy the initiator's invitation file to invitation_path; the node waits "
    "invitation_timeout_seconds for it."
)
INVITATION_INVALID_FIX = (
    "Copy the initiator's invitation file again, completely, to invitation_path."
)
INVITATION_BOOTSTRAP_FIX = (
    "Set bootstrap_uri in the node config to the initiator's bootstrap URI, or ask "
    "for an invitation made for this one."
)
INVITATION_DRAFT_FIX = (
    "Use the same draft file as the initiator, or ask the initiator for an "
    "invitation made from yours."
)
INVITATION_EXPIRED_FIX = (
    "Ask the initiator for a fresh invitation, or check this machine's clock."
)
CLOCK_FAIL_FIX = (
    "Enable NTP on this machine, for example `timedatectl set-ntp true` or chrony."
)
CLOCK_WARN_FIX = (
    "Make sure NTP is enabled; invitation expiry compares clocks across machines."
)


def system_clock_synchronized() -> bool | None:
    """Whether systemd reports the clock as NTP-synchronized.

    Returns None on other platforms, without timedatectl, or on any error.
    """
    if sys.platform != "linux" or shutil.which("timedatectl") is None:
        return None
    try:
        result = subprocess.run(
            ["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    answer = result.stdout.strip()
    if answer == "yes":
        return True
    if answer == "no":
        return False
    return None


async def diagnose(
    config_path: Path,
    *,
    task: TrainingTask | None = None,
    clock_synchronized: Callable[[], bool | None] = system_clock_synchronized,
) -> list[Finding]:
    """Run the checks in order and return one finding per check that ran.

    A bad config stops everything. A bad draft skips the checks that need it,
    and an unreachable bridge skips the node key and peer checks.
    """
    try:
        config = load_node_config(config_path)
    except ConfigError as error:
        return [Finding("config", Status.FAIL, one_line(str(error)), CONFIG_FIX)]
    findings = [
        Finding("config", Status.OK, f"{config_path} is valid; role {config.role}")
    ]

    draft_finding, draft = _check_draft(config.draft_path)
    findings.append(draft_finding)
    if draft is not None and task is not None:
        findings.append(_check_task(draft, task))
    if draft is not None:
        findings.append(_check_axl_build(draft))

    bridge_finding, topology = await _check_bridge(config.axl_bridge_url)
    findings.append(bridge_finding)
    if topology is not None:
        findings.append(_check_node_key(topology))
        findings.append(_check_peers(topology))

    if draft is not None and config.role is NodeRole.PARTICIPANT:
        findings.append(_check_invitation(config, draft))

    findings.append(_check_clock(clock_synchronized))
    return findings


def _check_draft(draft_path: Path) -> tuple[Finding, DraftRunSpec | None]:
    try:
        draft = load_draft(draft_path)
        check_environment(draft)
    except (DromeusError, ValueError, yaml.YAMLError) as error:
        return Finding("draft", Status.FAIL, one_line(str(error)), DRAFT_FIX), None
    detail = (
        f"run {draft.run_id}: {draft.participant_count} participants, "
        f"{draft.round_count} rounds"
    )
    return Finding("draft", Status.OK, detail), draft


def _check_task(draft: DraftRunSpec, task: TrainingTask) -> Finding:
    try:
        require_task_matches(draft, task)
    except DraftError as error:
        return Finding("task", Status.FAIL, one_line(str(error)), TASK_FIX)
    except TaskError as error:
        return Finding(
            "task",
            Status.FAIL,
            one_line(str(error)),
            "Fix the task definition in the --task module so it can be described.",
        )
    return Finding("task", Status.OK, "draft matches this task")


def _check_axl_build(draft: DraftRunSpec) -> Finding:
    pinned = draft.environment.axl_version
    bundled = bundled_axl_version()
    if bundled == pinned:
        return Finding(
            "axl build", Status.OK, f"draft pins AXL {pinned}, the bundled build"
        )
    detail = (
        f"draft pins AXL {pinned}; "
        + ("no AXL is bundled here" if bundled is None else f"bundled is {bundled}")
    )
    return Finding(
        "axl build",
        Status.WARN,
        detail,
        f"make sure the AXL at axl_bridge_url is build {pinned}",
    )


async def _check_bridge(base_url: str) -> tuple[Finding, dict[str, object] | None]:
    transport = AXLTransport(AXLBridgeConfig(base_url=base_url))
    try:
        topology = await transport.topology()
    except (TransportError, OSError, ValueError) as error:
        detail = f"cannot reach the AXL bridge at {base_url} ({one_line(str(error))})"
        return Finding("axl bridge", Status.FAIL, detail, BRIDGE_FIX), None
    return Finding("axl bridge", Status.OK, f"reachable at {base_url}"), topology


def _check_node_key(topology: dict[str, object]) -> Finding:
    key = topology.get("our_public_key")
    if isinstance(key, str) and key:
        return Finding("node key", Status.OK, f"public key {key[:16]}...")
    return Finding(
        "node key",
        Status.FAIL,
        "the AXL bridge did not report this node's public key",
        NODE_KEY_FIX,
    )


def _check_peers(topology: dict[str, object]) -> Finding:
    peers = topology.get("peers")
    entries: list[object] = cast(list[object], peers) if isinstance(peers, list) else []
    # AXL may list several links to one peer (one per direction); count peers.
    up = {key for entry in entries if (key := _up_peer_key(entry)) is not None}
    if not up:
        return Finding(
            "peers", Status.FAIL, f"no peer link up ({len(entries)} listed)", PEERS_FIX
        )
    return Finding("peers", Status.OK, f"connected to {len(up)} peer(s)")


def _up_peer_key(entry: object) -> str | None:
    if not isinstance(entry, dict):
        return None
    peer = cast(dict[str, object], entry)
    key = peer.get("public_key")
    return key if peer.get("up") is True and isinstance(key, str) and key else None


def _check_invitation(config: NodeConfig, draft: DraftRunSpec) -> Finding:
    path = config.invitation_path
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return Finding(
            "invitation",
            Status.WARN,
            f"not delivered yet: {path} does not exist",
            INVITATION_MISSING_FIX,
        )
    except OSError as error:
        return Finding(
            "invitation",
            Status.FAIL,
            f"cannot read {path}: {one_line(str(error))}",
            INVITATION_INVALID_FIX,
        )
    try:
        invitation = Invitation.model_validate_json(data)
    except ValidationError as error:
        return Finding(
            "invitation",
            Status.FAIL,
            f"{path} is incomplete or invalid: {one_line(str(error))}",
            INVITATION_INVALID_FIX,
        )
    if invitation.bootstrap_uri != config.bootstrap_uri:
        return Finding(
            "invitation",
            Status.FAIL,
            f"invitation is for bootstrap {invitation.bootstrap_uri}; "
            f"node config says {config.bootstrap_uri}",
            INVITATION_BOOTSTRAP_FIX,
        )
    if invitation.draft_hash != canonical_hash(draft):
        return Finding(
            "invitation",
            Status.FAIL,
            "invitation was made from a different draft than draft_path",
            INVITATION_DRAFT_FIX,
        )
    now = datetime.now(UTC)
    expires_at = invitation.enrollment_expires_at
    if expires_at is not None and expires_at <= now:
        return Finding(
            "invitation",
            Status.FAIL,
            f"invitation expired at {expires_at.isoformat(timespec='seconds')}; "
            f"this machine's UTC time is {now.isoformat(timespec='seconds')}",
            INVITATION_EXPIRED_FIX,
        )
    return Finding("invitation", Status.OK, f"invitation for run {invitation.run_id}")


def _check_clock(clock_synchronized: Callable[[], bool | None]) -> Finding:
    synchronized = clock_synchronized()
    if synchronized is True:
        return Finding("clock", Status.OK, "system clock is synchronized")
    if synchronized is False:
        return Finding(
            "clock", Status.FAIL, "system clock is not synchronized", CLOCK_FAIL_FIX
        )
    return Finding(
        "clock",
        Status.WARN,
        "could not tell whether the clock is synchronized",
        CLOCK_WARN_FIX,
    )
