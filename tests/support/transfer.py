"""Test helper: await one specific completed transfer."""

from __future__ import annotations

from dromeus.transport.transfer import ArtifactReceipt, TransferManager


async def wait_for_artifact(
    manager: TransferManager, transfer_id: str, *, timeout_seconds: float
) -> ArtifactReceipt:
    """Take the next completed artifact and require it to be ``transfer_id``."""
    receipt = await manager.next_artifact(timeout_seconds=timeout_seconds)
    assert receipt.transfer_id == transfer_id, (receipt.transfer_id, transfer_id)
    return receipt
