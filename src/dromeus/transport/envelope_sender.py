"""One node's run identity, stamped onto every outbound envelope."""

from __future__ import annotations

from dataclasses import dataclass, replace

from dromeus.manifests.models import (
    AlgorithmId,
    MessageId,
    PublicKey,
    RoundId,
    RunId,
    Sha256,
    TransportLimits,
)
from dromeus.protocol.codec import encode_envelope
from dromeus.protocol.models import MessageType, create_envelope
from dromeus.transport.outbound_scheduler import (
    OutboundScheduler,
    Priority,
    SendTiming,
)

UNSEALED_MANIFEST_HASH: Sha256 = "0" * 64


@dataclass(frozen=True)
class EnvelopeSender:
    """Send payloads as this node, in this run, under the manifest retry policy."""

    scheduler: OutboundScheduler
    local_public_key: PublicKey
    run_id: RunId
    manifest_hash: Sha256
    algorithm_id: AlgorithmId
    transport_limits: TransportLimits

    def for_manifest(self, manifest_hash: Sha256) -> EnvelopeSender:
        """Return the same sender bound to a sealed manifest."""
        return replace(self, manifest_hash=manifest_hash)

    async def send(
        self,
        destination: PublicKey,
        *,
        message_type: MessageType,
        message_id: MessageId,
        payload: bytes,
        priority: Priority,
        round_id: RoundId | None = None,
        correlation_id: MessageId | None = None,
    ) -> SendTiming:
        envelope = create_envelope(
            message_type=message_type,
            message_id=message_id,
            run_id=self.run_id,
            manifest_hash=self.manifest_hash,
            sender_public_key=self.local_public_key,
            algorithm_id=self.algorithm_id,
            round_id=round_id,
            correlation_id=correlation_id,
            payload=payload,
        )
        return await self.scheduler.send(
            destination,
            encode_envelope(envelope),
            priority=priority,
            retries=self.transport_limits.max_retries,
            retry_delay_seconds=self.transport_limits.retry_timeout_seconds,
        )


__all__ = ["UNSEALED_MANIFEST_HASH", "EnvelopeSender"]
