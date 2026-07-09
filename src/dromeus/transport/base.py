"""Transport interface shared by production and test adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from dromeus.manifests.models import PublicKey


class TransportError(RuntimeError):
    """The underlying byte transport failed."""


@dataclass(frozen=True)
class ReceivedBytes:
    """One authenticated inbound byte payload from the transport."""

    sender_public_key: PublicKey
    payload: bytes


class AsyncTransport(Protocol):
    """Small async seam used by the receiver and sender."""

    async def local_public_key(self) -> PublicKey: ...

    async def send(self, destination: PublicKey, payload: bytes) -> None: ...

    async def recv(self, timeout_seconds: float) -> ReceivedBytes | None: ...
