"""AXL HTTP bridge adapter."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dromeus.manifests.models import PublicKey
from dromeus.transport.base import ReceivedBytes, TransportError


def matches_yggdrasil_sender(public_key: str, bridge_sender: str) -> bool:
    try:
        sender_value = int(bridge_sender, 16)
        key_value = int(public_key, 16)
    except ValueError:
        return False
    trailing_ones = 0
    value = sender_value
    while value & 1:
        trailing_ones += 1
        value >>= 1
    known_bits = len(bridge_sender) * 4 - trailing_ones
    if known_bits < 112:
        return False
    unknown_mask = (1 << trailing_ones) - 1
    return key_value | unknown_mask == sender_value


@dataclass(frozen=True)
class AXLBridgeConfig:
    base_url: str


class AXLTransport:
    """Async wrapper over the local AXL `/send`, `/recv`, and `/topology` bridge."""

    def __init__(self, config: AXLBridgeConfig) -> None:
        self._config = config
        self._cached_public_key: PublicKey | None = None

    async def local_public_key(self) -> PublicKey:
        if self._cached_public_key is None:
            self._cached_public_key = await self._request_public_key()
        return self._cached_public_key

    async def send(self, destination: PublicKey, payload: bytes) -> None:
        await self._post_send(destination, payload)

    async def recv(self, timeout_seconds: float) -> ReceivedBytes | None:
        return await self._get_recv(timeout_seconds)

    async def _request_public_key(self) -> PublicKey:
        def load_topology() -> PublicKey:
            payload = self._load_topology()
            public_key = payload.get("our_public_key")
            if not isinstance(public_key, str):
                raise TransportError("AXL topology did not include our_public_key")
            return public_key

        return await asyncio.to_thread(load_topology)

    def _load_topology(self) -> dict[str, object]:
        request = Request(f"{self._config.base_url}/topology", method="GET")
        try:
            with urlopen(request, timeout=5.0) as response:
                body = response.read()
        except (HTTPError, URLError) as error:
            raise TransportError("failed to query AXL topology") from error
        payload = cast(object, json.loads(body))
        if not isinstance(payload, dict):
            raise TransportError("AXL topology response was not an object")
        return cast(dict[str, object], payload)

    def _resolve_sender(self, bridge_sender: str) -> PublicKey:
        topology = self._load_topology()
        candidates: set[str] = set()
        for section in ("peers", "tree"):
            entries = topology.get(section)
            if not isinstance(entries, list):
                continue
            for entry in cast(list[object], entries):
                if isinstance(entry, dict):
                    public_key = cast(dict[object, object], entry).get("public_key")
                    if isinstance(public_key, str):
                        candidates.add(public_key)
        if bridge_sender in candidates:
            return bridge_sender
        matches = [
            key for key in candidates if matches_yggdrasil_sender(key, bridge_sender)
        ]
        if len(matches) != 1:
            raise TransportError("AXL sender header did not resolve uniquely")
        return matches[0]

    async def _post_send(self, destination: PublicKey, payload: bytes) -> None:
        def send_request() -> None:
            request = Request(
                f"{self._config.base_url}/send",
                method="POST",
                data=payload,
                headers={"X-Destination-Peer-Id": destination},
            )
            try:
                with urlopen(request, timeout=5.0):
                    return
            except (HTTPError, URLError) as error:
                raise TransportError("AXL send failed") from error

        await asyncio.to_thread(send_request)

    async def _get_recv(self, timeout_seconds: float) -> ReceivedBytes | None:
        def recv_request() -> ReceivedBytes | None:
            request = Request(f"{self._config.base_url}/recv", method="GET")
            try:
                with urlopen(request, timeout=timeout_seconds) as response:
                    if response.status == 204:
                        return None
                    bridge_sender = response.headers.get("X-From-Peer-Id")
                    if bridge_sender is None:
                        raise TransportError("AXL recv response missing sender header")
                    return ReceivedBytes(
                        sender_public_key=self._resolve_sender(bridge_sender),
                        payload=response.read(),
                    )
            except HTTPError as error:
                if error.code == 204:
                    return None
                raise TransportError("AXL recv failed") from error
            except URLError as error:
                raise TransportError("AXL recv failed") from error

        return await asyncio.to_thread(recv_request)
