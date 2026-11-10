from __future__ import annotations

import asyncio
import hashlib
import math
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest
from support.application_fixture import application_draft
from support.in_memory_transport import InMemoryNetwork, InMemoryTransport
from support.transfer import wait_for_artifact

from dromeus.manifests.canonical import canonical_hash
from dromeus.manifests.models import (
    SealedManifest,
    Tensor,
    TensorSchema,
    TransportLimits,
)
from dromeus.membership.formation import FormationProtocol, seal_manifest
from dromeus.protocol.codec import encode_envelope, encode_message
from dromeus.protocol.models import (
    Chunk,
    MessageType,
    TransferBegin,
    TransferComplete,
    create_envelope,
)
from dromeus.transport.outbound_scheduler import OutboundScheduler
from dromeus.transport.receiver import Receiver, ReceiverPolicy
from dromeus.transport.transfer import TransferManager

MIB = 1024 * 1024
LARGE_BYTES = 129 * MIB
SCHEMA = TensorSchema(tensors=(Tensor(name="payload", dtype="int8", shape=(1,)),))


@dataclass
class _Events:
    rejected: list[Mapping[str, object]] = field(default_factory=lambda: [])

    def append(self, record: Mapping[str, object]) -> None:
        if record.get("event") == "transfer_message_rejected":
            self.rejected.append(record)


def _limits(
    *,
    artifact_bytes: int = LARGE_BYTES,
    store_bytes: int = 2 * LARGE_BYTES,
    concurrency: int = 2,
) -> TransportLimits:
    return TransportLimits(
        max_payload_bytes=max(artifact_bytes, store_bytes),
        max_retries=1,
        retry_timeout_seconds=0.1,
        chunk_size_bytes=MIB,
        window_size=2,
        max_message_payload_bytes=2 * MIB,
        max_artifact_bytes=artifact_bytes,
        max_concurrent_transfers=concurrency,
        # Chunk-window storage is deliberately much smaller than one artifact.
        max_inflight_bytes=2 * MIB,
        transfer_lifetime_seconds=10.0,
        artifact_store_capacity_bytes=store_bytes,
    )


@dataclass
class _Harness:
    manager: TransferManager
    peer: InMemoryTransport
    manifest: SealedManifest
    root: Path
    events: _Events

    def temporary(self, transfer_id: str) -> Path:
        return self.root / ".tmp" / f"{transfer_id}.part"

    async def send(
        self,
        message_type: MessageType,
        model: TransferBegin | Chunk | TransferComplete,
    ) -> None:
        await self.peer.send(
            "peer-1",
            encode_envelope(
                create_envelope(
                    message_type=message_type,
                    message_id=f"{message_type.value}-{model.transfer_id}",
                    run_id=self.manifest.run_id,
                    manifest_hash=canonical_hash(self.manifest),
                    sender_public_key="peer-0",
                    algorithm_id=self.manifest.algorithm_id,
                    correlation_id=model.transfer_id,
                    payload=encode_message(model),
                )
            ),
        )

    async def begin(
        self, transfer_id: str, size: int, *, digest: str = "0" * 64
    ) -> None:
        await self.send(
            MessageType.TRANSFER_BEGIN,
            TransferBegin(
                transfer_id=transfer_id,
                artifact_name="payload",
                total_size_bytes=size,
                total_sha256=digest,
                chunk_count=math.ceil(
                    size / self.manifest.transport.effective_chunk_size
                ),
                codec_id="identity",
                tensor_schema=SCHEMA,
            ),
        )

    async def wait_for(self, predicate: Callable[[], bool]) -> None:
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(0.001)


@asynccontextmanager
async def _harness(
    root: Path,
    limits: TransportLimits,
    *,
    formation: bool = True,
    explicit_incoming_bytes: int | None = None,
) -> AsyncGenerator[_Harness]:
    draft = application_draft().model_copy(update={"transport": limits})
    manifest = seal_manifest(
        draft=draft,
        participant_keys={f"peer-{index}" for index in range(4)},
        initial_checkpoint_hash="1" * 64,
        tensor_schema=SCHEMA,
    )
    network = InMemoryNetwork()
    peer = InMemoryTransport(network=network, public_key="peer-0")
    transport = InMemoryTransport(network=network, public_key="peer-1")
    events = _Events()
    if formation:
        protocol = FormationProtocol(
            transport=transport,
            draft=draft,
            environment=draft.environment,
            dataset=draft.dataset,
            transport_limits=limits,
            artifact_root=root,
            event_sink=events,
        )
        await protocol.start()
        # Reach the production formation construction, not a duplicated test factory.
        create_manager = cast(
            Callable[[str, SealedManifest, str], Awaitable[TransferManager]],
            getattr(protocol, "_create_transfer_manager"),
        )
        manager = await create_manager("peer-1", manifest, canonical_hash(manifest))
        try:
            yield _Harness(manager, peer, manifest, root, events)
        finally:
            await protocol.stop()
        return
    receiver = Receiver(
        transport,
        ReceiverPolicy(
            run_id=draft.run_id,
            algorithm_id=draft.algorithm_id,
            manifest_hash=canonical_hash(manifest),
            participant_keys=frozenset({f"peer-{index}" for index in range(4)}),
            max_payload_bytes=limits.message_payload_limit,
        ),
    )
    sender = OutboundScheduler(transport)
    arguments = (
        {}
        if explicit_incoming_bytes is None
        else {"max_inflight_bytes": explicit_incoming_bytes}
    )
    manager = TransferManager(
        local_public_key="peer-1",
        run_id=draft.run_id,
        manifest_hash=canonical_hash(manifest),
        algorithm_id=draft.algorithm_id,
        transport_limits=limits,
        receiver=receiver,
        sender=sender,
        artifact_root=root,
        event_sink=events,
        **arguments,
    )
    await receiver.start()
    await sender.start()
    await manager.start()
    try:
        yield _Harness(manager, peer, manifest, root, events)
    finally:
        await manager.stop()
        await sender.stop()
        await receiver.stop()


def test_formation_accepts_declared_large_artifact_without_allocating_payload(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        async with _harness(tmp_path, _limits()) as harness:
            await harness.begin("large-initial-checkpoint", LARGE_BYTES)
            path = harness.temporary("large-initial-checkpoint")
            await harness.wait_for(path.exists)
            assert path.stat().st_size == 0
            assert not harness.events.rejected
        assert not list((tmp_path / ".tmp").glob("*.part"))

    asyncio.run(scenario())


@pytest.mark.parametrize("explicit_incoming_bytes", [None, 128 * MIB])
def test_direct_manager_default_and_explicit_incoming_caps_are_unchanged(
    tmp_path: Path, explicit_incoming_bytes: int | None
) -> None:
    async def scenario() -> None:
        async with _harness(
            tmp_path,
            _limits(),
            formation=False,
            explicit_incoming_bytes=explicit_incoming_bytes,
        ) as harness:
            await harness.begin("large-checkpoint", LARGE_BYTES)
            await harness.wait_for(lambda: bool(harness.events.rejected))
            assert harness.events.rejected[-1]["error"] == (
                "local incoming reservation limit exceeded"
            )
            assert not harness.temporary("large-checkpoint").exists()

    asyncio.run(scenario())


def test_formation_still_limits_concurrent_incoming_artifacts(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with _harness(tmp_path, _limits(concurrency=1)) as harness:
            await harness.begin("first", LARGE_BYTES)
            await harness.wait_for(harness.temporary("first").exists)
            await harness.begin("second", 1)
            await harness.wait_for(lambda: bool(harness.events.rejected))
            assert harness.events.rejected[-1]["error"] == (
                "concurrent transfer limit exceeded"
            )
            assert not harness.temporary("second").exists()

    asyncio.run(scenario())


def test_formation_still_limits_each_artifact(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with _harness(tmp_path, _limits(artifact_bytes=128 * MIB)) as harness:
            await harness.begin("oversized", LARGE_BYTES)
            await harness.wait_for(lambda: bool(harness.events.rejected))
            assert harness.events.rejected[-1]["error"] == (
                "transfer exceeds manifest artifact limit"
            )
            assert not harness.temporary("oversized").exists()

    asyncio.run(scenario())


def test_completed_artifacts_still_consume_sealed_store_capacity(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        async with _harness(tmp_path, _limits(store_bytes=LARGE_BYTES)) as harness:
            payload = b"x"
            digest = hashlib.sha256(payload).hexdigest()
            await harness.begin("retained", len(payload), digest=digest)
            await harness.send(
                MessageType.CHUNK,
                Chunk(
                    transfer_id="retained",
                    chunk_index=0,
                    chunk_count=1,
                    chunk_sha256=digest,
                    data=payload,
                ),
            )
            await harness.send(
                MessageType.TRANSFER_COMPLETE,
                TransferComplete(transfer_id="retained", total_sha256=digest),
            )
            receipt = await wait_for_artifact(
                harness.manager, "retained", timeout_seconds=2
            )
            assert receipt.path.read_bytes() == payload
            await harness.begin("would-overfill-store", LARGE_BYTES)
            await harness.wait_for(lambda: bool(harness.events.rejected))
            assert harness.events.rejected[-1]["error"] == (
                "artifact store capacity exceeded"
            )
            assert not harness.temporary("would-overfill-store").exists()
            await harness.manager.release_receipt(receipt)

    asyncio.run(scenario())


def test_small_manifest_store_still_bounds_aggregate_incoming_size(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        async with _harness(
            tmp_path, _limits(artifact_bytes=8 * MIB, store_bytes=8 * MIB)
        ) as harness:
            await harness.begin("small-manifest-first", 8 * MIB)
            await harness.wait_for(harness.temporary("small-manifest-first").exists)
            await harness.begin("small-manifest-extra", 1)
            await harness.wait_for(lambda: bool(harness.events.rejected))
            assert harness.events.rejected[-1]["error"] == (
                "local incoming reservation limit exceeded"
            )
            assert not harness.temporary("small-manifest-extra").exists()

    asyncio.run(scenario())
