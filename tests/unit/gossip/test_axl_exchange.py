from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from support.gossip_fakes import DelayedPairReceiver, StubPairReceiver, StubPairSender
from support.in_memory_transport import (
    InMemoryFaults,
    InMemoryNetwork,
    InMemoryTransport,
)

from dromeus.algorithms.base import UpdateBundle
from dromeus.algorithms.codec import (
    NamedSafetensorsUpdateBundleCodec,
)
from dromeus.gossip.axl import AXLPairTransport
from dromeus.gossip.interfaces import (
    PairCommitError,
)
from dromeus.manifests.models import (
    Tensor,
    TensorSchema,
    TransportLimits,
)
from dromeus.protocol.codec import encode_message
from dromeus.protocol.models import (
    MessageType,
    PairCommitMessage,
    create_envelope,
)
from dromeus.transport.outbound_scheduler import OutboundScheduler
from dromeus.transport.receiver import Receiver
from dromeus.transport.transfer import TransferManager


def test_round_committed_rejects_payload_round_mismatch(tmp_path: Path) -> None:
    async def run() -> None:
        payload = encode_message(
            PairCommitMessage(round_id=1, checksum="1" * 64)
        )
        envelope = create_envelope(
            message_type=MessageType.ROUND_COMMITTED,
            message_id="round-committed-0",
            run_id="test-run",
            manifest_hash="0" * 64,
            sender_public_key="peer-1",
            algorithm_id="d-psgd",
            round_id=0,
            correlation_id="pair-round-0",
            payload=payload,
        )
        transport = AXLPairTransport(
            local_public_key="peer-0",
            run_id="test-run",
            manifest_hash="0" * 64,
            algorithm_id="d-psgd",
            transport_limits=TransportLimits(
                max_payload_bytes=1024,
                max_retries=0,
                retry_timeout_seconds=0.1,
            ),
            receiver=cast(Receiver, StubPairReceiver(envelope)),
            sender=cast(OutboundScheduler, StubPairSender()),
            transfer_manager=cast(TransferManager, object()),
            metadata_root=tmp_path,
        )

        with pytest.raises(PairCommitError, match="ROUND_COMMITTED round"):
            await transport.exchange_round_committed(
                peer="peer-1",
                round_id=0,
                state_checksum="2" * 64,
            )

    asyncio.run(run())


def test_update_ready_uses_transfer_lifetime_for_round_entry_skew(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        payload = encode_message(
            PairCommitMessage(round_id=0, checksum="1" * 64)
        )
        envelope = create_envelope(
            message_type=MessageType.UPDATE_READY,
            message_id="update-ready-0",
            run_id="test-run",
            manifest_hash="0" * 64,
            sender_public_key="peer-1",
            algorithm_id="d-psgd",
            round_id=0,
            correlation_id="pair-round-0",
            payload=payload,
        )
        receiver = DelayedPairReceiver(envelope, timeout_count=2)
        transport = AXLPairTransport(
            local_public_key="peer-0",
            run_id="test-run",
            manifest_hash="0" * 64,
            algorithm_id="d-psgd",
            transport_limits=TransportLimits(
                max_payload_bytes=1024,
                max_retries=1,
                retry_timeout_seconds=0.01,
                transfer_lifetime_seconds=0.05,
            ),
            receiver=cast(Receiver, receiver),
            sender=cast(OutboundScheduler, StubPairSender()),
            transfer_manager=cast(TransferManager, object()),
            metadata_root=tmp_path,
        )

        checksum = await transport.exchange_update_ready(
            peer="peer-1",
            round_id=0,
            bundle_checksum="2" * 64,
        )

        assert checksum == "1" * 64
        assert receiver.receive_count == 3

    asyncio.run(run())


def test_round_committed_recovers_from_silent_message_loss(tmp_path: Path) -> None:
    async def run() -> None:
        network = InMemoryNetwork()
        raw_transports = (
            InMemoryTransport(
                network=network,
                public_key="peer-0",
                faults=InMemoryFaults(drop_send_calls=frozenset({1})),
            ),
            InMemoryTransport(network=network, public_key="peer-1"),
        )
        receivers = tuple(Receiver(raw) for raw in raw_transports)
        senders = tuple(OutboundScheduler(raw) for raw in raw_transports)
        limits = TransportLimits(
            max_payload_bytes=1024,
            max_retries=1,
            retry_timeout_seconds=0.05,
        )
        transports = tuple(
            AXLPairTransport(
                local_public_key=f"peer-{index}",
                run_id="test-run",
                manifest_hash="0" * 64,
                algorithm_id="d-psgd",
                transport_limits=limits,
                receiver=receivers[index],
                sender=senders[index],
                transfer_manager=cast(TransferManager, object()),
                metadata_root=tmp_path / f"peer-{index}",
            )
            for index in range(2)
        )
        for receiver in receivers:
            await receiver.start()
        for sender in senders:
            await sender.start()
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    transports[0].exchange_round_committed(
                        peer="peer-1",
                        round_id=0,
                        state_checksum="1" * 64,
                    ),
                    transports[1].exchange_round_committed(
                        peer="peer-0",
                        round_id=0,
                        state_checksum="1" * 64,
                    ),
                ),
                timeout=1.0,
            )
            await asyncio.sleep(0.05)
            assert all(receiver.stats.rejected_messages == 0 for receiver in receivers)
        finally:
            await asyncio.gather(*(receiver.stop() for receiver in receivers))
            await asyncio.gather(*(sender.stop() for sender in senders))

    asyncio.run(run())


def test_transport_cancellation_waits_for_active_bundle_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        started = threading.Event()
        release = threading.Event()
        schema = TensorSchema(
            tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),)
        )
        codec = NamedSafetensorsUpdateBundleCodec(
            artifact_root=tmp_path / "local",
            run_id="test-run",
            manifest_hash="0" * 64,
            sender_public_key="peer-0",
            algorithm_id="d-psgd",
            artifact_schemas={"trained_weights": schema},
        )
        bundle = codec.encode(
            round_id=0,
            artifacts={
                "trained_weights": {
                    "weight": np.array([2.0], dtype=np.float32)
                }
            },
        )
        original_validate = UpdateBundle.validate_materialized

        def blocking_validate(
            value: UpdateBundle, max_bundle_bytes: int
        ) -> None:
            started.set()
            if not release.wait(timeout=1):
                raise RuntimeError("test did not release bundle validation")
            original_validate(value, max_bundle_bytes)

        monkeypatch.setattr(UpdateBundle, "validate_materialized", blocking_validate)
        transport = AXLPairTransport(
            local_public_key="peer-0",
            run_id="test-run",
            manifest_hash="0" * 64,
            algorithm_id="d-psgd",
            transport_limits=TransportLimits(
                max_payload_bytes=1024,
                max_retries=0,
                retry_timeout_seconds=0.1,
            ),
            receiver=cast(Receiver, object()),
            sender=cast(OutboundScheduler, object()),
            transfer_manager=cast(TransferManager, object()),
            metadata_root=tmp_path / "metadata",
        )

        task = asyncio.create_task(
            transport.exchange_update(
                peer="peer-1",
                round_id=0,
                bundle=bundle,
            )
        )
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        try:
            await asyncio.sleep(0.01)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
