from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError
from support.in_memory_transport import (
    InMemoryFaults,
    InMemoryNetwork,
    InMemoryTransport,
)
from support.transfer import wait_for_artifact

from dromeus.manifests.models import Tensor, TensorSchema, TransportLimits
from dromeus.protocol.models import CHUNK_FRAMING_BYTES
from dromeus.transport.envelope_sender import EnvelopeSender
from dromeus.transport.outbound_scheduler import OutboundScheduler
from dromeus.transport.receiver import Receiver, ReceiverPolicy
from dromeus.transport.transfer import (
    TransferError,
    TransferManager,
    TransferTiming,
)

SCHEMA = TensorSchema(tensors=(Tensor(name="payload", dtype="int8", shape=(1,)),))


def _limits(
    *,
    chunk_size: int = 256,
    window_size: int = 3,
    retry_timeout_seconds: float = 0.05,
) -> TransportLimits:
    return TransportLimits(
        max_payload_bytes=64 * 1024 * 1024,
        max_retries=2,
        retry_timeout_seconds=retry_timeout_seconds,
        chunk_size_bytes=chunk_size,
        window_size=window_size,
        max_message_payload_bytes=2 * 1024 * 1024,
        max_artifact_bytes=64 * 1024 * 1024,
        max_concurrent_transfers=2,
        max_inflight_bytes=8 * 1024 * 1024,
        transfer_lifetime_seconds=max(2.0, retry_timeout_seconds * 4),
        artifact_store_capacity_bytes=64 * 1024 * 1024,
    )


@dataclass(frozen=True, slots=True)
class _TransferResult:
    payload: bytes
    timing: TransferTiming


async def _transfer(
    root: Path,
    *,
    payload: bytes,
    limits: TransportLimits,
    sender_faults: InMemoryFaults | None = None,
    receiver_faults: InMemoryFaults | None = None,
) -> _TransferResult:
    network = InMemoryNetwork()
    sender_transport = InMemoryTransport(
        network=network,
        public_key="peer-0",
        faults=sender_faults,
    )
    receiver_transport = InMemoryTransport(
        network=network,
        public_key="peer-1",
        faults=receiver_faults,
    )
    sender_receiver = Receiver(
        sender_transport,
        ReceiverPolicy(
            run_id="multi-chunk",
            manifest_hash="1" * 64,
            algorithm_id="noloco",
            participant_keys=frozenset({"peer-0", "peer-1"}),
            max_payload_bytes=limits.message_payload_limit,
        ),
    )
    receiver_receiver = Receiver(
        receiver_transport,
        ReceiverPolicy(
            run_id="multi-chunk",
            manifest_hash="1" * 64,
            algorithm_id="noloco",
            participant_keys=frozenset({"peer-0", "peer-1"}),
            max_payload_bytes=limits.message_payload_limit,
        ),
    )
    sender_scheduler = OutboundScheduler(
        sender_transport,
        per_peer_in_flight=limits.effective_window_size,
    )
    receiver_scheduler = OutboundScheduler(
        receiver_transport,
        per_peer_in_flight=limits.effective_window_size,
    )
    sender_manager = TransferManager(
        sender=EnvelopeSender(
            scheduler=sender_scheduler,
            local_public_key="peer-0",
            run_id="multi-chunk",
            manifest_hash="1" * 64,
            algorithm_id="noloco",
            transport_limits=limits,
        ),
        receiver=sender_receiver,
        artifact_root=root / "sender",
    )
    receiver_manager = TransferManager(
        sender=EnvelopeSender(
            scheduler=receiver_scheduler,
            local_public_key="peer-1",
            run_id="multi-chunk",
            manifest_hash="1" * 64,
            algorithm_id="noloco",
            transport_limits=limits,
        ),
        receiver=receiver_receiver,
        artifact_root=root / "receiver",
    )
    services = (
        sender_receiver,
        receiver_receiver,
        sender_scheduler,
        receiver_scheduler,
        sender_manager,
        receiver_manager,
    )
    artifact = root / "payload.bin"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(payload)
    for service in services:
        await service.start()
    try:
        transfer_id = await sender_manager.send_artifact(
            destination="peer-1",
            artifact_name="payload",
            artifact_path=artifact,
            codec_id="identity",
            tensor_schema=SCHEMA,
            round_id=0,
        )
        receipt = await wait_for_artifact(receiver_manager, transfer_id,
            timeout_seconds=5.0,
        )
        received = receipt.path.read_bytes()
        await receiver_manager.release_receipt(receipt)
        assert sender_manager.last_timing is not None
        return _TransferResult(received, sender_manager.last_timing)
    finally:
        await sender_manager.stop()
        await receiver_manager.stop()
        await sender_scheduler.stop()
        await receiver_scheduler.stop()
        await sender_receiver.stop()
        await receiver_receiver.stop()


@pytest.mark.parametrize(
    ("sender_faults", "receiver_faults", "expects_retry"),
    [
        (InMemoryFaults(drop_send_calls=frozenset({3})), None, True),
        (InMemoryFaults(corrupt_send_calls=frozenset({3})), None, True),
        (InMemoryFaults(duplicate_send_calls=frozenset({2})), None, False),
        (InMemoryFaults(delay_send_calls={2: 0.03}), None, False),
        (None, InMemoryFaults(drop_send_calls=frozenset({2})), True),
    ],
)
def test_multichunk_window_survives_faults(
    tmp_path: Path,
    sender_faults: InMemoryFaults | None,
    receiver_faults: InMemoryFaults | None,
    expects_retry: bool,
) -> None:
    payload = bytes(range(251)) * 17
    result = asyncio.run(
        _transfer(
            tmp_path,
            payload=payload,
            limits=_limits(),
            sender_faults=sender_faults,
            receiver_faults=receiver_faults,
        )
    )

    assert result.payload == payload
    assert (result.timing.retry_count > 0) is expects_retry


def test_multichunk_transfers_resnet18_sized_artifact(tmp_path: Path) -> None:
    size_bytes = 45 * 1024 * 1024
    result = asyncio.run(
        _transfer(
            tmp_path,
            payload=b"x" * size_bytes,
            limits=_limits(
                chunk_size=1024 * 1024,
                window_size=4,
                retry_timeout_seconds=1.0,
            ),
        )
    )

    assert len(result.payload) == size_bytes
    assert result.timing.retry_count == 0


def test_multichunk_limits_reject_unbounded_window() -> None:
    with pytest.raises(ValidationError, match="window bytes"):
        TransportLimits(
            max_payload_bytes=4096,
            max_retries=1,
            retry_timeout_seconds=0.1,
            chunk_size_bytes=512,
            window_size=3,
            max_message_payload_bytes=1024,
            max_artifact_bytes=4096,
            max_concurrent_transfers=1,
            max_inflight_bytes=1024,
            transfer_lifetime_seconds=1.0,
            artifact_store_capacity_bytes=4096,
        )


def test_interrupted_multichunk_transfer_cleans_temporary_state(
    tmp_path: Path,
) -> None:
    values = _limits().model_dump(mode="python")
    values.update({"max_retries": 0, "transfer_lifetime_seconds": 0.1})
    limits = TransportLimits.model_validate(values)
    payload = b"interrupted" * 600

    with pytest.raises(TransferError, match="retries exhausted"):
        asyncio.run(
            _transfer(
                tmp_path / "failed",
                payload=payload,
                limits=limits,
                sender_faults=InMemoryFaults(
                    drop_send_calls=frozenset({3})
                ),
            )
        )
    assert not list((tmp_path / "failed" / "receiver" / ".tmp").glob("*.part"))

    recovered = asyncio.run(
        _transfer(
            tmp_path / "recovered",
            payload=payload,
            limits=limits,
        )
    )
    assert recovered.payload == payload


def _single_message_limits(*, chunk_size: int) -> TransportLimits:
    return TransportLimits(
        max_payload_bytes=64 * 1024,
        max_retries=1,
        retry_timeout_seconds=0.05,
        chunk_size_bytes=chunk_size,
        window_size=1,
        max_message_payload_bytes=4096,
        max_artifact_bytes=64 * 1024,
        max_concurrent_transfers=1,
        max_inflight_bytes=64 * 1024,
        transfer_lifetime_seconds=2.0,
        artifact_store_capacity_bytes=64 * 1024,
    )


def test_limits_reject_chunk_without_room_for_framing() -> None:
    with pytest.raises(ValidationError, match="chunk framing"):
        _single_message_limits(chunk_size=4096)


def test_largest_allowed_chunk_fits_message_payload_limit(tmp_path: Path) -> None:
    limits = _single_message_limits(chunk_size=4096 - CHUNK_FRAMING_BYTES)
    payload = bytes(range(256)) * 32

    result = asyncio.run(_transfer(tmp_path, payload=payload, limits=limits))

    assert result.payload == payload


def test_lost_transfer_completion_is_recovered(tmp_path: Path) -> None:
    """Dropping the final control message must not lose a fully sent artifact."""
    payload = bytes(range(200))
    # Sender calls: 1 begin, 2 the only chunk, 3 completion.
    result = asyncio.run(
        _transfer(
            tmp_path,
            payload=payload,
            limits=_limits(),
            sender_faults=InMemoryFaults(drop_send_calls=frozenset({3})),
        )
    )

    assert result.payload == payload


def test_lost_final_chunk_ack_is_resent_after_completion(tmp_path: Path) -> None:
    """A retried final chunk for a finished transfer is acknowledged again."""
    payload = bytes(range(200))
    # Receiver call 1 is the ack of the only chunk, sent after finalizing.
    result = asyncio.run(
        _transfer(
            tmp_path,
            payload=payload,
            limits=_limits(),
            receiver_faults=InMemoryFaults(drop_send_calls=frozenset({1})),
        )
    )

    assert result.payload == payload
    assert result.timing.retry_count > 0
