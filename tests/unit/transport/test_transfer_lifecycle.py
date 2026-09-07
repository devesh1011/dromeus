from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest
from support.in_memory_transport import (
    InMemoryFaults,
    InMemoryNetwork,
    InMemoryTransport,
)
from support.sample_manifest import manifest_data, write_checkpoint

from dromeus.manifests.models import SealedManifest
from dromeus.protocol.codec import encode_envelope, encode_message
from dromeus.protocol.models import (
    Chunk,
    MessageType,
    TransferBegin,
    create_envelope,
)
from dromeus.transport.outbound_scheduler import OutboundScheduler
from dromeus.transport.receiver import Receiver, ReceiverPolicy
from dromeus.transport.transfer import (
    TransferError,
    TransferManager,
)


def test_transfer_retries_duplicate_and_exhaustion(tmp_path: Path) -> None:
    asyncio.run(_test_transfer_retries_duplicate_and_exhaustion(tmp_path))


async def _test_transfer_retries_duplicate_and_exhaustion(tmp_path: Path) -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    transfer_limits = manifest.transport.model_copy(
        update={"retry_timeout_seconds": 0.05}
    )
    artifact = tmp_path / "payload.safetensors"
    write_checkpoint(artifact)
    artifact_size = artifact.stat().st_size
    network = InMemoryNetwork()
    sender_transport = InMemoryTransport(
        network=network,
        public_key="peer-0",
        faults=InMemoryFaults(drop_send_calls=frozenset({2})),
    )
    receiver_transport = InMemoryTransport(
        network=network,
        public_key="peer-1",
        faults=InMemoryFaults(
            drop_send_calls=frozenset({1}),
            duplicate_send_calls=frozenset({2}),
        ),
    )
    sender_receiver = Receiver(
        sender_transport,
        ReceiverPolicy(
            run_id=manifest.run_id,
            manifest_hash=manifest.draft_hash,
            algorithm_id=manifest.algorithm_id,
            participant_keys=frozenset({"peer-0", "peer-1"}),
            max_payload_bytes=manifest.transport.max_payload_bytes,
        ),
    )
    receiver_receiver = Receiver(
        receiver_transport,
        ReceiverPolicy(
            run_id=manifest.run_id,
            manifest_hash=manifest.draft_hash,
            algorithm_id=manifest.algorithm_id,
            participant_keys=frozenset({"peer-0", "peer-1"}),
            max_payload_bytes=manifest.transport.max_payload_bytes,
        ),
    )
    sender_scheduler = OutboundScheduler(sender_transport)
    receiver_scheduler = OutboundScheduler(receiver_transport)
    await sender_receiver.start()
    await receiver_receiver.start()
    await sender_scheduler.start()
    await receiver_scheduler.start()
    sender_manager = TransferManager(
        local_public_key="peer-0",
        run_id=manifest.run_id,
        manifest_hash=manifest.draft_hash,
        algorithm_id=manifest.algorithm_id,
        transport_limits=transfer_limits,
        receiver=sender_receiver,
        sender=sender_scheduler,
        artifact_root=tmp_path / "sender",
    )
    receiver_manager = TransferManager(
        local_public_key="peer-1",
        run_id=manifest.run_id,
        manifest_hash=manifest.draft_hash,
        algorithm_id=manifest.algorithm_id,
        transport_limits=transfer_limits,
        receiver=receiver_receiver,
        sender=receiver_scheduler,
        artifact_root=tmp_path / "receiver",
        max_inflight_bytes=artifact_size,
    )
    await sender_manager.start()
    await receiver_manager.start()
    malformed_transfer = create_envelope(
        message_type=MessageType.TRANSFER_BEGIN,
        message_id="malformed-transfer",
        run_id=manifest.run_id,
        manifest_hash=manifest.draft_hash,
        sender_public_key="peer-0",
        algorithm_id=manifest.algorithm_id,
        payload=b"not-msgpack",
    )
    malformed_ack = create_envelope(
        message_type=MessageType.CHUNK_ACK,
        message_id="malformed-ack",
        run_id=manifest.run_id,
        manifest_hash=manifest.draft_hash,
        sender_public_key="peer-1",
        algorithm_id=manifest.algorithm_id,
        payload=b"not-msgpack",
    )
    await sender_transport.send("peer-1", encode_envelope(malformed_transfer))
    await receiver_transport.send("peer-0", encode_envelope(malformed_ack))
    await asyncio.sleep(0.05)
    transfer_id = await sender_manager.send_artifact(
        destination="peer-1",
        artifact_name="payload",
        artifact_path=artifact,
        codec_id=manifest.codec_id,
        tensor_schema=manifest.tensor_schema,
    )
    receipt = await receiver_manager.wait_for_artifact(transfer_id, timeout_seconds=2.0)
    assert receipt.path.read_bytes() == artifact.read_bytes()
    assert sender_manager.last_timing is not None
    assert sender_manager.last_timing.elapsed_seconds >= 0
    assert sender_manager.last_timing.retry_count == 1
    await receiver_manager.release_receipt(receipt)

    opaque_artifact = tmp_path / "opaque.bin"
    opaque_artifact.write_bytes(b"opaque algorithm bytes")
    opaque_transfer_id = await sender_manager.send_artifact(
        destination="peer-1",
        artifact_name="opaque-update",
        artifact_path=opaque_artifact,
        codec_id="opaque-v2",
        tensor_schema=manifest.tensor_schema,
    )
    opaque_receipt = await receiver_manager.wait_for_artifact(
        opaque_transfer_id, timeout_seconds=2.0
    )
    assert opaque_receipt.codec_id == "opaque-v2"
    assert opaque_receipt.path.read_bytes() == opaque_artifact.read_bytes()
    await receiver_manager.release_receipt(opaque_receipt)

    partial_transfer_id = await sender_manager.send_artifact(
        destination="peer-1",
        artifact_name="partial-update",
        artifact_path=artifact,
        codec_id=manifest.codec_id,
        tensor_schema=manifest.tensor_schema,
        round_id=0,
    )
    partial_receipt = await receiver_manager.wait_for_artifact(
        partial_transfer_id, timeout_seconds=2.0
    )
    partial_bytes = artifact.read_bytes()
    partial_sha256 = hashlib.sha256(partial_bytes).hexdigest()
    pending_id = "pending-partial-update"
    pending_begin = TransferBegin(
        transfer_id=pending_id,
        artifact_name="pending-update",
        total_size_bytes=len(partial_bytes),
        total_sha256=partial_sha256,
        chunk_count=1,
        codec_id=manifest.codec_id,
        tensor_schema=manifest.tensor_schema,
    )
    pending_chunk = Chunk(
        transfer_id=pending_id,
        chunk_index=0,
        chunk_count=1,
        chunk_sha256=partial_sha256,
        data=partial_bytes,
    )
    for message_type, message_id, model in (
        (MessageType.TRANSFER_BEGIN, "pending-begin", pending_begin),
        (MessageType.CHUNK, "pending-chunk", pending_chunk),
    ):
        payload = encode_message(model)
        await sender_transport.send(
            "peer-1",
            encode_envelope(
                create_envelope(
                    message_type=message_type,
                    message_id=message_id,
                    run_id=manifest.run_id,
                    manifest_hash=manifest.draft_hash,
                    sender_public_key="peer-0",
                    algorithm_id=manifest.algorithm_id,
                    round_id=0,
                    correlation_id=pending_id,
                    payload=payload,
                )
            ),
        )
    await asyncio.sleep(0.05)
    assert list((tmp_path / "receiver" / ".tmp").glob("*.part"))
    await receiver_manager.discard_round_transfers(sender="peer-0", round_id=0)
    assert not partial_receipt.path.exists()
    assert not list((tmp_path / "receiver").glob("*.bin"))
    assert not list((tmp_path / "receiver" / ".tmp").glob("*.part"))
    with pytest.raises(TimeoutError):
        await receiver_manager.next_artifact(timeout_seconds=0.01)

    timeout_id = "timeout-partial-update"
    timeout_begin = pending_begin.model_copy(
        update={"transfer_id": timeout_id, "artifact_name": "timeout-update"}
    )
    timeout_chunk = pending_chunk.model_copy(update={"transfer_id": timeout_id})
    for message_type, message_id, model in (
        (MessageType.TRANSFER_BEGIN, "timeout-begin", timeout_begin),
        (MessageType.CHUNK, "timeout-chunk", timeout_chunk),
    ):
        await sender_transport.send(
            "peer-1",
            encode_envelope(
                create_envelope(
                    message_type=message_type,
                    message_id=message_id,
                    run_id=manifest.run_id,
                    manifest_hash=manifest.draft_hash,
                    sender_public_key="peer-0",
                    algorithm_id=manifest.algorithm_id,
                    round_id=1,
                    correlation_id=timeout_id,
                    payload=encode_message(model),
                )
            ),
        )
    await asyncio.sleep(0.35)
    assert not list((tmp_path / "receiver" / ".tmp").glob("*.part"))
    reusable_transfer_id = await sender_manager.send_artifact(
        destination="peer-1",
        artifact_name="capacity-reuse",
        artifact_path=artifact,
        codec_id=manifest.codec_id,
        tensor_schema=manifest.tensor_schema,
    )
    reusable_receipt = await receiver_manager.wait_for_artifact(
        reusable_transfer_id, timeout_seconds=2.0
    )
    await receiver_manager.release_receipt(reusable_receipt)

    corrupt_sender_transport = InMemoryTransport(
        network=network,
        public_key="peer-2",
        faults=InMemoryFaults(corrupt_send_calls=frozenset({2})),
    )
    corrupt_receiver_transport = InMemoryTransport(network=network, public_key="peer-3")
    corrupt_sender_receiver = Receiver(
        corrupt_sender_transport,
        ReceiverPolicy(
            run_id=manifest.run_id,
            manifest_hash=manifest.draft_hash,
            algorithm_id=manifest.algorithm_id,
            participant_keys=frozenset({"peer-2", "peer-3"}),
            max_payload_bytes=manifest.transport.max_payload_bytes,
        ),
    )
    corrupt_receiver_receiver = Receiver(
        corrupt_receiver_transport,
        ReceiverPolicy(
            run_id=manifest.run_id,
            manifest_hash=manifest.draft_hash,
            algorithm_id=manifest.algorithm_id,
            participant_keys=frozenset({"peer-2", "peer-3"}),
            max_payload_bytes=manifest.transport.max_payload_bytes,
        ),
    )
    corrupt_sender_scheduler = OutboundScheduler(corrupt_sender_transport)
    corrupt_receiver_scheduler = OutboundScheduler(corrupt_receiver_transport)
    await corrupt_sender_receiver.start()
    await corrupt_receiver_receiver.start()
    await corrupt_sender_scheduler.start()
    await corrupt_receiver_scheduler.start()
    transport_limits = manifest.transport.model_copy(
        update={"max_retries": 0, "retry_timeout_seconds": 0.05}
    )
    corrupt_sender_manager = TransferManager(
        local_public_key="peer-2",
        run_id=manifest.run_id,
        manifest_hash=manifest.draft_hash,
        algorithm_id=manifest.algorithm_id,
        transport_limits=transport_limits,
        receiver=corrupt_sender_receiver,
        sender=corrupt_sender_scheduler,
        artifact_root=tmp_path / "corrupt-sender",
    )
    corrupt_receiver_manager = TransferManager(
        local_public_key="peer-3",
        run_id=manifest.run_id,
        manifest_hash=manifest.draft_hash,
        algorithm_id=manifest.algorithm_id,
        transport_limits=transport_limits,
        receiver=corrupt_receiver_receiver,
        sender=corrupt_receiver_scheduler,
        artifact_root=tmp_path / "corrupt-receiver",
        max_inflight_bytes=manifest.transport.max_payload_bytes,
    )
    await corrupt_sender_manager.start()
    await corrupt_receiver_manager.start()
    with pytest.raises(TransferError):
        await corrupt_sender_manager.send_artifact(
            destination="peer-3",
            artifact_name="payload",
            artifact_path=artifact,
            codec_id=manifest.codec_id,
            tensor_schema=manifest.tensor_schema,
        )

    await asyncio.sleep(0.2)
    assert not list((tmp_path / "corrupt-receiver").glob("*.bin"))
    assert not list((tmp_path / "corrupt-receiver" / ".tmp").glob("*.part"))
    recovered_transfer_id = await corrupt_sender_manager.send_artifact(
        destination="peer-3",
        artifact_name="recovered",
        artifact_path=artifact,
        codec_id=manifest.codec_id,
        tensor_schema=manifest.tensor_schema,
    )
    recovered_receipt = await corrupt_receiver_manager.wait_for_artifact(
        recovered_transfer_id, timeout_seconds=2.0
    )
    await corrupt_receiver_manager.release_receipt(recovered_receipt)
    for manager in (
        sender_manager,
        receiver_manager,
        corrupt_sender_manager,
        corrupt_receiver_manager,
    ):
        await manager.stop()
    for scheduler in (
        sender_scheduler,
        receiver_scheduler,
        corrupt_sender_scheduler,
        corrupt_receiver_scheduler,
    ):
        await scheduler.stop()
    for receiver in (
        sender_receiver,
        receiver_receiver,
        corrupt_sender_receiver,
        corrupt_receiver_receiver,
    ):
        await receiver.stop()


def test_transfer_cancellation_and_shutdown_release_capacity(
    tmp_path: Path,
) -> None:
    asyncio.run(_test_transfer_cancellation_and_shutdown_release_capacity(tmp_path))


async def _test_transfer_cancellation_and_shutdown_release_capacity(
    tmp_path: Path,
) -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    limits = manifest.transport.model_copy(
        update={"max_retries": 0, "retry_timeout_seconds": 0.05}
    )
    artifact = tmp_path / "payload.bin"
    artifact.write_bytes(b"opaque payload")
    network = InMemoryNetwork()
    sender_transport = InMemoryTransport(
        network=network,
        public_key="peer-0",
        faults=InMemoryFaults(delay_send_calls={2: 0.3}),
    )
    receiver_transport = InMemoryTransport(network=network, public_key="peer-1")
    sender_receiver = Receiver(
        sender_transport,
        ReceiverPolicy(
            run_id=manifest.run_id,
            manifest_hash=manifest.draft_hash,
            algorithm_id=manifest.algorithm_id,
            participant_keys=frozenset({"peer-0", "peer-1"}),
            max_payload_bytes=limits.max_payload_bytes,
        ),
    )
    receiver_receiver = Receiver(
        receiver_transport,
        ReceiverPolicy(
            run_id=manifest.run_id,
            manifest_hash=manifest.draft_hash,
            algorithm_id=manifest.algorithm_id,
            participant_keys=frozenset({"peer-0", "peer-1"}),
            max_payload_bytes=limits.max_payload_bytes,
        ),
    )
    sender_scheduler = OutboundScheduler(sender_transport)
    receiver_scheduler = OutboundScheduler(receiver_transport)
    sender_manager = TransferManager(
        local_public_key="peer-0",
        run_id=manifest.run_id,
        manifest_hash=manifest.draft_hash,
        algorithm_id=manifest.algorithm_id,
        transport_limits=limits,
        receiver=sender_receiver,
        sender=sender_scheduler,
        artifact_root=tmp_path / "sender",
    )
    receiver_manager = TransferManager(
        local_public_key="peer-1",
        run_id=manifest.run_id,
        manifest_hash=manifest.draft_hash,
        algorithm_id=manifest.algorithm_id,
        transport_limits=limits,
        receiver=receiver_receiver,
        sender=receiver_scheduler,
        artifact_root=tmp_path / "receiver",
        max_inflight_bytes=artifact.stat().st_size,
    )
    for service in (
        sender_receiver,
        receiver_receiver,
        sender_scheduler,
        receiver_scheduler,
        sender_manager,
        receiver_manager,
    ):
        await service.start()

    canceled = asyncio.create_task(
        sender_manager.send_artifact(
            destination="peer-1",
            artifact_name="canceled",
            artifact_path=artifact,
            codec_id="opaque-v2",
            tensor_schema=manifest.tensor_schema,
        )
    )
    await asyncio.sleep(0.05)
    canceled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await canceled
    await asyncio.sleep(0.2)

    recovered_id = await sender_manager.send_artifact(
        destination="peer-1",
        artifact_name="after-cancel",
        artifact_path=artifact,
        codec_id="opaque-v2",
        tensor_schema=manifest.tensor_schema,
    )
    recovered = await receiver_manager.wait_for_artifact(
        recovered_id, timeout_seconds=2.0
    )
    await receiver_manager.release_receipt(recovered)

    payload = artifact.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    pending_id = "shutdown-partial"
    pending_begin = TransferBegin(
        transfer_id=pending_id,
        artifact_name="shutdown-partial",
        total_size_bytes=len(payload),
        total_sha256=digest,
        chunk_count=1,
        codec_id="opaque-v2",
        tensor_schema=manifest.tensor_schema,
    )
    pending_chunk = Chunk(
        transfer_id=pending_id,
        chunk_index=0,
        chunk_count=1,
        chunk_sha256=digest,
        data=payload,
    )
    for message_type, message_id, model in (
        (MessageType.TRANSFER_BEGIN, "shutdown-begin", pending_begin),
        (MessageType.CHUNK, "shutdown-chunk", pending_chunk),
    ):
        await sender_transport.send(
            "peer-1",
            encode_envelope(
                create_envelope(
                    message_type=message_type,
                    message_id=message_id,
                    run_id=manifest.run_id,
                    manifest_hash=manifest.draft_hash,
                    sender_public_key="peer-0",
                    algorithm_id=manifest.algorithm_id,
                    correlation_id=pending_id,
                    payload=encode_message(model),
                )
            ),
        )
    await asyncio.sleep(0.05)
    assert list((tmp_path / "receiver" / ".tmp").glob("*.part"))
    await receiver_manager.stop()
    assert not list((tmp_path / "receiver" / ".tmp").glob("*.part"))

    await sender_manager.stop()
    await sender_scheduler.stop()
    await receiver_scheduler.stop()
    await sender_receiver.stop()
    await receiver_receiver.stop()
