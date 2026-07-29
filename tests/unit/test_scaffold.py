import asyncio
import json

import pytest
from support.in_memory_transport import InMemoryNetwork, InMemoryTransport
from support.sample_manifest import manifest_data

import dromeus
from dromeus.manifests.models import SealedManifest
from dromeus.protocol.codec import encode_envelope
from dromeus.protocol.models import MessageType, create_envelope
from dromeus.telemetry.events import emit_event
from dromeus.transport.receiver import MessageChannel, Receiver, ReceiverPolicy


def test_import_and_structured_event(capsys: pytest.CaptureFixture[str]) -> None:
    assert dromeus.__version__ == "0.1.0"
    emit_event("test", run_id="run-1", round_id=2)

    output = capsys.readouterr().out
    assert json.loads(output)["run_id"] == "run-1"


def test_receiver_logs_correlation_ids(
    capsys: pytest.CaptureFixture[str],
) -> None:
    asyncio.run(_test_receiver_logs_correlation_ids(capsys))


async def _test_receiver_logs_correlation_ids(
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    network = InMemoryNetwork()
    sender = InMemoryTransport(network=network, public_key="peer-0")
    transport = InMemoryTransport(network=network, public_key="peer-1")
    receiver = Receiver(
        transport,
        ReceiverPolicy(
            run_id=manifest.run_id,
            manifest_hash=manifest.draft_hash,
            algorithm_id=manifest.algorithm_id,
            participant_keys=frozenset({"peer-0", "peer-1"}),
        ),
    )
    await receiver.start()
    envelope = create_envelope(
        message_type=MessageType.READY,
        message_id="message-1",
        run_id=manifest.run_id,
        manifest_hash=manifest.draft_hash,
        sender_public_key="peer-0",
        algorithm_id=manifest.algorithm_id,
        round_id=2,
        correlation_id="transfer-1",
        payload=b"ready",
    )
    await sender.send("peer-1", encode_envelope(envelope))
    await receiver.receive(MessageChannel.CONTROL, timeout_seconds=0.1)
    await receiver.stop()

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    received = next(
        record for record in records if record["event"] == "message_received"
    )
    expected = {
        "run_id": manifest.run_id,
        "message_id": "message-1",
        "transfer_id": "transfer-1",
        "peer_id": "peer-0",
        "round_id": 2,
    }
    assert {key: received[key] for key in expected} == expected
