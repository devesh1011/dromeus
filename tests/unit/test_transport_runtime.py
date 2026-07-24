from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from safetensors.numpy import (
    load_file as _load_file,  # pyright: ignore[reportUnknownVariableType]
)
from support.in_memory_transport import (
    InMemoryFaults,
    InMemoryNetwork,
    InMemoryTransport,
)
from support.sample_manifest import manifest_data, write_checkpoint

from dromeus.algorithms.dpsgd import DPSGDAdapter
from dromeus.manifests.models import DraftRunSpec, SealedManifest
from dromeus.membership.protocol import (
    FormationError,
    FormationResult,
    create_invitation,
)
from dromeus.persistence.run_store import RunStore
from dromeus.runtime import NodeRuntime, NodeState, TrainingConfig
from dromeus.transport.envelope import MessageType, create_envelope, encode_envelope
from dromeus.transport.receiver import MessageChannel, Receiver, ReceiverPolicy
from dromeus.transport.sender import OutboundScheduler
from dromeus.transport.transfer import ArtifactStore, TransferError, TransferManager

_load_checkpoint = cast(Callable[[str], dict[str, np.ndarray]], _load_file)


class RuntimeTrainer:
    def __init__(self) -> None:
        self._weights = {"layer.weight": np.zeros((2, 2), dtype=np.float32)}

    def load_checkpoint(self, path: Path) -> None:
        self._weights = {
            name: np.ascontiguousarray(value)
            for name, value in _load_checkpoint(str(path)).items()
        }

    def train_local_steps(self, step_count: int) -> None:
        self._weights["layer.weight"] += np.float32(step_count)

    def weights(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self._weights.items()}

    def load_weights(self, weights: dict[str, np.ndarray]) -> None:
        self._weights = {name: value.copy() for name, value in weights.items()}

    def evaluate(self) -> tuple[float, float]:
        return 0.0, 0.5


def test_future_round_message_id_is_deduplicated_per_sender() -> None:
    asyncio.run(_test_future_round_message_id_is_deduplicated_per_sender())


def test_future_round_consensus_sketch_is_buffered_until_round_advances() -> None:
    asyncio.run(_test_future_round_consensus_sketch_is_buffered())


async def _test_future_round_consensus_sketch_is_buffered() -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    network = InMemoryNetwork()
    sender_transport = InMemoryTransport(network=network, public_key="peer-0")
    receiver_transport = InMemoryTransport(network=network, public_key="peer-1")
    current_round = 0
    receiver = Receiver(
        receiver_transport,
        ReceiverPolicy(
            run_id=manifest.run_id,
            manifest_hash=manifest.draft_hash,
            algorithm_id=manifest.algorithm_id,
            participant_keys=frozenset({"peer-0", "peer-1"}),
            current_round=lambda: current_round,
        ),
    )
    await receiver.start()
    try:
        envelope = create_envelope(
            message_type=MessageType.CONSENSUS_SKETCH,
            message_id="future-consensus",
            run_id=manifest.run_id,
            manifest_hash=manifest.draft_hash,
            sender_public_key="peer-0",
            algorithm_id=manifest.algorithm_id,
            round_id=1,
            payload=b"sketch",
        )
        await sender_transport.send("peer-1", encode_envelope(envelope))
        await asyncio.sleep(0.05)
        with pytest.raises(TimeoutError):
            await receiver.receive(MessageChannel.TELEMETRY, timeout_seconds=0.01)

        current_round = 1
        await receiver.advance_round(1)
        received = await receiver.receive(
            MessageChannel.TELEMETRY, timeout_seconds=0.1
        )
        assert received.message_id == "future-consensus"
        assert receiver.stats.rejected_messages == 0
    finally:
        await receiver.stop()


async def _test_future_round_message_id_is_deduplicated_per_sender() -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    network = InMemoryNetwork()
    sender_transport = InMemoryTransport(network=network, public_key="peer-0")
    second_sender_transport = InMemoryTransport(network=network, public_key="peer-2")
    receiver_transport = InMemoryTransport(network=network, public_key="peer-1")
    current_round = 0
    receiver = Receiver(
        receiver_transport,
        ReceiverPolicy(
            run_id=manifest.run_id,
            manifest_hash=manifest.draft_hash,
            algorithm_id=manifest.algorithm_id,
            participant_keys=frozenset({"peer-0", "peer-1", "peer-2"}),
            current_round=lambda: current_round,
        ),
    )
    await receiver.start()
    envelope = create_envelope(
        message_type=MessageType.UPDATE_READY,
        message_id="future-update",
        run_id=manifest.run_id,
        manifest_hash=manifest.draft_hash,
        sender_public_key="peer-0",
        algorithm_id=manifest.algorithm_id,
        round_id=1,
        payload=b"update",
    )
    await sender_transport.send("peer-1", encode_envelope(envelope))
    await sender_transport.send("peer-1", encode_envelope(envelope))
    await second_sender_transport.send(
        "peer-1",
        encode_envelope(
            envelope.model_copy(update={"sender_public_key": "peer-2"})
        ),
    )
    await asyncio.sleep(0.05)
    with pytest.raises(TimeoutError):
        await receiver.receive(MessageChannel.PAIR_COMMIT, timeout_seconds=0.01)

    current_round = 1
    await receiver.advance_round(1)
    received = [
        await receiver.receive(MessageChannel.PAIR_COMMIT, timeout_seconds=0.1)
        for _ in range(2)
    ]
    assert {envelope.sender_public_key for envelope in received} == {
        "peer-0",
        "peer-2",
    }
    assert all(envelope.message_id == "future-update" for envelope in received)
    with pytest.raises(TimeoutError):
        await receiver.receive(MessageChannel.PAIR_COMMIT, timeout_seconds=0.01)
    await receiver.stop()


def test_future_round_transfer_sequence_is_buffered_until_round_advances() -> None:
    asyncio.run(_test_future_round_transfer_sequence_is_buffered())


async def _test_future_round_transfer_sequence_is_buffered() -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    network = InMemoryNetwork()
    sender_transport = InMemoryTransport(network=network, public_key="peer-0")
    receiver_transport = InMemoryTransport(network=network, public_key="peer-1")
    current_round = 0
    receiver = Receiver(
        receiver_transport,
        ReceiverPolicy(
            run_id=manifest.run_id,
            manifest_hash=manifest.draft_hash,
            algorithm_id=manifest.algorithm_id,
            participant_keys=frozenset({"peer-0", "peer-1"}),
            current_round=lambda: current_round,
        ),
    )
    envelopes = [
        create_envelope(
            message_type=message_type,
            message_id=f"future-transfer-{index}",
            correlation_id="future-transfer",
            run_id=manifest.run_id,
            manifest_hash=manifest.draft_hash,
            sender_public_key="peer-0",
            algorithm_id=manifest.algorithm_id,
            round_id=1,
            payload=b"transfer",
        )
        for index, message_type in enumerate(
            (
                MessageType.TRANSFER_BEGIN,
                MessageType.CHUNK,
                MessageType.TRANSFER_COMPLETE,
            )
        )
    ]
    await receiver.start()
    try:
        for envelope in envelopes:
            await sender_transport.send("peer-1", encode_envelope(envelope))
        await asyncio.sleep(0.05)
        assert receiver.stats.rejected_messages == 0
        second_update = create_envelope(
            message_type=MessageType.TRANSFER_BEGIN,
            message_id="second-future-transfer",
            correlation_id="second-future-transfer",
            run_id=manifest.run_id,
            manifest_hash=manifest.draft_hash,
            sender_public_key="peer-0",
            algorithm_id=manifest.algorithm_id,
            round_id=1,
            payload=b"second-transfer",
        )
        await sender_transport.send("peer-1", encode_envelope(second_update))
        await asyncio.sleep(0.05)
        assert receiver.stats.rejected_messages == 1

        current_round = 1
        await receiver.advance_round(1)
        received = [
            await receiver.receive(
                MessageChannel.TRANSFER,
                timeout_seconds=0.1,
            )
            for _ in envelopes
        ]
        assert [envelope.message_id for envelope in received] == [
            envelope.message_id for envelope in envelopes
        ]
    finally:
        await receiver.stop()


def test_in_memory_transport_and_formation(tmp_path: Path) -> None:
    asyncio.run(_test_in_memory_transport_and_formation(tmp_path))


async def _test_in_memory_transport_and_formation(tmp_path: Path) -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    draft_data = manifest.model_dump(mode="python")
    for field in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del draft_data[field]
    draft_data["transport"]["max_retries"] = 1
    draft_data["transport"]["retry_timeout_seconds"] = 0.05
    draft = DraftRunSpec.model_validate(draft_data)
    network = InMemoryNetwork()
    transports: list[InMemoryTransport] = []
    nodes: list[NodeRuntime] = []
    for index in range(5):
        transport = InMemoryTransport(network=network, public_key=f"peer-{index}")
        transports.append(transport)
        store = ArtifactStore(tmp_path / f"artifacts-{index}")
        node = NodeRuntime(
            transport=transport,
            draft=draft,
            environment=manifest.environment,
            dataset=manifest.dataset,
            artifact_store=store,
        )
        nodes.append(node)
    checkpoint = tmp_path / "checkpoint.safetensors"
    write_checkpoint(checkpoint)
    invitation = create_invitation(
        draft=draft,
        initiator_public_key=await transports[0].local_public_key(),
        bootstrap_uri="axl://bootstrap",
    )
    tasks: list[asyncio.Task[FormationResult]] = [
        asyncio.create_task(
            nodes[0].initiate(
                bootstrap_uri="axl://bootstrap",
                checkpoint_path=checkpoint,
                tensor_schema=manifest.tensor_schema,
            )
        )
    ]
    tasks.extend(
        asyncio.create_task(node.join(invitation=invitation)) for node in nodes[1:]
    )
    outcomes = list(
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), timeout=2.0
        )
    )
    results = [outcome for outcome in outcomes if isinstance(outcome, FormationResult)]
    failures = [outcome for outcome in outcomes if isinstance(outcome, FormationError)]
    assert len(results) == 4
    assert len(failures) == 1
    assert sum(node.state is NodeState.READY for node in nodes) == 4
    assert sum(node.state is NodeState.FAILED for node in nodes) == 1
    hashes = {result.manifest_hash for result in results}
    assert len(hashes) == 1
    for result in results:
        assert result.checkpoint_path.read_bytes() == checkpoint.read_bytes()
    for node in nodes:
        await node.stop()
        assert node.state is NodeState.STOPPED


def test_runtime_runs_training_after_in_memory_formation(tmp_path: Path) -> None:
    asyncio.run(_test_runtime_runs_training_after_in_memory_formation(tmp_path))


async def _test_runtime_runs_training_after_in_memory_formation(
    tmp_path: Path,
) -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    draft_data = manifest.model_dump(mode="python")
    for field in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del draft_data[field]
    draft_data["round_count"] = 2
    draft_data["local_steps"] = 1
    draft_data["transport"]["max_retries"] = 1
    draft_data["transport"]["retry_timeout_seconds"] = 0.05
    draft = DraftRunSpec.model_validate(draft_data)
    network = InMemoryNetwork()
    transports = [
        InMemoryTransport(network=network, public_key=f"peer-{index}")
        for index in range(5)
    ]
    checkpoint = tmp_path / "checkpoint.safetensors"
    write_checkpoint(checkpoint)
    nodes: list[NodeRuntime] = []
    for index, transport in enumerate(transports):
        training = None
        if index < 4:
            training_trainer = RuntimeTrainer()
            training = TrainingConfig(
                algorithm=DPSGDAdapter(
                    trainer=training_trainer,
                    tensor_schema=manifest.tensor_schema,
                    local_steps=1,
                ),
                load_checkpoint=training_trainer.load_checkpoint,
                run_store=RunStore(tmp_path / f"run-{index}"),
                artifact_root=tmp_path / f"rounds-{index}",
            )
        nodes.append(
            NodeRuntime(
                transport=transport,
                draft=draft,
                environment=manifest.environment,
                dataset=manifest.dataset,
                artifact_store=ArtifactStore(tmp_path / f"artifacts-{index}"),
                training=training,
            )
        )
    invitation = create_invitation(
        draft=draft,
        initiator_public_key=await transports[0].local_public_key(),
        bootstrap_uri="axl://bootstrap",
    )
    formation_tasks: list[asyncio.Task[FormationResult]] = [
        asyncio.create_task(
            nodes[0].initiate(
                bootstrap_uri="axl://bootstrap",
                checkpoint_path=checkpoint,
                tensor_schema=manifest.tensor_schema,
            )
        )
    ]
    formation_tasks.extend(
        asyncio.create_task(node.join(invitation=invitation)) for node in nodes[1:]
    )
    outcomes = await asyncio.wait_for(
        asyncio.gather(*formation_tasks, return_exceptions=True), timeout=2.0
    )
    assert sum(isinstance(outcome, FormationResult) for outcome in outcomes) == 4
    assert sum(isinstance(outcome, FormationError) for outcome in outcomes) == 1
    assert all(
        isinstance(outcome, (FormationResult, FormationError))
        for outcome in outcomes
    )

    commits = await asyncio.wait_for(
        asyncio.gather(*(node.run() for node in nodes[:4])), timeout=5.0
    )
    assert all(len(records) == 2 for records in commits)
    assert all(node.state is NodeState.COMPLETE for node in nodes[:4])
    for index in range(4):
        state = RunStore(tmp_path / f"run-{index}").load_state()
        assert state["committed_round"] == 1
        assert len(state["metrics"]) == 2
        assert len(state["consensus"]) == 2
        assert [record["round_id"] for record in state["consensus"]] == [0, 1]
        assert all(record["sketch_count"] == 4 for record in state["consensus"])
        assert state["terminal"] == {
            "result": "complete",
            "diagnostics": {"committed_rounds": 2},
        }
    await nodes[4].stop()
    assert nodes[4].state is NodeState.STOPPED


def test_transfer_retries_duplicate_and_exhaustion(tmp_path: Path) -> None:
    asyncio.run(_test_transfer_retries_duplicate_and_exhaustion(tmp_path))


async def _test_transfer_retries_duplicate_and_exhaustion(tmp_path: Path) -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    network = InMemoryNetwork()
    sender_transport = InMemoryTransport(network=network, public_key="peer-0")
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
        transport_limits=manifest.transport,
        receiver=sender_receiver,
        sender=sender_scheduler,
        artifact_store=ArtifactStore(tmp_path / "sender"),
    )
    receiver_manager = TransferManager(
        local_public_key="peer-1",
        run_id=manifest.run_id,
        manifest_hash=manifest.draft_hash,
        algorithm_id=manifest.algorithm_id,
        transport_limits=manifest.transport,
        receiver=receiver_receiver,
        sender=receiver_scheduler,
        artifact_store=ArtifactStore(tmp_path / "receiver"),
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
    artifact = tmp_path / "payload.safetensors"
    write_checkpoint(artifact)
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
    assert sender_manager.last_timing.retry_count == 0

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
        artifact_store=ArtifactStore(tmp_path / "corrupt-sender"),
    )
    corrupt_receiver_manager = TransferManager(
        local_public_key="peer-3",
        run_id=manifest.run_id,
        manifest_hash=manifest.draft_hash,
        algorithm_id=manifest.algorithm_id,
        transport_limits=transport_limits,
        receiver=corrupt_receiver_receiver,
        sender=corrupt_receiver_scheduler,
        artifact_store=ArtifactStore(tmp_path / "corrupt-receiver"),
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
