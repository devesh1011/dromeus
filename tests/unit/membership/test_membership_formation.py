from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from support.application_fixture import application_draft
from support.in_memory_transport import (
    InMemoryNetwork,
    InMemoryTransport,
)
from support.runtime_fakes import RecordingInMemoryTransport
from support.sample_manifest import manifest_data, write_checkpoint

from dromeus.manifests.canonical import canonical_hash, canonical_json
from dromeus.manifests.models import DraftRunSpec, SealedManifest, Tensor, TensorSchema
from dromeus.membership.formation import (
    FormationError,
    FormationProtocol,
    FormationResult,
    ReadyValidationError,
    create_invitation,
    seal_manifest,
    validate_ready,
)
from dromeus.protocol.codec import decode_envelope, encode_envelope, encode_message
from dromeus.protocol.models import JoinAccepted, MessageType, create_envelope
from dromeus.runtime import (
    NodeRuntime,
    NodeState,
)


def test_validate_ready_accepts_matching_environment() -> None:
    manifest = SealedManifest.model_validate(manifest_data())

    validate_ready(
        manifest=manifest,
        local_public_key="peer-0",
        environment=manifest.environment,
        dataset=manifest.dataset,
        checkpoint_hash=manifest.initial_checkpoint_hash,
    )


def test_environment_mismatch_prevents_ready() -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    environment = manifest.environment.model_copy(update={"axl_version": "2.0.0"})

    with pytest.raises(
        ReadyValidationError, match="environment fingerprint does not match manifest"
    ):
        validate_ready(
            manifest=manifest,
            local_public_key="peer-0",
            environment=environment,
            dataset=manifest.dataset,
            checkpoint_hash=manifest.initial_checkpoint_hash,
        )


def test_v3_sealing_assigns_deterministic_indices_for_eight_members() -> None:
    base = SealedManifest.model_validate(manifest_data())
    draft_data = base.model_dump(mode="python")
    for field in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del draft_data[field]
    draft_data.update(
        {
            "manifest_version": 3,
            "algorithm_id": "noloco",
            "model_id": "resnet18-gn",
            "optimizer": "adam",
            "training": {
                "batch_size": 128,
                "momentum": 0.0,
                "weight_decay": 0.0,
                "learning_rate_milestones": [],
                "learning_rate_gamma": 0.1,
                "crop_padding": 0,
                "normalize": True,
                "final_consensus_rounds": 0,
            },
            "algorithm_config": {
                "alpha": 0.5,
                "beta": 0.7,
                "gamma": 0.7,
                "inner_steps": 50,
                "adam": {
                    "learning_rate": 0.001,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "epsilon": 1e-8,
                    "gradient_clip_norm": 1.0,
                },
            },
            "artifact_codecs": [
                {"artifact_name": "outer_gradient", "codec_id": "identity-v1"},
                {"artifact_name": "slow_weights", "codec_id": "identity-v1"},
            ],
            "transport": {
                **draft_data["transport"],
                "chunk_size_bytes": 1_048_576,
                "window_size": 4,
            },
            "dataset": {
                **draft_data["dataset"],
                "partition_sample_counts": [6_250] * 8,
                "node_index_partitions": list(range(8)),
            },
        }
    )
    draft = DraftRunSpec.model_validate(draft_data)
    keys = {f"peer-{index}" for index in range(8)}

    invitation = create_invitation(
        draft=draft,
        initiator_public_key="peer-0",
        bootstrap_uri="axl://bootstrap",
    )
    manifest = seal_manifest(
        draft=draft,
        participant_keys=keys,
        initial_checkpoint_hash=base.initial_checkpoint_hash,
        tensor_schema=base.tensor_schema,
    )

    assert invitation.expected_participant_count == 8
    assert [participant.node_index for participant in manifest.participants] == list(
        range(8)
    )
    assert [participant.public_key for participant in manifest.participants] == sorted(
        keys
    )


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
        node = NodeRuntime(
            transport=transport,
            draft=draft,
            environment=manifest.environment,
            dataset=manifest.dataset,
            artifact_root=tmp_path / f"artifacts-{index}",
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


def test_in_memory_transport_and_eight_member_formation(tmp_path: Path) -> None:
    asyncio.run(_test_in_memory_transport_and_eight_member_formation(tmp_path))


async def _test_in_memory_transport_and_eight_member_formation(
    tmp_path: Path,
) -> None:
    base = SealedManifest.model_validate(manifest_data())
    draft_data = base.model_dump(mode="python")
    for field in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del draft_data[field]
    draft_data.update(
        {
            "manifest_version": 3,
            "algorithm_id": "noloco",
            "model_id": "resnet18-gn",
            "optimizer": "adam",
            "training": {
                "batch_size": 128,
                "momentum": 0.0,
                "weight_decay": 0.0,
                "learning_rate_milestones": [],
                "learning_rate_gamma": 0.1,
                "crop_padding": 0,
                "normalize": True,
                "final_consensus_rounds": 0,
            },
            "algorithm_config": {
                "alpha": 0.5,
                "beta": 0.7,
                "gamma": 0.7,
                "inner_steps": 50,
                "adam": {
                    "learning_rate": 0.001,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "epsilon": 1e-8,
                    "gradient_clip_norm": 1.0,
                },
            },
            "artifact_codecs": [
                {"artifact_name": "outer_gradient", "codec_id": "identity-v1"},
                {"artifact_name": "slow_weights", "codec_id": "identity-v1"},
            ],
            "transport": {
                **draft_data["transport"],
                "max_retries": 1,
                "retry_timeout_seconds": 0.05,
                "chunk_size_bytes": 1_048_576,
                "window_size": 4,
            },
            "dataset": {
                **draft_data["dataset"],
                "partition_sample_counts": [6_250] * 8,
                "node_index_partitions": list(range(8)),
            },
        }
    )
    draft = DraftRunSpec.model_validate(draft_data)
    network = InMemoryNetwork()
    transports = [
        InMemoryTransport(network=network, public_key=f"peer-{index}")
        for index in range(8)
    ]
    nodes = [
        NodeRuntime(
            transport=transport,
            draft=draft,
            environment=base.environment,
            dataset=draft.dataset,
            artifact_root=tmp_path / f"artifacts-{index}",
        )
        for index, transport in enumerate(transports)
    ]
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
                tensor_schema=base.tensor_schema,
            )
        )
    ]
    tasks.extend(
        asyncio.create_task(node.join(invitation=invitation)) for node in nodes[1:]
    )
    outcomes = await asyncio.wait_for(
        asyncio.gather(*tasks, return_exceptions=True),
        timeout=4.0,
    )
    results = [outcome for outcome in outcomes if isinstance(outcome, FormationResult)]
    assert len(results) == 8
    assert all(node.state is NodeState.READY for node in nodes)
    assert len({result.manifest_hash for result in results}) == 1
    assert all(
        result.manifest.participants == results[0].manifest.participants
        for result in results
    )
    for node in nodes:
        await node.stop()


@pytest.mark.parametrize("recompute_hash", [False, True])
def test_altered_sealed_policy_never_reaches_ready(
    tmp_path: Path, recompute_hash: bool
) -> None:
    draft = application_draft()
    schema = TensorSchema(tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),))
    sealed = seal_manifest(
        draft=draft,
        participant_keys={f"peer-{rank}" for rank in range(4)},
        initial_checkpoint_hash="a" * 64,
        tensor_schema=schema,
    )
    data = sealed.model_dump(mode="python")
    data["algorithm_config"]["beta"] = 0.9
    if recompute_hash:
        other = DraftRunSpec.model_validate(
            {name: data[name] for name in DraftRunSpec.model_fields}
        )
        data["draft_hash"] = canonical_hash(other)
    altered = SealedManifest.model_validate(data)
    network = InMemoryNetwork()
    initiator = InMemoryTransport(network=network, public_key="peer-0")
    transport = RecordingInMemoryTransport(network=network, public_key="peer-1")
    participant = FormationProtocol(
        transport=transport,
        draft=draft,
        environment=draft.environment,
        dataset=draft.dataset,
        transport_limits=draft.transport,
        artifact_root=tmp_path,
        local_tensor_schema=schema,
    )
    invitation = create_invitation(
        draft=draft, initiator_public_key="peer-0", bootstrap_uri="axl://integrity-test"
    )

    async def run() -> None:
        await participant.start()
        try:
            for message_type, payload in (
                (
                    MessageType.JOIN_ACCEPTED,
                    encode_message(JoinAccepted(draft_hash=canonical_hash(draft))),
                ),
                (MessageType.MANIFEST_SEALED, canonical_json(altered)),
            ):
                await initiator.send(
                    "peer-1",
                    encode_envelope(
                        create_envelope(
                            message_type=message_type,
                            message_id=f"altered-{message_type.value}",
                            run_id=draft.run_id,
                            manifest_hash="0" * 64,
                            sender_public_key="peer-0",
                            algorithm_id=draft.algorithm_id,
                            payload=payload,
                        )
                    ),
                )
            with pytest.raises(FormationError, match="draft"):
                await asyncio.wait_for(
                    participant.join(invitation=invitation), timeout=2
                )
        finally:
            await participant.stop()

    asyncio.run(run())
    sent = [
        decode_envelope(
            payload, authenticated_sender="peer-1", participant_keys=None
        ).message_type
        for _, payload in transport.sent_payloads
    ]
    assert MessageType.JOIN_REQUEST in sent
    assert MessageType.READY not in sent
    assert not list(tmp_path.rglob("*.safetensors"))
