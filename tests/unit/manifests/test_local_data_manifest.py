from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError
from support.in_memory_transport import InMemoryNetwork, InMemoryTransport
from support.sample_manifest import manifest_data, write_checkpoint

from dromeus.manifests.canonical import (
    canonical_hash,
    canonical_json,
    parse_draft_yaml,
    parse_sealed_json,
)
from dromeus.manifests.models import (
    ApplicationTaskContract,
    DraftRunSpec,
    SealedManifest,
    Tensor,
    TensorSchema,
)
from dromeus.membership.formation import (
    FormationError,
    FormationProtocol,
    FormationResult,
    ReadyValidationError,
    create_invitation,
    seal_manifest,
    validate_ready,
)
from dromeus.protocol.codec import decode_envelope
from dromeus.protocol.models import MessageType

SEALED_FIELDS = (
    "draft_hash",
    "participants",
    "initial_checkpoint_hash",
    "tensor_schema",
)


def _draft(participant_count: int = 4) -> DraftRunSpec:
    data = manifest_data()
    for field in SEALED_FIELDS:
        del data[field]
    data["expected_participant_count"] = participant_count
    return DraftRunSpec.model_validate(data)


def _manifest(participant_count: int = 4) -> SealedManifest:
    base = SealedManifest.model_validate(manifest_data())
    return seal_manifest(
        draft=_draft(participant_count),
        participant_keys={f"peer-{index}" for index in range(participant_count)},
        initial_checkpoint_hash=base.initial_checkpoint_hash,
        tensor_schema=base.tensor_schema,
    )


@pytest.mark.parametrize("participant_count", [4, 8, 16])
def test_task_and_explicit_membership_round_trip(participant_count: int) -> None:
    draft = _draft(participant_count)
    manifest = _manifest(participant_count)
    invitation = create_invitation(
        draft=draft,
        initiator_public_key="peer-0",
        bootstrap_uri="axl://bootstrap",
    )

    assert parse_draft_yaml(canonical_json(draft)) == draft
    assert parse_sealed_json(canonical_json(manifest)) == manifest
    assert manifest.manifest_version == 7
    assert manifest.protocol_version == 1
    assert draft.participant_count == manifest.participant_count == participant_count
    assert invitation.expected_participant_count == participant_count
    assert manifest.expected_participant_count == participant_count
    assert manifest.draft_hash == canonical_hash(draft)
    assert [member.public_key for member in manifest.participants] == sorted(
        f"peer-{index}" for index in range(participant_count)
    )
    assert [member.node_index for member in manifest.participants] == list(
        range(participant_count)
    )
    assert isinstance(draft.dataset, ApplicationTaskContract)
    assert draft.dataset.objective == "equal-node"


def test_canonical_identity_remains_frozen() -> None:
    """Catch added defaults or encoding changes before extending M3 manifests."""
    assert canonical_hash(_draft()) == (
        "bf38d7a90cf9d2e8c63c30b5669da6206293a50786b54a83b2a7a8a2c5d3b0b2"
    )
    assert canonical_hash(_manifest()) == (
        "12ddaac3a7df7b6c98090695eb4ec3555e399b4c78db1eeb1ae4265e74a005e2"
    )


@pytest.mark.parametrize("participant_count", [1, 2, 3, 5, 7, 15, 17, 32])
def test_invalid_fixed_group_sizes_are_rejected(participant_count: int) -> None:
    with pytest.raises(ValidationError):
        _draft(participant_count)


def test_expected_participant_count_is_required_and_canonical() -> None:
    data = manifest_data()
    for field in SEALED_FIELDS:
        del data[field]
    del data["expected_participant_count"]
    with pytest.raises(ValidationError, match="expected_participant_count"):
        DraftRunSpec.model_validate(data)

    manifest = SealedManifest.model_validate(manifest_data())
    assert b'"expected_participant_count":4' in canonical_json(manifest)


@pytest.mark.parametrize("explicit_none", [False, True])
def test_container_identity_is_optional_for_host_execution(explicit_none: bool) -> None:
    data = _draft().model_dump(mode="python")
    if explicit_none:
        data["environment"]["container_image_digest"] = None
    else:
        del data["environment"]["container_image_digest"]
    draft = DraftRunSpec.model_validate(data)
    base = _manifest()
    sealed = seal_manifest(
        draft=draft,
        participant_keys={member.public_key for member in base.participants},
        initial_checkpoint_hash=base.initial_checkpoint_hash,
        tensor_schema=base.tensor_schema,
    )

    assert draft.environment.container_image_digest is None
    assert b"container_image_digest" not in canonical_json(draft)
    assert parse_sealed_json(canonical_json(sealed)) == sealed


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_id", "cifar10"),
        ("dataset_id", "application-v1"),
        ("objective", "sample-weighted"),
        ("definition_hash", "not-a-hash"),
        ("task_id", ""),
        ("sample_count", 10),
        ("dataset_path", "/private/node-0/data"),
        ("fingerprint", "b" * 64),
    ],
)
def test_application_task_rejects_invalid_or_private_contract_fields(
    field: str, value: object
) -> None:
    data = dict(manifest_data()["dataset"])
    data[field] = value
    with pytest.raises(ValidationError):
        ApplicationTaskContract.model_validate(data)


@pytest.mark.parametrize(
    "change",
    [
        {"task_id": "other-task"},
        {"definition_hash": "b" * 64},
    ],
)
def test_incompatible_task_meaning_changes_identity_and_blocks_ready(
    change: dict[str, object],
) -> None:
    manifest = _manifest()
    incompatible = ApplicationTaskContract.model_validate(
        {**manifest.dataset.model_dump(mode="python"), **change}
    )

    assert canonical_hash(incompatible) != canonical_hash(manifest.dataset)
    with pytest.raises(ReadyValidationError, match="dataset contract"):
        validate_ready(
            manifest=manifest,
            local_public_key="peer-0",
            environment=manifest.environment,
            dataset=incompatible,
            checkpoint_hash=manifest.initial_checkpoint_hash,
            local_tensor_schema=manifest.tensor_schema,
        )


def test_model_schema_mismatch_blocks_ready() -> None:
    manifest = _manifest()
    local_schema = TensorSchema(
        tensors=(Tensor(name="layer.weight", dtype="float32", shape=(3, 2)),)
    )

    with pytest.raises(ReadyValidationError, match="local model tensor schema"):
        validate_ready(
            manifest=manifest,
            local_public_key="peer-0",
            environment=manifest.environment,
            dataset=manifest.dataset,
            checkpoint_hash=manifest.initial_checkpoint_hash,
            local_tensor_schema=local_schema,
        )


def test_sealing_rejects_missing_participant() -> None:
    base = _manifest()
    with pytest.raises(FormationError, match="declared membership"):
        seal_manifest(
            draft=_draft(8),
            participant_keys={f"peer-{index}" for index in range(4)},
            initial_checkpoint_hash=base.initial_checkpoint_hash,
            tensor_schema=base.tensor_schema,
        )


class _RecordingTransport(InMemoryTransport):
    def __init__(self, *, network: InMemoryNetwork, public_key: str) -> None:
        super().__init__(network=network, public_key=public_key)
        self.sent_types: list[MessageType] = []

    async def send(self, destination: str, payload: bytes) -> None:
        self.sent_types.append(
            decode_envelope(
                payload,
                authenticated_sender=await self.local_public_key(),
                participant_keys=None,
                max_payload_bytes=8_388_608,
            ).message_type
        )
        await super().send(destination, payload)


@pytest.mark.parametrize("participant_count", [4, 8, 16])
def test_local_task_completes_fixed_group_formation(
    participant_count: int, tmp_path: Path
) -> None:
    asyncio.run(_form_local_group(participant_count, tmp_path, bad_schema=False))


def test_formation_does_not_send_ready_for_incompatible_local_model(
    tmp_path: Path,
) -> None:
    asyncio.run(_form_local_group(4, tmp_path, bad_schema=True))


async def _form_local_group(
    participant_count: int, root: Path, *, bad_schema: bool
) -> None:
    draft = _draft(participant_count)
    manifest = _manifest(participant_count)
    checkpoint = root / "checkpoint.safetensors"
    write_checkpoint(checkpoint)
    network = InMemoryNetwork()
    transports = [
        _RecordingTransport(network=network, public_key=f"peer-{index}")
        for index in range(participant_count)
    ]
    incompatible_schema = TensorSchema(
        tensors=(Tensor(name="layer.weight", dtype="float32", shape=(3, 2)),)
    )
    nodes = [
        FormationProtocol(
            transport=transport,
            draft=draft,
            environment=draft.environment,
            dataset=draft.dataset,
            transport_limits=draft.transport,
            artifact_root=root / f"node-{index}",
            local_tensor_schema=(
                incompatible_schema
                if bad_schema and index == 1
                else manifest.tensor_schema
            ),
        )
        for index, transport in enumerate(transports)
    ]
    invitation = create_invitation(
        draft=draft,
        initiator_public_key="peer-0",
        bootstrap_uri="axl://bootstrap",
    )
    tasks: list[asyncio.Task[FormationResult]] = []
    try:
        for node in nodes:
            await node.start()
        tasks.append(
            asyncio.create_task(
                nodes[0].initiate(
                    bootstrap_uri="axl://bootstrap",
                    checkpoint_path=checkpoint,
                    tensor_schema=manifest.tensor_schema,
                )
            )
        )
        tasks.extend(
            asyncio.create_task(node.join(invitation=invitation)) for node in nodes[1:]
        )
        if bad_schema:
            with pytest.raises(ReadyValidationError, match="local model tensor schema"):
                await asyncio.wait_for(tasks[1], timeout=5)
            assert MessageType.READY not in transports[1].sent_types
        else:
            results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
            assert len({result.manifest_hash for result in results}) == 1
            for result in results:
                assert result.manifest.participant_count == participant_count
                assert result.manifest.manifest_version == 7
                assert result.checkpoint_path.read_bytes() == checkpoint.read_bytes()
            assert all(
                MessageType.READY in transport.sent_types
                for transport in transports[1:]
            )
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for node in nodes:
            await node.stop()
