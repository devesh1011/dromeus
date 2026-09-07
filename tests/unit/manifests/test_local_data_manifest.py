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
    ClassificationTaskContract,
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


def _task() -> ClassificationTaskContract:
    return ClassificationTaskContract(
        dataset_id="local-classification-v1",
        input_shape=(2,),
        input_dtype="float32",
        label_names=("cat", "truck"),
        preprocessing_hash="a" * 64,
    )


def _draft(participant_count: int = 4) -> DraftRunSpec:
    data = manifest_data()
    for field in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del data[field]
    data.update(
        manifest_version=4,
        expected_participant_count=participant_count,
        dataset=_task(),
    )
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
def test_v4_task_and_explicit_membership_round_trip(participant_count: int) -> None:
    draft = _draft(participant_count)
    manifest = _manifest(participant_count)
    invitation = create_invitation(
        draft=draft,
        initiator_public_key="peer-0",
        bootstrap_uri="axl://bootstrap",
    )

    assert parse_draft_yaml(canonical_json(draft)) == draft
    assert parse_sealed_json(canonical_json(manifest)) == manifest
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
    assert isinstance(draft.dataset, ClassificationTaskContract)
    assert draft.dataset.class_count == 2
    assert draft.dataset.objective == "equal-node"
    with pytest.raises(ValueError, match="requires.*CIFAR-10"):
        draft.require_iid_dataset()


@pytest.mark.parametrize("participant_count", [1, 2, 3, 5, 7, 15, 17, 32])
def test_v4_rejects_invalid_fixed_group_sizes(participant_count: int) -> None:
    with pytest.raises(ValidationError):
        _draft(participant_count)


@pytest.mark.parametrize(
    ("version", "local_data", "explicit_count", "message"),
    [
        (3, True, None, "v3 requires the CIFAR-10"),
        (3, False, 4, "v3 derives membership"),
        (4, False, 4, "v4 requires a local classification"),
        (4, True, None, "v4 requires expected participant count"),
    ],
)
def test_versions_do_not_silently_reinterpret_data_or_membership(
    version: int, local_data: bool, explicit_count: int | None, message: str
) -> None:
    data = _draft().model_dump(mode="python")
    data.update(
        manifest_version=version,
        dataset=_task() if local_data else manifest_data()["dataset"],
        expected_participant_count=explicit_count,
    )
    with pytest.raises(ValidationError, match=message):
        DraftRunSpec.model_validate(data)


def test_v3_keeps_legacy_membership_out_of_canonical_bytes() -> None:
    manifest = SealedManifest.model_validate(manifest_data())

    assert manifest.manifest_version == 3
    assert manifest.participant_count == 4
    assert manifest.require_iid_dataset().participant_count == 4
    assert b"expected_participant_count" not in canonical_json(manifest)


@pytest.mark.parametrize("explicit_none", [False, True])
def test_v3_requires_retained_container_identity(explicit_none: bool) -> None:
    data = manifest_data()
    if explicit_none:
        data["environment"]["container_image_digest"] = None
    else:
        del data["environment"]["container_image_digest"]
    with pytest.raises(ValidationError, match="v3 requires a container image digest"):
        SealedManifest.model_validate(data)


@pytest.mark.parametrize("explicit_none", [False, True])
def test_v4_host_execution_can_omit_container_identity(explicit_none: bool) -> None:
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
        ("input_shape", ()),
        ("input_shape", (2, 0)),
        ("input_dtype", "float64"),
        ("label_names", ("cat",)),
        ("label_names", ("cat", "cat")),
        ("label_names", ("cat", "")),
        ("label_names", ("cat", "  ")),
        ("objective", "sample-weighted"),
        ("sample_count", 10),
        ("dataset_path", "/private/node-0/data"),
        ("fingerprint", "b" * 64),
    ],
)
def test_local_task_rejects_invalid_or_private_contract_fields(
    field: str, value: object
) -> None:
    data = _task().model_dump(mode="python")
    data[field] = value
    with pytest.raises(ValidationError):
        ClassificationTaskContract.model_validate(data)


@pytest.mark.parametrize(
    "change",
    [
        {"label_names": ("truck", "cat")},
        {"preprocessing_hash": "b" * 64},
        {"input_shape": (3,)},
    ],
)
def test_incompatible_task_meaning_changes_identity_and_blocks_ready(
    change: dict[str, object],
) -> None:
    task_data = _task().model_dump(mode="python")
    task_data.update(change)
    incompatible = ClassificationTaskContract.model_validate(task_data)
    manifest = _manifest()

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


def test_v4_sealing_rejects_missing_participant() -> None:
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
            asyncio.create_task(node.join(invitation=invitation))
            for node in nodes[1:]
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
                assert result.manifest.manifest_version == 4
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
