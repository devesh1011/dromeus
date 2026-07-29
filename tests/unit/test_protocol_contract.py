from __future__ import annotations

import json
from pathlib import Path

import pytest

from dromeus.protocol.codec import (
    ProtocolDecodeError,
    decode_envelope,
    decode_message,
    encode_envelope,
    encode_message,
)
from dromeus.protocol.models import (
    Chunk,
    ChunkAck,
    DomainModel,
    Envelope,
    JoinAccepted,
    JoinRequest,
    MessageType,
    PairCommitMessage,
    ReadyMessage,
    RunFailedMessage,
    StartMessage,
    Tensor,
    TensorSchema,
    TransferBegin,
    TransferComplete,
    create_envelope,
)

GOLDEN = Path(__file__).parents[1] / "golden" / "protocol_v1.json"


def _models() -> dict[str, DomainModel]:
    schema = TensorSchema(
        tensors=(Tensor(name="layer.weight", dtype="float32", shape=(2, 2)),)
    )
    return {
        "join_request": JoinRequest(draft_hash="1" * 64),
        "join_accepted": JoinAccepted(draft_hash="2" * 64),
        "ready": ReadyMessage(manifest_hash="3" * 64),
        "start": StartMessage(manifest_hash="4" * 64),
        "transfer_begin": TransferBegin(
            transfer_id="transfer-1",
            artifact_name="artifact",
            total_size_bytes=4,
            total_sha256="5" * 64,
            chunk_count=1,
            codec_id="safetensors-v1",
            tensor_schema=schema,
        ),
        "chunk": Chunk(
            transfer_id="transfer-1",
            chunk_index=0,
            chunk_count=1,
            chunk_sha256="6" * 64,
            data=b"abcd",
        ),
        "chunk_ack": ChunkAck(
            transfer_id="transfer-1",
            chunk_index=0,
            chunk_sha256="6" * 64,
        ),
        "transfer_complete": TransferComplete(
            transfer_id="transfer-1",
            total_sha256="5" * 64,
        ),
        "pair_commit": PairCommitMessage(round_id=7, checksum="7" * 64),
        "run_failed": RunFailedMessage(
            round_id=7,
            error_type="PairCommitError",
            reason="deadline",
        ),
    }


def test_protocol_v1_payload_bytes_match_golden_fixture() -> None:
    golden = json.loads(GOLDEN.read_text())

    assert {
        name: encode_message(model).hex() for name, model in _models().items()
    } == {name: value for name, value in golden.items() if name != "envelope"}


def test_protocol_v1_envelope_bytes_match_golden_fixture() -> None:
    golden = json.loads(GOLDEN.read_text())
    envelope = create_envelope(
        message_type=MessageType.CHUNK,
        message_id="message-1",
        run_id="run-001",
        manifest_hash="a" * 64,
        sender_public_key="peer-0",
        algorithm_id="dpsgd",
        round_id=2,
        correlation_id="transfer-1",
        payload=b"weights",
    )

    assert encode_envelope(envelope).hex() == golden["envelope"]


@pytest.mark.parametrize(
    "encoded",
    (
        "8ba56d61676963c40444524d31b070726f746f636f6c5f76657273696f6e02ac6d6573736167655f74797065a54348554e4baa6d6573736167655f6964a96d6573736167652d31a672756e5f6964a772756e2d303031ad6d616e69666573745f68617368d94061616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161b173656e6465725f7075626c69635f6b6579a6706565722d30ac616c676f726974686d5f6964a56470736764ae7061796c6f61645f6c656e67746800ae7061796c6f61645f736861323536d94065336230633434323938666331633134396166626634633839393666623932343237616534316534363439623933346361343935393931623738353262383535a77061796c6f6164c400",
        "8ba56d61676963c40444524d31b070726f746f636f6c5f76657273696f6e01ac6d6573736167655f74797065a7554e4b4e4f574eaa6d6573736167655f6964a96d6573736167652d31a672756e5f6964a772756e2d303031ad6d616e69666573745f68617368d94061616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161b173656e6465725f7075626c69635f6b6579a6706565722d30ac616c676f726974686d5f6964a56470736764ae7061796c6f61645f6c656e67746800ae7061796c6f61645f736861323536d94065336230633434323938666331633134396166626634633839393666623932343237616534316534363439623933346361343935393931623738353262383535a77061796c6f6164c400",
    ),
)
def test_unknown_protocol_version_and_message_type_are_rejected(
    encoded: str,
) -> None:
    with pytest.raises(ProtocolDecodeError):
        decode_envelope(
            bytes.fromhex(encoded),
            authenticated_sender="peer-0",
            participant_keys=frozenset({"peer-0"}),
        )


def test_unknown_payload_fields_and_oversized_payloads_are_rejected() -> None:
    unknown_field = bytes.fromhex(
        "82aa64726166745f68617368d94031313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131a56578747261c3"
    )
    with pytest.raises(ProtocolDecodeError):
        decode_message(unknown_field, JoinRequest, max_bytes=len(unknown_field))

    valid = encode_message(JoinRequest(draft_hash="1" * 64))
    with pytest.raises(ProtocolDecodeError):
        decode_message(valid, JoinRequest, max_bytes=len(valid) - 1)


def test_payload_decode_returns_strict_model() -> None:
    encoded = encode_message(ReadyMessage(manifest_hash="3" * 64))

    decoded = decode_message(encoded, ReadyMessage, max_bytes=len(encoded))

    assert isinstance(decoded, ReadyMessage)
    assert decoded.manifest_hash == "3" * 64
    assert Envelope.model_fields["protocol_version"].default == 1


def test_transfer_begin_supports_resnet32_tensor_count() -> None:
    schema = TensorSchema(
        tensors=tuple(
            Tensor(name=f"layer-{index}", dtype="float32", shape=(1,))
            for index in range(157)
        )
    )
    begin = TransferBegin(
        transfer_id="transfer-resnet32",
        artifact_name="initial-checkpoint",
        total_size_bytes=1,
        total_sha256="5" * 64,
        chunk_count=1,
        codec_id="safetensors-v1",
        tensor_schema=schema,
    )
    encoded = encode_message(begin)

    assert decode_message(encoded, TransferBegin, max_bytes=len(encoded)) == begin
