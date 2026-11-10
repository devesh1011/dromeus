from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from dromeus.algorithms.codec import (
    BitmapTopKInt8Codec,
    DenseInt8Codec,
    IdentityCodec,
    UpdateCodec,
    describe_update_codec,
    save_safetensors,
    validate_tensor_map,
)
from dromeus.algorithms.noloco import NoLoCoAlgorithm
from dromeus.manifests.canonical import canonical_hash, file_sha256
from dromeus.manifests.models import NoLoCoConfig, Tensor, TensorSchema

_CODECS = (
    "safetensors",
    "identity",
    "dense-int8",
    "topk-bitmap-int8",
)
# Captured from the preserved M2 source at 0eff9eb, using the same input tensors.
_FROZEN_ARTIFACT_HASHES = {
    "safetensors": (
        "42e1562bf5c8737fb4ed7f02ea5a927927913a35d28e3e0840c038f5a35f7bc5"
    ),
    "identity": "42e1562bf5c8737fb4ed7f02ea5a927927913a35d28e3e0840c038f5a35f7bc5",
    "dense-int8": "9472869cf0223ade0743b31f726943ce71b63845775223fe2572fc0a63896561",
    "topk-bitmap-int8": (
        "79473fe29dc35f3ac753a1ee9437e2d33fb9795679d72c3bf1e5528c28771eb0"
    ),
}


def _schema(size: int = 10) -> TensorSchema:
    return TensorSchema(
        tensors=(Tensor(name="weight", dtype="float32", shape=(size,)),)
    )


def _codec(name: str, schema: TensorSchema) -> UpdateCodec:
    if name == "dense-int8":
        return DenseInt8Codec(schema)
    if name == "topk-bitmap-int8":
        return BitmapTopKInt8Codec(schema, top_k_fraction=0.4)
    return IdentityCodec(name)


class _Trainer:
    def __init__(self) -> None:
        self._weights = {
            "weight": np.array(
                [1.0, -4.0, 4.0, 0.5, -3.0, 2.0, 0.0, 0.25, -0.1, 0.1],
                dtype=np.float32,
            )
        }

    def weights(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self._weights.items()}

    def load_weights(self, weights: dict[str, np.ndarray]) -> None:
        self._weights = {name: value.copy() for name, value in weights.items()}

    def train_local_steps(self, step_count: int) -> None:
        self._weights["weight"] *= np.float32(0.9)

    @property
    def local_loss(self) -> None:
        return None

    def evaluate(self) -> None:
        return None


def _config() -> NoLoCoConfig:
    return NoLoCoConfig(alpha=0.5, beta=0.7, gamma=0.7, inner_steps=50)


def _algorithm(codec: UpdateCodec) -> NoLoCoAlgorithm:
    return NoLoCoAlgorithm(
        trainer=_Trainer(),
        tensor_schema=_schema(),
        config=_config(),
        artifact_codecs={
            "outer_gradient": codec,
            "slow_weights": IdentityCodec("identity"),
        },
    )


@pytest.mark.parametrize("codec_id", _CODECS)
def test_common_codec_interface_preserves_frozen_wire_bytes(
    tmp_path: Path, codec_id: str
) -> None:
    schema = _schema()
    codec = _codec(codec_id, schema)
    description = describe_update_codec(codec, schema)
    encoded = codec.encode(_Trainer().weights())
    validate_tensor_map(encoded, description.encoded_schema)
    validate_tensor_map(codec.decode(encoded), schema)
    path = tmp_path / "artifact.safetensors"
    save_safetensors(encoded, str(path))

    assert description.codec_id == codec_id
    assert description.lossy == ("int8" in codec_id)
    assert file_sha256(path) == _FROZEN_ARTIFACT_HASHES[codec_id]
    if not description.lossy:
        assert description.encoded_schema == schema
        assert codec.encoded_schema_for(_schema(2)) == _schema(2)


@pytest.mark.parametrize("codec_id", _CODECS)
def test_noloco_binds_all_declared_codec_ids_and_schemas(
    tmp_path: Path, codec_id: str
) -> None:
    codec = _codec(codec_id, _schema())
    algorithm = _algorithm(codec)
    algorithm.configure_bundle_codec(
        artifact_root=tmp_path,
        run_id="codec-interface",
        manifest_hash="0" * 64,
        sender_public_key="node-0",
        algorithm_id="noloco",
    )
    algorithm.pre_local(0)
    algorithm.local_training()
    bundle = algorithm.post_local_bundle()
    try:
        algorithm.validate_peer(bundle)
        item = bundle.metadata.artifacts[0]
        assert item.codec_id == codec_id
        assert item.logical_schema_hash == canonical_hash(_schema())
        assert item.encoded_schema_hash == canonical_hash(
            codec.encoded_schema_for(_schema())
        )
        for changes in (
            {"codec_id": "another-codec"},
            {"logical_schema_hash": "f" * 64},
            {"encoded_schema_hash": "f" * 64},
        ):
            invalid = replace(
                bundle,
                metadata=bundle.metadata.model_copy(
                    update={
                        "artifacts": (
                            item.model_copy(update=changes),
                            *bundle.metadata.artifacts[1:],
                        )
                    }
                ),
            )
            with pytest.raises(ValueError, match="binding does not match"):
                algorithm.validate_peer(invalid)
    finally:
        algorithm.release_bundle(bundle)
    assert not tuple(tmp_path.glob("*.safetensors"))


def test_noloco_rejects_codec_mismatch_with_manifest() -> None:
    with pytest.raises(ValueError, match="do not match manifest"):
        NoLoCoAlgorithm(
            trainer=_Trainer(),
            tensor_schema=_schema(),
            config=_config(),
            artifact_codecs={
                "outer_gradient": IdentityCodec("identity"),
                "slow_weights": IdentityCodec("identity"),
            },
            manifest_codec_ids={
                "outer_gradient": "dense-int8",
                "slow_weights": "identity",
            },
        )


def _external_codec() -> SimpleNamespace:
    identity = IdentityCodec()
    return SimpleNamespace(
        codec_id=identity.codec_id,
        lossy=identity.lossy,
        encoded_schema_for=identity.encoded_schema_for,
        encode=identity.encode,
        decode=identity.decode,
        state_dict=identity.state_dict,
        load_state_dict=identity.load_state_dict,
    )


@pytest.mark.parametrize("capability", ("codec_id", "lossy", "encoded_schema_for"))
def test_algorithms_reject_missing_capabilities_at_construction(
    capability: str,
) -> None:
    codec = _external_codec()
    delattr(codec, capability)
    with pytest.raises(TypeError, match="complete UpdateCodec interface"):
        _algorithm(cast(UpdateCodec, codec))


def _invalid_schema(_logical_schema: TensorSchema) -> None:
    return None


@pytest.mark.parametrize(
    ("capability", "value", "message"),
    (
        ("codec_id", "", "nonempty string"),
        ("codec_id", 1, "nonempty string"),
        ("lossy", 1, "must be boolean"),
        ("lossy", None, "must be boolean"),
        ("encoded_schema_for", _invalid_schema, "encoded tensor schema"),
    ),
)
def test_algorithms_reject_malformed_capabilities_at_construction(
    capability: str,
    value: object,
    message: str,
) -> None:
    codec = _external_codec()
    setattr(codec, capability, value)
    with pytest.raises(TypeError, match=message):
        _algorithm(cast(UpdateCodec, codec))


@pytest.mark.parametrize("codec_id", _CODECS[2:])
def test_algorithms_reject_codec_for_another_model_at_construction(
    codec_id: str,
) -> None:
    with pytest.raises(ValueError, match="logical schema does not match algorithm"):
        _algorithm(_codec(codec_id, _schema(2)))
