from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from safetensors.numpy import (
    save_file as save_safetensors,  # pyright: ignore[reportUnknownVariableType]
)

from benchmarks.noloco.compression import measure_bundle_wire_bytes
from dromeus.algorithms.codec import (
    DenseInt8Codec,
    NamedSafetensorsUpdateBundleCodec,
    TopKInt8Codec,
)
from dromeus.algorithms.noloco import NoLoCoAlgorithm
from dromeus.manifests.models import (
    AdamSettings,
    ArtifactCodec,
    NoLoCoConfig,
    Tensor,
    TensorSchema,
    TransportLimits,
    UpdateCodecBinding,
)
from dromeus.training.resnet18_groupnorm import build_model, floating_model_state


def _schema(*, size: int = 8) -> TensorSchema:
    return TensorSchema(
        tensors=(Tensor(name="weight", dtype="float32", shape=(size,)),)
    )


@pytest.mark.parametrize(
    "artifact_names",
    [
        ("trained_weights",),
        ("outer_gradient", "slow_weights"),
    ],
    ids=("dpsgd-cardinality-one", "noloco-cardinality-two"),
)
def test_named_bundle_codec_covers_all_algorithm_cardinalities_and_frozen_bytes(
    tmp_path: Path,
    artifact_names: tuple[str, ...],
) -> None:
    schema = _schema(size=2)
    artifacts = {
        name: {
            "weight": np.array([index + 1.0, index + 2.0], dtype=np.float32)
        }
        for index, name in enumerate(artifact_names)
    }
    artifact_schemas = {name: schema for name in artifact_names}
    codec = NamedSafetensorsUpdateBundleCodec(
        artifact_root=tmp_path / "bundles",
        run_id="cardinality-test",
        manifest_hash="0" * 64,
        sender_public_key="peer-0",
        algorithm_id="dpsgd" if len(artifact_names) == 1 else "noloco",
        artifact_schemas=artifact_schemas,
    )
    artifact_schemas["late-mutation"] = schema

    assert set(codec.artifact_schemas) == set(artifact_names)
    with pytest.raises(TypeError):
        cast(dict[str, TensorSchema], codec.artifact_schemas)["mutation"] = schema

    bundle = codec.encode(round_id=3, artifacts=artifacts)
    try:
        decoded = codec.decode(bundle)

        assert tuple(item.name for item in bundle.metadata.artifacts) == tuple(
            sorted(artifact_names)
        )
        assert set(decoded) == set(artifact_names)
        for item, materialized in zip(
            bundle.metadata.artifacts,
            bundle.artifacts,
            strict=True,
        ):
            expected = tmp_path / f"expected-{item.name}.safetensors"
            save_safetensors(artifacts[item.name], str(expected))
            assert materialized.path.read_bytes() == expected.read_bytes()
            assert np.array_equal(
                decoded[item.name]["weight"],
                artifacts[item.name]["weight"],
            )
    finally:
        codec.release(bundle)


def test_dense_int8_format_and_roundtrip_are_frozen() -> None:
    schema = _schema()
    codec = DenseInt8Codec(schema)
    source = {
        "weight": np.array(
            [-2.0, -1.0, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0],
            dtype=np.float32,
        )
    }

    encoded = codec.encode(source)
    decoded = codec.decode(encoded)

    assert codec.codec_id == "dense-int8-v1"
    assert codec.lossy
    assert tuple(item.name for item in codec.encoded_schema.tensors) == (
        "weight.__q",
        "weight.__scale",
        "weight.__zero_point",
    )
    assert encoded["weight.__q"].dtype == np.int8
    assert encoded["weight.__scale"].dtype == np.float32
    assert encoded["weight.__zero_point"].dtype == np.int32
    assert int(encoded["weight.__zero_point"][0]) == 0
    assert float(encoded["weight.__scale"][0]) == pytest.approx(2.0 / 127.0)
    assert np.max(np.abs(decoded["weight"] - source["weight"])) <= (
        float(encoded["weight.__scale"][0]) / 2.0 + 1e-7
    )
    repeated = codec.encode(source)
    assert all(np.array_equal(encoded[name], repeated[name]) for name in encoded)


def test_dense_int8_zero_range_and_malformed_metadata_reject() -> None:
    codec = DenseInt8Codec(_schema(size=4))
    encoded = codec.encode({"weight": np.zeros(4, dtype=np.float32)})

    assert float(encoded["weight.__scale"][0]) == 1.0
    assert np.array_equal(encoded["weight.__q"], np.zeros(4, dtype=np.int8))

    invalid = {name: value.copy() for name, value in encoded.items()}
    invalid["weight.__scale"][0] = np.float32(0.0)
    with pytest.raises(ValueError, match="scale"):
        codec.decode(invalid)

    invalid = {name: value.copy() for name, value in encoded.items()}
    invalid["weight.__zero_point"][0] = np.int32(1)
    with pytest.raises(ValueError, match="zero point"):
        codec.decode(invalid)


def test_topk_int8_selection_ties_indices_and_roundtrip_are_deterministic() -> None:
    schema = _schema(size=10)
    codec = TopKInt8Codec(schema, top_k_fraction=0.2)
    source = {
        "weight": np.array(
            [1.0, -4.0, 4.0, 0.5, -3.0, 2.0, 0.0, 0.25, -0.1, 0.1],
            dtype=np.float32,
        )
    }

    first = codec.encode(source)
    second = codec.encode(source)
    decoded = codec.decode(first)["weight"]

    assert codec.codec_id == "topk-int8-v1"
    assert codec.lossy
    assert tuple(item.name for item in codec.encoded_schema.tensors) == (
        "weight.__indices",
        "weight.__scale",
        "weight.__values",
        "weight.__zero_point",
    )
    assert np.array_equal(first["weight.__indices"], np.array([1, 2], np.int32))
    assert first["weight.__indices"].dtype == np.int32
    assert first["weight.__values"].dtype == np.int8
    assert all(np.array_equal(first[name], second[name]) for name in first)
    assert np.count_nonzero(decoded) == 2
    assert decoded[1] == pytest.approx(-4.0)
    assert decoded[2] == pytest.approx(4.0)


def test_topk_int8_uses_dense_fallback_when_indices_cost_more() -> None:
    codec = TopKInt8Codec(_schema(size=3), top_k_fraction=0.5)
    encoded = codec.encode(
        {"weight": np.array([1.0, -2.0, 3.0], dtype=np.float32)}
    )

    assert tuple(item.name for item in codec.encoded_schema.tensors) == (
        "weight.__q",
        "weight.__scale",
        "weight.__zero_point",
    )
    assert set(encoded) == {
        "weight.__q",
        "weight.__scale",
        "weight.__zero_point",
    }


def test_topk_int8_rejects_invalid_indices_and_schema() -> None:
    codec = TopKInt8Codec(_schema(size=10), top_k_fraction=0.2)
    encoded = codec.encode(
        {"weight": np.arange(10, dtype=np.float32)}
    )

    duplicate = {name: value.copy() for name, value in encoded.items()}
    duplicate["weight.__indices"][:] = np.int32(1)
    with pytest.raises(ValueError, match="strictly increasing"):
        codec.decode(duplicate)

    out_of_range = {name: value.copy() for name, value in encoded.items()}
    out_of_range["weight.__indices"][-1] = np.int32(10)
    with pytest.raises(ValueError, match="range"):
        codec.decode(out_of_range)

    wrong_dtype = {name: value.copy() for name, value in encoded.items()}
    wrong_dtype["weight.__indices"] = wrong_dtype[
        "weight.__indices"
    ].astype(np.int64)
    with pytest.raises(ValueError, match="dtype"):
        codec.decode(wrong_dtype)


class _Trainer:
    def __init__(self, slow: np.ndarray, fast: np.ndarray) -> None:
        self._weights = {"weight": slow.copy()}
        self._fast = fast.copy()

    def train_local_steps(self, step_count: int) -> None:
        assert step_count == 50
        self._weights = {"weight": self._fast.copy()}

    def weights(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self._weights.items()}

    def load_weights(self, weights: dict[str, np.ndarray]) -> None:
        self._weights = {name: value.copy() for name, value in weights.items()}

    @property
    def local_loss(self) -> None:
        return None

    def evaluate(self) -> None:
        return None

    def checkpoint_tensors(self) -> dict[str, np.ndarray]:
        return {
            "weight": self._weights["weight"].copy(),
            "completed_steps": np.array([0], dtype=np.int64),
        }

    def load_checkpoint_tensors(self, state: dict[str, np.ndarray]) -> None:
        self.load_weights({"weight": state["weight"]})


def _noloco_config() -> NoLoCoConfig:
    return NoLoCoConfig(
        alpha=0.5,
        beta=0.7,
        gamma=0.7,
        inner_steps=50,
        adam=AdamSettings(
            learning_rate=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            gradient_clip_norm=1.0,
        ),
    )


def _compressed_algorithm(
    root: Path,
    *,
    sender: str,
    slow: np.ndarray,
    fast: np.ndarray,
) -> NoLoCoAlgorithm:
    schema = _schema(size=slow.size)
    outer = TopKInt8Codec(schema, top_k_fraction=0.2)
    slow_codec = DenseInt8Codec(schema)
    return NoLoCoAlgorithm(
        trainer=_Trainer(slow, fast),
        tensor_schema=schema,
        config=_noloco_config(),
        artifact_codecs={
            "outer_gradient": outer,
            "slow_weights": slow_codec,
        },
        manifest_codec_ids={
            "outer_gradient": outer.codec_id,
            "slow_weights": slow_codec.codec_id,
        },
        bundle_codec=NamedSafetensorsUpdateBundleCodec(
            artifact_root=root,
            run_id="compressed-trace",
            manifest_hash="0" * 64,
            sender_public_key=sender,
            algorithm_id="noloco",
            artifact_schemas={
                "outer_gradient": outer.encoded_schema,
                "slow_weights": slow_codec.encoded_schema,
            },
        ),
    )


def test_compressed_two_node_trace_uses_decoded_values_and_error_feedback(
    tmp_path: Path,
) -> None:
    first_original = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32
    )
    second_original = np.array(
        [2, 1, 4, 3, 6, 5, 8, 7, 10, 9], dtype=np.float32
    )
    first_fast = np.array(
        [0, 2, 3, 4, 5, 6, 7, 8, 7, 10], dtype=np.float32
    )
    first = _compressed_algorithm(
        tmp_path / "first",
        sender="first",
        slow=first_original,
        fast=first_fast,
    )
    second = _compressed_algorithm(
        tmp_path / "second",
        sender="second",
        slow=second_original,
        fast=np.array([2, 0, 4, 3, 6, 5, 6, 7, 10, 8], dtype=np.float32),
    )
    for algorithm in (first, second):
        algorithm.pre_local(0)
        algorithm.local_training()
    first_bundle = first.post_local_bundle()
    second_bundle = second.post_local_bundle()
    try:
        first_local = first.validate_peer(first_bundle)
        second_local = second.validate_peer(second_bundle)
        first_snapshot = first.peer_apply(first.validate_peer(second_bundle))
        second_snapshot = second.peer_apply(second.validate_peer(first_bundle))
    finally:
        first.release_bundle(first_bundle)
        second.release_bundle(second_bundle)

    first_gradient = first_local.artifacts["outer_gradient"]["weight"]
    second_gradient = second_local.artifacts["outer_gradient"]["weight"]
    shared_gradient_term = np.float32(-0.7 / 2.0) * (
        first_gradient + second_gradient
    )
    first_slow = first_local.artifacts["slow_weights"]["weight"]
    second_slow = second_local.artifacts["slow_weights"]["weight"]
    first_expected = first_original + shared_gradient_term - np.float32(
        0.7 / 2.0
    ) * (first_slow - second_slow)
    second_expected = second_original + shared_gradient_term - np.float32(
        0.7 / 2.0
    ) * (second_slow - first_slow)

    assert np.allclose(first_snapshot.weights["weight"], first_expected, atol=1e-6)
    assert np.allclose(second_snapshot.weights["weight"], second_expected, atol=1e-6)
    first_residual = first.checkpoint_tensors()[
        "noloco.v1.error_feedback_residual.weight"
    ]
    assert np.count_nonzero(first_residual) > 0
    assert np.isfinite(first_residual).all()

    for algorithm in (first, second):
        algorithm.pre_local(1)
        algorithm.local_training()
    first_bundle = first.post_local_bundle()
    second_bundle = second.post_local_bundle()
    try:
        first_local_second = first.validate_peer(first_bundle)
        first.peer_apply(first.validate_peer(second_bundle))
        second.peer_apply(second.validate_peer(first_bundle))
    finally:
        first.release_bundle(first_bundle)
        second.release_bundle(second_bundle)
    next_residual = first.checkpoint_tensors()[
        "noloco.v1.error_feedback_residual.weight"
    ]
    expected_input = (
        np.asarray(first_snapshot.weights["weight"], dtype=np.float32)
        - first_fast
        + first_residual
    ).astype(np.float32)
    outer_codec = first.artifact_codecs["outer_gradient"]
    expected_decoded = outer_codec.decode(
        outer_codec.encode({"weight": expected_input})
    )["weight"]
    assert np.array_equal(
        first_local_second.artifacts["outer_gradient"]["weight"],
        expected_decoded,
    )
    assert np.isfinite(next_residual).all()
    assert not np.array_equal(next_residual, first_residual)

    restored = _compressed_algorithm(
        tmp_path / "restored",
        sender="restored",
        slow=np.asarray(first.snapshot().weights["weight"]),
        fast=np.zeros(10, dtype=np.float32),
    )
    restored.load_checkpoint_tensors(first.checkpoint_tensors())
    assert np.array_equal(
        restored.checkpoint_tensors()[
            "noloco.v1.error_feedback_residual.weight"
        ],
        next_residual,
    )


def test_compressed_observations_report_error_feedback_norms_and_ratio(
    tmp_path: Path,
) -> None:
    slow = np.arange(1, 6, dtype=np.float32)
    algorithm = _compressed_algorithm(
        tmp_path / "observations",
        sender="observations",
        slow=slow,
        fast=np.zeros_like(slow),
    )

    algorithm.pre_local(0)
    algorithm.local_training()
    bundle = algorithm.post_local_bundle()
    try:
        observations = algorithm.observations()
    finally:
        algorithm.release_bundle(bundle)

    assert observations.error_feedback_residual_l2_norm == pytest.approx(
        math.sqrt(30.0), rel=1e-6
    )
    assert observations.error_feedback_signal_l2_norm == pytest.approx(
        math.sqrt(55.0), rel=1e-6
    )
    assert observations.error_feedback_residual_to_signal_ratio == pytest.approx(
        math.sqrt(30.0 / 55.0), rel=1e-6
    )


def test_manifest_codec_settings_freeze_supported_combinations() -> None:
    outer = ArtifactCodec(
        artifact_name="outer_gradient",
        codec_id="topk-int8-v1",
        top_k_fraction=0.01,
        lossy_allowed=True,
    )
    slow = ArtifactCodec(
        artifact_name="slow_weights",
        codec_id="dense-int8-v1",
        lossy_allowed=True,
    )

    assert outer.top_k_fraction == 0.01
    assert slow.top_k_fraction is None
    with pytest.raises(ValueError, match="only valid for outer gradient"):
        ArtifactCodec(
            artifact_name="slow_weights",
            codec_id="topk-int8-v1",
            top_k_fraction=0.01,
            lossy_allowed=True,
        )


def test_resnet18_complete_wire_bundle_exceeds_five_x_compression(
    tmp_path: Path,
) -> None:
    model = build_model(seed=17)
    schema = TensorSchema(
        tensors=tuple(
            Tensor(name=name, dtype="float32", shape=tuple(value.shape))
            for name, value in floating_model_state(model).items()
        )
    )
    slow = {
        name: value.detach().cpu().numpy().astype(np.float32, copy=True)
        for name, value in floating_model_state(model).items()
    }
    gradient = {name: np.zeros_like(value) for name, value in slow.items()}
    logical = {"outer_gradient": gradient, "slow_weights": slow}
    identity_bundle_codec = NamedSafetensorsUpdateBundleCodec(
        artifact_root=tmp_path / "identity",
        run_id="compression-measurement",
        manifest_hash="1" * 64,
        sender_public_key="measurement-node",
        algorithm_id="noloco",
        artifact_schemas={"outer_gradient": schema, "slow_weights": schema},
    )
    outer_codec = TopKInt8Codec(schema, top_k_fraction=0.01)
    slow_codec = DenseInt8Codec(schema)
    compressed_bundle_codec = NamedSafetensorsUpdateBundleCodec(
        artifact_root=tmp_path / "compressed",
        run_id="compression-measurement",
        manifest_hash="1" * 64,
        sender_public_key="measurement-node",
        algorithm_id="noloco",
        artifact_schemas={
            "outer_gradient": outer_codec.encoded_schema,
            "slow_weights": slow_codec.encoded_schema,
        },
    )
    identity = identity_bundle_codec.encode(round_id=0, artifacts=logical)
    compressed = compressed_bundle_codec.encode(
        round_id=0,
        artifacts={
            "outer_gradient": outer_codec.encode(gradient),
            "slow_weights": slow_codec.encode(slow),
        },
        codec_bindings={
            "outer_gradient": UpdateCodecBinding(
                codec_id=outer_codec.codec_id,
                codec_version=1,
                logical_schema=schema,
            ),
            "slow_weights": UpdateCodecBinding(
                codec_id=slow_codec.codec_id,
                codec_version=1,
                logical_schema=schema,
            ),
        },
    )
    transport = TransportLimits(
        max_payload_bytes=128 * 1024 * 1024,
        max_retries=2,
        retry_timeout_seconds=1.0,
        chunk_size_bytes=1024 * 1024,
        window_size=4,
        max_message_payload_bytes=2 * 1024 * 1024,
        max_artifact_bytes=64 * 1024 * 1024,
        max_concurrent_transfers=4,
        max_inflight_bytes=8 * 1024 * 1024,
        transfer_lifetime_seconds=30.0,
        artifact_store_capacity_bytes=128 * 1024 * 1024,
    )
    try:
        identity_measurement = measure_bundle_wire_bytes(
            identity,
            transport=transport,
            logical_artifact_schemas={
                "outer_gradient": schema,
                "slow_weights": schema,
            },
        )
        compressed_measurement = measure_bundle_wire_bytes(
            compressed,
            transport=transport,
            logical_artifact_schemas={
                "outer_gradient": schema,
                "slow_weights": schema,
            },
        )
    finally:
        identity_bundle_codec.release(identity)
        compressed_bundle_codec.release(compressed)

    ratio = identity_measurement.wire_bytes / compressed_measurement.wire_bytes
    assert identity_measurement.raw_artifact_bytes == 89_391_696
    assert compressed_measurement.raw_artifact_bytes == (
        identity_measurement.raw_artifact_bytes
    )
    assert identity_measurement.protocol_overhead_bytes > 0
    assert compressed_measurement.protocol_overhead_bytes > 0
    assert (
        identity_measurement.protocol_overhead_bytes
        + identity_measurement.metadata_carrier_bytes
        + identity_measurement.encoded_artifact_bytes
        == identity_measurement.wire_bytes
    )
    assert identity_measurement.encoded_artifact_bytes > 85 * 1024 * 1024
    assert ratio >= 5.0
