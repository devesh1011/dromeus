from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import dromeus.algorithms.codec as codec_module
from dromeus.algorithms.base import UpdateBundle
from dromeus.algorithms.codec import IdentityCodec, SafetensorsUpdateBundleCodec
from dromeus.algorithms.dpsgd import DPSGDAdapter
from dromeus.manifests.models import Tensor, TensorSchema


class Trainer:
    def __init__(self, value: float) -> None:
        self._weights = {"weight": np.array([value], dtype=np.float32)}

    def train_local_steps(self, step_count: int) -> None:
        self._weights["weight"] += np.float32(step_count)

    def weights(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self._weights.items()}

    def load_weights(self, weights: dict[str, np.ndarray]) -> None:
        self._weights = {name: value.copy() for name, value in weights.items()}


def test_m1_identity_codec_has_no_state_and_copies_tensors() -> None:
    codec = IdentityCodec()
    source = {"weight": np.array([1.0], dtype=np.float32)}

    encoded = codec.encode(source)
    encoded["weight"][0] = 2.0

    assert source["weight"][0] == 1.0
    assert codec.decode(encoded)["weight"][0] == 2.0
    assert codec.state_dict() == {}
    with pytest.raises(ValueError, match="no state"):
        codec.load_state_dict({"residual": np.array([1.0])})


def test_safetensors_bundle_codec_materializes_decodes_and_releases(
    tmp_path: Path,
) -> None:
    schema = TensorSchema(
        tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),)
    )
    codec = SafetensorsUpdateBundleCodec(
        artifact_root=tmp_path / "bundles",
        run_id="run-001",
        manifest_hash="1" * 64,
        sender_public_key="peer-0",
        algorithm_id="dpsgd",
        tensor_schema=schema,
    )
    source = {"weight": np.array([3.0], dtype=np.float32)}

    bundle = codec.encode(round_id=4, tensors=source)
    decoded = codec.decode(bundle)

    assert bundle.metadata.round_id == 4
    assert bundle.metadata.artifacts[0].codec_id == "safetensors"
    assert bundle.metadata.artifacts[0].codec_version == 1
    assert bundle.artifacts[0].path.is_file()
    assert np.array_equal(decoded["weight"], source["weight"])
    encoded_size = bundle.metadata.artifacts[0].size_bytes
    bundle.validate_materialized(encoded_size)
    with pytest.raises(ValueError, match="exceeds"):
        bundle.validate_materialized(encoded_size - 1)
    with bundle.artifacts[0].path.open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(ValueError, match="size mismatch"):
        bundle.validate_materialized(encoded_size + 1)

    path = bundle.artifacts[0].path
    codec.release(bundle)
    assert not path.exists()


def test_bundle_size_limit_is_aggregate_across_artifacts(tmp_path: Path) -> None:
    schema = TensorSchema(
        tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),)
    )
    codec = SafetensorsUpdateBundleCodec(
        artifact_root=tmp_path,
        run_id="run-001",
        manifest_hash="1" * 64,
        sender_public_key="peer-0",
        algorithm_id="dpsgd",
        tensor_schema=schema,
    )
    first = codec.encode(
        round_id=0, tensors={"weight": np.array([1.0], dtype=np.float32)}
    )
    second = codec.encode(
        round_id=0, tensors={"weight": np.array([2.0], dtype=np.float32)}
    )
    combined = UpdateBundle(
        metadata=first.metadata.model_copy(
            update={
                "artifacts": (
                    first.metadata.artifacts[0].model_copy(
                        update={"name": "first"}
                    ),
                    second.metadata.artifacts[0].model_copy(
                        update={"name": "second"}
                    ),
                )
            }
        ),
        artifacts=(first.artifacts[0], second.artifacts[0]),
    )

    with pytest.raises(ValueError, match="exceeds"):
        combined.validate_materialized(
            max(
                artifact.size_bytes for artifact in combined.metadata.artifacts
            )
        )
    codec.release(combined)


def test_bundle_encode_failure_removes_partial_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema = TensorSchema(
        tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),)
    )
    codec = SafetensorsUpdateBundleCodec(
        artifact_root=tmp_path,
        run_id="run-001",
        manifest_hash="1" * 64,
        sender_public_key="peer-0",
        algorithm_id="dpsgd",
        tensor_schema=schema,
    )

    def fail_after_write(_values: object, path: str) -> None:
        Path(path).write_bytes(b"partial")
        raise OSError("forced write failure")

    monkeypatch.setattr(codec_module, "save_safetensors", fail_after_write)
    with pytest.raises(OSError, match="forced"):
        codec.encode(
            round_id=0,
            tensors={"weight": np.array([1.0], dtype=np.float32)},
        )
    assert not any(path.is_file() for path in tmp_path.iterdir())


def test_dpsgd_algorithm_state_round_trips_model_round_and_codec() -> None:
    schema = TensorSchema(
        tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),)
    )
    first = DPSGDAdapter(trainer=Trainer(1.0), tensor_schema=schema, local_steps=2)
    first.pre_local(3)
    first.local_training()
    state = first.state_dict()

    second = DPSGDAdapter(trainer=Trainer(99.0), tensor_schema=schema, local_steps=2)
    second.load_state_dict(state)

    assert second.snapshot().round_id == 3
    assert second.snapshot().phase == "post-local"
    assert np.array_equal(second.snapshot().weights["weight"], np.array([3.0]))
    with pytest.raises(TypeError):
        second.snapshot().weights["weight"] = np.array([4.0])  # type: ignore[index]
    with pytest.raises(ValueError):
        second.snapshot().weights["weight"][0] = 4.0
