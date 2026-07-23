from __future__ import annotations

import numpy as np
import pytest

from dromeus.algorithms.codec import IdentityCodec
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
