"""Checkpoint temporaries must not depend on cyclic GC for reclamation."""

from __future__ import annotations

import gc
import weakref
from collections.abc import Iterator

import numpy as np
import pytest
import torch

from dromeus.training.private_state import decode_state, encode_state


@pytest.fixture
def without_cyclic_collection() -> Iterator[None]:
    # Isolate reference ownership, not a proposed runtime GC configuration.
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.collect()
        if enabled:
            gc.enable()


@pytest.mark.usefixtures("without_cyclic_collection")
def test_encoded_optimizer_arrays_release_with_last_owner() -> None:
    encoded = encode_state(
        {"state": {0: {"exp_avg": torch.ones(16), "exp_avg_sq": torch.zeros(16)}}}
    )
    references = [weakref.ref(value) for value in encoded.values()]
    del encoded
    assert all(reference() is None for reference in references)


@pytest.mark.usefixtures("without_cyclic_collection")
def test_decoded_state_does_not_retain_checkpoint_input_arrays() -> None:
    # Hand-built input isolates decoding from the encoder's ownership behavior.
    arrays = {
        "tensor.0": np.arange(16, dtype=np.float32),
        "structure": np.frombuffer(b'{"tensor":"tensor.0"}', dtype=np.uint8).copy(),
    }
    references = [weakref.ref(value) for value in arrays.values()]
    decoded = decode_state(arrays)
    del arrays
    assert all(reference() is None for reference in references)
    assert torch.equal(decoded, torch.arange(16, dtype=torch.float32))


def test_checkpoint_structure_and_tensor_bytes_remain_stable() -> None:
    encoded = encode_state({"x": (torch.tensor([1.0, -2.0]), [None, True, 3])})
    assert tuple(encoded) == ("tensor.0", "structure")
    assert encoded["structure"].tobytes() == (
        b'{"dict":[["x",{"tuple":[{"tensor":"tensor.0"},{"list":[null,true,3]}]}]]}'
    )
    np.testing.assert_array_equal(encoded["tensor.0"], np.array([1, -2], np.float32))
    decoded = decode_state(encoded)
    assert torch.equal(decoded["x"][0], torch.tensor([1.0, -2.0]))
    assert decoded["x"][1] == [None, True, 3]
