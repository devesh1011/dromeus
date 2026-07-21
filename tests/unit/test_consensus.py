from __future__ import annotations

import asyncio
from collections.abc import Mapping

import numpy as np
import pytest

from dromeus.telemetry.consensus import (
    ConsensusSketchBuffer,
    ConsensusSketchError,
    ConsensusSketchPublisher,
    count_sketch,
    decode_sketch,
    encode_sketch,
    exact_normalized_rms_distance,
    normalized_rms_consensus_distance,
)


def test_count_sketch_is_seeded_and_round_trips_as_16_kib_fp32() -> None:
    first: Mapping[str, np.ndarray] = {
        "b": np.array([3.0], dtype=np.float32),
        "a": np.array([1.0, 2.0], dtype=np.float32),
    }
    second = {name: first[name] for name in reversed(first)}

    sketch = count_sketch(first, seed=9)
    assert np.array_equal(sketch, count_sketch(second, seed=9))
    assert sketch.dtype == np.float32
    assert sketch.shape == (4096,)
    assert len(encode_sketch(sketch)) == 4096 * 4
    assert np.array_equal(decode_sketch(encode_sketch(sketch)), sketch)


def test_sketch_distance_matches_exact_distance_without_collisions() -> None:
    models = [
        {"weight": np.array([1.0, 0.0], dtype=np.float32)},
        {"weight": np.array([3.0, 0.0], dtype=np.float32)},
        {"weight": np.array([2.0, 0.0], dtype=np.float32)},
        {"weight": np.array([2.0, 0.0], dtype=np.float32)},
    ]
    sketches = [count_sketch(model, seed=9) for model in models]

    exact = exact_normalized_rms_distance(models)
    approximate = normalized_rms_consensus_distance(sketches)
    assert exact == pytest.approx(0.3535533906)
    assert approximate == pytest.approx(exact)


def test_sketch_buffer_computes_once_all_sealed_members_arrive() -> None:
    buffer = ConsensusSketchBuffer(
        participant_keys=("peer-0", "peer-1", "peer-2", "peer-3"), seed=9
    )
    sketch = np.zeros(4096, dtype=np.float32)
    for peer in ("peer-0", "peer-1", "peer-2"):
        assert buffer.add(round_id=4, sender_public_key=peer, sketch=sketch) is None
    result = buffer.add(round_id=4, sender_public_key="peer-3", sketch=sketch)
    assert result is not None
    assert result.round_id == 4
    assert result.normalized_rms == 0.0
    assert buffer.add(round_id=4, sender_public_key="peer-3", sketch=sketch) == result
    with pytest.raises(ConsensusSketchError, match="not a sealed participant"):
        buffer.add(round_id=4, sender_public_key="outsider", sketch=sketch)


def test_sketch_publisher_is_bounded_and_non_blocking() -> None:
    published: list[tuple[int, np.ndarray]] = []

    async def publish(round_id: int, sketch: np.ndarray) -> None:
        published.append((round_id, sketch))

    async def run() -> None:
        publisher = ConsensusSketchPublisher(seed=9, publish=publish, max_queue_size=1)
        assert publisher.submit(
            round_id=0, weights={"weight": np.array([1.0], dtype=np.float32)}
        )
        assert not publisher.submit(
            round_id=1, weights={"weight": np.array([2.0], dtype=np.float32)}
        )
        await publisher.start()
        await asyncio.sleep(0.05)
        await publisher.stop()
        assert [round_id for round_id, _ in published] == [0]

    asyncio.run(run())
