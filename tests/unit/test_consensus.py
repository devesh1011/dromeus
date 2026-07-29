from __future__ import annotations

import asyncio
from collections.abc import Mapping

import numpy as np
import pytest

from dromeus.telemetry.consensus import (
    ConsensusDistance,
    ConsensusSketchBuffer,
    ConsensusSketchError,
    ConsensusSketchMessage,
    ConsensusSketchPublisher,
    LiveConsensusTelemetry,
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
        participant_keys=("peer-0", "peer-1", "peer-2", "peer-3")
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


def test_sketch_buffer_evicts_old_incomplete_rounds() -> None:
    buffer = ConsensusSketchBuffer(
        participant_keys=("peer-0", "peer-1"), max_pending_rounds=1
    )
    sketch = np.zeros(4096, dtype=np.float32)
    assert buffer.add(round_id=0, sender_public_key="peer-0", sketch=sketch) is None
    assert buffer.add(round_id=1, sender_public_key="peer-0", sketch=sketch) is None
    assert buffer.pending_rounds() == (1,)
    assert buffer.dropped == 1


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
        assert publisher.dropped == 1

    asyncio.run(run())


def test_sketch_publisher_shutdown_is_bounded() -> None:
    async def run() -> None:
        async def publish(_round_id: int, _sketch: np.ndarray) -> None:
            await asyncio.sleep(10)

        publisher = ConsensusSketchPublisher(seed=9, publish=publish)
        assert publisher.submit(
            round_id=0, weights={"weight": np.array([1.0], dtype=np.float32)}
        )
        await publisher.start()
        assert not await publisher.stop(timeout_seconds=0.01)
        assert publisher.dropped >= 1

    asyncio.run(run())


def test_live_consensus_joins_local_and_remote_sketches() -> None:
    async def run() -> None:
        incoming: asyncio.Queue[ConsensusSketchMessage] = asyncio.Queue()
        published: list[int] = []
        distances: list[ConsensusDistance] = []
        completed = asyncio.Event()

        async def receive(timeout_seconds: float) -> ConsensusSketchMessage:
            return await asyncio.wait_for(incoming.get(), timeout_seconds)

        async def publish(round_id: int, sketch: np.ndarray) -> None:
            assert sketch.dtype == np.float32
            published.append(round_id)

        def on_distance(distance: ConsensusDistance) -> None:
            distances.append(distance)
            completed.set()

        telemetry = LiveConsensusTelemetry(
            local_public_key="peer-0",
            participant_keys=[f"peer-{index}" for index in range(4)],
            seed=9,
            receive=receive,
            publish=publish,
            on_distance=on_distance,
        )
        await telemetry.start()
        assert telemetry.submit(
            round_id=0,
            weights={"weight": np.array([0.0], dtype=np.float32)},
        )
        for index in range(1, 4):
            sketch = count_sketch(
                {"weight": np.array([float(index)], dtype=np.float32)}, seed=9
            )
            await incoming.put(
                ConsensusSketchMessage(
                    sender_public_key=f"peer-{index}",
                    round_id=0,
                    payload=encode_sketch(sketch),
                )
            )
        await asyncio.wait_for(completed.wait(), timeout=1.0)
        await telemetry.stop()

        assert published == [0]
        assert len(distances) == 1
        assert distances[0].round_id == 0
        assert distances[0].sketch_count == 4
        assert distances[0].normalized_rms == pytest.approx(0.7453559925)
        assert telemetry.result(0) == distances[0]

    asyncio.run(run())


def test_live_consensus_waits_for_inflight_remote_sketches_on_stop() -> None:
    async def run() -> None:
        incoming: asyncio.Queue[ConsensusSketchMessage] = asyncio.Queue()
        distances: list[ConsensusDistance] = []

        async def receive(timeout_seconds: float) -> ConsensusSketchMessage:
            return await asyncio.wait_for(incoming.get(), timeout_seconds)

        async def publish(round_id: int, sketch: np.ndarray) -> None:
            del round_id, sketch

        telemetry = LiveConsensusTelemetry(
            local_public_key="peer-0",
            participant_keys=[f"peer-{index}" for index in range(4)],
            seed=9,
            receive=receive,
            publish=publish,
            on_distance=distances.append,
        )
        await telemetry.start()
        assert telemetry.submit(
            round_id=0,
            weights={"weight": np.array([0.0], dtype=np.float32)},
        )

        async def deliver_remote_sketches() -> None:
            await asyncio.sleep(1.1)
            for index in range(1, 4):
                sketch = count_sketch(
                    {"weight": np.array([float(index)], dtype=np.float32)},
                    seed=9,
                )
                await incoming.put(
                    ConsensusSketchMessage(
                        sender_public_key=f"peer-{index}",
                        round_id=0,
                        payload=encode_sketch(sketch),
                    )
                )

        delivery = asyncio.create_task(deliver_remote_sketches())
        await telemetry.stop()
        await delivery

        assert [distance.round_id for distance in distances] == [0]

    asyncio.run(run())
