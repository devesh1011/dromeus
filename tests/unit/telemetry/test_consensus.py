from __future__ import annotations

import asyncio
import gc
import hashlib
import tracemalloc
import weakref
from collections.abc import Mapping

import numpy as np
import pytest

from dromeus.telemetry.consensus import (
    ConsensusObservation,
    ConsensusSketchBuffer,
    ConsensusSketchError,
    ConsensusSketchMessage,
    ConsensusSketchPublisher,
    LiveConsensusTelemetry,
    consensus_observation,
    count_sketch,
    decode_sketch,
    encode_sketch,
    exact_consensus_observation,
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

    exact = exact_consensus_observation(models, round_id=0).normalized_rms
    approximate = consensus_observation(sketches, round_id=0).normalized_rms
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


@pytest.mark.parametrize("world_size", (4, 8, 16))
def test_sketch_buffer_matches_direct_reference_with_missing_and_late_arrivals(
    world_size: int,
) -> None:
    members = tuple(f"peer-{index:02d}" for index in range(world_size))
    sketches = {
        member: count_sketch(
            {"weight": np.array([float(index), 1.0], dtype=np.float32)},
            seed=17,
        )
        for index, member in enumerate(members)
    }
    buffer = ConsensusSketchBuffer(participant_keys=members)

    for member in reversed(members[1:]):
        assert (
            buffer.add(
                round_id=3,
                sender_public_key=member,
                sketch=sketches[member],
            )
            is None
        )
    assert buffer.pending_rounds() == (3,)
    result = buffer.add(
        round_id=3,
        sender_public_key=members[0],
        sketch=sketches[members[0]],
    )

    assert result is not None
    assert result.sketch_count == world_size
    assert result.normalized_rms == pytest.approx(
        consensus_observation(
            [sketches[member] for member in sorted(members)], round_id=3
        ).normalized_rms
    )
    assert buffer.pending_rounds() == ()


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
        distances: list[ConsensusObservation] = []
        completed = asyncio.Event()

        async def receive(timeout_seconds: float) -> ConsensusSketchMessage:
            return await asyncio.wait_for(incoming.get(), timeout_seconds)

        async def publish(round_id: int, sketch: np.ndarray) -> None:
            assert sketch.dtype == np.float32
            published.append(round_id)

        def on_distance(distance: ConsensusObservation) -> None:
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
        distances: list[ConsensusObservation] = []

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


def test_observation_separates_spread_from_denominator_collapse() -> None:
    sketches = [np.array([1, 0], dtype=np.float32), np.array([-1, 0], dtype=np.float32)]
    measured = consensus_observation(sketches, round_id=3)
    exact = exact_consensus_observation([{"weight": x} for x in sketches], round_id=3)
    assert measured == exact
    assert measured.absolute_rms_spread == 1 and measured.mean_sketch_norm == 0
    assert measured.denominator_degenerate
    assert measured.normalized_rms == pytest.approx(1 / 1e-12)


def test_individual_distances_bind_members_not_arrival_order() -> None:
    buffer = ConsensusSketchBuffer(participant_keys=("d", "b", "a", "c"), size=1)
    for member, value in (("c", 4.0), ("a", 0.0), ("d", 8.0)):
        assert (
            buffer.add(
                round_id=0,
                sender_public_key=member,
                sketch=np.array([value], dtype=np.float32),
            )
            is None
        )
    result = buffer.add(
        round_id=0, sender_public_key="b", sketch=np.array([0.0], dtype=np.float32)
    )
    assert isinstance(result, ConsensusObservation)
    # Mean=3; distances by member: a=3, b=3, c=1, d=5.
    assert [(m.public_key, m.absolute_distance) for m in result.member_distances] == [
        ("a", 3),
        ("b", 3),
        ("c", 1),
        ("d", 5),
    ]
    assert [m.normalized_distance for m in result.member_distances] == pytest.approx(
        [1, 1, 1 / 3, 5 / 3]
    )
    assert result.absolute_rms_spread == pytest.approx(np.sqrt(11))
    assert (
        buffer.add(
            round_id=0, sender_public_key="a", sketch=np.array([0.0], dtype=np.float32)
        )
        == result
    )


@pytest.mark.parametrize("keys", [("a",), ("a", "a"), ("", "b")])
def test_member_measurements_reject_ambiguous_identity(keys: tuple[str, ...]) -> None:
    from dromeus.telemetry.consensus import consensus_observation

    with pytest.raises(ConsensusSketchError, match="participant keys"):
        consensus_observation(
            [np.array([1], dtype=np.float32), np.array([-1], dtype=np.float32)],
            round_id=0,
            participant_keys=keys,
        )


def test_individual_distances_expose_degenerate_mean() -> None:
    from dromeus.telemetry.consensus import consensus_observation

    result = consensus_observation(
        [np.array([1], dtype=np.float32), np.array([-1], dtype=np.float32)],
        round_id=0,
        participant_keys=("a", "b"),
    )
    assert result.denominator_degenerate
    assert [m.absolute_distance for m in result.member_distances] == [1, 1]
    assert [m.normalized_distance for m in result.member_distances] == [1e12, 1e12]


def _legacy_projection_reference(
    weights: Mapping[str, np.ndarray], *, seed: int, size: int = 4096
) -> np.ndarray:
    """Frozen pre-chunking numerical recipe, independent of new private helpers."""
    values = np.concatenate(
        [
            np.asarray(weights[name]).astype(np.float64, copy=False).reshape(-1)
            for name in sorted(weights)
        ]
    )
    normalized = int.from_bytes(
        hashlib.sha256(str(seed).encode()).digest()[:8], "little"
    )
    rng = np.random.default_rng(normalized)
    buckets = rng.integers(0, size, size=values.size)
    signs = np.where(
        rng.integers(0, 2, size=values.size, dtype=np.int8) == 0, -1.0, 1.0
    )
    result = np.zeros(size, dtype=np.float32)
    np.add.at(result, buckets, values * signs)
    return result


@pytest.mark.parametrize(
    "count", [0, 1, 3, 63, 64, 65, 1023, 1024, 1025, 65535, 65536, 65537]
)
@pytest.mark.parametrize("size", [1, 17, 4096])
@pytest.mark.parametrize("seed", [-3, 0, 17, 2**65 + 7])
def test_count_sketch_preserves_legacy_bytes_across_global_boundaries(
    count: int,
    size: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(91)
    first, second = count // 3, count // 4
    weights = {
        "z-f64": rng.standard_normal(count - first - second),
        "a-strided-f32": rng.standard_normal((first, 2)).astype(np.float32)[:, 0],
        "m-f16": rng.standard_normal(second).astype(np.float16),
        "empty": np.empty((0, 2), dtype=np.float32),
    }
    expected = _legacy_projection_reference(weights, seed=seed, size=size)
    actual = count_sketch(weights, seed=seed, size=size)
    assert actual.dtype == np.float32 and actual.tobytes() == expected.tobytes()


def test_count_sketch_scalar_fortran_layout_and_zero_signs_preserve_bytes() -> None:
    values = {
        "scalar": np.asarray(-0.0, dtype=np.float32),
        "matrix": np.asfortranarray(np.arange(35, dtype=np.float64).reshape(5, 7)),
        "reverse": np.linspace(-1, 1, 129, dtype=np.float32)[::-1],
    }
    assert (
        count_sketch(values, seed=0).tobytes()
        == _legacy_projection_reference(values, seed=0).tobytes()
    )


def test_count_sketch_projection_temporary_budget_is_bounded() -> None:
    values = {
        "b": np.ones(524288, dtype=np.float32),
        "a": np.arange(524288, dtype=np.float32),
    }
    # Exclude lazy imports/RNG setup and input storage from the temporary budget.
    count_sketch({"warmup": np.ones(1, dtype=np.float32)}, seed=17)
    expected = _legacy_projection_reference(values, seed=17)
    tracemalloc.start()
    try:
        actual = count_sketch(values, seed=17)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert actual.tobytes() == expected.tobytes()
    assert peak <= 8 * 1024**2, f"projection temporary allocation exceeded8MiB: {peak}"


@pytest.mark.parametrize("failing_callback", ["none", "on_sketch", "publish"])
def test_open_publisher_releases_completed_snapshot_before_next_input(
    monkeypatch: pytest.MonkeyPatch,
    failing_callback: str,
) -> None:
    import dromeus.telemetry.consensus as module

    async def exercise() -> None:
        copied: list[weakref.ReferenceType[np.ndarray]] = []
        published: list[bytes] = []
        done = asyncio.Event()
        original = getattr(module, "_copy_weights")

        def observe_copy(values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
            result: dict[str, np.ndarray] = original(values)
            copied.extend(weakref.ref(value) for value in result.values())
            return result

        async def on_sketch(_round: int, _sketch: np.ndarray) -> None:
            if failing_callback == "on_sketch":
                raise RuntimeError("public fixture callback failure")

        async def publish(_round: int, sketch: np.ndarray) -> None:
            published.append(sketch.tobytes())
            done.set()
            if failing_callback == "publish":
                raise RuntimeError("public fixture publication failure")

        monkeypatch.setattr(module, "_copy_weights", observe_copy)
        publisher = ConsensusSketchPublisher(
            seed=17, publish=publish, on_sketch=on_sketch
        )
        await publisher.start()
        try:
            weights = {"weight": np.array([1.0], dtype=np.float32)}
            references = [weakref.ref(value) for value in weights.values()]
            expected = _legacy_projection_reference(weights, seed=17).tobytes()
            assert publisher.submit(round_id=0, weights=weights)
            del weights
            await asyncio.wait_for(done.wait(), timeout=2)
            await asyncio.wait_for(getattr(publisher, "_queue").join(), timeout=2)
            await asyncio.sleep(0.22)
            gc.collect()
            assert publisher.pending == 0 and not getattr(publisher, "_task").done()
            assert not any(ref() is not None for ref in references), (
                "idle publisher retained input"
            )
            assert not any(ref() is not None for ref in copied), (
                "idle publisher retained snapshot"
            )
            assert published == [expected]
            assert publisher.dropped == int(failing_callback != "none")
        finally:
            await publisher.stop(timeout_seconds=2)

    asyncio.run(exercise())
