from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import numpy as np
import pytest
from support.gossip_fakes import (
    BlockingTrainer,
    HangingPairTransport,
    InMemoryPairTransport,
    InvalidObservationTrainer,
    LinearTrainer,
    ReadinessGuardTransport,
    RecordingBundleCodec,
    RecordingFailureBroadcaster,
    RecordingMetricsPublisher,
    RecordingPublisher,
    RejectingCommitTransport,
    SharedPairChannel,
    SlowEvaluationTrainer,
    StaticPairTransport,
    TimedPairTransport,
)
from support.gossip_fakes import make_algorithm as _algorithm

from dromeus.algorithms.codec import (
    NamedSafetensorsUpdateBundleCodec,
)
from dromeus.algorithms.dpsgd import DPSGDAdapter, checksum_tensors
from dromeus.gossip.engine import GossipEngine
from dromeus.gossip.interfaces import (
    EvaluationMetrics,
    PairCommitError,
    RoundCommit,
    RunFailure,
)
from dromeus.gossip.peer_scheduler import PeerScheduler
from dromeus.manifests.models import (
    Tensor,
    TensorSchema,
    UpdateCodecBinding,
)


def test_pair_timeout_fails_once_and_reports_diagnostics(tmp_path: Path) -> None:
    failures: list[RunFailure] = []
    broadcasted: list[RunFailure] = []
    metrics = RecordingMetricsPublisher([], [])

    async def run() -> None:
        schema = TensorSchema(
            tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),)
        )
        codec = RecordingBundleCodec(
            NamedSafetensorsUpdateBundleCodec(
                artifact_root=tmp_path / "peer-0",
                run_id="test-run",
                manifest_hash="0" * 64,
                sender_public_key="peer-0",
                algorithm_id="d-psgd",
                artifact_schemas={"trained_weights": schema},
            )
        )
        engine = GossipEngine(
            local_public_key="peer-0",
            round_count=1,
            scheduler=PeerScheduler(["peer-0", "peer-1"], seed=8),
            algorithm=DPSGDAdapter(
                trainer=LinearTrainer(1.0),
                tensor_schema=schema,
                local_steps=1,
                bundle_codec=codec,
            ),
            transport=HangingPairTransport("peer-0", SharedPairChannel.create()),
            commit_callback=lambda commit: None,
            timeout_seconds=0.01,
            failure_callback=failures.append,
            failure_broadcaster=RecordingFailureBroadcaster(broadcasted),
            metrics_publisher=metrics,
        )

        with pytest.raises(PairCommitError, match="deadline"):
            await engine.run()
        assert engine.failure == failures[0]
        assert len(failures) == 1
        assert broadcasted == failures
        assert metrics.failures[0][0] == 0
        assert len(codec.released) == 1

    asyncio.run(run())


def test_engine_confirms_durability_only_after_peer_confirmation(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        schema = TensorSchema(
            tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),)
        )
        peer_codec = NamedSafetensorsUpdateBundleCodec(
            artifact_root=tmp_path / "peer",
            run_id="test-run",
            manifest_hash="0" * 64,
            sender_public_key="peer-1",
            algorithm_id="d-psgd",
            artifact_schemas={"trained_weights": schema},
        )
        peer_bundle = peer_codec.encode(
            round_id=0,
            artifacts={
                "trained_weights": {
                    "weight": np.array([3.0], dtype=np.float32)
                }
            },
            codec_bindings={
                "trained_weights": UpdateCodecBinding(
                    codec_id="safetensors-v1",
                    codec_version=1,
                    logical_schema=schema,
                )
            },
        )
        prepared: list[RoundCommit] = []
        confirmed: list[RoundCommit] = []
        engine = GossipEngine(
            local_public_key="peer-0",
            round_count=1,
            scheduler=PeerScheduler(["peer-0", "peer-1"], seed=8),
            algorithm=_algorithm(
                key="peer-0",
                trainer=LinearTrainer(1.0),
                schema=schema,
                artifact_root=tmp_path / "local",
            ),
            transport=RejectingCommitTransport(peer_bundle),
            commit_callback=prepared.append,
            confirm_callback=confirmed.append,
        )

        with pytest.raises(PairCommitError, match="peer did not confirm"):
            await engine.run()
        assert len(prepared) == 1
        assert confirmed == []

    asyncio.run(run())


@pytest.mark.parametrize("failure", [None, "corruption", "validation"])
def test_release_runs_after_bundle_outcomes(
    tmp_path: Path, failure: str | None
) -> None:
    async def run() -> None:
        schema = TensorSchema(
            tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),)
        )
        local_delegate = NamedSafetensorsUpdateBundleCodec(
            artifact_root=tmp_path / "local",
            run_id="test-run",
            manifest_hash="0" * 64,
            sender_public_key="peer-0",
            algorithm_id="d-psgd",
            artifact_schemas={"trained_weights": schema},
        )
        codec = RecordingBundleCodec(
            local_delegate, validation_error=failure == "validation"
        )
        peer_codec = NamedSafetensorsUpdateBundleCodec(
            artifact_root=tmp_path / "peer",
            run_id="test-run",
            manifest_hash="0" * 64,
            sender_public_key="peer-1",
            algorithm_id="d-psgd",
            artifact_schemas={"trained_weights": schema},
        )
        peer_bundle = peer_codec.encode(
            round_id=0,
            artifacts={
                "trained_weights": {
                    "weight": np.array([3.0], dtype=np.float32)
                }
            },
            codec_bindings={
                "trained_weights": UpdateCodecBinding(
                    codec_id="safetensors-v1",
                    codec_version=1,
                    logical_schema=schema,
                )
            },
        )
        if failure == "corruption":
            with peer_bundle.artifacts[0].path.open("ab") as handle:
                handle.write(b"corrupt")
        engine = GossipEngine(
            local_public_key="peer-0",
            round_count=1,
            scheduler=PeerScheduler(["peer-0", "peer-1"], seed=8),
            algorithm=DPSGDAdapter(
                trainer=LinearTrainer(1.0),
                tensor_schema=schema,
                local_steps=1,
                bundle_codec=codec,
            ),
            transport=StaticPairTransport(peer_bundle),
            commit_callback=lambda commit: None,
        )
        if failure is None:
            await engine.run()
        else:
            with pytest.raises(PairCommitError, match="validation"):
                await engine.run()
        assert len(codec.released) == 2
        assert not any(path.is_file() for path in tmp_path.rglob("*"))

    asyncio.run(run())


def test_pair_readiness_is_exchanged_before_bulk_transfer(tmp_path: Path) -> None:
    async def run() -> None:
        schema = TensorSchema(
            tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),)
        )
        local_codec = NamedSafetensorsUpdateBundleCodec(
            artifact_root=tmp_path / "local",
            run_id="test-run",
            manifest_hash="0" * 64,
            sender_public_key="peer-0",
            algorithm_id="d-psgd",
            artifact_schemas={"trained_weights": schema},
        )
        peer_codec = NamedSafetensorsUpdateBundleCodec(
            artifact_root=tmp_path / "peer",
            run_id="test-run",
            manifest_hash="0" * 64,
            sender_public_key="peer-1",
            algorithm_id="d-psgd",
            artifact_schemas={"trained_weights": schema},
        )
        peer_bundle = peer_codec.encode(
            round_id=0,
            artifacts={
                "trained_weights": {
                    "weight": np.array([3.0], dtype=np.float32)
                }
            },
            codec_bindings={
                "trained_weights": UpdateCodecBinding(
                    codec_id="safetensors-v1",
                    codec_version=1,
                    logical_schema=schema,
                )
            },
        )
        transport = ReadinessGuardTransport(peer_bundle)
        engine = GossipEngine(
            local_public_key="peer-0",
            round_count=1,
            scheduler=PeerScheduler(["peer-0", "peer-1"], seed=8),
            algorithm=DPSGDAdapter(
                trainer=LinearTrainer(1.0),
                tensor_schema=schema,
                local_steps=1,
                bundle_codec=local_codec,
            ),
            transport=transport,
            commit_callback=lambda commit: None,
        )

        await engine.run()

        assert transport.calls == ["ready", "update"]

    asyncio.run(run())


def test_release_runs_after_cancellation(tmp_path: Path) -> None:
    async def run() -> None:
        schema = TensorSchema(
            tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),)
        )
        codec = RecordingBundleCodec(
            NamedSafetensorsUpdateBundleCodec(
                artifact_root=tmp_path / "local",
                run_id="test-run",
                manifest_hash="0" * 64,
                sender_public_key="peer-0",
                algorithm_id="d-psgd",
                artifact_schemas={"trained_weights": schema},
            )
        )
        engine = GossipEngine(
            local_public_key="peer-0",
            round_count=1,
            scheduler=PeerScheduler(["peer-0", "peer-1"], seed=8),
            algorithm=DPSGDAdapter(
                trainer=LinearTrainer(1.0),
                tensor_schema=schema,
                local_steps=1,
                bundle_codec=codec,
            ),
            transport=HangingPairTransport("peer-0", SharedPairChannel.create()),
            commit_callback=lambda commit: None,
        )
        task = asyncio.create_task(engine.run())
        while not any(path.is_file() for path in tmp_path.rglob("*")):
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(codec.released) == 1
        assert not any(path.is_file() for path in tmp_path.rglob("*"))

    asyncio.run(run())


def test_cancellation_waits_for_active_model_mutation(tmp_path: Path) -> None:
    async def run() -> None:
        started = threading.Event()
        release = threading.Event()
        trainer = BlockingTrainer(1.0, started=started, release=release)
        schema = TensorSchema(
            tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),)
        )
        engine = GossipEngine(
            local_public_key="peer-0",
            round_count=1,
            scheduler=PeerScheduler(["peer-0", "peer-1"], seed=8),
            algorithm=_algorithm(
                key="peer-0",
                trainer=trainer,
                schema=schema,
                artifact_root=tmp_path / "local",
            ),
            transport=HangingPairTransport("peer-0", SharedPairChannel.create()),
            commit_callback=lambda commit: None,
        )

        task = asyncio.create_task(engine.run())
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        try:
            await asyncio.sleep(0.01)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert np.array_equal(
            trainer.weights()["weight"], np.array([2.0], dtype=np.float32)
        )

    asyncio.run(run())


def test_engine_publishes_round_timings_without_waiting_for_metric_writer(
    tmp_path: Path,
) -> None:
    schema = TensorSchema(tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),))
    channel = SharedPairChannel.create()
    metrics = RecordingMetricsPublisher([], [])
    evaluations: list[EvaluationMetrics] = []

    async def run() -> None:
        engines: list[GossipEngine] = []
        for key, value in (("peer-0", 1.0), ("peer-1", 3.0)):
            engines.append(
                GossipEngine(
                    local_public_key=key,
                    round_count=1,
                    scheduler=PeerScheduler(["peer-0", "peer-1"], seed=8),
                    algorithm=_algorithm(
                        key=key,
                        trainer=LinearTrainer(value),
                        schema=schema,
                        artifact_root=tmp_path / key,
                    ),
                    transport=TimedPairTransport(key, channel),
                    commit_callback=lambda commit: None,
                    evaluation_callback=evaluations.append,
                    metrics_publisher=metrics,
                )
            )
        await asyncio.gather(*(engine.run() for engine in engines))

    asyncio.run(run())
    assert len(metrics.timings) == 2
    assert all(timing.transfer_id == "transfer-0" for timing in metrics.timings)
    assert all(timing.retries == 2 for timing in metrics.timings)
    assert all(timing.local_loss == 0.25 for timing in metrics.timings)
    assert all(timing.transfer_seconds >= 0 for timing in metrics.timings)
    assert all(timing.peer_wait_seconds >= 0 for timing in metrics.timings)
    assert all(timing.mixing_seconds >= 0 for timing in metrics.timings)
    assert all(timing.evaluation_seconds >= 0 for timing in metrics.timings)
    assert all(timing.evaluation_accuracy == 0.5 for timing in metrics.timings)


def test_observation_failures_never_control_round_commit(tmp_path: Path) -> None:
    schema = TensorSchema(tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),))
    channel = SharedPairChannel.create()
    metrics = RecordingMetricsPublisher([], [])
    commits: dict[str, list[RoundCommit]] = {"peer-0": [], "peer-1": []}
    evaluations: list[EvaluationMetrics] = []

    async def run() -> None:
        engines = [
            GossipEngine(
                local_public_key=key,
                round_count=1,
                scheduler=PeerScheduler(["peer-0", "peer-1"], seed=8),
                algorithm=_algorithm(
                    key=key,
                    trainer=InvalidObservationTrainer(value),
                    schema=schema,
                    artifact_root=tmp_path / key,
                ),
                transport=InMemoryPairTransport(key, channel),
                commit_callback=commits[key].append,
                evaluation_callback=evaluations.append,
                metrics_publisher=metrics,
            )
            for key, value in (("peer-0", 1.0), ("peer-1", 3.0))
        ]
        await asyncio.gather(*(engine.run() for engine in engines))

    asyncio.run(run())

    assert all(len(records) == 1 for records in commits.values())
    assert evaluations == []
    assert len(metrics.timings) == 2
    assert all(timing.local_loss is None for timing in metrics.timings)
    assert all(timing.evaluation_loss is None for timing in metrics.timings)
    assert all(timing.evaluation_accuracy is None for timing in metrics.timings)


def test_two_nodes_complete_pair_commit_without_group_barrier(
    tmp_path: Path,
) -> None:
    schema = TensorSchema(tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),))
    channel = SharedPairChannel.create()
    commits: dict[str, list[RoundCommit]] = {"peer-0": [], "peer-1": []}
    trainers: dict[str, LinearTrainer] = {}
    publisher = RecordingPublisher([])

    async def run() -> None:
        engines: list[GossipEngine] = []
        for key, value in (("peer-0", 1.0), ("peer-1", 3.0)):
            trainer = LinearTrainer(value)
            trainers[key] = trainer
            algorithm = _algorithm(
                key=key,
                trainer=trainer,
                schema=schema,
                artifact_root=tmp_path / key,
            )
            transport = InMemoryPairTransport(key, channel)
            engines.append(
                GossipEngine(
                    local_public_key=key,
                    round_count=1,
                    scheduler=PeerScheduler(["peer-0", "peer-1"], seed=8),
                    algorithm=algorithm,
                    transport=transport,
                    commit_callback=commits[key].append,
                    consensus_publisher=publisher if key == "peer-0" else None,
                )
            )
        await asyncio.gather(*(engine.run() for engine in engines))
        assert all(len(records) == 1 for records in commits.values())
        assert all(
            np.array_equal(trainer.weights()["weight"], np.array([3.0]))
            for trainer in trainers.values()
        )
        assert all(
            len(record.local_bundle_digest) == 64
            and len(record.peer_bundle_digest) == 64
            and len(record.state_checksum) == 64
            for records in commits.values()
            for record in records
        )
        assert all(
            not hasattr(record, snapshot_field)
            for records in commits.values()
            for record in records
            for snapshot_field in ("pre_local", "post_local", "post_mix")
        )
        assert publisher.rounds == [0]

    asyncio.run(run())


def test_evaluation_runs_every_five_rounds_and_on_final_round(
    tmp_path: Path,
) -> None:
    schema = TensorSchema(tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),))
    channel = SharedPairChannel.create()
    evaluations: dict[str, list[EvaluationMetrics]] = {"peer-0": [], "peer-1": []}

    async def run() -> None:
        engines: list[GossipEngine] = []
        for key, value in (("peer-0", 1.0), ("peer-1", 3.0)):
            algorithm = _algorithm(
                key=key,
                trainer=LinearTrainer(value),
                schema=schema,
                artifact_root=tmp_path / key,
            )
            engines.append(
                GossipEngine(
                    local_public_key=key,
                    round_count=6,
                    scheduler=PeerScheduler(["peer-0", "peer-1"], seed=8),
                    algorithm=algorithm,
                    transport=InMemoryPairTransport(key, channel),
                    commit_callback=lambda commit: None,
                    evaluation_callback=evaluations[key].append,
                )
            )
        await asyncio.gather(*(engine.run() for engine in engines))

    asyncio.run(run())
    assert [metric.round_id for metric in evaluations["peer-0"]] == [4, 5]
    assert [metric.round_id for metric in evaluations["peer-1"]] == [4, 5]
    assert all(
        metric.accuracy == 0.5
        for values in evaluations.values()
        for metric in values
    )


def test_evaluation_is_outside_pair_deadline(tmp_path: Path) -> None:
    schema = TensorSchema(tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),))
    channel = SharedPairChannel.create()

    async def run() -> None:
        engines = [
            GossipEngine(
                local_public_key=key,
                round_count=1,
                scheduler=PeerScheduler(["peer-0", "peer-1"], seed=8),
                algorithm=_algorithm(
                    key=key,
                    trainer=SlowEvaluationTrainer(value),
                    schema=schema,
                    artifact_root=tmp_path / key,
                ),
                transport=InMemoryPairTransport(key, channel),
                commit_callback=lambda commit: None,
                timeout_seconds=0.05,
                evaluation_callback=lambda metrics: None,
            )
            for key, value in (("peer-0", 1.0), ("peer-1", 3.0))
        ]
        await asyncio.gather(*(engine.run() for engine in engines))

    asyncio.run(run())


def test_tensor_checksum_is_stable() -> None:
    tensors = {"weight": np.array([2.0], dtype=np.float32)}
    assert (
        checksum_tensors(tensors)
        == "fd734a524f800f74e9b9ac0d0134f90237a95dbc2854a847382d01fed960bfb4"
    )
