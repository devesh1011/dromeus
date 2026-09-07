from __future__ import annotations

import asyncio
from pathlib import Path

from support.gossip_fakes import (
    ConvexTrainer,
    DivergentPairTransport,
    InMemoryPairTransport,
    LinearTrainer,
    NoLoCoConvexTrainer,
    SharedPairChannel,
)
from support.gossip_fakes import make_algorithm as _algorithm

from dromeus.algorithms.codec import (
    DenseInt8Codec,
    NamedSafetensorsUpdateBundleCodec,
    TopKInt8Codec,
)
from dromeus.algorithms.noloco import NoLoCoAlgorithm
from dromeus.gossip.engine import GossipEngine
from dromeus.gossip.interfaces import (
    EvaluationMetrics,
    RoundCommit,
)
from dromeus.gossip.peer_scheduler import PeerScheduler
from dromeus.manifests.models import (
    AdamSettings,
    NoLoCoConfig,
    Tensor,
    TensorSchema,
)


def test_four_in_memory_nodes_reduce_a_shared_convex_objective(
    tmp_path: Path,
) -> None:
    schema = TensorSchema(tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),))
    channel = SharedPairChannel.create()
    trainers = {
        f"peer-{index}": ConvexTrainer(float(8 - index * 2)) for index in range(4)
    }

    async def run() -> None:
        engines = [
            GossipEngine(
                local_public_key=key,
                round_count=3,
                scheduler=PeerScheduler(list(trainers), seed=8),
                algorithm=_algorithm(
                    key=key,
                    trainer=trainer,
                    schema=schema,
                    artifact_root=tmp_path / key,
                ),
                transport=InMemoryPairTransport(key, channel),
                commit_callback=lambda commit: None,
            )
            for key, trainer in trainers.items()
        ]
        await asyncio.gather(*(engine.run() for engine in engines))

    asyncio.run(run())
    assert all(
        abs(float(trainer.weights()["weight"][0])) < 8.0
        for trainer in trainers.values()
    )


def test_four_noloco_nodes_reduce_toy_convex_objective_with_divergent_states(
    tmp_path: Path,
) -> None:
    schema = TensorSchema(tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),))
    config = NoLoCoConfig(
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
    channel = SharedPairChannel.create()
    trainers = {
        f"peer-{index}": NoLoCoConvexTrainer(float(8 - index * 2))
        for index in range(4)
    }
    initial_objective = sum(
        float(trainer.weights()["weight"][0]) ** 2 / 2
        for trainer in trainers.values()
    )
    commits: dict[str, list[RoundCommit]] = {key: [] for key in trainers}
    evaluations: dict[str, list[EvaluationMetrics]] = {key: [] for key in trainers}

    async def run() -> None:
        engines = [
            GossipEngine(
                local_public_key=key,
                round_count=3,
                scheduler=PeerScheduler(list(trainers), seed=8),
                algorithm=NoLoCoAlgorithm(
                    trainer=trainer,
                    tensor_schema=schema,
                    config=config,
                    bundle_codec=NamedSafetensorsUpdateBundleCodec(
                        artifact_root=tmp_path / key,
                        run_id="test-run",
                        manifest_hash="0" * 64,
                        sender_public_key=key,
                        algorithm_id="noloco",
                        artifact_schemas={
                            "outer_gradient": schema,
                            "slow_weights": schema,
                        },
                    ),
                ),
                transport=DivergentPairTransport(key, channel),
                commit_callback=commits[key].append,
                evaluation_callback=evaluations[key].append,
            )
            for key, trainer in trainers.items()
        ]
        await asyncio.gather(*(engine.run() for engine in engines))

    asyncio.run(run())
    final_objective = sum(
        float(trainer.weights()["weight"][0]) ** 2 / 2
        for trainer in trainers.values()
    )

    assert final_objective < initial_objective
    assert all(len(records) == 3 for records in commits.values())
    assert len({records[-1].state_checksum for records in commits.values()}) > 1
    assert all(
        [metric.round_id for metric in records] == [2]
        for records in evaluations.values()
    )
    assert all(
        metric.accuracy == 0.5
        for records in evaluations.values()
        for metric in records
    )


def test_four_noloco_nodes_compressed_smoke_records_error_feedback(
    tmp_path: Path,
) -> None:
    schema = TensorSchema(tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),))
    config = NoLoCoConfig(
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
    channel = SharedPairChannel.create()
    trainers = {
        f"peer-{index}": NoLoCoConvexTrainer(float(8 - index * 2))
        for index in range(4)
    }
    commits: dict[str, list[RoundCommit]] = {key: [] for key in trainers}

    async def run() -> None:
        engines: list[GossipEngine] = []
        for key, trainer in trainers.items():
            outer_codec = TopKInt8Codec(schema, top_k_fraction=0.01)
            slow_codec = DenseInt8Codec(schema)
            engines.append(
                GossipEngine(
                    local_public_key=key,
                    round_count=2,
                    scheduler=PeerScheduler(list(trainers), seed=8),
                    algorithm=NoLoCoAlgorithm(
                        trainer=trainer,
                        tensor_schema=schema,
                        config=config,
                        artifact_codecs={
                            "outer_gradient": outer_codec,
                            "slow_weights": slow_codec,
                        },
                        manifest_codec_ids={
                            "outer_gradient": outer_codec.codec_id,
                            "slow_weights": slow_codec.codec_id,
                        },
                        bundle_codec=NamedSafetensorsUpdateBundleCodec(
                            artifact_root=tmp_path / key,
                            run_id="compressed-smoke",
                            manifest_hash="0" * 64,
                            sender_public_key=key,
                            algorithm_id="noloco",
                            artifact_schemas={
                                "outer_gradient": outer_codec.encoded_schema,
                                "slow_weights": slow_codec.encoded_schema,
                            },
                        ),
                    ),
                    transport=DivergentPairTransport(key, channel),
                    commit_callback=commits[key].append,
                )
            )
        await asyncio.gather(*(engine.run() for engine in engines))

    asyncio.run(run())

    assert all(len(records) == 2 for records in commits.values())
    assert all(
        record.error_feedback_residual_l2_norm is not None
        and record.error_feedback_signal_l2_norm is not None
        and record.error_feedback_residual_to_signal_ratio is not None
        and record.error_feedback_residual_l2_norm >= 0
        and record.error_feedback_signal_l2_norm >= 0
        and record.error_feedback_residual_to_signal_ratio >= 0
        for records in commits.values()
        for record in records
    )
    assert not any(tmp_path.rglob("*.safetensors"))


def test_two_final_consensus_stages_exactly_average_four_nodes(
    tmp_path: Path,
) -> None:
    schema = TensorSchema(tensors=(Tensor(name="weight", dtype="float32", shape=(1,)),))
    channel = SharedPairChannel.create()
    training_calls = {f"peer-{index}": 0 for index in range(4)}

    class CountingTrainer(LinearTrainer):
        def __init__(self, key: str, value: float) -> None:
            super().__init__(value)
            self._key = key

        def train_local_steps(self, step_count: int) -> None:
            training_calls[self._key] += 1
            super().train_local_steps(step_count)

    trainers = {
        key: CountingTrainer(key, float(index * 2))
        for index, key in enumerate(training_calls)
    }

    async def run() -> None:
        engines = [
            GossipEngine(
                local_public_key=key,
                round_count=3,
                scheduler=PeerScheduler(
                    list(trainers),
                    seed=8,
                    training_round_count=1,
                    final_consensus_rounds=2,
                ),
                algorithm=_algorithm(
                    key=key,
                    trainer=trainer,
                    schema=schema,
                    artifact_root=tmp_path / key,
                    training_round_count=1,
                ),
                transport=InMemoryPairTransport(key, channel),
                commit_callback=lambda commit: None,
            )
            for key, trainer in trainers.items()
        ]
        await asyncio.gather(*(engine.run() for engine in engines))
        assert all(len(engine.commits) == 3 for engine in engines)
        assert all(engine.commits[-1].phase == "final-consensus" for engine in engines)

    asyncio.run(run())
    assert training_calls == {key: 1 for key in trainers}
    assert {
        float(trainer.weights()["weight"][0]) for trainer in trainers.values()
    } == {4.0}
