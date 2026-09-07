from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from support.in_memory_transport import (
    InMemoryNetwork,
    InMemoryTransport,
)
from support.runtime_fakes import (
    FailingRunStore,
    LifecycleRuntime,
    RecordingInMemoryTransport,
    RuntimeTrainer,
)
from support.sample_manifest import manifest_data, write_checkpoint

from dromeus.algorithms.codec import (
    BitmapTopKInt8Codec,
    DenseInt8Codec,
    TopKInt8Codec,
)
from dromeus.algorithms.dpsgd import DPSGDAdapter
from dromeus.algorithms.noloco import NoLoCoAlgorithm
from dromeus.manifests.canonical import canonical_hash
from dromeus.manifests.models import DraftRunSpec, SealedManifest
from dromeus.membership.formation import (
    FormationError,
    FormationResult,
    create_invitation,
)
from dromeus.persistence.run_store import RunStore
from dromeus.protocol.codec import decode_envelope
from dromeus.protocol.models import (
    MessageType,
)
from dromeus.runtime import (
    FailureConfig,
    InitiatorFormation,
    NodeRunResult,
    NodeRuntime,
    NodeState,
    ParticipantFormation,
    TrainingConfig,
    build_algorithm,
)


def test_runtime_lifecycle_fails_ready_hook_before_run(tmp_path: Path) -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    result = FormationResult(
        manifest=manifest,
        manifest_hash=canonical_hash(manifest),
        checkpoint_path=tmp_path / "checkpoint.safetensors",
    )
    runtime = LifecycleRuntime(result)
    trainer = RuntimeTrainer()
    training = TrainingConfig(
        algorithm=DPSGDAdapter(
            trainer=trainer,
            tensor_schema=manifest.tensor_schema,
            local_steps=1,
        ),
        load_checkpoint=trainer.load_checkpoint,
        run_store=RunStore(tmp_path / "run"),
        artifact_root=tmp_path / "rounds",
    )

    def build_training(_: FormationResult) -> TrainingConfig:
        runtime.events.append("factory")
        return training

    async def reject_ready(_: FormationResult) -> None:
        runtime.events.append("ready")
        raise RuntimeError("ready rejected")

    with pytest.raises(RuntimeError, match="ready rejected"):
        asyncio.run(
            runtime.run_to_completion(
                formation=InitiatorFormation(
                    bootstrap_uri="axl://bootstrap",
                    checkpoint_path=result.checkpoint_path,
                    tensor_schema=manifest.tensor_schema,
                ),
                training_factory=build_training,
                ready_hook=reject_ready,
            )
        )

    assert runtime.events == [
        "initiate",
        "factory",
        "configure",
        "ready",
        "fail:ready rejected",
        "stop",
    ]


def test_runtime_lifecycle_does_not_fail_after_completion_hook(
    tmp_path: Path,
) -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    result = FormationResult(
        manifest=manifest,
        manifest_hash=canonical_hash(manifest),
        checkpoint_path=tmp_path / "checkpoint.safetensors",
    )
    runtime = LifecycleRuntime(result)
    trainer = RuntimeTrainer()
    training = TrainingConfig(
        algorithm=DPSGDAdapter(
            trainer=trainer,
            tensor_schema=manifest.tensor_schema,
            local_steps=1,
        ),
        load_checkpoint=trainer.load_checkpoint,
        run_store=RunStore(tmp_path / "run"),
        artifact_root=tmp_path / "rounds",
    )

    def build_training(_: FormationResult) -> TrainingConfig:
        runtime.events.append("factory")
        return training

    async def reject_completion(_: NodeRunResult) -> None:
        runtime.events.append("complete")
        raise RuntimeError("completion evidence unavailable")

    with pytest.raises(RuntimeError, match="completion evidence unavailable"):
        asyncio.run(
            runtime.run_to_completion(
                formation=InitiatorFormation(
                    bootstrap_uri="axl://bootstrap",
                    checkpoint_path=result.checkpoint_path,
                    tensor_schema=manifest.tensor_schema,
                ),
                training_factory=build_training,
                completion_hook=reject_completion,
            )
        )

    assert runtime.events == [
        "initiate",
        "factory",
        "configure",
        "run",
        "complete",
        "stop",
    ]


def test_runtime_builds_noloco_from_sealed_manifest() -> None:
    data = manifest_data()
    data.update(
        {
            "algorithm_id": "noloco",
            "optimizer": "adam",
            "algorithm_config": {
                "alpha": 0.5,
                "beta": 0.7,
                "gamma": 0.7,
                "inner_steps": 50,
                "adam": {
                    "learning_rate": 0.001,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "epsilon": 1e-8,
                    "gradient_clip_norm": 1.0,
                },
            },
            "artifact_codecs": [
                {"artifact_name": "outer_gradient", "codec_id": "identity-v1"},
                {"artifact_name": "slow_weights", "codec_id": "identity-v1"},
            ],
            "transport": {
                **data["transport"],
                "chunk_size_bytes": 1024,
                "window_size": 1,
            },
        }
    )
    draft_data = data.copy()
    for field in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del draft_data[field]
    data["draft_hash"] = canonical_hash(DraftRunSpec.model_validate(draft_data))
    manifest = SealedManifest.model_validate(data)

    algorithm = build_algorithm(manifest=manifest, trainer=RuntimeTrainer())

    assert isinstance(algorithm, NoLoCoAlgorithm)
    assert {
        name: codec.codec_id for name, codec in algorithm.artifact_codecs.items()
    } == {
        "outer_gradient": "identity-v1",
        "slow_weights": "identity-v1",
    }


@pytest.mark.parametrize(
    ("codec_id", "codec_type", "fraction"),
    (
        ("topk-int8-v1", TopKInt8Codec, 0.01),
        ("topk-bitmap-int8-v2", BitmapTopKInt8Codec, 0.4),
    ),
)
def test_runtime_builds_compressed_noloco_codecs_from_manifest(
    codec_id: str,
    codec_type: type[TopKInt8Codec] | type[BitmapTopKInt8Codec],
    fraction: float,
) -> None:
    data = manifest_data()
    data.update(
        {
            "algorithm_id": "noloco",
            "optimizer": "adam",
            "algorithm_config": {
                "alpha": 0.5,
                "beta": 0.7,
                "gamma": 0.7,
                "inner_steps": 50,
                "adam": {
                    "learning_rate": 0.001,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "epsilon": 1e-8,
                    "gradient_clip_norm": 1.0,
                },
            },
            "artifact_codecs": [
                {
                    "artifact_name": "outer_gradient",
                    "codec_id": codec_id,
                    "top_k_fraction": fraction,
                    "lossy_allowed": True,
                },
                {
                    "artifact_name": "slow_weights",
                    "codec_id": "dense-int8-v1",
                    "lossy_allowed": True,
                },
            ],
            "transport": {
                **data["transport"],
                "chunk_size_bytes": 1024,
                "window_size": 1,
            },
        }
    )
    draft_data = data.copy()
    for field in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del draft_data[field]
    data["draft_hash"] = canonical_hash(DraftRunSpec.model_validate(draft_data))
    manifest = SealedManifest.model_validate(data)

    algorithm = build_algorithm(manifest=manifest, trainer=RuntimeTrainer())

    assert isinstance(algorithm, NoLoCoAlgorithm)
    assert isinstance(algorithm.artifact_codecs["outer_gradient"], codec_type)
    assert isinstance(algorithm.artifact_codecs["slow_weights"], DenseInt8Codec)
    outer = algorithm.artifact_codecs["outer_gradient"]
    assert isinstance(outer, codec_type)
    assert outer.top_k_fraction == fraction


def test_runtime_runs_training_after_in_memory_formation(tmp_path: Path) -> None:
    asyncio.run(_test_runtime_runs_training_after_in_memory_formation(tmp_path))


async def _test_runtime_runs_training_after_in_memory_formation(
    tmp_path: Path,
) -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    draft_data = manifest.model_dump(mode="python")
    for field in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del draft_data[field]
    draft_data["round_count"] = 2
    draft_data["local_steps"] = 1
    draft_data["transport"]["max_retries"] = 1
    draft_data["transport"]["retry_timeout_seconds"] = 0.05
    draft = DraftRunSpec.model_validate(draft_data)
    network = InMemoryNetwork()
    transports = [
        InMemoryTransport(network=network, public_key=f"peer-{index}")
        for index in range(5)
    ]
    checkpoint = tmp_path / "checkpoint.safetensors"
    write_checkpoint(checkpoint)
    nodes: list[NodeRuntime] = []
    training_configs: list[TrainingConfig] = []
    for index, transport in enumerate(transports):
        training_trainer = RuntimeTrainer()
        training_configs.append(
            TrainingConfig(
                algorithm=DPSGDAdapter(
                    trainer=training_trainer,
                    tensor_schema=manifest.tensor_schema,
                    local_steps=1,
                ),
                load_checkpoint=training_trainer.load_checkpoint,
                run_store=RunStore(tmp_path / f"run-{index}"),
                artifact_root=tmp_path / f"rounds-{index}",
            )
        )
        nodes.append(
            NodeRuntime(
                transport=transport,
                draft=draft,
                environment=manifest.environment,
                dataset=manifest.dataset,
                artifact_root=tmp_path / f"artifacts-{index}",
                failure=FailureConfig.for_run_root(tmp_path / f"failure-{index}"),
            )
        )
    invitation = create_invitation(
        draft=draft,
        initiator_public_key=await transports[0].local_public_key(),
        bootstrap_uri="axl://bootstrap",
    )
    formations = [
        InitiatorFormation(
            bootstrap_uri="axl://bootstrap",
            checkpoint_path=checkpoint,
            tensor_schema=manifest.tensor_schema,
        ),
        *(ParticipantFormation(invitation=invitation) for _ in nodes[1:]),
    ]
    lifecycle_events: list[list[str]] = [[] for _ in nodes]

    async def run_lifecycle(index: int) -> NodeRunResult:
        def build_training(_: FormationResult) -> TrainingConfig:
            lifecycle_events[index].append("training")
            return training_configs[index]

        async def ready(_: FormationResult) -> None:
            lifecycle_events[index].append("ready")

        async def complete(_: NodeRunResult) -> None:
            lifecycle_events[index].append("complete")

        return await nodes[index].run_to_completion(
            formation=formations[index],
            training_factory=build_training,
            ready_hook=ready,
            completion_hook=complete,
        )

    outcomes = await asyncio.wait_for(
        asyncio.gather(
            *(run_lifecycle(index) for index in range(len(nodes))),
            return_exceptions=True,
        ),
        timeout=5.0,
    )
    assert sum(isinstance(outcome, NodeRunResult) for outcome in outcomes) == 4
    assert sum(isinstance(outcome, FormationError) for outcome in outcomes) == 1
    assert all(
        isinstance(outcome, (NodeRunResult, FormationError)) for outcome in outcomes
    )
    assert all(
        len(outcome.commits) == 2
        for outcome in outcomes
        if isinstance(outcome, NodeRunResult)
    )
    assert all(node.state is NodeState.STOPPED for node in nodes)
    successful_indices = [
        index
        for index, outcome in enumerate(outcomes)
        if isinstance(outcome, NodeRunResult)
    ]
    for index in successful_indices:
        assert lifecycle_events[index] == ["training", "ready", "complete"]
        state = RunStore(tmp_path / f"run-{index}").load_state()
        assert state.committed_round == 1
        assert len(state.metrics) == 2
        assert all(record["local_loss"] == 0.25 for record in state.metrics)
        assert len(state.transfer_diagnostics) == 2
        assert all(
            isinstance(record["transfer_id"], str)
            and bool(record["transfer_id"])
            and isinstance(record["retries"], int)
            and int(record["retries"]) >= 0
            for record in state.transfer_diagnostics
        )
        assert len(state.consensus) == 2
        assert [record.round_id for record in state.consensus] == [0, 1]
        assert all(record.sketch_count == 4 for record in state.consensus)
        assert state.terminal is not None
        assert state.terminal.result == "complete"
        assert state.terminal.diagnostics == {"committed_rounds": 2}


def test_runtime_persists_and_broadcasts_failure_before_run(tmp_path: Path) -> None:
    asyncio.run(
        _test_runtime_persists_and_broadcasts_failure_before_run(
            tmp_path,
            persistence_fails=False,
        )
    )


def test_runtime_broadcasts_when_failure_persistence_fails(tmp_path: Path) -> None:
    asyncio.run(
        _test_runtime_persists_and_broadcasts_failure_before_run(
            tmp_path,
            persistence_fails=True,
        )
    )


async def _test_runtime_persists_and_broadcasts_failure_before_run(
    tmp_path: Path,
    *,
    persistence_fails: bool,
) -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    draft_data = manifest.model_dump(mode="python")
    for field in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del draft_data[field]
    draft_data["transport"]["max_retries"] = 1
    draft_data["transport"]["retry_timeout_seconds"] = 0.05
    draft = DraftRunSpec.model_validate(draft_data)
    network = InMemoryNetwork()
    transports = [
        RecordingInMemoryTransport(network=network, public_key=f"peer-{index}")
        for index in range(4)
    ]
    blocked_artifact_parent = tmp_path / "blocked-artifact-parent"
    blocked_artifact_parent.write_text("not a directory", encoding="utf-8")
    nodes: list[NodeRuntime] = []
    for index, transport in enumerate(transports):
        trainer = RuntimeTrainer()
        run_store = (
            FailingRunStore(tmp_path / f"run-{index}")
            if persistence_fails and index == 0
            else RunStore(tmp_path / f"run-{index}")
        )
        training = (
            None
            if index == 0
            else TrainingConfig(
                algorithm=DPSGDAdapter(
                    trainer=trainer,
                    tensor_schema=manifest.tensor_schema,
                    local_steps=1,
                ),
                load_checkpoint=trainer.load_checkpoint,
                run_store=run_store,
                artifact_root=tmp_path / f"rounds-{index}",
            )
        )
        nodes.append(
            NodeRuntime(
                transport=transport,
                draft=draft,
                environment=manifest.environment,
                dataset=manifest.dataset,
                artifact_root=tmp_path / f"artifacts-{index}",
                training=training,
                failure=FailureConfig(
                    run_store=run_store,
                    artifact_root=(
                        blocked_artifact_parent / "rounds"
                        if persistence_fails and index == 0
                        else tmp_path / f"rounds-{index}"
                    ),
                ),
            )
        )
    checkpoint = tmp_path / "checkpoint.safetensors"
    write_checkpoint(checkpoint)
    invitation = create_invitation(
        draft=draft,
        initiator_public_key=await transports[0].local_public_key(),
        bootstrap_uri="axl://bootstrap",
    )
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                nodes[0].initiate(
                    bootstrap_uri="axl://bootstrap",
                    checkpoint_path=checkpoint,
                    tensor_schema=manifest.tensor_schema,
                ),
                *(node.join(invitation=invitation) for node in nodes[1:]),
            ),
            timeout=2.0,
        )
        assert len(results) == 4
        transports[0].sent_payloads.clear()

        if persistence_fails:
            with pytest.raises(OSError, match="run store unavailable"):
                await nodes[0].fail_before_run(RuntimeError("topology unavailable"))
        else:
            await nodes[0].fail_before_run(RuntimeError("topology unavailable"))
        for _ in range(100):
            if all(node.state is NodeState.FAILED for node in nodes):
                break
            await asyncio.sleep(0.01)

        assert all(node.state is NodeState.FAILED for node in nodes)
        if not persistence_fails:
            state = RunStore(tmp_path / "run-0").load_state()
            assert state.committed_round == -1
            assert state.terminal is not None
            assert state.terminal.result == "failed"
            assert state.terminal.diagnostics == {
                "error_type": "RuntimeError",
                "error": "topology unavailable",
            }
        participant_keys = frozenset(f"peer-{index}" for index in range(4))
        sent_envelopes = [
            decode_envelope(
                payload,
                authenticated_sender="peer-0",
                participant_keys=participant_keys,
            )
            for _, payload in transports[0].sent_payloads
        ]
        failure_envelopes = [
            envelope
            for envelope in sent_envelopes
            if envelope.message_type is MessageType.RUN_FAILED
        ]
        assert len(failure_envelopes) == 3
        assert {envelope.round_id for envelope in failure_envelopes} == {0}
        for index in range(1, 4):
            peer_state = RunStore(tmp_path / f"run-{index}").load_state()
            assert peer_state.terminal is not None
            assert peer_state.terminal.result == "failed"
            assert peer_state.terminal.diagnostics == {
                "error_type": "PeerRunFailureError",
                "error": (
                    "peer peer-0 failed at round 0: RuntimeError: topology unavailable"
                ),
            }
    finally:
        await asyncio.gather(*(node.stop() for node in nodes))
