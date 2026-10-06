"""Exercise detector composition through four real runtime lifecycles."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Mapping
from pathlib import Path

import pytest
from support.in_memory_transport import InMemoryNetwork, InMemoryTransport
from support.runtime_fakes import RuntimeTrainer
from support.sample_manifest import manifest_data, write_checkpoint

from dromeus.algorithms.dpsgd import DPSGDAdapter
from dromeus.manifests.models import DraftRunSpec, SealedManifest
from dromeus.membership.formation import FormationResult, create_invitation
from dromeus.persistence.run_store import RunStore
from dromeus.protocol.codec import decode_envelope
from dromeus.protocol.models import MessageType
from dromeus.runtime import (
    FailureConfig,
    InitiatorFormation,
    NodeRuntime,
    ParticipantFormation,
    TrainingConfig,
)
from dromeus.telemetry.divergence import DetectorState
from dromeus.telemetry.events import JsonlEventSink, emit_event
from dromeus.telemetry.evidence import (
    ConsensusObservationEvidence,
    DivergenceStatusEvidence,
    EvidenceLog,
)


@pytest.mark.parametrize("delivery", ["normal", "missing", "blocked", "failed"])
def test_warning_delivery_never_stops_four_runtime_nodes(
    tmp_path: Path, delivery: str
) -> None:
    release = threading.Event()
    sinks: list[JsonlEventSink] = []

    async def run() -> None:
        manifest = SealedManifest.model_validate(manifest_data())
        values = manifest.model_dump(
            mode="python", include=set(DraftRunSpec.model_fields)
        )
        values.update(
            manifest_version=5,
            round_count=8,
            local_steps=1,
            divergence_detection=dict(
                threshold_set_id="runtime-synthetic-v1",
                warmup_rounds=0,
                window_rounds=4,
                patience_windows=2,
                distance_floor=0.01,
                growth_ratio=2,
                min_log_slope=0.1,
                severe_distance=2,
                severe_patience=2,
                recovery_windows=2,
                recovery_distance=0.3,
                max_lag_rounds=16 if delivery != "missing" else 1,
            ),
        )
        draft = DraftRunSpec.model_validate(values)
        network = InMemoryNetwork()

        class Transport(InMemoryTransport):
            async def send(self, destination: str, payload: bytes) -> None:
                if (
                    delivery == "missing"
                    and decode_envelope(
                        payload,
                        authenticated_sender=await self.local_public_key(),
                        participant_keys=None,
                    ).message_type
                    == MessageType.CONSENSUS_SKETCH
                ):
                    return
                await super().send(destination, payload)

        class Sink:
            def __init__(self, target: JsonlEventSink) -> None:
                self.target = target
                self.lock = threading.Lock()

            def append(self, record: Mapping[str, object]) -> None:
                with self.lock:
                    if record.get("event") in {
                        "consensus_observation",
                        "divergence_status",
                    }:
                        if delivery == "blocked":
                            release.wait(8)
                        if delivery == "failed":
                            raise OSError("sink unavailable")
                    self.target.append(record)

        checkpoint = tmp_path / "initial.safetensors"
        write_checkpoint(checkpoint)
        nodes: list[NodeRuntime] = []
        configs: list[TrainingConfig] = []
        for rank in range(4):
            trainer = RuntimeTrainer()
            transport = Transport(network=network, public_key=f"peer-{rank}")
            sink = JsonlEventSink(tmp_path / f"node-{rank}.jsonl")
            sinks.append(sink)
            nodes.append(
                NodeRuntime(
                    transport=transport,
                    draft=draft,
                    environment=draft.environment,
                    dataset=draft.dataset,
                    artifact_root=tmp_path / f"formation-{rank}",
                    event_sink=Sink(sink),
                    failure=FailureConfig.for_run_root(tmp_path / f"failure-{rank}"),
                )
            )
            configs.append(
                TrainingConfig(
                    algorithm=DPSGDAdapter(
                        trainer=trainer,
                        tensor_schema=manifest.tensor_schema,
                        local_steps=1,
                    ),
                    load_checkpoint=trainer.load_checkpoint,
                    run_store=RunStore(tmp_path / f"store-{rank}"),
                    artifact_root=tmp_path / f"rounds-{rank}",
                )
            )
        invitation = create_invitation(
            draft=draft, initiator_public_key="peer-0", bootstrap_uri="axl://test"
        )

        async def node(rank: int) -> None:
            def training(_: FormationResult) -> TrainingConfig:
                return configs[rank]

            def complete(_: object) -> None:
                emit_event("node_complete", sink=nodes[rank].event_sink)

            outcome = await nodes[rank].run_to_completion(
                formation=InitiatorFormation(
                    bootstrap_uri="axl://test",
                    checkpoint_path=checkpoint,
                    tensor_schema=manifest.tensor_schema,
                )
                if rank == 0
                else ParticipantFormation(invitation=invitation),
                training_factory=training,
                completion_hook=complete,
            )
            assert len(outcome.commits) == 8
            state = nodes[rank].divergence_status
            assert state is not None and state.detection_round == 7
            if delivery == "missing":
                assert state.state == DetectorState.INSUFFICIENT_DATA
                assert state.missing_rounds > 0
            else:
                assert state.state == DetectorState.HEALTHY
            if delivery in {"blocked", "failed"}:
                assert nodes[rank].divergence_dropped > 0
            if delivery == "normal":
                assert any(
                    json.loads(line)["event"] == "node_complete"
                    for line in sinks[rank].path.read_text().splitlines()
                )
                log = EvidenceLog.open(
                    sinks[rank].path,
                    run_id=draft.run_id,
                    manifest_hash=outcome.formation.manifest_hash,
                )
                assert (
                    len(
                        [
                            x
                            for x in log.records
                            if isinstance(x, ConsensusObservationEvidence)
                        ]
                    )
                    == 8
                )
                changes = [
                    x for x in log.records if isinstance(x, DivergenceStatusEvidence)
                ]
                assert len(changes) == 1 and changes[0].state == "healthy"

        started = asyncio.get_running_loop().time()
        await asyncio.wait_for(asyncio.gather(*(node(i) for i in range(4))), timeout=12)

        if delivery == "blocked":
            assert asyncio.get_running_loop().time() - started < 6

    try:
        asyncio.run(run())
    finally:
        release.set()


def test_created_runtime_stop_closes_event_delivery(tmp_path: Path) -> None:
    from dromeus.telemetry.events import BoundedEventSink

    async def run() -> None:
        values = manifest_data()
        for k in (
            "participants",
            "draft_hash",
            "initial_checkpoint_hash",
            "tensor_schema",
        ):
            del values[k]
        values.update(
            manifest_version=5,
            divergence_detection=dict(
                threshold_set_id="stop-test",
                warmup_rounds=0,
                window_rounds=4,
                patience_windows=2,
                distance_floor=0.01,
                growth_ratio=2,
                min_log_slope=0.1,
                severe_distance=2,
                severe_patience=2,
                recovery_windows=2,
                recovery_distance=0.3,
                max_lag_rounds=1,
            ),
        )
        draft = DraftRunSpec.model_validate(values)
        runtime = NodeRuntime(
            transport=InMemoryTransport(network=InMemoryNetwork(), public_key="peer-0"),
            draft=draft,
            environment=draft.environment,
            dataset=draft.dataset,
            artifact_root=tmp_path,
        )
        sink = runtime.event_sink
        assert isinstance(sink, BoundedEventSink)
        await runtime.stop()
        before = sink.dropped
        sink.append({"event": "after_stop"})
        assert sink.dropped == before + 1

    asyncio.run(run())
