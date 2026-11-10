"""Opt-in final telemetry coordination through the complete NodeRuntime lifecycle."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from support.in_memory_transport import InMemoryNetwork, InMemoryTransport
from support.runtime_fakes import RuntimeTrainer
from support.sample_manifest import manifest_data, write_checkpoint

from dromeus.manifests.models import DraftRunSpec, SealedManifest
from dromeus.membership.formation import FormationResult, create_invitation
from dromeus.persistence.run_store import RunStore
from dromeus.protocol.codec import decode_envelope
from dromeus.protocol.models import MessageType
from dromeus.runtime import (
    ConsensusCompletionPolicy,
    ConsensusCompletionReceipt,
    FailureConfig,
    InitiatorFormation,
    NodeRunResult,
    NodeRuntime,
    ParticipantFormation,
    TrainingConfig,
    build_algorithm,
)
from dromeus.transport.interface import ReceivedBytes


def test_run_waits_collective_final_receipts_before_terminal_and_transport_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def exercise() -> None:
        manifest = SealedManifest.model_validate(manifest_data())
        data = manifest.model_dump(mode="python")
        for name in (
            "draft_hash",
            "participants",
            "initial_checkpoint_hash",
            "tensor_schema",
        ):
            del data[name]
        data.update(round_count=1)
        data["transport"].update(max_retries=1, retry_timeout_seconds=0.05)
        draft = DraftRunSpec.model_validate(data)
        network = InMemoryNetwork()
        keys = tuple(f"peer-{i}" for i in range(4))
        received: set[tuple[str, str]] = set()
        receipts: dict[str, ConsensusCompletionReceipt] = {}
        reports: set[str] = set()
        complete = asyncio.Event()
        cleanup_records: list[tuple[int, int]] = []

        class ObservedTransport(InMemoryTransport):
            async def recv(self, timeout_seconds: float) -> ReceivedBytes | None:
                value = await super().recv(timeout_seconds)
                if value is not None:
                    envelope = decode_envelope(
                        value.payload,
                        authenticated_sender=value.sender_public_key,
                        participant_keys=frozenset(keys),
                    )
                    if envelope.message_type is MessageType.CONSENSUS_SKETCH:
                        received.add(
                            (envelope.sender_public_key, await self.local_public_key())
                        )
                return value

        old_report = cast(Any, NodeRuntime)._record_consensus

        async def report(node: Any, distance: Any) -> None:
            await asyncio.sleep(
                0.02
            )  # report completion is distinct from buffer completion
            await old_report(node, distance)
            reports.add(node._local_public_key)

        monkeypatch.setattr(NodeRuntime, "_record_consensus", report)
        old_cleanup = cast(Any, NodeRuntime)._cleanup

        async def cleanup(node: Any) -> None:
            cleanup_records.append((len(receipts), len(received)))
            await old_cleanup(node)

        monkeypatch.setattr(NodeRuntime, "_cleanup", cleanup)

        async def barrier(receipt: ConsensusCompletionReceipt, deadline: float) -> None:
            assert receipt.participant_keys == receipt.received_member_keys == keys
            assert receipt.report_completed and not receipt.dropped
            assert receipt.node_public_key in reports
            # This executes inside run(), before terminal completion is persisted.
            rank = keys.index(receipt.node_public_key)
            assert RunStore(tmp_path / f"run-{rank}").load_state().terminal is None
            receipts[receipt.node_public_key] = receipt
            if len(receipts) == 4:
                complete.set()
            async with asyncio.timeout_at(deadline):
                await complete.wait()

        transports = [ObservedTransport(network=network, public_key=k) for k in keys]
        nodes = [
            NodeRuntime(
                transport=t,
                draft=draft,
                environment=manifest.environment,
                dataset=manifest.dataset,
                artifact_root=tmp_path / f"artifacts-{i}",
                failure=FailureConfig.for_run_root(tmp_path / f"failure-{i}"),
            )
            for i, t in enumerate(transports)
        ]
        checkpoint = tmp_path / "checkpoint.safetensors"
        write_checkpoint(checkpoint)
        invitation = create_invitation(
            draft=draft, initiator_public_key=keys[0], bootstrap_uri="axl://bootstrap"
        )
        formations = [
            InitiatorFormation(
                bootstrap_uri="axl://bootstrap",
                checkpoint_path=checkpoint,
                tensor_schema=manifest.tensor_schema,
            ),
            *(ParticipantFormation(invitation=invitation) for _ in range(3)),
        ]

        async def run(i: int) -> NodeRunResult:
            trainer = RuntimeTrainer()

            def build(_formed: FormationResult) -> TrainingConfig:
                return TrainingConfig(
                    algorithm=build_algorithm(manifest=manifest, trainer=trainer),
                    load_checkpoint=trainer.load_checkpoint,
                    run_store=RunStore(tmp_path / f"run-{i}"),
                    artifact_root=tmp_path / f"rounds-{i}",
                    consensus_completion=ConsensusCompletionPolicy(2.0, barrier),
                )

            return await nodes[i].run_to_completion(
                formation=formations[i], training_factory=build
            )

        results = await asyncio.wait_for(asyncio.gather(*(run(i) for i in range(4))), 5)
        assert len(results) == 4 and all(len(v.commits) == 1 for v in results)
        assert len(received) == 12 and len(receipts) == len(reports) == 4
        assert cleanup_records == [(4, 12)] * 4

    asyncio.run(exercise())
