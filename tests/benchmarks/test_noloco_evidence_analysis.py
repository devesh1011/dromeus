from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.noloco.evidence_analysis import (
    encoded_artifact_bytes_from_metadata_roots,
    reconcile_official_evidence,
)
from dromeus.manifests.canonical import materialize_bundle_metadata
from dromeus.manifests.models import (
    OpaqueArtifactMetadata,
    OpaqueUpdateBundleMetadata,
)
from dromeus.telemetry.evidence import (
    ConsensusSketchSentEvidence,
    EvidenceLog,
    EvidenceRecord,
    RoundMetricsEvidence,
    TransferMessageSentEvidence,
)


def _log(*, node_index: int, world_size: int, round_count: int) -> EvidenceLog:
    node_id = f"peer-{node_index}"
    records: list[EvidenceRecord] = []
    for round_id in range(round_count):
        records.extend(
            (
                RoundMetricsEvidence(
                    run_id="run-1",
                    manifest_hash="a" * 64,
                    node_id=node_id,
                    message_id=f"metric-{node_index}-{round_id}",
                    transfer_id=f"transfer-{node_index}-{round_id}",
                    peer_id=f"peer-{(node_index + 1) % world_size}",
                    round_id=round_id,
                    local_compute_seconds=1.0,
                    peer_wait_seconds=0.5,
                    transfer_seconds=0.25,
                    mixing_seconds=0.1,
                    evaluation_seconds=0.05,
                    retries=1,
                    encoded_artifact_bytes=900,
                ),
                TransferMessageSentEvidence(
                    run_id="run-1",
                    manifest_hash="a" * 64,
                    node_id=node_id,
                    message_id=f"chunk-{node_index}-{round_id}",
                    transfer_id=f"transfer-{node_index}-{round_id}",
                    peer_id=f"peer-{(node_index + 1) % world_size}",
                    round_id=round_id,
                    message_type="CHUNK",
                    payload_bytes=1000,
                    queue_seconds=0.01,
                    send_seconds=0.2,
                    retry_count=1,
                    completion_seconds=0.21,
                ),
                ConsensusSketchSentEvidence(
                    run_id="run-1",
                    manifest_hash="a" * 64,
                    node_id=node_id,
                    message_id=f"sketch-{node_index}-{round_id}",
                    round_id=round_id,
                    payload_bytes=16 * 1024,
                    recipient_count=world_size - 1,
                    successful_recipient_count=world_size - 1,
                    retry_count=0,
                ),
            )
        )
    return EvidenceLog(
        path=Path(f"node-{node_index}.jsonl"),
        run_id="run-1",
        manifest_hash="a" * 64,
        node_id=node_id,
        records=tuple(records),
    )


@pytest.mark.parametrize("world_size", (4, 8, 16))
def test_reconcile_official_evidence_separates_update_and_telemetry_bytes(
    world_size: int,
) -> None:
    logs = tuple(
        _log(node_index=index, world_size=world_size, round_count=2)
        for index in range(world_size)
    )
    encoded = {
        (f"peer-{node_index}", round_id): 900
        for node_index in range(world_size)
        for round_id in range(2)
    }

    report = reconcile_official_evidence(
        logs=logs,
        world_size=world_size,
        round_count=2,
        countsketch_interval_rounds=1,
        raw_artifact_bytes_per_round=9000,
        encoded_artifact_bytes=encoded,
    )

    assert report.passed
    assert len(report.rounds) == world_size * 2
    assert report.rounds[0].compression_ratio == 10.0
    assert report.rounds[0].update_payload_bytes == 1000
    assert report.rounds[0].telemetry_payload_bytes == (world_size - 1) * 16384
    assert report.rounds[0].total_round_seconds == pytest.approx(1.9)


def test_reconcile_official_evidence_rejects_missing_telemetry() -> None:
    log = _log(node_index=0, world_size=4, round_count=1)
    incomplete = EvidenceLog(
        path=log.path,
        run_id=log.run_id,
        manifest_hash=log.manifest_hash,
        node_id=log.node_id,
        records=tuple(
            record
            for record in log.records
            if not isinstance(record, ConsensusSketchSentEvidence)
        ),
    )

    with pytest.raises(ValueError, match="telemetry cadence"):
        reconcile_official_evidence(
            logs=(
                incomplete,
                *(
                    _log(node_index=index, world_size=4, round_count=1)
                    for index in range(1, 4)
                ),
            ),
            world_size=4,
            round_count=1,
            countsketch_interval_rounds=1,
            raw_artifact_bytes_per_round=9000,
            encoded_artifact_bytes={
                (f"peer-{index}", 0): 900 for index in range(4)
            },
        )


def test_encoded_artifact_accounting_reads_canonical_bundle_metadata(
    tmp_path: Path,
) -> None:
    roots: list[Path] = []
    for rank in range(4):
        root = tmp_path / f"rank-{rank}"
        roots.append(root)
        materialize_bundle_metadata(
            OpaqueUpdateBundleMetadata(
                run_id="run-1",
                manifest_hash="a" * 64,
                sender_public_key=f"peer-{rank}",
                algorithm_id="noloco",
                round_id=3,
                artifacts=(
                    OpaqueArtifactMetadata(
                        name="outer_gradient",
                        size_bytes=400,
                        sha256="b" * 64,
                        codec_id="topk-bitmap-int8-v2",
                        codec_version=2,
                        logical_schema_hash="c" * 64,
                        encoded_schema_hash="d" * 64,
                    ),
                    OpaqueArtifactMetadata(
                        name="slow_weights",
                        size_bytes=500,
                        sha256="e" * 64,
                        codec_id="dense-int8-v1",
                        codec_version=1,
                        logical_schema_hash="f" * 64,
                        encoded_schema_hash="0" * 64,
                    ),
                ),
            ),
            root,
        )

    assert encoded_artifact_bytes_from_metadata_roots(roots) == {
        (f"peer-{rank}", 3): 900 for rank in range(4)
    }


def test_reconcile_official_evidence_enforces_compressed_residual_bounds() -> None:
    logs: list[EvidenceLog] = []
    for index in range(4):
        log = _log(node_index=index, world_size=4, round_count=1)
        logs.append(
            EvidenceLog(
                path=log.path,
                run_id=log.run_id,
                manifest_hash=log.manifest_hash,
                node_id=log.node_id,
                records=tuple(
                    record.model_copy(
                        update={
                            "error_feedback_residual_l2_norm": 3.7,
                            "error_feedback_residual_to_signal_ratio": 0.3,
                        }
                    )
                    if isinstance(record, RoundMetricsEvidence)
                    else record
                    for record in log.records
                ),
            )
        )

    with pytest.raises(ValueError, match="residual bound"):
        reconcile_official_evidence(
            logs=logs,
            world_size=4,
            round_count=1,
            countsketch_interval_rounds=1,
            raw_artifact_bytes_per_round=9000,
            encoded_artifact_bytes={(f"peer-{index}", 0): 900 for index in range(4)},
            require_error_feedback=True,
            residual_l2_bound=3.6,
            residual_to_signal_ratio_bound=0.35,
        )
