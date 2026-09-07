"""Strict official NoLoCo bandwidth, timing, and telemetry reconciliation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from dromeus.manifests.canonical import parse_bundle_metadata
from dromeus.telemetry.evidence import (
    ConsensusSketchSentEvidence,
    EvidenceLog,
    RoundMetricsEvidence,
    TransferMessageSentEvidence,
)


@dataclass(frozen=True, slots=True)
class RoundEvidenceAccounting:
    node_id: str
    round_id: int
    raw_artifact_bytes: int
    encoded_artifact_bytes: int
    compression_ratio: float
    update_payload_bytes: int
    telemetry_payload_bytes: int
    update_retry_count: int
    telemetry_retry_count: int
    local_compute_seconds: float
    peer_wait_seconds: float
    transfer_seconds: float
    mixing_seconds: float
    evaluation_seconds: float
    total_round_seconds: float


@dataclass(frozen=True, slots=True)
class EvidenceReconciliationReport:
    passed: bool
    world_size: int
    round_count: int
    maximum_residual_l2_norm: float | None
    maximum_residual_to_signal_ratio: float | None
    rounds: tuple[RoundEvidenceAccounting, ...]


def encoded_artifact_bytes_from_metadata_roots(
    roots: Sequence[Path],
) -> dict[tuple[str, int], int]:
    """Read canonical bundle metadata and total encoded artifact bytes."""
    accounting: dict[tuple[str, int], int] = {}
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"bundle metadata root is missing: {root}")
        for path in sorted(root.glob("*.safetensors")):
            metadata = parse_bundle_metadata(path)
            key = (metadata.sender_public_key, metadata.round_id)
            size_bytes = sum(artifact.size_bytes for artifact in metadata.artifacts)
            existing = accounting.setdefault(key, size_bytes)
            if existing != size_bytes:
                raise ValueError("duplicate bundle metadata accounting does not match")
    if not accounting:
        raise ValueError("bundle metadata accounting is empty")
    return accounting


def reconcile_official_evidence(
    *,
    logs: Sequence[EvidenceLog],
    world_size: int,
    round_count: int,
    countsketch_interval_rounds: int,
    raw_artifact_bytes_per_round: int,
    encoded_artifact_bytes: Mapping[tuple[str, int], int] | None = None,
    require_error_feedback: bool = False,
    residual_l2_bound: float | None = None,
    residual_to_signal_ratio_bound: float | None = None,
) -> EvidenceReconciliationReport:
    """Require complete node-round evidence and reconcile distinct byte classes."""
    if world_size not in {4, 8, 16} or len(logs) != world_size:
        raise ValueError("official evidence logs do not match world size")
    if round_count <= 0 or countsketch_interval_rounds <= 0:
        raise ValueError("official evidence cadence is invalid")
    if raw_artifact_bytes_per_round <= 0:
        raise ValueError("raw artifact bytes must be positive")
    if require_error_feedback and (
        residual_l2_bound is None
        or residual_l2_bound <= 0
        or residual_to_signal_ratio_bound is None
        or residual_to_signal_ratio_bound <= 0
    ):
        raise ValueError("compressed evidence residual bounds are invalid")
    node_ids = tuple(log.node_id for log in logs)
    if any(node_id is None for node_id in node_ids) or len(set(node_ids)) != world_size:
        raise ValueError("official evidence node identities are incomplete")
    expected_telemetry_rounds = set(
        range(0, round_count, countsketch_interval_rounds)
    )
    accounting: list[RoundEvidenceAccounting] = []
    residual_l2_values: list[float] = []
    residual_ratio_values: list[float] = []
    for log in logs:
        node_id = log.node_id
        if node_id is None:
            raise ValueError("official evidence node identity is missing")
        metrics_by_round: dict[int, list[RoundMetricsEvidence]] = defaultdict(list)
        updates_by_round: dict[int, list[TransferMessageSentEvidence]] = defaultdict(
            list
        )
        telemetry_by_round: dict[int, list[ConsensusSketchSentEvidence]] = defaultdict(
            list
        )
        for record in log.records:
            if isinstance(record, RoundMetricsEvidence):
                metrics_by_round[record.round_id].append(record)
            elif isinstance(record, TransferMessageSentEvidence):
                if record.round_id is not None:
                    updates_by_round[record.round_id].append(record)
            elif isinstance(record, ConsensusSketchSentEvidence):
                telemetry_by_round[record.round_id].append(record)
        if set(telemetry_by_round) != expected_telemetry_rounds or any(
            len(records) != 1 for records in telemetry_by_round.values()
        ):
            raise ValueError("consensus telemetry cadence is incomplete")
        for round_id in range(round_count):
            metrics = metrics_by_round.get(round_id, [])
            if len(metrics) != 1:
                raise ValueError("round timing evidence is incomplete")
            encoded = (
                encoded_artifact_bytes.get((node_id, round_id))
                if encoded_artifact_bytes is not None
                else metrics[0].encoded_artifact_bytes
            )
            if encoded is None or encoded <= 0:
                raise ValueError("encoded artifact accounting is incomplete")
            update_records = updates_by_round.get(round_id, [])
            update_payload = sum(record.payload_bytes for record in update_records)
            if update_payload < encoded:
                raise ValueError("transport payload is smaller than encoded artifacts")
            telemetry_records = telemetry_by_round.get(round_id, [])
            telemetry_payload = sum(
                record.payload_bytes * record.successful_recipient_count
                for record in telemetry_records
            )
            metric = metrics[0]
            if require_error_feedback:
                residual_l2 = metric.error_feedback_residual_l2_norm
                residual_ratio = metric.error_feedback_residual_to_signal_ratio
                if residual_l2 is None or residual_ratio is None:
                    raise ValueError("compressed residual evidence is incomplete")
                residual_l2_values.append(residual_l2)
                residual_ratio_values.append(residual_ratio)
                if (
                    residual_l2 > cast(float, residual_l2_bound)
                    or residual_ratio
                    > cast(float, residual_to_signal_ratio_bound)
                ):
                    raise ValueError("compressed residual bound was exceeded")
            durations = (
                metric.local_compute_seconds,
                metric.peer_wait_seconds,
                metric.transfer_seconds,
                metric.mixing_seconds,
                metric.evaluation_seconds,
            )
            accounting.append(
                RoundEvidenceAccounting(
                    node_id=node_id,
                    round_id=round_id,
                    raw_artifact_bytes=raw_artifact_bytes_per_round,
                    encoded_artifact_bytes=encoded,
                    compression_ratio=raw_artifact_bytes_per_round / encoded,
                    update_payload_bytes=update_payload,
                    telemetry_payload_bytes=telemetry_payload,
                    update_retry_count=sum(
                        record.retry_count for record in update_records
                    ),
                    telemetry_retry_count=sum(
                        record.retry_count for record in telemetry_records
                    ),
                    local_compute_seconds=metric.local_compute_seconds,
                    peer_wait_seconds=metric.peer_wait_seconds,
                    transfer_seconds=metric.transfer_seconds,
                    mixing_seconds=metric.mixing_seconds,
                    evaluation_seconds=metric.evaluation_seconds,
                    total_round_seconds=sum(durations),
                )
            )
    if encoded_artifact_bytes is not None and set(encoded_artifact_bytes) != {
        (node_id, round_id)
        for node_id in node_ids
        if node_id is not None
        for round_id in range(round_count)
    }:
        raise ValueError("encoded artifact accounting contains unexpected entries")
    return EvidenceReconciliationReport(
        passed=True,
        world_size=world_size,
        round_count=round_count,
        maximum_residual_l2_norm=(
            max(residual_l2_values) if residual_l2_values else None
        ),
        maximum_residual_to_signal_ratio=(
            max(residual_ratio_values) if residual_ratio_values else None
        ),
        rounds=tuple(
            sorted(accounting, key=lambda item: (item.node_id, item.round_id))
        ),
    )


__all__ = [
    "EvidenceReconciliationReport",
    "RoundEvidenceAccounting",
    "encoded_artifact_bytes_from_metadata_roots",
    "reconcile_official_evidence",
]
