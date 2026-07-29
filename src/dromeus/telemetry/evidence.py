"""Strict benchmark evidence records over mixed diagnostic JSONL logs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
)

from dromeus.manifests.models import (
    Identifier,
    MessageId,
    PublicKey,
    RoundId,
    RunId,
    Sha256,
    TransferId,
)
from dromeus.telemetry.events import EventSink

EVIDENCE_VERSION = 1


class EvidenceError(ValueError):
    """Official benchmark evidence is invalid, incompatible, or inconsistent."""


class _EvidenceModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    evidence_version: Literal[1] = EVIDENCE_VERSION
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    run_id: RunId
    manifest_hash: Sha256
    node_id: PublicKey

    @field_validator("timestamp")
    @classmethod
    def timestamp_has_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("evidence timestamp must include a timezone")
        return value


class BenchmarkNodeReadyEvidence(_EvidenceModel):
    event: Literal["benchmark_node_ready"] = "benchmark_node_ready"
    benchmark_seed: int
    transport: Literal["axl"]


class RoundMetricsEvidence(_EvidenceModel):
    event: Literal["round_metrics"] = "round_metrics"
    message_id: MessageId
    transfer_id: TransferId | None = None
    peer_id: PublicKey
    round_id: RoundId
    local_loss: float | None = Field(default=None, ge=0)
    evaluation_loss: float | None = Field(default=None, ge=0)
    evaluation_accuracy: float | None = Field(default=None, ge=0, le=1)
    local_compute_seconds: float = Field(ge=0)
    peer_wait_seconds: float = Field(ge=0)
    transfer_seconds: float = Field(ge=0)
    mixing_seconds: float = Field(ge=0)
    evaluation_seconds: float = Field(ge=0)
    retries: int = Field(ge=0)


class ConsensusDistanceEvidence(_EvidenceModel):
    event: Literal["consensus_distance"] = "consensus_distance"
    message_id: MessageId
    round_id: RoundId
    normalized_rms: float = Field(ge=0)
    sketch_count: int = Field(ge=1)


class TransferMessageSentEvidence(_EvidenceModel):
    event: Literal["transfer_message_sent"] = "transfer_message_sent"
    message_id: MessageId
    transfer_id: TransferId | None = None
    peer_id: PublicKey
    round_id: RoundId | None = None
    message_type: Identifier
    payload_bytes: int = Field(ge=0)
    queue_seconds: float = Field(ge=0)
    send_seconds: float = Field(ge=0)
    retry_count: int = Field(ge=0)
    completion_seconds: float = Field(ge=0)


class RunFailedEvidence(_EvidenceModel):
    event: Literal["run_failed"] = "run_failed"
    message_id: MessageId
    round_id: RoundId
    error_type: Identifier
    error: str = Field(min_length=1, max_length=1024)


type EvidenceRecord = (
    BenchmarkNodeReadyEvidence
    | RoundMetricsEvidence
    | ConsensusDistanceEvidence
    | TransferMessageSentEvidence
    | RunFailedEvidence
)
_EVIDENCE_ADAPTER: TypeAdapter[EvidenceRecord] = TypeAdapter(EvidenceRecord)
_EVIDENCE_EVENTS = frozenset(
    {
        "benchmark_node_ready",
        "round_metrics",
        "consensus_distance",
        "transfer_message_sent",
        "run_failed",
    }
)


@dataclass(frozen=True, slots=True)
class EvidenceLog:
    """One strictly decoded node evidence stream from a mixed JSONL log."""

    path: Path
    run_id: str
    manifest_hash: str
    node_id: str | None
    records: tuple[EvidenceRecord, ...]

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        run_id: str,
        manifest_hash: str,
    ) -> EvidenceLog:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise EvidenceError(f"evidence log is unreadable: {path}") from error
        records: list[EvidenceRecord] = []
        for line_number, line in enumerate(lines, start=1):
            try:
                value = _json_object(line)
            except EvidenceError as error:
                raise EvidenceError(
                    f"invalid JSONL record at {path}:{line_number}"
                ) from error
            event = value.get("event")
            if (
                "evidence_version" not in value
                and event not in _EVIDENCE_EVENTS
            ):
                continue
            try:
                record = decode_evidence(line)
            except EvidenceError as error:
                raise EvidenceError(
                    f"invalid evidence at {path}:{line_number}"
                ) from error
            if record.run_id != run_id:
                raise EvidenceError(f"evidence run id mismatch in {path}")
            if record.manifest_hash != manifest_hash:
                raise EvidenceError(f"evidence manifest hash mismatch in {path}")
            records.append(record)
        node_ids = {record.node_id for record in records}
        if len(node_ids) > 1:
            raise EvidenceError(f"evidence node id mismatch in {path}")
        for record_type in (
            RoundMetricsEvidence,
            ConsensusDistanceEvidence,
            RunFailedEvidence,
        ):
            rounds = [
                record.round_id
                for record in records
                if isinstance(record, record_type)
            ]
            if len(rounds) != len(set(rounds)):
                raise EvidenceError(f"duplicate evidence round in {path}")
        return cls(
            path=path,
            run_id=run_id,
            manifest_hash=manifest_hash,
            node_id=next(iter(node_ids), None),
            records=tuple(records),
        )


def evidence_record(record: EvidenceRecord) -> dict[str, object]:
    """Return one JSON-compatible strict evidence mapping."""
    return cast(
        dict[str, object],
        record.model_dump(mode="json", exclude_none=True),
    )


def encode_evidence(record: EvidenceRecord) -> bytes:
    """Encode one evidence record as deterministic strict JSON."""
    return json.dumps(
        evidence_record(record),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def decode_evidence(data: bytes | str) -> EvidenceRecord:
    """Strictly decode one supported evidence record."""
    value = _json_object(data)
    version = value.get("evidence_version")
    if version != EVIDENCE_VERSION:
        raise EvidenceError(f"unsupported evidence version: {version}")
    try:
        return _EVIDENCE_ADAPTER.validate_json(
            json.dumps(value, allow_nan=False, separators=(",", ":")),
            strict=True,
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise EvidenceError("evidence record is invalid") from error


def append_evidence(sink: EventSink | None, record: EvidenceRecord) -> bool:
    """Best-effort evidence write that never propagates sink failures."""
    if sink is None:
        return False
    try:
        sink.append(evidence_record(record))
    except Exception:
        return False
    return True


def _json_object(data: bytes | str) -> dict[str, object]:
    try:
        value = cast(
            object,
            json.loads(data, parse_constant=_reject_json_constant),
        )
    except (TypeError, ValueError) as error:
        raise EvidenceError("evidence JSON is invalid") from error
    if not isinstance(value, dict):
        raise EvidenceError("evidence record must be an object")
    entries = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in entries):
        raise EvidenceError("evidence object keys must be strings")
    return cast(dict[str, object], entries)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


__all__ = [
    "EVIDENCE_VERSION",
    "BenchmarkNodeReadyEvidence",
    "ConsensusDistanceEvidence",
    "EvidenceError",
    "EvidenceLog",
    "EvidenceRecord",
    "RoundMetricsEvidence",
    "RunFailedEvidence",
    "TransferMessageSentEvidence",
    "append_evidence",
    "decode_evidence",
    "encode_evidence",
    "evidence_record",
]
