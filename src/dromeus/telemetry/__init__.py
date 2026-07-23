"""Runtime telemetry."""

from dromeus.telemetry.consensus import (
    COUNT_SKETCH_SIZE,
    ConsensusDistance,
    ConsensusSketchBuffer,
    ConsensusSketchError,
    ConsensusSketchPublisher,
    count_sketch,
    exact_normalized_rms_distance,
    normalized_rms_consensus_distance,
)
from dromeus.telemetry.events import EventSink, JsonlEventSink, emit_event
from dromeus.telemetry.metrics import (
    JsonlMetricsPublisher,
    MetricsPublisher,
    RoundTiming,
)
from dromeus.telemetry.report import (
    ConsensusReportError,
    ConsensusRoundReport,
    ExactConsensusReport,
    build_exact_consensus_report,
)

__all__ = [
    "COUNT_SKETCH_SIZE",
    "ConsensusDistance",
    "ConsensusSketchBuffer",
    "ConsensusSketchError",
    "ConsensusSketchPublisher",
    "EventSink",
    "JsonlEventSink",
    "JsonlMetricsPublisher",
    "MetricsPublisher",
    "RoundTiming",
    "ConsensusReportError",
    "ConsensusRoundReport",
    "ExactConsensusReport",
    "build_exact_consensus_report",
    "count_sketch",
    "emit_event",
    "exact_normalized_rms_distance",
    "normalized_rms_consensus_distance",
]
