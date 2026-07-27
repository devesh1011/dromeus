"""Deterministic aggregate reporting for the M1 CIFAR-10 benchmark."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from benchmarks.cifar10.fedavg_reference import FedAvgResult
from dromeus.manifests.canonical import canonical_hash
from dromeus.manifests.models import SealedManifest
from dromeus.telemetry.report import ExactConsensusReport, build_exact_consensus_report

JsonObject = dict[str, object]
Event = dict[str, object]


class BenchmarkReportError(ValueError):
    """Benchmark inputs are incomplete, incompatible, or failed."""


@dataclass(frozen=True, slots=True)
class SummaryStats:
    """Population summary for one scalar benchmark measurement."""

    mean: float
    stddev: float
    minimum: float
    maximum: float

    def as_dict(self) -> JsonObject:
        return {
            "mean": self.mean,
            "stddev": self.stddev,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Machine-readable aggregate plus links to deterministic chart artifacts."""

    run_id: str
    manifest_hash: str
    environment: Mapping[str, object]
    node_count: int
    dpsgd_final_accuracy: SummaryStats
    fedavg: JsonObject
    accuracy_curve: tuple[JsonObject, ...]
    loss_curve: tuple[JsonObject, ...]
    consensus: tuple[JsonObject, ...]
    round_timing: Mapping[str, object]
    transport: Mapping[str, object]
    connectivity: Mapping[str, object]
    failures: tuple[JsonObject, ...]
    exact_consensus: ExactConsensusReport
    mean_within_fedavg_3pp: bool
    no_node_more_than_5pp_below: bool

    @property
    def aggregate_pass(self) -> bool:
        return self.mean_within_fedavg_3pp and self.no_node_more_than_5pp_below

    def as_dict(self) -> JsonObject:
        return {
            "run_id": self.run_id,
            "manifest_hash": self.manifest_hash,
            "environment": dict(self.environment),
            "node_count": self.node_count,
            "dpsgd_final_accuracy": self.dpsgd_final_accuracy.as_dict(),
            "fedavg": dict(self.fedavg),
            "accuracy_curve": list(self.accuracy_curve),
            "loss_curve": list(self.loss_curve),
            "consensus": list(self.consensus),
            "round_timing": dict(self.round_timing),
            "transport": dict(self.transport),
            "connectivity": dict(self.connectivity),
            "failures": list(self.failures),
            "exact_consensus": self.exact_consensus.as_dict(),
            "criteria": {
                "mean_within_fedavg_3pp": self.mean_within_fedavg_3pp,
                "no_node_more_than_5pp_below": self.no_node_more_than_5pp_below,
                "aggregate_pass": self.aggregate_pass,
            },
        }

    def write_artifacts(self, output_dir: Path) -> None:
        """Write JSON, human-readable Markdown, and linked SVG charts."""
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "report.json").write_text(
            json.dumps(self.as_dict(), allow_nan=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        (output_dir / "metrics.svg").write_text(
            self.render_metrics_svg(), encoding="utf-8"
        )
        self.exact_consensus.write_svg(output_dir / "consensus.svg")
        (output_dir / "report.md").write_text(self.render_markdown(), encoding="utf-8")

    def render_markdown(self) -> str:
        """Render a concise report that links charts and preserves provenance."""
        status = "PASS" if self.aggregate_pass else "FAIL"
        return "\n".join(
            (
                f"# CIFAR-10 benchmark report: {status}",
                "",
                f"- run: `{self.run_id}`",
                f"- manifest hash: `{self.manifest_hash}`",
                f"- nodes: {self.node_count}",
                f"- D-PSGD final accuracy mean: {self.dpsgd_final_accuracy.mean:.6f}",
                f"- FedAvg final accuracy: {self._fedavg_accuracy:.6f}",
                "",
                "[Accuracy and loss curves](metrics.svg)",
                "",
                "[Exact consensus curves](consensus.svg)",
                "",
            )
        )

    @property
    def _fedavg_accuracy(self) -> float:
        value = self.fedavg.get("final_accuracy")
        assert isinstance(value, float)
        return value

    def render_metrics_svg(self, *, width: int = 900, height: int = 620) -> str:
        """Render deterministic accuracy and loss curves.

        The benchmark has no plotting dependency, so this emits plain SVG.
        """
        if width <= 0 or height <= 0:
            raise ValueError("SVG dimensions must be positive")
        rounds = sorted(
            {
                int(point["round_id"])
                for point in (*self.accuracy_curve, *self.loss_curve)
                if isinstance(point.get("round_id"), int)
            }
            | {
                int(round_value["round_id"])
                for round_value in cast(list[object], self.fedavg.get("rounds", []))
                if isinstance(round_value, Mapping)
                and isinstance(round_value.get("round_id"), int)
            }
        )
        rounds = rounds or [0]
        fedavg_rounds = {
            int(round_value["round_id"]): round_value
            for round_value in cast(list[object], self.fedavg.get("rounds", []))
            if isinstance(round_value, Mapping)
            and isinstance(round_value.get("round_id"), int)
        }

        lines = [
            (
                '<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{width}" height="{height}" '
                f'viewBox="0 0 {width} {height}">'
            ),
            '<rect width="100%" height="100%" fill="white"/>',
            '<text x="30" y="25" font-family="sans-serif" font-size="16">'
            "CIFAR-10 accuracy and loss</text>",
        ]
        lines.extend(
            _render_panel(
                title="accuracy",
                top=45,
                width=width,
                height=250,
                rounds=rounds,
                series=(
                    (
                        "D-PSGD mean",
                        _curve_values(self.accuracy_curve, "mean_accuracy"),
                        "#2563eb",
                    ),
                    ("FedAvg", _fedavg_values(fedavg_rounds, "accuracy"), "#dc2626"),
                ),
            )
        )
        lines.extend(
            _render_panel(
                title="loss",
                top=330,
                width=width,
                height=250,
                rounds=rounds,
                series=(
                    (
                        "D-PSGD local loss",
                        _curve_values(self.loss_curve, "mean_local_loss"),
                        "#2563eb",
                    ),
                    (
                        "D-PSGD eval loss",
                        _curve_values(self.loss_curve, "mean_evaluation_loss"),
                        "#16a34a",
                    ),
                    ("FedAvg", _fedavg_values(fedavg_rounds, "loss"), "#dc2626"),
                ),
            )
        )
        lines.append("</svg>")
        return "\n".join(lines)


def build_benchmark_report(
    *,
    run_roots: Sequence[Path],
    event_logs: Sequence[Path],
    fedavg: FedAvgResult | float,
) -> BenchmarkReport:
    """Aggregate four compatible completed runs and one FedAvg reference."""
    if len(run_roots) != 4 or len(event_logs) != 4:
        raise BenchmarkReportError(
            "exactly four run stores and event logs are required"
        )

    manifests = [_read_manifest(root) for root in run_roots]
    manifest_hashes = [canonical_hash(manifest) for manifest in manifests]
    if len(set(manifest_hashes)) != 1:
        raise BenchmarkReportError("run stores do not share one manifest hash")
    environments = [
        manifest.environment.model_dump(mode="json") for manifest in manifests
    ]
    if any(environment != environments[0] for environment in environments[1:]):
        raise BenchmarkReportError(
            "run stores do not share one environment fingerprint"
        )
    if any(manifest.run_id != manifests[0].run_id for manifest in manifests[1:]):
        raise BenchmarkReportError("run stores do not share one run id")

    for root in run_roots:
        _require_completed_state(root)
    all_events = [
        _read_events(path, run_id=manifests[0].run_id, manifest_hash=manifest_hashes[0])
        for path in event_logs
    ]
    if any(
        event.get("event") in {"run_failed", "formation_failed"}
        for events in all_events
        for event in events
    ):
        raise BenchmarkReportError("failed run cannot be included")

    metrics = [
        event
        for events in all_events
        for event in events
        if event.get("event") == "round_metrics"
    ]
    if not metrics:
        raise BenchmarkReportError("event logs contain no round metrics")
    for event in metrics:
        if (
            event.get("run_id") != manifests[0].run_id
            or event.get("manifest_hash") != manifest_hashes[0]
        ):
            raise BenchmarkReportError("round metric is missing run provenance")
    node_metrics = _group_node_metrics(metrics)
    expected_nodes = {
        participant.public_key for participant in manifests[0].participants
    }
    if set(node_metrics) != expected_nodes:
        raise BenchmarkReportError("round metrics do not cover sealed participants")

    final_rounds = {
        node_id: _latest_round(node_events, node_id)
        for node_id, node_events in node_metrics.items()
    }
    if len(set(final_rounds.values())) != 1:
        raise BenchmarkReportError("nodes do not share one final round")
    final_round = next(iter(final_rounds.values()))
    final_accuracies = [
        _metric_value_for_round(
            node_metrics[node_id],
            "evaluation_accuracy",
            final_round,
            node_id,
        )
        for node_id in sorted(node_metrics)
    ]
    dpsgd_final_accuracy = _summary(final_accuracies)
    fedavg_payload = _fedavg_payload(fedavg)
    fedavg_accuracy = cast(float, fedavg_payload["final_accuracy"])
    accuracy_curve = _metric_curve(metrics, "evaluation_accuracy", "accuracy")
    loss_curve = _loss_curve(metrics)
    consensus = _consensus_curve(all_events)
    round_timing = _round_timing(metrics)
    transport = _transport_summary(all_events)
    connectivity = _connectivity(metrics)
    failures = _failure_summary(all_events)
    exact_consensus = build_exact_consensus_report(run_roots)
    return BenchmarkReport(
        run_id=manifests[0].run_id,
        manifest_hash=manifest_hashes[0],
        environment=cast(Mapping[str, object], environments[0]),
        node_count=4,
        dpsgd_final_accuracy=dpsgd_final_accuracy,
        fedavg=fedavg_payload,
        accuracy_curve=accuracy_curve,
        loss_curve=loss_curve,
        consensus=consensus,
        round_timing=round_timing,
        transport=transport,
        connectivity=connectivity,
        failures=failures,
        exact_consensus=exact_consensus,
        mean_within_fedavg_3pp=abs(dpsgd_final_accuracy.mean - fedavg_accuracy) <= 0.03,
        no_node_more_than_5pp_below=min(final_accuracies) >= fedavg_accuracy - 0.05,
    )


def _read_manifest(root: Path) -> SealedManifest:
    try:
        value = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        return SealedManifest.model_validate(value)
    except (OSError, ValueError, TypeError) as error:
        raise BenchmarkReportError(f"invalid run manifest at {root}") from error


def _require_completed_state(root: Path) -> None:
    try:
        state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BenchmarkReportError(f"invalid run state at {root}") from error
    if not isinstance(state, Mapping):
        raise BenchmarkReportError(f"invalid run state at {root}")
    terminal = state.get("terminal")
    if not isinstance(terminal, Mapping) or terminal.get("result") != "complete":
        raise BenchmarkReportError(f"run is not complete: {root}")


def _read_events(path: Path, *, run_id: str, manifest_hash: str) -> list[Event]:
    events: list[Event] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise BenchmarkReportError(f"event log is unreadable: {path}") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except ValueError as error:
            raise BenchmarkReportError(
                f"invalid event JSON at {path}:{line_number}"
            ) from error
        if not isinstance(value, dict) or not isinstance(value.get("event"), str):
            raise BenchmarkReportError(f"invalid event record at {path}:{line_number}")
        event = cast(Event, value)
        if event.get("run_id") not in (None, run_id):
            raise BenchmarkReportError(f"event run id mismatch in {path}")
        if event.get("manifest_hash") not in (None, manifest_hash):
            raise BenchmarkReportError(f"event manifest hash mismatch in {path}")
        events.append(event)
    if not events:
        raise BenchmarkReportError(f"event log is empty: {path}")
    return events


def _group_node_metrics(metrics: Sequence[Event]) -> dict[str, list[Event]]:
    grouped: dict[str, list[Event]] = defaultdict(list)
    for event in metrics:
        node_id = event.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            raise BenchmarkReportError("round metric is missing node_id")
        grouped[node_id].append(event)
    return dict(grouped)


def _latest_round(metrics: Sequence[Event], node_id: str) -> int:
    rounds = [
        int(event["round_id"])
        for event in metrics
        if isinstance(event.get("round_id"), int)
    ]
    if not rounds:
        raise BenchmarkReportError(f"node {node_id} has no metric rounds")
    return max(rounds)


def _metric_value_for_round(
    metrics: Sequence[Event], field: str, round_id: int, node_id: str
) -> float:
    values = [
        float(event[field])
        for event in metrics
        if event.get("round_id") == round_id
        and isinstance(event.get(field), (int, float))
        and not isinstance(event.get(field), bool)
    ]
    if not values:
        raise BenchmarkReportError(
            f"node {node_id} has no {field} for final round {round_id}"
        )
    value = values[-1]
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise BenchmarkReportError(f"node {node_id} has invalid {field}")
    return value


def _summary(values: Sequence[float]) -> SummaryStats:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise BenchmarkReportError("cannot summarize empty or invalid values")
    return SummaryStats(
        mean=float(array.mean()),
        stddev=float(array.std()),
        minimum=float(array.min()),
        maximum=float(array.max()),
    )


def _metric_curve(
    metrics: Sequence[Event], field: str, prefix: str
) -> tuple[JsonObject, ...]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for event in metrics:
        round_id = event.get("round_id")
        value = event.get(field)
        if (
            isinstance(round_id, int)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            grouped[round_id].append(float(value))
    result: list[JsonObject] = []
    for round_id in sorted(grouped):
        values = _summary(grouped[round_id])
        result.append(
            {
                "round_id": round_id,
                f"mean_{prefix}": values.mean,
                f"minimum_{prefix}": values.minimum,
                f"maximum_{prefix}": values.maximum,
            }
        )
    return tuple(result)


def _loss_curve(metrics: Sequence[Event]) -> tuple[JsonObject, ...]:
    local = _metric_curve(metrics, "local_loss", "local_loss")
    evaluation = _metric_curve(metrics, "evaluation_loss", "evaluation_loss")
    grouped: dict[int, JsonObject] = {}
    for point in (*local, *evaluation):
        round_id = cast(int, point["round_id"])
        grouped.setdefault(round_id, {"round_id": round_id}).update(point)
    return tuple(grouped[round_id] for round_id in sorted(grouped))


def _consensus_curve(all_events: Sequence[Sequence[Event]]) -> tuple[JsonObject, ...]:
    grouped: dict[int, list[float]] = defaultdict(list)
    sketches: dict[int, list[int]] = defaultdict(list)
    for events in all_events:
        for event in events:
            if event.get("event") != "consensus_distance":
                continue
            round_id = event.get("round_id")
            distance = event.get("normalized_rms")
            sketch_count = event.get("sketch_count")
            if (
                isinstance(round_id, int)
                and isinstance(distance, (int, float))
                and not isinstance(distance, bool)
            ):
                grouped[round_id].append(float(distance))
                if isinstance(sketch_count, int):
                    sketches[round_id].append(sketch_count)
    result: list[JsonObject] = []
    for round_id in sorted(grouped):
        result.append(
            {
                "round_id": round_id,
                "mean_normalized_rms": _summary(grouped[round_id]).mean,
                "minimum_normalized_rms": min(grouped[round_id]),
                "maximum_normalized_rms": max(grouped[round_id]),
                "sketch_count": min(sketches[round_id]) if sketches[round_id] else None,
            }
        )
    return tuple(result)


def _round_timing(metrics: Sequence[Event]) -> dict[str, object]:
    fields = (
        "local_compute_seconds",
        "peer_wait_seconds",
        "transfer_seconds",
        "mixing_seconds",
        "evaluation_seconds",
    )
    result: dict[str, object] = {}
    for field in fields:
        values = [
            float(event[field])
            for event in metrics
            if isinstance(event.get(field), (int, float))
            and not isinstance(event.get(field), bool)
        ]
        if values:
            result[field] = _summary(values).as_dict()
    return result


def _transport_summary(all_events: Sequence[Sequence[Event]]) -> dict[str, object]:
    events = [
        event
        for records in all_events
        for event in records
        if event.get("event") == "transfer_message_sent"
    ]
    fields = ("queue_seconds", "send_seconds", "completion_seconds")
    timings: dict[str, object] = {}
    for field in fields:
        values = [
            float(event[field])
            for event in events
            if isinstance(event.get(field), (int, float))
            and not isinstance(event.get(field), bool)
        ]
        if values:
            timings[field] = _summary(values).as_dict()
    retries = [
        int(event["retry_count"])
        for event in events
        if isinstance(event.get("retry_count"), int)
        and not isinstance(event.get("retry_count"), bool)
    ]
    return {
        "transfer_count": len(events),
        "retry_count_total": sum(retries),
        "retry_count_maximum": max(retries, default=0),
        "timings_seconds": timings,
        "goodput_bytes_per_second": None,
    }


def _connectivity(metrics: Sequence[Event]) -> dict[str, object]:
    edges: dict[tuple[str, str], int] = defaultdict(int)
    for event in metrics:
        node_id = event.get("node_id")
        peer_id = event.get("peer_id")
        if isinstance(node_id, str) and isinstance(peer_id, str):
            edges[(node_id, peer_id)] += 1
    edge_records = [
        {"node_id": node_id, "peer_id": peer_id, "count": count}
        for (node_id, peer_id), count in sorted(edges.items())
    ]
    return {"edge_count": len(edge_records), "edges": edge_records}


def _failure_summary(all_events: Sequence[Sequence[Event]]) -> tuple[JsonObject, ...]:
    result: list[JsonObject] = []
    for events in all_events:
        for event in events:
            name = event["event"]
            if not (
                isinstance(name, str)
                and (name.endswith("_rejected") or name == "formation_failed")
            ):
                continue
            result.append(
                {
                    key: event[key]
                    for key in (
                        "event",
                        "node_id",
                        "peer_id",
                        "round_id",
                        "error_type",
                        "error",
                    )
                    if key in event
                }
            )
    result.sort(key=lambda value: json.dumps(value, sort_keys=True))
    return tuple(result)


def _fedavg_payload(fedavg: FedAvgResult | float) -> JsonObject:
    payload = (
        fedavg.as_dict()
        if isinstance(fedavg, FedAvgResult)
        else {"final_accuracy": fedavg}
    )
    value = payload.get("final_accuracy")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise BenchmarkReportError("FedAvg result has no final accuracy")
    final_accuracy = float(value)
    if not math.isfinite(final_accuracy) or not 0 <= final_accuracy <= 1:
        raise BenchmarkReportError("FedAvg final accuracy is invalid")
    payload["final_accuracy"] = final_accuracy
    return payload


def _curve_values(curve: Sequence[JsonObject], field: str) -> dict[int, float]:
    return {
        cast(int, point["round_id"]): cast(float, point[field])
        for point in curve
        if isinstance(point.get("round_id"), int)
        and isinstance(point.get(field), (int, float))
    }


def _fedavg_values(
    rounds: Mapping[int, Mapping[str, object]], field: str
) -> dict[int, float]:
    return {
        round_id: float(value[field])
        for round_id, value in rounds.items()
        if isinstance(value.get(field), (int, float))
    }


def _render_panel(
    *,
    title: str,
    top: int,
    width: int,
    height: int,
    rounds: Sequence[int],
    series: Sequence[tuple[str, Mapping[int, float], str]],
) -> list[str]:
    left, right, bottom = 70, 30, 35
    plot_width = width - left - right
    plot_height = height - bottom - 25
    values = [value for _, points, _ in series for value in points.values()]
    maximum = max(values, default=1.0)
    maximum = maximum if maximum > 0 else 1.0
    last_round = max(rounds, default=0)

    def point(round_id: int, value: float) -> tuple[float, float]:
        x = left + (plot_width * round_id / last_round if last_round else 0)
        y = top + 25 + plot_height * (1 - value / maximum)
        return x, y

    output = [
        (
            f'<text x="{left}" y="{top + 15}" '
            f'font-family="sans-serif" font-size="14">{title}</text>'
        ),
        (
            f'<line x1="{left}" y1="{top + 25 + plot_height}" '
            f'x2="{width - right}" y2="{top + 25 + plot_height}" stroke="#333"/>'
        ),
        (
            f'<line x1="{left}" y1="{top + 25}" '
            f'x2="{left}" y2="{top + 25 + plot_height}" stroke="#333"/>'
        ),
    ]
    for series_index, (label, points, color) in enumerate(series):
        commands = [
            f"{'M' if point_index == 0 else 'L'} {x:.2f},{y:.2f}"
            for point_index, round_id in enumerate(rounds)
            if round_id in points
            for x, y in (point(round_id, points[round_id]),)
        ]
        if commands:
            output.append(
                f'<path d="{" ".join(commands)}" fill="none" '
                f'stroke="{color}" stroke-width="2"/>'
            )
        output.append(
            f'<text x="{width - 170}" y="{top + 15 + 16 * series_index}" '
            f'font-family="sans-serif" font-size="12" fill="{color}">{label}</text>'
        )
    return output


__all__ = [
    "BenchmarkReport",
    "BenchmarkReportError",
    "SummaryStats",
    "build_benchmark_report",
]
