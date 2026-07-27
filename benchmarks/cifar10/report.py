"""Deterministic aggregate reporting for the M1 CIFAR-10 benchmark."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import cast

import numpy as np

from benchmarks.cifar10.fedavg_reference import FedAvgResult, load_fedavg_result
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
class SeedBenchmarkInput:
    """Archived D-PSGD and FedAvg inputs for one benchmark seed."""

    seed: int
    run_roots: tuple[Path, ...]
    event_logs: tuple[Path, ...]
    fedavg_result_path: Path


@dataclass(frozen=True, slots=True)
class ThreeSeedReport:
    """Aggregate report for the three frozen benchmark seeds."""

    seeds: tuple[BenchmarkReport, ...]
    dpsgd_final_accuracy: SummaryStats
    fedavg_final_accuracy: SummaryStats
    aggregate_pass: bool

    def as_dict(self) -> JsonObject:
        return {
            "seeds": [report.as_dict() for report in self.seeds],
            "dpsgd_final_accuracy": self.dpsgd_final_accuracy.as_dict(),
            "fedavg_final_accuracy": self.fedavg_final_accuracy.as_dict(),
            "aggregate_pass": self.aggregate_pass,
        }

    def write_artifacts(self, output_dir: Path) -> None:
        """Write each seed report and one deterministic aggregate report."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for report in self.seeds:
            report.write_artifacts(output_dir / f"seed-{report.seed}")
        (output_dir / "report.json").write_text(
            json.dumps(self.as_dict(), allow_nan=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        links = "\n".join(
            f"- [seed {report.seed} report](seed-{report.seed}/report.md)"
            for report in self.seeds
        )
        (output_dir / "report.md").write_text(
            "\n".join(
                (
                    "# CIFAR-10 three-seed benchmark report",
                    "",
                    f"- status: {'PASS' if self.aggregate_pass else 'FAIL'}",
                    f"- D-PSGD mean across seeds: {self.dpsgd_final_accuracy.mean:.6f}",
                    (
                        f"- FedAvg mean across seeds: "
                        f"{self.fedavg_final_accuracy.mean:.6f}"
                    ),
                    "",
                    links,
                    "",
                )
            ),
            encoding="utf-8",
        )


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Machine-readable aggregate plus links to deterministic chart artifacts."""

    seed: int
    run_id: str
    manifest_hash: str
    environment: Mapping[str, object]
    configuration: Mapping[str, object]
    node_count: int
    dpsgd_final_accuracy: SummaryStats
    final_node_accuracies: tuple[float, ...]
    fedavg: JsonObject
    accuracy_curve: tuple[JsonObject, ...]
    loss_curve: tuple[JsonObject, ...]
    consensus: tuple[JsonObject, ...]
    round_timing: Mapping[str, object]
    transport: Mapping[str, object]
    connectivity: Mapping[str, object]
    topology: Mapping[str, object]
    hardware: tuple[JsonObject, ...]
    failures: tuple[JsonObject, ...]
    exact_consensus: ExactConsensusReport
    mean_within_fedavg_3pp: bool
    no_node_more_than_5pp_below: bool
    run_roots: tuple[Path, ...]
    event_logs: tuple[Path, ...]
    fedavg_result_path: Path | None

    @property
    def aggregate_pass(self) -> bool:
        return (
            self.mean_within_fedavg_3pp
            and self.no_node_more_than_5pp_below
            and self.consensus_evidence_pass
        )

    @property
    def consensus_evidence_pass(self) -> bool:
        return bool(self.consensus) and self.exact_consensus.mixing_non_increasing

    @property
    def consensus_comparison(self) -> JsonObject:
        """Report the observed final sketch-vs-checkpoint distance error."""
        if not self.consensus:
            return {"available": False}
        approximate = self.consensus[-1]["mean_normalized_rms"]
        exact = self.exact_consensus.final_normalized_distance
        if not isinstance(approximate, (int, float)):
            return {"available": False}
        return {
            "available": True,
            "approximate_final_normalized_rms": float(approximate),
            "exact_final_normalized_rms": exact,
            "absolute_difference": abs(float(approximate) - exact),
        }

    def as_dict(self) -> JsonObject:
        return {
            "seed": self.seed,
            "run_id": self.run_id,
            "manifest_hash": self.manifest_hash,
            "environment": dict(self.environment),
            "configuration": dict(self.configuration),
            "node_count": self.node_count,
            "dpsgd_final_accuracy": self.dpsgd_final_accuracy.as_dict(),
            "final_node_accuracies": list(self.final_node_accuracies),
            "fedavg": dict(self.fedavg),
            "accuracy_curve": list(self.accuracy_curve),
            "loss_curve": list(self.loss_curve),
            "consensus": list(self.consensus),
            "consensus_comparison": self.consensus_comparison,
            "round_timing": dict(self.round_timing),
            "transport": dict(self.transport),
            "connectivity": dict(self.connectivity),
            "topology": dict(self.topology),
            "hardware": list(self.hardware),
            "failures": list(self.failures),
            "exact_consensus": self.exact_consensus.as_dict(),
            "criteria": {
                "mean_within_fedavg_3pp": self.mean_within_fedavg_3pp,
                "no_node_more_than_5pp_below": self.no_node_more_than_5pp_below,
                "consensus_evidence_pass": self.consensus_evidence_pass,
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
        provenance = self._provenance(output_dir)
        (output_dir / "metrics.svg").write_text(
            _add_provenance(self.render_metrics_svg(), provenance), encoding="utf-8"
        )
        (output_dir / "approximate-consensus.svg").write_text(
            _add_provenance(self.render_consensus_svg(), provenance),
            encoding="utf-8",
        )
        (output_dir / "consensus.svg").write_text(
            _add_provenance(self.exact_consensus.render_svg(), provenance),
            encoding="utf-8",
        )
        (output_dir / "timing.svg").write_text(
            _add_provenance(self.render_timing_svg(), provenance), encoding="utf-8"
        )
        (output_dir / "goodput.svg").write_text(
            _add_provenance(self.render_goodput_svg(), provenance), encoding="utf-8"
        )
        (output_dir / "provenance.json").write_text(
            json.dumps(provenance, allow_nan=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "report.md").write_text(self.render_markdown(), encoding="utf-8")

    def _provenance(self, output_dir: Path) -> JsonObject:
        return {
            "manifest_hash": self.manifest_hash,
            "environment": dict(self.environment),
            "manifest_files": [
                str((root / "manifest.json").resolve()) for root in self.run_roots
            ],
            "topology_files": [
                str((root / f"topology-{phase}.json").resolve())
                for root in self.run_roots
                for phase in ("ready", "complete")
            ],
            "hardware_files": [
                str((root / "hardware.json").resolve()) for root in self.run_roots
            ],
            "run_stores": [str(root.resolve()) for root in self.run_roots],
            "event_logs": [str(path.resolve()) for path in self.event_logs],
            "fedavg_results": (
                [str(self.fedavg_result_path.resolve())]
                if self.fedavg_result_path is not None
                else []
            ),
            "report_directory": str(output_dir.resolve()),
        }

    def render_markdown(self) -> str:
        """Render a concise report that links charts and preserves provenance."""
        status = "PASS" if self.aggregate_pass else "FAIL"
        consensus_error = self.consensus_comparison.get(
            "absolute_difference", "unavailable"
        )
        return "\n".join(
            (
                f"# CIFAR-10 benchmark report: {status}",
                "",
                f"- run: `{self.run_id}`",
                f"- manifest hash: `{self.manifest_hash}`",
                f"- nodes: {self.node_count}",
                f"- D-PSGD final accuracy mean: {self.dpsgd_final_accuracy.mean:.6f}",
                f"- FedAvg final accuracy: {self.fedavg_accuracy:.6f}",
                "",
                "[Accuracy and loss curves](metrics.svg)",
                "",
                "[Approximate consensus curves](approximate-consensus.svg)",
                "",
                "[Exact consensus curves](consensus.svg)",
                "",
                "[AXL latency and round timing](timing.svg)",
                "",
                "[AXL payload goodput](goodput.svg)",
                "",
                f"Observed final approximate/exact consensus error: {consensus_error}",
                "",
                "[Raw-data provenance](provenance.json)",
                "",
            )
        )

    def render_consensus_svg(self, *, width: int = 900, height: int = 420) -> str:
        """Render approximate normalized-RMS consensus over rounds."""
        values = {
            _integer_value(point, "round_id"): _numeric_value(
                point, "mean_normalized_rms"
            )
            for point in self.consensus
        }
        return _render_simple_svg(
            title="Approximate consensus distance",
            y_label="normalized RMS",
            series=(("approximate", values, "#2563eb"),),
            width=width,
            height=height,
        )

    def render_timing_svg(self, *, width: int = 900, height: int = 520) -> str:
        """Render round timing and AXL transfer latency in seconds."""
        round_curve = _mapping_sequence(self.round_timing.get("curve", []))
        transport_curve = _mapping_sequence(self.transport.get("curve", []))
        round_values = {
            _integer_value(point, "round_id"): _numeric_value(point, "total_seconds")
            for point in round_curve
        }
        transport_values = {
            _integer_value(point, "round_id"): _numeric_value(
                point, "mean_completion_seconds"
            )
            for point in transport_curve
        }
        return _render_simple_svg(
            title="AXL latency and round timing",
            y_label="seconds",
            series=(
                ("round total seconds", round_values, "#2563eb"),
                ("AXL completion seconds", transport_values, "#dc2626"),
            ),
            width=width,
            height=height,
        )

    def render_goodput_svg(self, *, width: int = 900, height: int = 420) -> str:
        """Render payload goodput in bytes per second."""
        transport_curve = _mapping_sequence(self.transport.get("curve", []))
        goodput_values = {
            _integer_value(point, "round_id"): _numeric_value(
                point, "mean_goodput_bytes_per_second"
            )
            for point in transport_curve
            if _is_number(point.get("mean_goodput_bytes_per_second"))
        }
        return _render_simple_svg(
            title="AXL payload goodput",
            y_label="bytes per second",
            series=(("goodput bytes/second", goodput_values, "#16a34a"),),
            width=width,
            height=height,
        )

    @property
    def fedavg_accuracy(self) -> float:
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
                _integer_value(point, "round_id")
                for point in (*self.accuracy_curve, *self.loss_curve)
                if isinstance(point.get("round_id"), int)
            }
            | {
                _integer_value(round_value, "round_id")
                for round_value in _mapping_sequence(self.fedavg.get("rounds", []))
                if isinstance(round_value.get("round_id"), int)
            }
        )
        rounds = rounds or [0]
        fedavg_rounds = {
            _integer_value(round_value, "round_id"): round_value
            for round_value in _mapping_sequence(self.fedavg.get("rounds", []))
            if isinstance(round_value.get("round_id"), int)
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
    fedavg: FedAvgResult,
    seed: int,
    fedavg_result_path: Path | None = None,
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

    for root, manifest in zip(run_roots, manifests, strict=True):
        _require_completed_state(root, manifest)
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
    benchmark_nodes: set[str] = set()
    for events in all_events:
        ready = [
            event for event in events if event.get("event") == "benchmark_node_ready"
        ]
        if len(ready) != 1:
            raise BenchmarkReportError(
                "each event log must contain one benchmark node provenance record"
            )
        record = ready[0]
        if record.get("benchmark_seed") != seed:
            raise BenchmarkReportError("D-PSGD benchmark seed does not match report")
        if record.get("transport") != "axl":
            raise BenchmarkReportError("official D-PSGD transport is not AXL")
        node_id = record.get("node_id")
        if (
            record.get("run_id") != manifests[0].run_id
            or record.get("manifest_hash") != manifest_hashes[0]
            or not isinstance(node_id, str)
        ):
            raise BenchmarkReportError("benchmark node provenance is incomplete")
        benchmark_nodes.add(node_id)

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
    topology = _topology_summary(run_roots, expected_nodes)
    hardware = tuple(_read_hardware(root) for root in run_roots)
    if {cast(str, value["node_id"]) for value in hardware} != expected_nodes:
        raise BenchmarkReportError(
            "hardware metadata does not cover sealed participants"
        )
    if benchmark_nodes != expected_nodes:
        raise BenchmarkReportError(
            "benchmark provenance does not cover sealed participants"
        )
    if set(node_metrics) != expected_nodes:
        raise BenchmarkReportError("round metrics do not cover sealed participants")

    final_rounds = {
        node_id: _latest_round(node_events, node_id)
        for node_id, node_events in node_metrics.items()
    }
    if len(set(final_rounds.values())) != 1:
        raise BenchmarkReportError("nodes do not share one final round")
    final_round = next(iter(final_rounds.values()))
    expected_rounds = set(range(manifests[0].round_count))
    if any(
        {
            _integer_value(event, "round_id")
            for event in node_events
            if isinstance(event.get("round_id"), int)
        }
        != expected_rounds
        for node_events in node_metrics.values()
    ):
        raise BenchmarkReportError("round metrics are incomplete")
    _validate_dpsgd_evaluation_schedule(
        node_metrics,
        round_count=manifests[0].round_count,
    )
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
    _validate_fedavg_config(manifests[0], fedavg, seed)
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
        seed=seed,
        run_id=manifests[0].run_id,
        manifest_hash=manifest_hashes[0],
        environment=cast(Mapping[str, object], environments[0]),
        configuration=_manifest_configuration(manifests[0]),
        node_count=4,
        dpsgd_final_accuracy=dpsgd_final_accuracy,
        final_node_accuracies=tuple(final_accuracies),
        fedavg=fedavg_payload,
        accuracy_curve=accuracy_curve,
        loss_curve=loss_curve,
        consensus=consensus,
        round_timing=round_timing,
        transport=transport,
        connectivity=connectivity,
        topology=topology,
        hardware=hardware,
        failures=failures,
        exact_consensus=exact_consensus,
        mean_within_fedavg_3pp=abs(dpsgd_final_accuracy.mean - fedavg_accuracy) <= 0.03,
        no_node_more_than_5pp_below=min(final_accuracies) >= fedavg_accuracy - 0.05,
        run_roots=tuple(run_roots),
        event_logs=tuple(event_logs),
        fedavg_result_path=fedavg_result_path,
    )


def build_three_seed_report(
    inputs: Sequence[SeedBenchmarkInput],
) -> ThreeSeedReport:
    """Build one report per seed and one aggregate across exactly three seeds."""
    if len(inputs) != 3 or len({item.seed for item in inputs}) != 3:
        raise BenchmarkReportError("exactly three distinct seed inputs are required")
    try:
        reports = tuple(
            build_benchmark_report(
                run_roots=item.run_roots,
                event_logs=item.event_logs,
                fedavg=load_fedavg_result(item.fedavg_result_path),
                seed=item.seed,
                fedavg_result_path=item.fedavg_result_path,
            )
            for item in sorted(inputs, key=lambda value: value.seed)
        )
    except ValueError as error:
        raise BenchmarkReportError("FedAvg raw result file is invalid") from error
    signatures = {_shared_configuration_signature(report) for report in reports}
    if len(signatures) != 1:
        raise BenchmarkReportError(
            "seed manifests do not share one benchmark configuration"
        )
    dpsgd = _summary(
        [accuracy for report in reports for accuracy in report.final_node_accuracies]
    )
    fedavg = _summary([report.fedavg_accuracy for report in reports])
    return ThreeSeedReport(
        seeds=reports,
        dpsgd_final_accuracy=dpsgd,
        fedavg_final_accuracy=fedavg,
        aggregate_pass=(
            abs(dpsgd.mean - fedavg.mean) <= 0.03
            and all(report.no_node_more_than_5pp_below for report in reports)
            and all(report.consensus_evidence_pass for report in reports)
        ),
    )


def _shared_configuration_signature(report: BenchmarkReport) -> str:
    configuration = dict(report.configuration)
    sketch = cast(Mapping[str, object], configuration["consensus_sketch"])
    configuration["consensus_sketch"] = {"size": sketch["size"]}
    return json.dumps(configuration, sort_keys=True)


def _validate_fedavg_config(
    manifest: SealedManifest, result: FedAvgResult, seed: int
) -> None:
    config = result.config
    if config.trainer_seed != seed:
        raise BenchmarkReportError("FedAvg seed does not match benchmark seed")
    if (
        config.model_id != manifest.model_id
        or config.model_definition_hash != manifest.model_definition_hash
        or config.dataset != manifest.dataset
        or config.environment != manifest.environment
        or config.local_steps != manifest.local_steps
        or config.round_count != manifest.round_count
        or config.learning_rate != manifest.learning_rate
    ):
        raise BenchmarkReportError("FedAvg frozen configuration mismatches manifest")
    if (
        config.data_source != "torchvision-cifar10"
        or config.test_sample_count != 10_000
        or config.evaluation_interval != 5
        or config.batch_size != 32
        or config.device != "cpu"
        or not config.augment
    ):
        raise BenchmarkReportError("FedAvg trainer configuration mismatches D-PSGD")
    if result.initial_checkpoint_hash != manifest.initial_checkpoint_hash:
        raise BenchmarkReportError("FedAvg initialization mismatches manifest")
    if len(result.rounds) != config.round_count:
        raise BenchmarkReportError("FedAvg result has an incomplete round archive")
    if tuple(round_result.round_id for round_result in result.rounds) != tuple(
        range(config.round_count)
    ):
        raise BenchmarkReportError("FedAvg round ids are not contiguous")
    if any(len(round_result.local_losses) != 4 for round_result in result.rounds):
        raise BenchmarkReportError("FedAvg result does not contain four local losses")
    for round_result in result.rounds:
        should_evaluate = (
            (round_result.round_id + 1) % config.evaluation_interval == 0
            or round_result.round_id + 1 == config.round_count
        )
        has_evaluation = (
            round_result.loss is not None and round_result.accuracy is not None
        )
        if should_evaluate != has_evaluation:
            raise BenchmarkReportError("FedAvg evaluation schedule is incomplete")
        if has_evaluation and (
            not math.isfinite(cast(float, round_result.loss))
            or not math.isfinite(cast(float, round_result.accuracy))
            or not 0 <= cast(float, round_result.accuracy) <= 1
        ):
            raise BenchmarkReportError("FedAvg evaluation metrics are invalid")


def _manifest_configuration(manifest: SealedManifest) -> JsonObject:
    """Return the shared comparison config, excluding run-specific identities."""
    return {
        "algorithm_id": manifest.algorithm_id,
        "model_id": manifest.model_id,
        "model_definition_hash": manifest.model_definition_hash,
        "dataset": manifest.dataset.model_dump(mode="json"),
        "environment": manifest.environment.model_dump(mode="json"),
        "local_steps": manifest.local_steps,
        "round_count": manifest.round_count,
        "optimizer": manifest.optimizer,
        "learning_rate": manifest.learning_rate,
        "codec_id": manifest.codec_id,
        "transport": manifest.transport.model_dump(mode="json"),
        "consensus_sketch": manifest.consensus_sketch.model_dump(mode="json"),
        "tensor_schema": manifest.tensor_schema.model_dump(mode="json"),
    }


def _add_provenance(svg: str, provenance: Mapping[str, object]) -> str:
    metadata = escape(json.dumps(dict(provenance), sort_keys=True))
    raw_paths = [
        value
        for key in (
            "event_logs",
            "manifest_files",
            "topology_files",
            "fedavg_results",
        )
        for value in cast(list[object], provenance.get(key, []))
        if isinstance(value, str)
    ]
    anchors = "".join(
        f'<a href="{escape(Path(path).as_uri(), quote=True)}">raw artifact</a>'
        for path in raw_paths
    )
    marker = ">"
    insertion = f"<metadata>{metadata}</metadata>{anchors}"
    return svg.replace(marker, f">{insertion}", 1)


def _render_simple_svg(
    *,
    title: str,
    y_label: str,
    series: Sequence[tuple[str, Mapping[int, float], str]],
    width: int,
    height: int,
) -> str:
    if width <= 0 or height <= 0:
        raise ValueError("SVG dimensions must be positive")
    rounds = sorted({round_id for _, values, _ in series for round_id in values}) or [0]
    left, right, top, bottom = 70, 30, 45, 50
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max(
        (value for _, values, _ in series for value in values.values()), default=1.0
    )
    maximum = maximum if maximum > 0 else 1.0
    last_round = max(rounds)

    def point(round_id: int, value: float) -> tuple[float, float]:
        x = left + (plot_width * round_id / last_round if last_round else 0)
        y = top + plot_height * (1 - value / maximum)
        return x, y

    lines = [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{left}" y="25" font-family="sans-serif" '
            f'font-size="16">{title}</text>'
        ),
        (
            f'<text x="15" y="{top + plot_height / 2}" '
            f'transform="rotate(-90 15 {top + plot_height / 2})" '
            f'font-family="sans-serif" font-size="12">{y_label}</text>'
        ),
        (
            f'<line x1="{left}" y1="{top + plot_height}" '
            f'x2="{width - right}" y2="{top + plot_height}" stroke="#333"/>'
        ),
        (
            f'<line x1="{left}" y1="{top}" '
            f'x2="{left}" y2="{top + plot_height}" stroke="#333"/>'
        ),
    ]
    for index, (label, values, color) in enumerate(series):
        commands: list[str] = []
        for round_id in rounds:
            if round_id not in values:
                continue
            x, y = point(round_id, values[round_id])
            commands.append(f"{'M' if not commands else 'L'} {x:.2f},{y:.2f}")
        if commands:
            lines.append(
                f'<path d="{" ".join(commands)}" fill="none" '
                f'stroke="{color}" stroke-width="2"/>'
            )
        lines.append(
            f'<text x="{width - 220}" y="{25 + 16 * index}" '
            f'font-family="sans-serif" font-size="12" fill="{color}">{label}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines)


def _read_manifest(root: Path) -> SealedManifest:
    try:
        value = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        return SealedManifest.model_validate(value)
    except (OSError, ValueError, TypeError) as error:
        raise BenchmarkReportError(f"invalid run manifest at {root}") from error


def _require_completed_state(root: Path, manifest: SealedManifest) -> None:
    try:
        state = cast(
            object,
            json.loads((root / "state.json").read_text(encoding="utf-8")),
        )
    except (OSError, ValueError) as error:
        raise BenchmarkReportError(f"invalid run state at {root}") from error
    if not isinstance(state, dict):
        raise BenchmarkReportError(f"invalid run state at {root}")
    state_record = cast(JsonObject, state)
    if state_record.get("manifest_hash") != canonical_hash(manifest):
        raise BenchmarkReportError(f"manifest hash mismatch in {root}")
    terminal = state_record.get("terminal")
    if not isinstance(terminal, dict):
        raise BenchmarkReportError(f"run is not complete: {root}")
    terminal_record = cast(JsonObject, terminal)
    if terminal_record.get("result") != "complete":
        raise BenchmarkReportError(f"run is not complete: {root}")
    if state_record.get("committed_round") != manifest.round_count - 1:
        raise BenchmarkReportError(f"run is incomplete: {root}")


def _read_topology(root: Path, *, phase: str) -> tuple[str, tuple[str, ...]]:
    path = root / f"topology-{phase}.json"
    try:
        value = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as error:
        raise BenchmarkReportError(f"invalid topology snapshots at {root}") from error
    if not isinstance(value, dict):
        raise BenchmarkReportError(f"invalid topology snapshots at {root}")
    topology = cast(JsonObject, value)
    public_key = topology.get("our_public_key")
    if not isinstance(public_key, str):
        raise BenchmarkReportError(f"invalid topology snapshots at {root}")
    peers = topology.get("peers")
    if not isinstance(peers, list):
        raise BenchmarkReportError(f"invalid topology snapshots at {root}")
    peer_keys: list[str] = []
    for peer in cast(list[object], peers):
        if not isinstance(peer, dict):
            raise BenchmarkReportError(f"invalid topology snapshots at {root}")
        peer_record = cast(JsonObject, peer)
        peer_key = peer_record.get("public_key")
        if not isinstance(peer_key, str):
            raise BenchmarkReportError(f"invalid topology snapshots at {root}")
        peer_keys.append(peer_key)
    return public_key, tuple(peer_keys)


def _read_hardware(root: Path) -> JsonObject:
    path = root / "hardware.json"
    try:
        value = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as error:
        raise BenchmarkReportError(f"invalid hardware metadata at {root}") from error
    if not isinstance(value, dict):
        raise BenchmarkReportError(f"invalid hardware metadata at {root}")
    hardware = cast(JsonObject, value)
    for field in (
        "node_id",
        "provider",
        "region",
        "machine_type",
        "cpu_model",
        "accelerator",
    ):
        if not isinstance(hardware.get(field), str) or not hardware[field]:
            raise BenchmarkReportError(f"invalid hardware metadata at {root}")
    for field in ("cpu_count", "memory_bytes"):
        field_value = hardware.get(field)
        if (
            not isinstance(field_value, int)
            or isinstance(field_value, bool)
            or field_value <= 0
        ):
            raise BenchmarkReportError(f"invalid hardware metadata at {root}")
    return hardware


def _topology_summary(
    run_roots: Sequence[Path], expected_nodes: set[str]
) -> JsonObject:
    snapshots = {
        phase: tuple(_read_topology(root, phase=phase) for root in run_roots)
        for phase in ("ready", "complete")
    }
    for phase, records in snapshots.items():
        if {public_key for public_key, _ in records} != expected_nodes:
            raise BenchmarkReportError(
                f"{phase} topology snapshots do not cover sealed participants"
            )
    participant_edges: set[tuple[str, str]] = set()
    external_peers: set[str] = set()
    for public_key, peers in snapshots["complete"]:
        for peer in peers:
            if peer in expected_nodes and peer != public_key:
                edge = (public_key, peer) if public_key < peer else (peer, public_key)
                participant_edges.add(edge)
            elif peer not in expected_nodes:
                external_peers.add(peer)
    possible_edges = len(expected_nodes) * (len(expected_nodes) - 1) // 2
    if len(participant_edges) == possible_edges:
        classification = "direct-participant-mesh"
    elif participant_edges:
        classification = "partial-participant-mesh"
    elif external_peers:
        classification = "relay-only"
    else:
        classification = "isolated"
    return {
        "classification": classification,
        "participant_edge_count": len(participant_edges),
        "possible_participant_edges": possible_edges,
        "external_peer_count": len(external_peers),
        "ready_snapshot_count": len(snapshots["ready"]),
        "complete_snapshot_count": len(snapshots["complete"]),
    }


def _read_events(path: Path, *, run_id: str, manifest_hash: str) -> list[Event]:
    events: list[Event] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise BenchmarkReportError(f"event log is unreadable: {path}") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            value = cast(object, json.loads(line))
        except ValueError as error:
            raise BenchmarkReportError(
                f"invalid event JSON at {path}:{line_number}"
            ) from error
        if not isinstance(value, dict):
            raise BenchmarkReportError(f"invalid event record at {path}:{line_number}")
        event = cast(Event, value)
        if not isinstance(event.get("event"), str):
            raise BenchmarkReportError(f"invalid event record at {path}:{line_number}")
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
        _integer_value(event, "round_id")
        for event in metrics
        if isinstance(event.get("round_id"), int)
    ]
    if not rounds:
        raise BenchmarkReportError(f"node {node_id} has no metric rounds")
    return max(rounds)


def _validate_dpsgd_evaluation_schedule(
    node_metrics: Mapping[str, Sequence[Event]],
    *,
    round_count: int,
) -> None:
    for events in node_metrics.values():
        if len(events) != round_count:
            raise BenchmarkReportError("round metrics are incomplete")
        for event in events:
            round_id = _integer_value(event, "round_id")
            should_evaluate = (
                (round_id + 1) % 5 == 0 or round_id + 1 == round_count
            )
            loss = event.get("evaluation_loss")
            accuracy = event.get("evaluation_accuracy")
            has_loss = loss is not None
            has_accuracy = accuracy is not None
            if has_loss != has_accuracy or should_evaluate != has_loss:
                raise BenchmarkReportError(
                    "D-PSGD evaluation schedule is incomplete or invalid"
                )
            if has_loss and (
                not isinstance(loss, (int, float))
                or isinstance(loss, bool)
                or not math.isfinite(float(loss))
                or not isinstance(accuracy, (int, float))
                or isinstance(accuracy, bool)
                or not math.isfinite(float(accuracy))
                or not 0 <= float(accuracy) <= 1
            ):
                raise BenchmarkReportError(
                    "D-PSGD evaluation schedule is incomplete or invalid"
                )


def _metric_value_for_round(
    metrics: Sequence[Event], field: str, round_id: int, node_id: str
) -> float:
    values = [
        _numeric_value(event, field)
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
            _numeric_value(event, field)
            for event in metrics
            if isinstance(event.get(field), (int, float))
            and not isinstance(event.get(field), bool)
        ]
        if values:
            result[field] = _summary(values).as_dict()
    by_round: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for event in metrics:
        round_id = event.get("round_id")
        if not isinstance(round_id, int):
            continue
        for field in fields:
            value = event.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                by_round[round_id][field].append(float(value))
    result["curve"] = [
        {
            "round_id": round_id,
            "total_seconds": sum(
                _summary(by_round[round_id][field]).mean
                for field in fields
                if by_round[round_id][field]
            ),
        }
        for round_id in sorted(by_round)
    ]
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
            _numeric_value(event, field)
            for event in events
            if isinstance(event.get(field), (int, float))
            and not isinstance(event.get(field), bool)
        ]
        if values:
            timings[field] = _summary(values).as_dict()
    retries = [
        _integer_value(event, "retry_count")
        for event in events
        if isinstance(event.get("retry_count"), int)
        and not isinstance(event.get("retry_count"), bool)
    ]
    payload_sizes = [
        _integer_value(event, "payload_bytes")
        for event in events
        if isinstance(event.get("payload_bytes"), int)
        and not isinstance(event.get("payload_bytes"), bool)
        and _integer_value(event, "payload_bytes") >= 0
    ]
    goodputs = [
        _integer_value(event, "payload_bytes")
        / _numeric_value(event, "completion_seconds")
        for event in events
        if isinstance(event.get("payload_bytes"), int)
        and isinstance(event.get("completion_seconds"), (int, float))
        and not isinstance(event.get("completion_seconds"), bool)
        and _numeric_value(event, "completion_seconds") > 0
    ]
    by_round: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for event in events:
        round_id = event.get("round_id")
        if not isinstance(round_id, int):
            continue
        for field in fields:
            value = event.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                by_round[round_id][field].append(float(value))
        if (
            isinstance(event.get("payload_bytes"), int)
            and isinstance(event.get("completion_seconds"), (int, float))
            and _numeric_value(event, "completion_seconds") > 0
        ):
            by_round[round_id]["goodput_bytes_per_second"].append(
                _integer_value(event, "payload_bytes")
                / _numeric_value(event, "completion_seconds")
            )
    curve: list[JsonObject] = []
    for round_id in sorted(by_round):
        values = by_round[round_id]
        curve.append(
            {
                "round_id": round_id,
                "mean_completion_seconds": _summary(values["completion_seconds"]).mean
                if values["completion_seconds"]
                else 0.0,
                "mean_goodput_bytes_per_second": _summary(
                    values["goodput_bytes_per_second"]
                ).mean
                if values["goodput_bytes_per_second"]
                else None,
            }
        )
    return {
        "transfer_count": len(events),
        "retry_count_total": sum(retries),
        "retry_count_maximum": max(retries, default=0),
        "timings_seconds": timings,
        "payload_bytes_total": sum(payload_sizes),
        "goodput_bytes_per_second": _summary(goodputs).as_dict() if goodputs else None,
        "curve": curve,
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


def _fedavg_payload(fedavg: FedAvgResult) -> JsonObject:
    payload = fedavg.as_dict()
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
        round_id: _numeric_value(value, field)
        for round_id, value in rounds.items()
        if isinstance(value.get(field), (int, float))
    }


def _mapping_sequence(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        raise BenchmarkReportError("report curve is invalid")
    result: list[Mapping[str, object]] = []
    for item in cast(list[object], value):
        if not isinstance(item, Mapping):
            raise BenchmarkReportError("report curve is invalid")
        result.append(cast(Mapping[str, object], item))
    return tuple(result)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numeric_value(value: Mapping[str, object], field: str) -> float:
    field_value = value.get(field)
    if not _is_number(field_value):
        raise BenchmarkReportError(f"report value {field} is invalid")
    return float(cast(int | float, field_value))


def _integer_value(value: Mapping[str, object], field: str) -> int:
    field_value = value.get(field)
    if not isinstance(field_value, int) or isinstance(field_value, bool):
        raise BenchmarkReportError(f"report value {field} is invalid")
    return field_value


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
    "SeedBenchmarkInput",
    "SummaryStats",
    "ThreeSeedReport",
    "build_benchmark_report",
    "build_three_seed_report",
]
