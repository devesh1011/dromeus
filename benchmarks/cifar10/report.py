"""Deterministic aggregate reporting for the M1 CIFAR-10 benchmark."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np

from benchmarks.cifar10.fedavg_reference import FedAvgResult, load_fedavg_result
from benchmarks.cifar10.plots import write_line_plot, write_panel_plot
from dromeus.manifests.models import SealedManifest
from dromeus.persistence.archive import RunArchive, RunArchiveError
from dromeus.telemetry.evidence import (
    BenchmarkNodeReadyEvidence,
    ConsensusDistanceEvidence,
    EvidenceError,
    EvidenceLog,
    RoundMetricsEvidence,
    RunFailedEvidence,
    TransferMessageSentEvidence,
)

JsonObject = dict[str, object]
MetricField = Literal[
    "local_loss",
    "evaluation_loss",
    "evaluation_accuracy",
]


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

    @property
    def publication_ready(self) -> bool:
        return self.aggregate_pass and all(
            report.publication_ready for report in self.seeds
        )

    def as_dict(self) -> JsonObject:
        return {
            "seeds": [report.as_dict() for report in self.seeds],
            "dpsgd_final_accuracy": self.dpsgd_final_accuracy.as_dict(),
            "fedavg_final_accuracy": self.fedavg_final_accuracy.as_dict(),
            "aggregate_pass": self.aggregate_pass,
            "publication_ready": self.publication_ready,
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
                    (
                        "- publication ready: "
                        f"{'yes' if self.publication_ready else 'no'}"
                    ),
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
    mean_within_fedavg_3pp: bool
    no_node_more_than_5pp_below: bool
    minimum_accuracy_90: bool
    run_roots: tuple[Path, ...]
    event_logs: tuple[Path, ...]
    fedavg_result_path: Path | None

    @property
    def aggregate_pass(self) -> bool:
        return (
            self.mean_within_fedavg_3pp
            and self.no_node_more_than_5pp_below
            and self.minimum_accuracy_90
            and self.consensus_evidence_pass
        )

    @property
    def quality_gate_required(self) -> bool:
        return True

    @property
    def publication_ready(self) -> bool:
        return self.aggregate_pass and self.minimum_accuracy_90

    @property
    def consensus_evidence_pass(self) -> bool:
        return bool(self.consensus)

    @property
    def final_approximate_consensus_distance(self) -> float | None:
        if not self.consensus:
            return None
        value = self.consensus[-1]["mean_normalized_rms"]
        return float(value) if isinstance(value, (int, float)) else None

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
            "final_approximate_consensus_distance": (
                self.final_approximate_consensus_distance
            ),
            "round_timing": dict(self.round_timing),
            "transport": dict(self.transport),
            "connectivity": dict(self.connectivity),
            "topology": dict(self.topology),
            "hardware": list(self.hardware),
            "failures": list(self.failures),
            "criteria": {
                "mean_within_fedavg_3pp": self.mean_within_fedavg_3pp,
                "no_node_more_than_5pp_below": self.no_node_more_than_5pp_below,
                "minimum_accuracy_90": self.minimum_accuracy_90,
                "quality_gate_required": self.quality_gate_required,
                "consensus_evidence_pass": self.consensus_evidence_pass,
                "aggregate_pass": self.aggregate_pass,
                "publication_ready": self.publication_ready,
            },
        }

    def write_artifacts(self, output_dir: Path) -> None:
        """Write JSON, human-readable Markdown, and Matplotlib PNG charts."""
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "report.json").write_text(
            json.dumps(self.as_dict(), allow_nan=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        provenance = self._provenance(output_dir)
        self._write_plots(output_dir, provenance)
        (output_dir / "provenance.json").write_text(
            json.dumps(provenance, allow_nan=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "report.md").write_text(self.render_markdown(), encoding="utf-8")

    def _write_plots(
        self, output_dir: Path, provenance: Mapping[str, object]
    ) -> None:
        fedavg_rounds = {
            _integer_value(round_value, "round_id"): round_value
            for round_value in _mapping_sequence(self.fedavg.get("rounds", []))
            if isinstance(round_value.get("round_id"), int)
        }
        write_panel_plot(
            output_dir / "metrics.png",
            title="CIFAR-10 accuracy and loss",
            panels=(
                (
                    "Accuracy",
                    "Accuracy",
                    (
                        (
                            "D-PSGD mean",
                            _curve_values(self.accuracy_curve, "mean_accuracy"),
                            "#2563eb",
                        ),
                        (
                            "FedAvg",
                            _fedavg_values(fedavg_rounds, "accuracy"),
                            "#dc2626",
                        ),
                    ),
                ),
                (
                    "Loss",
                    "Cross-entropy loss",
                    (
                        (
                            "D-PSGD local loss",
                            _curve_values(self.loss_curve, "mean_local_loss"),
                            "#2563eb",
                        ),
                        (
                            "D-PSGD evaluation loss",
                            _curve_values(self.loss_curve, "mean_evaluation_loss"),
                            "#16a34a",
                        ),
                        (
                            "FedAvg",
                            _fedavg_values(fedavg_rounds, "loss"),
                            "#dc2626",
                        ),
                    ),
                ),
            ),
            provenance=provenance,
        )
        write_line_plot(
            output_dir / "consensus.png",
            title="Approximate consensus distance",
            y_label="Normalized RMS",
            series=(
                (
                    "Mean",
                    _curve_values(self.consensus, "mean_normalized_rms"),
                    "#2563eb",
                ),
                (
                    "Smoothed mean",
                    _curve_values(
                        self.consensus, "smoothed_mean_normalized_rms"
                    ),
                    "#16a34a",
                ),
            ),
            provenance=provenance,
        )
        round_curve = _mapping_sequence(self.round_timing.get("curve", []))
        transport_curve = _mapping_sequence(self.transport.get("curve", []))
        write_line_plot(
            output_dir / "timing.png",
            title="AXL latency and round timing",
            y_label="Seconds",
            series=(
                (
                    "Round total",
                    _curve_values(round_curve, "total_seconds"),
                    "#2563eb",
                ),
                (
                    "AXL completion",
                    _curve_values(transport_curve, "mean_completion_seconds"),
                    "#dc2626",
                ),
            ),
            provenance=provenance,
        )
        write_line_plot(
            output_dir / "goodput.png",
            title="AXL payload goodput",
            y_label="Bytes per second",
            series=(
                (
                    "Goodput",
                    _curve_values(
                        transport_curve, "mean_goodput_bytes_per_second"
                    ),
                    "#16a34a",
                ),
            ),
            provenance=provenance,
        )

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
        return "\n".join(
            (
                f"# CIFAR-10 benchmark report: {status}",
                "",
                f"- run: `{self.run_id}`",
                f"- manifest hash: `{self.manifest_hash}`",
                f"- nodes: {self.node_count}",
                f"- D-PSGD final accuracy mean: {self.dpsgd_final_accuracy.mean:.6f}",
                f"- FedAvg final accuracy: {self.fedavg_accuracy:.6f}",
                (
                    "- absolute accuracy gate (all nodes >= 90%): "
                    f"{'PASS' if self.minimum_accuracy_90 else 'FAIL'}"
                ),
                (
                    "- publication ready: "
                    f"{'yes' if self.publication_ready else 'no'}"
                ),
                "",
                "[Accuracy and loss curves](metrics.png)",
                "",
                "[Approximate consensus distance](consensus.png)",
                "",
                "[AXL latency and round timing](timing.png)",
                "",
                "[AXL payload goodput](goodput.png)",
                "",
                (
                    "Final approximate consensus distance: "
                    f"{self.final_approximate_consensus_distance}"
                ),
                "",
                "[Raw-data provenance](provenance.json)",
                "",
            )
        )

    @property
    def fedavg_accuracy(self) -> float:
        value = self.fedavg.get("final_accuracy")
        assert isinstance(value, float)
        return value


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

    archives = [_read_archive(root) for root in run_roots]
    manifests = [archive.manifest for archive in archives]
    manifest_hashes = [archive.manifest_hash for archive in archives]
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

    for archive in archives:
        try:
            archive.require_complete(_total_dpsgd_round_count(archive.manifest))
        except RunArchiveError as error:
            raise BenchmarkReportError(
                f"run is not complete: {archive.root}"
            ) from error
        try:
            archive.verify_checkpoint_integrity()
        except RunArchiveError as error:
            raise BenchmarkReportError(
                f"checkpoint integrity failed: {archive.root}"
            ) from error
    evidence_logs = [
        _read_evidence_log(
            path,
            run_id=manifests[0].run_id,
            manifest_hash=manifest_hashes[0],
        )
        for path in event_logs
    ]
    if any(
        isinstance(record, RunFailedEvidence)
        for log in evidence_logs
        for record in log.records
    ):
        raise BenchmarkReportError("failed run cannot be included")
    benchmark_nodes: set[str] = set()
    for log in evidence_logs:
        ready = [
            record
            for record in log.records
            if isinstance(record, BenchmarkNodeReadyEvidence)
        ]
        if len(ready) != 1:
            raise BenchmarkReportError(
                "each event log must contain one benchmark node provenance record"
            )
        record = ready[0]
        if record.benchmark_seed != seed:
            raise BenchmarkReportError("D-PSGD benchmark seed does not match report")
        if record.transport != "axl":
            raise BenchmarkReportError("official D-PSGD transport is not AXL")
        benchmark_nodes.add(record.node_id)

    metrics = [
        record
        for log in evidence_logs
        for record in log.records
        if isinstance(record, RoundMetricsEvidence)
    ]
    if not metrics:
        raise BenchmarkReportError("event logs contain no round metrics")
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
    total_round_count = _total_dpsgd_round_count(manifests[0])
    expected_rounds = set(range(total_round_count))
    if any(
        {event.round_id for event in node_events}
        != expected_rounds
        for node_events in node_metrics.values()
    ):
        raise BenchmarkReportError("round metrics are incomplete")
    _validate_dpsgd_evaluation_schedule(
        node_metrics,
        training_round_count=manifests[0].round_count,
        total_round_count=total_round_count,
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
    consensus = _consensus_curve(
        evidence_logs,
        expected_nodes=expected_nodes,
        expected_rounds=expected_rounds,
    )
    round_timing = _round_timing(metrics)
    transport = _transport_summary(evidence_logs)
    connectivity = _connectivity(metrics)
    failures = _failure_summary(evidence_logs)
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
        mean_within_fedavg_3pp=abs(dpsgd_final_accuracy.mean - fedavg_accuracy) <= 0.03,
        no_node_more_than_5pp_below=min(final_accuracies) >= fedavg_accuracy - 0.05,
        minimum_accuracy_90=min(final_accuracies) >= 0.90,
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
        aggregate_pass=all(report.aggregate_pass for report in reports),
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
        or config.training != manifest.training
    ):
        raise BenchmarkReportError("FedAvg frozen configuration mismatches manifest")
    expected_data_source = (
        "huggingface-uoft-cs-cifar10"
        if manifest.training is not None
        else "torchvision-cifar10"
    )
    if (
        config.data_source != expected_data_source
        or config.test_sample_count != 10_000
        or config.evaluation_interval != 5
        or config.batch_size
        != (manifest.training.batch_size if manifest.training is not None else 32)
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
        "training": (
            manifest.training.model_dump(mode="json")
            if manifest.training is not None
            else None
        ),
        "tensor_schema": manifest.tensor_schema.model_dump(mode="json"),
    }


def _read_archive(root: Path) -> RunArchive:
    try:
        return RunArchive.open(root)
    except RunArchiveError as error:
        raise BenchmarkReportError(f"invalid run archive at {root}") from error


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


def _read_evidence_log(
    path: Path, *, run_id: str, manifest_hash: str
) -> EvidenceLog:
    try:
        return EvidenceLog.open(
            path,
            run_id=run_id,
            manifest_hash=manifest_hash,
        )
    except EvidenceError as error:
        raise BenchmarkReportError(f"invalid evidence log: {path}") from error


def _group_node_metrics(
    metrics: Sequence[RoundMetricsEvidence],
) -> dict[str, list[RoundMetricsEvidence]]:
    grouped: dict[str, list[RoundMetricsEvidence]] = defaultdict(list)
    for event in metrics:
        grouped[event.node_id].append(event)
    return dict(grouped)


def _latest_round(
    metrics: Sequence[RoundMetricsEvidence], node_id: str
) -> int:
    rounds = [event.round_id for event in metrics]
    if not rounds:
        raise BenchmarkReportError(f"node {node_id} has no metric rounds")
    return max(rounds)


def _validate_dpsgd_evaluation_schedule(
    node_metrics: Mapping[str, Sequence[RoundMetricsEvidence]],
    *,
    training_round_count: int,
    total_round_count: int,
) -> None:
    for events in node_metrics.values():
        if len(events) != total_round_count:
            raise BenchmarkReportError("round metrics are incomplete")
        for event in events:
            round_id = event.round_id
            should_evaluate = (
                (
                    round_id < training_round_count
                    and (
                        (round_id + 1) % 5 == 0
                        or round_id + 1 == training_round_count
                    )
                )
                or round_id + 1 == total_round_count
            )
            loss = event.evaluation_loss
            accuracy = event.evaluation_accuracy
            has_loss = loss is not None
            has_accuracy = accuracy is not None
            if has_loss != has_accuracy or should_evaluate != has_loss:
                raise BenchmarkReportError(
                    "D-PSGD evaluation schedule is incomplete or invalid"
                )
            if loss is not None and accuracy is not None and (
                not math.isfinite(loss)
                or not math.isfinite(accuracy)
                or not 0 <= accuracy <= 1
            ):
                raise BenchmarkReportError(
                    "D-PSGD evaluation schedule is incomplete or invalid"
                )


def _total_dpsgd_round_count(manifest: SealedManifest) -> int:
    final_rounds = (
        manifest.training.final_consensus_rounds
        if manifest.training is not None
        else 0
    )
    return manifest.round_count + final_rounds


def _metric_value_for_round(
    metrics: Sequence[RoundMetricsEvidence],
    field: MetricField,
    round_id: int,
    node_id: str,
) -> float:
    values = [
        value
        for event in metrics
        if event.round_id == round_id
        and (value := getattr(event, field)) is not None
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
    metrics: Sequence[RoundMetricsEvidence],
    field: MetricField,
    prefix: str,
) -> tuple[JsonObject, ...]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for event in metrics:
        value = getattr(event, field)
        if value is not None:
            grouped[event.round_id].append(value)
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


def _loss_curve(
    metrics: Sequence[RoundMetricsEvidence],
) -> tuple[JsonObject, ...]:
    local = _metric_curve(metrics, "local_loss", "local_loss")
    evaluation = _metric_curve(metrics, "evaluation_loss", "evaluation_loss")
    grouped: dict[int, JsonObject] = {}
    for point in (*local, *evaluation):
        round_id = cast(int, point["round_id"])
        grouped.setdefault(round_id, {"round_id": round_id}).update(point)
    return tuple(grouped[round_id] for round_id in sorted(grouped))


def _consensus_curve(
    evidence_logs: Sequence[EvidenceLog],
    *,
    expected_nodes: set[str],
    expected_rounds: set[int],
    smoothing_window: int = 3,
) -> tuple[JsonObject, ...]:
    if smoothing_window <= 0:
        raise ValueError("smoothing_window must be positive")
    grouped: dict[int, list[float]] = defaultdict(list)
    sketches: dict[int, list[int]] = defaultdict(list)
    observed: dict[str, set[int]] = defaultdict(set)
    for log in evidence_logs:
        for record in log.records:
            if not isinstance(record, ConsensusDistanceEvidence):
                continue
            grouped[record.round_id].append(record.normalized_rms)
            sketches[record.round_id].append(record.sketch_count)
            observed[record.node_id].add(record.round_id)
    if set(observed) != expected_nodes or any(
        rounds != expected_rounds for rounds in observed.values()
    ):
        raise BenchmarkReportError("consensus evidence is incomplete")
    means: list[float] = []
    result: list[JsonObject] = []
    for round_id in sorted(grouped):
        if (
            len(grouped[round_id]) != len(expected_nodes)
            or set(sketches[round_id]) != {len(expected_nodes)}
        ):
            raise BenchmarkReportError("consensus evidence is inconsistent")
        mean = _summary(grouped[round_id]).mean
        means.append(mean)
        window = means[-smoothing_window:]
        result.append(
            {
                "round_id": round_id,
                "mean_normalized_rms": mean,
                "smoothed_mean_normalized_rms": sum(window) / len(window),
                "minimum_normalized_rms": min(grouped[round_id]),
                "maximum_normalized_rms": max(grouped[round_id]),
                "sketch_count": len(expected_nodes),
            }
        )
    return tuple(result)


def _round_timing(
    metrics: Sequence[RoundMetricsEvidence],
) -> dict[str, object]:
    fields = (
        "local_compute_seconds",
        "peer_wait_seconds",
        "transfer_seconds",
        "mixing_seconds",
        "evaluation_seconds",
    )
    result: dict[str, object] = {}
    for field in fields:
        values = [float(getattr(event, field)) for event in metrics]
        if values:
            result[field] = _summary(values).as_dict()
    by_round: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for event in metrics:
        for field in fields:
            by_round[event.round_id][field].append(
                float(getattr(event, field))
            )
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


def _transport_summary(
    evidence_logs: Sequence[EvidenceLog],
) -> dict[str, object]:
    events = [
        record
        for log in evidence_logs
        for record in log.records
        if isinstance(record, TransferMessageSentEvidence)
    ]
    fields = ("queue_seconds", "send_seconds", "completion_seconds")
    timings: dict[str, object] = {}
    for field in fields:
        values = [float(getattr(event, field)) for event in events]
        if values:
            timings[field] = _summary(values).as_dict()
    retries = [event.retry_count for event in events]
    payload_sizes = [event.payload_bytes for event in events]
    goodputs = [
        event.payload_bytes / event.completion_seconds
        for event in events
        if event.completion_seconds > 0
    ]
    by_round: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for event in events:
        if event.round_id is None:
            continue
        for field in fields:
            by_round[event.round_id][field].append(
                float(getattr(event, field))
            )
        if event.completion_seconds > 0:
            by_round[event.round_id]["goodput_bytes_per_second"].append(
                event.payload_bytes / event.completion_seconds
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


def _connectivity(
    metrics: Sequence[RoundMetricsEvidence],
) -> dict[str, object]:
    edges: dict[tuple[str, str], int] = defaultdict(int)
    for event in metrics:
        edges[(event.node_id, event.peer_id)] += 1
    edge_records = [
        {"node_id": node_id, "peer_id": peer_id, "count": count}
        for (node_id, peer_id), count in sorted(edges.items())
    ]
    return {"edge_count": len(edge_records), "edges": edge_records}


def _failure_summary(
    evidence_logs: Sequence[EvidenceLog],
) -> tuple[JsonObject, ...]:
    result: list[JsonObject] = [
        {
            "event": record.event,
            "node_id": record.node_id,
            "round_id": record.round_id,
            "error_type": record.error_type,
            "error": record.error,
        }
        for log in evidence_logs
        for record in log.records
        if isinstance(record, RunFailedEvidence)
    ]
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


def _curve_values(
    curve: Sequence[Mapping[str, object]], field: str
) -> dict[int, float]:
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


__all__ = [
    "BenchmarkReport",
    "BenchmarkReportError",
    "SeedBenchmarkInput",
    "SummaryStats",
    "ThreeSeedReport",
    "build_benchmark_report",
    "build_three_seed_report",
]
