from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from support.sample_manifest import manifest_data

from benchmarks.cifar10.fedavg_reference import FedAvgResult, FedAvgRound
from benchmarks.cifar10.report import (
    BenchmarkReportError,
    build_benchmark_report,
)
from dromeus.manifests.canonical import canonical_hash
from dromeus.manifests.models import SealedManifest
from dromeus.persistence.run_store import RunStore
from dromeus.telemetry.events import JsonlEventSink, event_record


def _write_inputs(root: Path) -> tuple[list[Path], list[Path]]:
    manifest = SealedManifest.model_validate(manifest_data())
    run_roots: list[Path] = []
    event_logs: list[Path] = []
    accuracies = (0.64, 0.65, 0.66, 0.67)
    for index, accuracy in enumerate(accuracies):
        run_root = root / f"node-{index}"
        store = RunStore(run_root)
        manifest_hash = store.initialize(manifest)
        store.persist_commit(
            committed_round=0,
            algorithm_state={"weight": np.array([1], dtype=np.float32)},
            pre_mix_state={"weight": np.array([index], dtype=np.float32)},
            post_mix_state={"weight": np.array([1], dtype=np.float32)},
            state_checksum=f"{index + 1:064x}",
            schedule={"round_id": 0, "peer": f"peer-{(index + 1) % 4}"},
        )
        store.record_terminal("complete", {"committed_rounds": 1})
        log_path = root / f"node-{index}.jsonl"
        sink = JsonlEventSink(log_path)
        sink.append(
            event_record(
                "round_metrics",
                run_id=manifest.run_id,
                manifest_hash=manifest_hash,
                node_id=f"peer-{index}",
                peer_id=f"peer-{(index + 1) % 4}",
                round_id=0,
                local_loss=1.0 - index / 10,
                evaluation_loss=0.8 - index / 10,
                evaluation_accuracy=accuracy,
                local_compute_seconds=1.0,
                peer_wait_seconds=0.2,
                transfer_seconds=0.3,
                mixing_seconds=0.1,
                evaluation_seconds=0.4,
                retries=index,
            )
        )
        sink.append(
            event_record(
                "consensus_distance",
                run_id=manifest.run_id,
                manifest_hash=manifest_hash,
                node_id=f"peer-{index}",
                round_id=0,
                normalized_rms=0.2 + index / 100,
                sketch_count=4,
            )
        )
        sink.append(
            event_record(
                "transfer_message_sent",
                run_id=manifest.run_id,
                manifest_hash=manifest_hash,
                node_id=f"peer-{index}",
                peer_id=f"peer-{(index + 1) % 4}",
                round_id=0,
                queue_seconds=0.05,
                send_seconds=0.1,
                retry_count=index,
                completion_seconds=0.2,
            )
        )
        run_roots.append(run_root)
        event_logs.append(log_path)
    return run_roots, event_logs


def _fedavg_result() -> FedAvgResult:
    return FedAvgResult(
        rounds=(
            FedAvgRound(
                round_id=0,
                local_losses=(1.0, 0.9, 0.8, 0.7),
                loss=0.75,
                accuracy=0.66,
            ),
        )
    )


def test_benchmark_report_aggregates_metrics_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    run_roots, event_logs = _write_inputs(tmp_path)

    report = build_benchmark_report(
        run_roots=run_roots,
        event_logs=event_logs,
        fedavg=_fedavg_result(),
    )

    assert report.dpsgd_final_accuracy.mean == pytest.approx(0.655)
    assert report.mean_within_fedavg_3pp
    assert report.no_node_more_than_5pp_below
    assert report.aggregate_pass
    assert report.connectivity["edge_count"] == 4
    assert report.transport["transfer_count"] == 4
    assert report.consensus[0]["mean_normalized_rms"] == pytest.approx(0.215)

    output = tmp_path / "report"
    report.write_artifacts(output)
    payload = json.loads((output / "report.json").read_text())
    assert payload == report.as_dict()
    assert (output / "metrics.svg").read_text().startswith("<svg")
    assert (output / "consensus.svg").read_text().startswith("<svg")
    assert "metrics.svg" in (output / "report.md").read_text()


def test_benchmark_report_rejects_failed_runs(tmp_path: Path) -> None:
    run_roots, event_logs = _write_inputs(tmp_path)
    manifest = SealedManifest.model_validate(manifest_data())
    JsonlEventSink(event_logs[0]).append(
        event_record(
            "run_failed",
            run_id=manifest.run_id,
            manifest_hash=canonical_hash(manifest),
            node_id="peer-0",
            round_id=0,
            error_type="TestError",
            error="failed",
        )
    )

    with pytest.raises(BenchmarkReportError, match="failed run"):
        build_benchmark_report(
            run_roots=run_roots,
            event_logs=event_logs,
            fedavg=_fedavg_result(),
        )
