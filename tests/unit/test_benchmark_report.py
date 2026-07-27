from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from support.sample_manifest import manifest_data

from benchmarks.cifar10.fedavg_reference import (
    FedAvgConfig,
    FedAvgResult,
    FedAvgRound,
)
from benchmarks.cifar10.report import (
    BenchmarkReportError,
    SeedBenchmarkInput,
    build_benchmark_report,
    build_three_seed_report,
)
from dromeus.manifests.canonical import canonical_hash
from dromeus.manifests.models import DraftRunSpec, SealedManifest
from dromeus.persistence.run_store import RunStore
from dromeus.telemetry.events import JsonlEventSink, event_record


def _write_inputs(root: Path, *, seed: int = 17) -> tuple[list[Path], list[Path]]:
    data = manifest_data()
    data["local_steps"] = 1
    data["round_count"] = 1
    draft_data = data.copy()
    del draft_data["participants"]
    del draft_data["initial_checkpoint_hash"]
    del draft_data["tensor_schema"]
    del draft_data["draft_hash"]
    data["draft_hash"] = canonical_hash(DraftRunSpec.model_validate(draft_data))
    manifest = SealedManifest.model_validate(data)
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
        for phase in ("ready", "complete"):
            (run_root / f"topology-{phase}.json").write_text(
                json.dumps(
                    {
                        "our_public_key": f"peer-{index}",
                        "peers": [
                            {"public_key": f"peer-{(index + 1) % 4}"}
                        ],
                    }
                ),
                encoding="utf-8",
            )
        log_path = root / f"node-{index}.jsonl"
        sink = JsonlEventSink(log_path)
        sink.append(
            event_record(
                "benchmark_node_ready",
                run_id=manifest.run_id,
                manifest_hash=manifest_hash,
                node_id=f"peer-{index}",
                benchmark_seed=seed,
                transport="axl",
            )
        )
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
                payload_bytes=100,
            )
        )
        run_roots.append(run_root)
        event_logs.append(log_path)
    return run_roots, event_logs


def _fedavg_result(seed: int = 17) -> FedAvgResult:
    return FedAvgResult(
        rounds=(
            FedAvgRound(
                round_id=0,
                local_losses=(1.0, 0.9, 0.8, 0.7),
                loss=0.75,
                accuracy=0.66,
            ),
        ),
        config=FedAvgConfig(
            local_steps=1,
            round_count=1,
            learning_rate=0.1,
            trainer_seed=seed,
        ),
        initial_checkpoint_hash="2" * 64,
    )


def test_benchmark_report_aggregates_metrics_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    run_roots, event_logs = _write_inputs(tmp_path)

    report = build_benchmark_report(
        run_roots=run_roots,
        event_logs=event_logs,
        fedavg=_fedavg_result(),
        seed=17,
    )

    assert report.dpsgd_final_accuracy.mean == pytest.approx(0.655)
    assert report.mean_within_fedavg_3pp
    assert report.no_node_more_than_5pp_below
    assert report.aggregate_pass
    assert report.connectivity["edge_count"] == 4
    assert report.transport["transfer_count"] == 4
    assert report.transport["payload_bytes_total"] == 400
    assert report.transport["goodput_bytes_per_second"] is not None
    assert report.consensus[0]["mean_normalized_rms"] == pytest.approx(0.215)
    assert report.consensus_comparison["available"] is True

    output = tmp_path / "report"
    report.write_artifacts(output)
    payload = json.loads((output / "report.json").read_text())
    assert payload == report.as_dict()
    assert (output / "metrics.svg").read_text().startswith("<svg")
    assert (output / "approximate-consensus.svg").read_text().startswith("<svg")
    assert (output / "consensus.svg").read_text().startswith("<svg")
    assert (output / "timing.svg").read_text().startswith("<svg")
    assert (output / "goodput.svg").read_text().startswith("<svg")
    assert event_logs[0].as_uri() in (output / "metrics.svg").read_text()
    assert event_logs[0].as_uri() in (output / "approximate-consensus.svg").read_text()
    assert event_logs[0].as_uri() in (output / "consensus.svg").read_text()
    assert event_logs[0].as_uri() in (output / "timing.svg").read_text()
    assert event_logs[0].as_uri() in (output / "goodput.svg").read_text()
    assert "metrics.svg" in (output / "report.md").read_text()


def test_benchmark_report_rejects_failed_runs(tmp_path: Path) -> None:
    run_roots, event_logs = _write_inputs(tmp_path)
    manifest = SealedManifest.model_validate(
        json.loads((run_roots[0] / "manifest.json").read_text())
    )
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
            seed=17,
        )


def test_benchmark_report_rejects_incomplete_run_store(tmp_path: Path) -> None:
    run_roots, event_logs = _write_inputs(tmp_path)
    state_path = run_roots[0] / "state.json"
    state = json.loads(state_path.read_text())
    state["terminal"] = None
    state_path.write_text(json.dumps(state))

    with pytest.raises(BenchmarkReportError, match="not complete"):
        build_benchmark_report(
            run_roots=run_roots,
            event_logs=event_logs,
            fedavg=_fedavg_result(),
            seed=17,
        )


def test_benchmark_report_rejects_dpsgd_seed_without_node_provenance(
    tmp_path: Path,
) -> None:
    run_roots, event_logs = _write_inputs(tmp_path)
    records = [
        json.loads(line)
        for line in event_logs[0].read_text(encoding="utf-8").splitlines()
    ]
    records[0]["benchmark_seed"] = 23
    event_logs[0].write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkReportError, match="benchmark seed"):
        build_benchmark_report(
            run_roots=run_roots,
            event_logs=event_logs,
            fedavg=_fedavg_result(),
            seed=17,
        )


def test_benchmark_report_rejects_missing_topology_snapshot(tmp_path: Path) -> None:
    run_roots, event_logs = _write_inputs(tmp_path)
    (run_roots[0] / "topology-complete.json").unlink()

    with pytest.raises(BenchmarkReportError, match="topology snapshots"):
        build_benchmark_report(
            run_roots=run_roots,
            event_logs=event_logs,
            fedavg=_fedavg_result(),
            seed=17,
        )


def test_three_seed_report_requires_three_compatible_seed_inputs(
    tmp_path: Path,
) -> None:
    inputs: list[SeedBenchmarkInput] = []
    for seed in (17, 23, 29):
        run_roots, event_logs = _write_inputs(
            tmp_path / f"seed-{seed}", seed=seed
        )
        inputs.append(
            SeedBenchmarkInput(
                seed=seed,
                partition_seed=7,
                run_roots=tuple(run_roots),
                event_logs=tuple(event_logs),
                fedavg=_fedavg_result(seed),
                fedavg_result_path=(tmp_path / f"seed-{seed}" / "fedavg.json"),
            )
        )
        inputs[-1].fedavg_result_path.write_text(
            json.dumps(inputs[-1].fedavg.as_dict())
        )

    report = build_three_seed_report(inputs)

    assert [seed_report.seed for seed_report in report.seeds] == [17, 23, 29]
    assert report.aggregate_pass
    assert report.dpsgd_final_accuracy.mean == pytest.approx(0.655)
    assert report.fedavg_final_accuracy.mean == pytest.approx(0.66)
