from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from safetensors.numpy import (
    save_file as _save_file,  # pyright: ignore[reportUnknownVariableType]
)
from support.sample_manifest import manifest_data

from benchmarks.cifar10.fedavg_reference import (
    FedAvgConfig,
    FedAvgResult,
    FedAvgRound,
)
from benchmarks.cifar10.report import (
    BenchmarkReportError,
    SeedBenchmarkInput,
    _consensus_curve,  # pyright: ignore[reportPrivateUsage]
    build_benchmark_report,
    build_submission_benchmark_report,
    build_three_seed_report,
)
from dromeus.manifests.canonical import canonical_hash
from dromeus.manifests.models import DraftRunSpec, SealedManifest
from dromeus.persistence.run_store import RunStore
from dromeus.telemetry.events import JsonlEventSink
from dromeus.telemetry.evidence import (
    BenchmarkNodeReadyEvidence,
    ConsensusDistanceEvidence,
    EvidenceLog,
    RoundMetricsEvidence,
    RunFailedEvidence,
    TransferMessageSentEvidence,
    append_evidence,
)


def _write_inputs(
    root: Path,
    *,
    seed: int = 17,
    round_count: int = 1,
    consensus_sketch_seed: int = 9,
    accuracies: tuple[float, float, float, float] = (0.94, 0.95, 0.96, 0.97),
) -> tuple[list[Path], list[Path]]:
    data = manifest_data()
    data["local_steps"] = 1
    data["round_count"] = round_count
    data["consensus_sketch"]["seed"] = consensus_sketch_seed
    draft_data = data.copy()
    del draft_data["participants"]
    del draft_data["initial_checkpoint_hash"]
    del draft_data["tensor_schema"]
    del draft_data["draft_hash"]
    data["draft_hash"] = canonical_hash(DraftRunSpec.model_validate(draft_data))
    manifest = SealedManifest.model_validate(data)
    run_roots: list[Path] = []
    event_logs: list[Path] = []
    for index, accuracy in enumerate(accuracies):
        run_root = root / f"node-{index}"
        store = RunStore(run_root)
        manifest_hash = store.initialize(manifest)
        for round_id in range(round_count):
            store.persist_commit(
                committed_round=round_id,
                algorithm_state={"weight": np.array([1], dtype=np.float32)},
                state_checksum=f"{index * round_count + round_id + 1:064x}",
                schedule={
                    "round_id": round_id,
                    "peer": f"peer-{(index + 1) % 4}",
                },
            )
        store.record_terminal("complete", {"committed_rounds": round_count})
        for phase in ("ready", "complete"):
            (run_root / f"topology-{phase}.json").write_text(
                json.dumps(
                    {
                        "our_public_key": f"peer-{index}",
                        "peers": [{"public_key": f"peer-{(index + 1) % 4}"}],
                    }
                ),
                encoding="utf-8",
            )
        (run_root / "hardware.json").write_text(
            json.dumps(
                {
                    "node_id": f"peer-{index}",
                    "provider": "local",
                    "region": "local",
                    "machine_type": "test",
                    "cpu_model": "test-cpu",
                    "cpu_count": 4,
                    "memory_bytes": 8 * 1024 * 1024 * 1024,
                    "accelerator": "none",
                }
            ),
            encoding="utf-8",
        )
        log_path = root / f"node-{index}.jsonl"
        sink = JsonlEventSink(log_path)
        assert append_evidence(
            sink,
            BenchmarkNodeReadyEvidence(
                run_id=manifest.run_id,
                manifest_hash=manifest_hash,
                node_id=f"peer-{index}",
                benchmark_seed=seed,
                transport="axl",
            )
        )
        for round_id in range(round_count):
            should_evaluate = (
                (round_id + 1) % 5 == 0 or round_id + 1 == round_count
            )
            assert append_evidence(
                sink,
                RoundMetricsEvidence(
                    run_id=manifest.run_id,
                    manifest_hash=manifest_hash,
                    node_id=f"peer-{index}",
                    message_id=f"metric-peer-{index}-{round_id}",
                    peer_id=f"peer-{(index + 1) % 4}",
                    round_id=round_id,
                    local_loss=1.0 - index / 10,
                    evaluation_loss=0.8 - index / 10 if should_evaluate else None,
                    evaluation_accuracy=accuracy if should_evaluate else None,
                    local_compute_seconds=1.0,
                    peer_wait_seconds=0.2,
                    transfer_seconds=0.3,
                    mixing_seconds=0.1,
                    evaluation_seconds=0.4,
                    retries=index,
                )
            )
            assert append_evidence(
                sink,
                ConsensusDistanceEvidence(
                    run_id=manifest.run_id,
                    manifest_hash=manifest_hash,
                    node_id=f"peer-{index}",
                    message_id=f"consensus-peer-{index}-{round_id}",
                    round_id=round_id,
                    normalized_rms=0.2 + index / 100 + round_id / 10,
                    sketch_count=4,
                )
            )
            assert append_evidence(
                sink,
                TransferMessageSentEvidence(
                    run_id=manifest.run_id,
                    manifest_hash=manifest_hash,
                    node_id=f"peer-{index}",
                    message_id=f"transfer-peer-{index}-{round_id}",
                    peer_id=f"peer-{(index + 1) % 4}",
                    round_id=round_id,
                    message_type="CHUNK",
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


def _fedavg_result(
    seed: int = 17,
    *,
    round_count: int = 1,
    consensus_sketch_seed: int = 9,
    accuracy: float = 0.96,
) -> FedAvgResult:
    data = manifest_data()
    data["local_steps"] = 1
    data["round_count"] = round_count
    data["consensus_sketch"]["seed"] = consensus_sketch_seed
    manifest = SealedManifest.model_validate(data)
    return FedAvgResult(
        rounds=tuple(
            FedAvgRound(
                round_id=round_id,
                local_losses=(1.0, 0.9, 0.8, 0.7),
                loss=(
                    0.75
                    if (round_id + 1) % 5 == 0 or round_id + 1 == round_count
                    else None
                ),
                accuracy=(
                    accuracy
                    if (round_id + 1) % 5 == 0 or round_id + 1 == round_count
                    else None
                ),
            )
            for round_id in range(round_count)
        ),
        config=FedAvgConfig.from_manifest(
            manifest,
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

    assert report.dpsgd_final_accuracy.mean == pytest.approx(0.955)
    assert report.mean_within_fedavg_3pp
    assert report.no_node_more_than_5pp_below
    assert report.minimum_accuracy_90
    assert report.aggregate_pass
    assert report.connectivity["edge_count"] == 4
    assert report.topology["classification"] == "partial-participant-mesh"
    assert report.topology["participant_edge_count"] == 4
    assert len(report.hardware) == 4
    assert report.transport["transfer_count"] == 4
    assert report.transport["payload_bytes_total"] == 400
    assert report.transport["goodput_bytes_per_second"] is not None
    assert report.consensus[0]["mean_normalized_rms"] == pytest.approx(0.215)
    assert report.consensus[0]["smoothed_mean_normalized_rms"] == pytest.approx(0.215)
    assert report.final_approximate_consensus_distance == pytest.approx(0.215)
    state = json.loads((run_roots[0] / "state.json").read_text())
    assert "pre_mix_checkpoints" not in state
    assert "post_mix_checkpoints" not in state

    output = tmp_path / "report"
    report.write_artifacts(output)
    payload = json.loads((output / "report.json").read_text())
    assert payload == report.as_dict()
    png_signature = b"\x89PNG\r\n\x1a\n"
    for name in (
        "metrics.png",
        "consensus.png",
        "timing.png",
        "goodput.png",
    ):
        assert (output / name).read_bytes().startswith(png_signature)
    provenance = json.loads((output / "provenance.json").read_text())
    assert str(event_logs[0].resolve()) in provenance["event_logs"]
    assert "metrics.png" in (output / "report.md").read_text()


def test_consensus_plot_data_reproduces_from_jsonl_alone(tmp_path: Path) -> None:
    run_roots, event_logs = _write_inputs(tmp_path, round_count=4)

    report = build_benchmark_report(
        run_roots=run_roots,
        event_logs=event_logs,
        fedavg=_fedavg_result(round_count=4),
        seed=17,
    )

    assert report.final_approximate_consensus_distance == pytest.approx(0.515)
    assert report.consensus[-1]["smoothed_mean_normalized_rms"] == pytest.approx(
        0.415
    )


def test_benchmark_report_rejects_missing_consensus_evidence(tmp_path: Path) -> None:
    run_roots, event_logs = _write_inputs(tmp_path, round_count=2)
    records = event_logs[0].read_text(encoding="utf-8").splitlines()
    event_logs[0].write_text(
        "\n".join(
            line
            for line in records
            if not (
                json.loads(line).get("event") == "consensus_distance"
                and json.loads(line).get("round_id") == 1
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkReportError, match="consensus evidence"):
        build_benchmark_report(
            run_roots=run_roots,
            event_logs=event_logs,
            fedavg=_fedavg_result(round_count=2),
            seed=17,
        )


def test_submission_consensus_allows_partial_terminal_round(tmp_path: Path) -> None:
    run_roots, event_logs = _write_inputs(tmp_path)
    manifest = SealedManifest.model_validate_json(
        (run_roots[0] / "manifest.json").read_text()
    )
    manifest_hash = canonical_hash(manifest)
    for index in range(3):
        assert append_evidence(
            JsonlEventSink(event_logs[index]),
            ConsensusDistanceEvidence(
                run_id=manifest.run_id,
                manifest_hash=manifest_hash,
                node_id=f"peer-{index}",
                message_id=f"terminal-consensus-peer-{index}",
                round_id=1,
                normalized_rms=0.01,
                sketch_count=4,
            ),
        )
    logs = tuple(
        EvidenceLog.open(
            path,
            run_id=manifest.run_id,
            manifest_hash=manifest_hash,
        )
        for path in event_logs
    )

    consensus = _consensus_curve(
        logs,
        expected_nodes={f"peer-{index}" for index in range(4)},
        expected_rounds={0, 1},
        allow_incomplete_terminal_round=True,
    )

    assert [point["round_id"] for point in consensus] == [0]


def test_relative_parity_cannot_pass_below_90_percent(tmp_path: Path) -> None:
    run_roots, event_logs = _write_inputs(
        tmp_path,
        accuracies=(0.24, 0.25, 0.26, 0.27),
    )

    report = build_benchmark_report(
        run_roots=run_roots,
        event_logs=event_logs,
        fedavg=_fedavg_result(accuracy=0.26),
        seed=17,
    )

    assert report.mean_within_fedavg_3pp
    assert report.no_node_more_than_5pp_below
    assert not report.minimum_accuracy_90
    assert not report.aggregate_pass
    assert not report.publication_ready
    assert report.quality_gate_required


def test_benchmark_report_rejects_failed_runs(tmp_path: Path) -> None:
    run_roots, event_logs = _write_inputs(tmp_path)
    manifest = SealedManifest.model_validate(
        json.loads((run_roots[0] / "manifest.json").read_text())
    )
    assert append_evidence(
        JsonlEventSink(event_logs[0]),
        RunFailedEvidence(
            run_id=manifest.run_id,
            manifest_hash=canonical_hash(manifest),
            node_id="peer-0",
            message_id="run-failed-peer-0-0",
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


def test_benchmark_report_rejects_legacy_archive_without_hashes(
    tmp_path: Path,
) -> None:
    run_roots, event_logs = _write_inputs(tmp_path)
    state_path = run_roots[0] / "state.json"
    state = json.loads(state_path.read_text())
    del state["archive_version"]
    del state["prepared_commit"]
    checkpoint = state["algorithm_state"]
    state["algorithm_state"] = checkpoint["path"]
    state["pre_mix_checkpoints"] = {"0": checkpoint["path"]}
    state["post_mix_checkpoints"] = {"0": checkpoint["path"]}
    state_path.write_text(json.dumps(state))

    with pytest.raises(BenchmarkReportError, match="checkpoint integrity"):
        build_benchmark_report(
            run_roots=run_roots,
            event_logs=event_logs,
            fedavg=_fedavg_result(),
            seed=17,
        )


def test_submission_report_accepts_legacy_final_checkpoint_only(
    tmp_path: Path,
) -> None:
    run_roots, event_logs = _write_inputs(tmp_path)
    for run_root in run_roots:
        state_path = run_root / "state.json"
        state = json.loads(state_path.read_text())
        manifest = SealedManifest.model_validate_json(
            (run_root / "manifest.json").read_text()
        )
        algorithm_state = cast(dict[str, object], state["algorithm_state"])
        checkpoint_relative_path = algorithm_state["path"]
        assert isinstance(checkpoint_relative_path, str)
        checkpoint_path = run_root / checkpoint_relative_path
        _save_file(
            {
                tensor.name: np.ones(tensor.shape, dtype=tensor.dtype)
                for tensor in manifest.tensor_schema.tensors
            }
            | {"__dromeus_training__.completed_steps": np.array([1])},
            checkpoint_path,
        )
        del state["archive_version"]
        del state["prepared_commit"]
        state["algorithm_state"] = checkpoint_relative_path
        state["pre_mix_checkpoints"] = {
            "0": "checkpoints/missing-pre-mix.safetensors"
        }
        state["post_mix_checkpoints"] = {
            "0": "checkpoints/missing-post-mix.safetensors"
        }
        state_path.write_text(json.dumps(state), encoding="utf-8")
    for event_log in event_logs:
        records = [json.loads(line) for line in event_log.read_text().splitlines()]
        for record in records:
            record.pop("evidence_version")
        records.insert(0, {"event": "node_start", "run_id": "ignored"})
        event_log.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )

    report = build_submission_benchmark_report(
        run_roots=run_roots,
        event_logs=event_logs,
        fedavg=_fedavg_result(),
        seed=17,
    )

    assert report.aggregate_pass


def test_submission_report_rejects_final_checkpoint_hash_mismatch(
    tmp_path: Path,
) -> None:
    run_roots, event_logs = _write_inputs(tmp_path)
    state = json.loads((run_roots[0] / "state.json").read_text())
    algorithm_state = cast(dict[str, object], state["algorithm_state"])
    checkpoint_relative_path = algorithm_state["path"]
    assert isinstance(checkpoint_relative_path, str)
    (run_roots[0] / checkpoint_relative_path).write_bytes(b"tampered")

    with pytest.raises(BenchmarkReportError, match="final checkpoint integrity"):
        build_submission_benchmark_report(
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


def test_benchmark_report_rejects_missing_hardware_metadata(tmp_path: Path) -> None:
    run_roots, event_logs = _write_inputs(tmp_path)
    (run_roots[0] / "hardware.json").unlink()

    with pytest.raises(BenchmarkReportError, match="hardware metadata"):
        build_benchmark_report(
            run_roots=run_roots,
            event_logs=event_logs,
            fedavg=_fedavg_result(),
            seed=17,
        )


@pytest.mark.parametrize(
    ("round_id", "evaluation_loss", "evaluation_accuracy"),
    (
        (4, None, None),
        (0, 0.8, 0.6),
        (0, 0.8, None),
    ),
)
def test_benchmark_report_rejects_invalid_dpsgd_evaluation_schedule(
    tmp_path: Path,
    round_id: int,
    evaluation_loss: float | None,
    evaluation_accuracy: float | None,
) -> None:
    run_roots, event_logs = _write_inputs(tmp_path, round_count=6)
    records = [
        json.loads(line)
        for line in event_logs[0].read_text(encoding="utf-8").splitlines()
    ]
    metric = next(
        record
        for record in records
        if record.get("event") == "round_metrics"
        and record.get("round_id") == round_id
    )
    metric["evaluation_loss"] = evaluation_loss
    metric["evaluation_accuracy"] = evaluation_accuracy
    event_logs[0].write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkReportError, match="D-PSGD evaluation schedule"):
        build_benchmark_report(
            run_roots=run_roots,
            event_logs=event_logs,
            fedavg=_fedavg_result(),
            seed=17,
        )


@pytest.mark.parametrize(
    ("round_id", "loss", "accuracy"),
    (
        (4, None, None),
        (0, 0.75, 0.96),
    ),
)
def test_benchmark_report_rejects_invalid_fedavg_evaluation_schedule(
    tmp_path: Path,
    round_id: int,
    loss: float | None,
    accuracy: float | None,
) -> None:
    run_roots, event_logs = _write_inputs(tmp_path, round_count=6)
    result = _fedavg_result(round_count=6)
    rounds = list(result.rounds)
    rounds[round_id] = FedAvgRound(
        round_id=round_id,
        local_losses=rounds[round_id].local_losses,
        loss=loss,
        accuracy=accuracy,
    )

    with pytest.raises(BenchmarkReportError, match="FedAvg evaluation schedule"):
        build_benchmark_report(
            run_roots=run_roots,
            event_logs=event_logs,
            fedavg=FedAvgResult(
                rounds=tuple(rounds),
                config=result.config,
                initial_checkpoint_hash=result.initial_checkpoint_hash,
            ),
            seed=17,
        )


def test_three_seed_report_requires_three_compatible_seed_inputs(
    tmp_path: Path,
) -> None:
    inputs: list[SeedBenchmarkInput] = []
    for seed in (17, 23, 29):
        run_roots, event_logs = _write_inputs(
            tmp_path / f"seed-{seed}",
            seed=seed,
            consensus_sketch_seed=seed,
        )
        inputs.append(
            SeedBenchmarkInput(
                seed=seed,
                run_roots=tuple(run_roots),
                event_logs=tuple(event_logs),
                fedavg_result_path=(tmp_path / f"seed-{seed}" / "fedavg.json"),
            )
        )
        _fedavg_result(seed, consensus_sketch_seed=seed).write(
            inputs[-1].fedavg_result_path
        )

    report = build_three_seed_report(inputs)

    assert [seed_report.seed for seed_report in report.seeds] == [17, 23, 29]
    assert report.aggregate_pass
    assert report.dpsgd_final_accuracy.mean == pytest.approx(0.955)
    assert report.fedavg_final_accuracy.mean == pytest.approx(0.96)


def test_three_seed_report_rejects_mismatched_raw_fedavg_seed(
    tmp_path: Path,
) -> None:
    inputs: list[SeedBenchmarkInput] = []
    for seed in (17, 23, 29):
        run_roots, event_logs = _write_inputs(tmp_path / f"seed-{seed}", seed=seed)
        result_path = tmp_path / f"seed-{seed}" / "fedavg.json"
        _fedavg_result(99 if seed == 17 else seed).write(result_path)
        inputs.append(
            SeedBenchmarkInput(
                seed=seed,
                run_roots=tuple(run_roots),
                event_logs=tuple(event_logs),
                fedavg_result_path=result_path,
            )
        )

    with pytest.raises(BenchmarkReportError, match="FedAvg raw result"):
        build_three_seed_report(inputs)
