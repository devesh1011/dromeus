"""Run the frozen synthetic non-IID development suite through production nodes."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import torch

from benchmarks.federated.controls import (
    ControlResult,
    Trajectories,
    load_tensors,
    run_control,
)
from benchmarks.federated.data_plan import PROFILES, DataPlan, create_data_plan
from benchmarks.federated.evidence import (
    InMemoryTransport,
    MeasuredTransport,
    TrajectoryStore,
    compare_trajectories,
    dispersion,
    source_provenance,
)
from benchmarks.federated.local_axl import (
    AXL_COMMIT,
    LocalAXLNode,
    ensure_axl_binary,
    local_axl_cluster,
)
from benchmarks.federated.workload import (
    Experiment,
    build_draft,
    build_model,
    evaluate,
    load_experiment,
    local_data,
    sample_budget,
    summarize_nodes,
    write_json,
)
from dromeus.adapters.classification.runtime import (
    prepare_local_training,
)
from dromeus.adapters.classification.torch_trainer import create_initial_checkpoint
from dromeus.manifests.canonical import canonical_json, file_sha256
from dromeus.membership.formation import FormationResult, create_invitation
from dromeus.persistence.archive import RunArchive
from dromeus.runtime import (
    FailureConfig,
    InitiatorFormation,
    NodeRuntime,
    ParticipantFormation,
    TrainingConfig,
)
from dromeus.telemetry.events import JsonlEventSink
from dromeus.telemetry.evidence import RoundMetricsEvidence
from dromeus.telemetry.metrics import JsonlMetricsPublisher
from dromeus.transport.axl import AXLBridgeConfig, AXLTransport
from dromeus.transport.interface import AsyncTransport, ReceivedBytes


@dataclass
class RuntimeResult:
    report: dict[str, Any]
    trajectories: Trajectories | None


async def _runtime_run(
    *,
    root: Path,
    experiment: Experiment,
    plan: DataPlan,
    variant: str,
    keys: tuple[str, ...],
    bridges: tuple[LocalAXLNode, ...] | None,
    checkpoint_path: Path,
    source_commit: str,
) -> RuntimeResult:
    root.mkdir(parents=True, exist_ok=False)
    draft = build_draft(
        experiment,
        plan,
        variant=variant,
        run_id=f"federated-v1-{plan.profile}-{variant}-{uuid4().hex[:10]}",
        source_commit=source_commit,
        axl_version=f"axl-{AXL_COMMIT}" if bridges else "in-memory-development",
    )
    (root / "draft.json").write_bytes(canonical_json(draft))
    queues = {key: asyncio.Queue[ReceivedBytes]() for key in keys}
    transports: list[MeasuredTransport] = []
    prepared = [
        prepare_local_training(
            draft=draft,
            data=local_data(plan, rank),
            model=build_model(experiment.seed),
            model_definition=experiment.definition,
            seed=experiment.seed,
        )
        for rank in range(plan.world_size)
    ]
    stores = [
        TrajectoryStore(root / f"node-{rank}" / "run-store", experiment.rounds)
        for rank in range(plan.world_size)
    ]
    sinks = [
        JsonlEventSink(root / f"node-{rank}" / "logs" / "dromeus.jsonl")
        for rank in range(plan.world_size)
    ]
    for rank, key in enumerate(keys):
        underlying: AsyncTransport = (
            AXLTransport(AXLBridgeConfig(base_url=bridges[rank].bridge_url))
            if bridges
            else InMemoryTransport(key, queues)
        )
        if await underlying.local_public_key() != key:
            raise ValueError(
                "live public keys do not match the recorded sorted membership"
            )
        transports.append(MeasuredTransport(underlying))
    nodes = [
        NodeRuntime(
            transport=transports[rank],
            draft=draft,
            environment=draft.environment,
            dataset=draft.dataset,
            artifact_root=root / f"node-{rank}" / "formation",
            event_sink=sinks[rank],
            failure=FailureConfig(stores[rank], root / f"node-{rank}" / "rounds"),
            local_tensor_schema=prepared[rank].tensor_schema,
        )
        for rank in range(plan.world_size)
    ]
    invitation = create_invitation(
        draft=draft,
        initiator_public_key=keys[0],
        bootstrap_uri="axl://local-development-full-mesh",
    )
    initial = load_tensors(str(checkpoint_path))
    started = time.monotonic()

    async def run_rank(rank: int) -> None:
        def factory(result: FormationResult) -> TrainingConfig:
            metrics = JsonlMetricsPublisher(
                sink=sinks[rank],
                run_id=draft.run_id,
                manifest_hash=result.manifest_hash,
                node_id=keys[rank],
            )
            config = prepared[rank].build_config(
                result=result,
                local_public_key=keys[rank],
                run_root=root / f"node-{rank}",
                metrics_publisher=metrics,
                evaluation_interval=1,
            )
            stores[rank].training_started = time.monotonic()
            return replace(config, run_store=stores[rank])

        await nodes[rank].run_to_completion(
            formation=InitiatorFormation(
                bootstrap_uri=invitation.bootstrap_uri,
                checkpoint_path=checkpoint_path,
                tensor_schema=prepared[0].tensor_schema,
            )
            if rank == 0
            else ParticipantFormation(invitation=invitation),
            training_factory=factory,
        )

    errors: list[str] = []
    try:
        outcomes = await asyncio.wait_for(
            asyncio.gather(
                *(run_rank(rank) for rank in range(plan.world_size)),
                return_exceptions=True,
            ),
            timeout=300.0,
        )
        errors = [
            f"{type(result).__name__}: {result}"
            for result in outcomes
            if isinstance(result, BaseException)
        ]
    except Exception:
        errors.append(traceback.format_exc())
    finally:
        await asyncio.gather(*(node.stop() for node in nodes), return_exceptions=True)
        for rank, transport in enumerate(transports):
            write_json(root / f"node-{rank}" / "wire-messages.json", transport.records)
    if errors:
        report = {
            "status": "failed",
            "run_id": draft.run_id,
            "errors": errors,
            "seconds": time.monotonic() - started,
        }
        write_json(root / "result.json", report)
        return RuntimeResult(report, None)

    trajectories: Trajectories = []
    node_reports: list[dict[str, object]] = []
    metric_rows: list[RoundMetricsEvidence] = []
    for rank, store in enumerate(stores):
        archive = RunArchive.open(root / f"node-{rank}" / "run-store")
        state = store.load_state()
        if (
            state.committed_round != experiment.rounds - 1
            or state.terminal is None
            or state.terminal.result != "complete"
        ):
            raise RuntimeError("runtime did not retain the complete frozen horizon")
        assert archive.algorithm_state is not None
        weights = store.snapshots[experiment.rounds - 1]
        trajectories.append(
            [initial, *(store.snapshots[step] for step in range(experiment.rounds))]
        )
        steps = experiment.rounds * experiment.settings["inner_steps"]
        round_metrics: list[RoundMetricsEvidence] = []
        for line in sinks[rank].path.read_text().splitlines():
            record = cast(dict[str, Any], json.loads(line))
            if record.get("event") == "round_metrics":
                metric = RoundMetricsEvidence.model_validate_json(line)
                if metric.run_id != draft.run_id or metric.node_id != keys[rank]:
                    raise ValueError("round metrics belong to another run or node")
                round_metrics.append(metric)
        if sorted(metric.round_id for metric in round_metrics) != list(
            range(experiment.rounds)
        ):
            raise ValueError("retained round timing/byte evidence is incomplete")
        metric_rows.extend(round_metrics)
        node_reports.append(
            {
                "rank": rank,
                "public_key": keys[rank],
                "local_heldout": evaluate(weights, plan.nodes[rank].evaluation.path),
                "common_heldout": evaluate(weights, plan.common_evaluation.path),
                "optimizer_steps": steps,
                "sample_presentations": sample_budget(
                    plan.nodes[rank].train.sample_count,
                    steps=steps,
                    batch_size=experiment.settings["batch_size"],
                ),
                "metrics": [item.model_dump(mode="json") for item in round_metrics],
                "transfer_diagnostics": [
                    dict(item) for item in state.transfer_diagnostics
                ],
                "commit_intervals_seconds": store.commit_intervals(),
                ("commit_interval_note"): (
                    "first interval starts at training construction and includes "
                    "readiness/setup; later intervals include actual compute, "
                    "transport, evaluation and commit persistence"
                ),
                "wire": transports[rank].summary(),
                "final_checkpoint_sha256": state.algorithm_state.sha256
                if state.algorithm_state
                else None,
            }
        )
    wire_totals = {"telemetry": 0, "round-protocol": 0, "formation-control": 0}
    for measured in transports:
        for record in measured.records:
            if record["accepted_by_transport"]:
                wire_totals[str(record["category"])] += int(str(record["bytes"]))
    intervals = [interval for store in stores for interval in store.commit_intervals()]
    report = {
        "status": "complete",
        "run_id": draft.run_id,
        "variant": variant,
        "transport": "real-local-axl" if bridges else "in-memory-development",
        "manifest_hash": nodes[0].formation_result.manifest_hash,
        "data_plan_sha256": plan.sha256,
        "experiment_sha256": experiment.sha256,
        "seconds": time.monotonic() - started,
        "weight_dispersion": dispersion(trajectories),
        "accepted_wire_bytes": wire_totals,
        "raw_update_tensor_bytes": sum(value.nbytes for value in initial.values())
        * 2
        * plan.world_size
        * experiment.rounds,
        "encoded_artifact_bytes": sum(
            row.encoded_artifact_bytes or 0 for row in metric_rows
        ),
        "round_retries": sum(row.retries for row in metric_rows),
        "maximum_error_feedback_residual_l2": max(
            row.error_feedback_residual_l2_norm or 0.0 for row in metric_rows
        ),
        "maximum_error_feedback_residual_to_signal": max(
            row.error_feedback_residual_to_signal_ratio or 0.0 for row in metric_rows
        ),
        "mean_commit_interval_seconds": sum(intervals) / len(intervals),
        **summarize_nodes(node_reports),
    }
    write_json(root / "result.json", report)
    return RuntimeResult(report, trajectories)


def assess_profile(
    experiment: Experiment,
    identity: RuntimeResult,
    compressed: RuntimeResult,
    reference: Trajectories,
) -> dict[str, Any]:
    gates = experiment.settings["gates"]
    utility: dict[str, bool] = {}
    for name, result in (("identity", identity), ("bitmap-int8", compressed)):
        utility[name] = result.report["status"] == "complete" and (
            result.report["common_mean_accuracy"]
            >= gates["common_heldout_mean_accuracy_min"]
            and result.report["common_worst_accuracy"]
            >= gates["common_heldout_worst_accuracy_min"]
        )
    completed = (
        identity.trajectories is not None and compressed.trajectories is not None
    )
    gap = (
        float(
            identity.report["common_mean_accuracy"]
            - compressed.report["common_mean_accuracy"]
        )
        if completed
        else None
    )
    parity = (
        compare_trajectories(
            identity.trajectories,
            reference,
            atol=gates["reference_atol"],
            rtol=gates["reference_rtol"],
        )
        if identity.trajectories is not None
        else {"passed": False, "reason": "identity runtime failed"}
    )
    passed = (
        all(utility.values())
        and gap is not None
        and gap <= gates["compressed_accuracy_gap_max"]
        and parity["passed"]
    )
    return {
        "decision": "pass" if passed else "refine" if completed else "reject",
        "utility_passed": utility,
        "identity_minus_compressed_accuracy": gap,
        "compression_gap_passed": gap is not None
        and gap <= gates["compressed_accuracy_gap_max"],
        "identity_reference_parity": parity,
        "wire_ratio_identity_to_compressed": (
            identity.report["accepted_wire_bytes"]["round-protocol"]
            / compressed.report["accepted_wire_bytes"]["round-protocol"]
            if completed
            else None
        ),
    }


def _number(value: object) -> str:
    return "n/a" if value is None else f"{float(cast(Any, value)):.3f}"


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Synthetic local-data development evaluation",
        "",
        f"Transport: **{result['transport']}**. Seed 17; four local CPU workers; "
        "eight rounds of 50 Adam steps.",
        "All training nodes are separate NodeRuntime instances in one Python "
        "process, with thread-offloaded CPU work. AXL mode adds four real AXL "
        "subprocesses on that same host.",
        "",
        (
            "This is synthetic development evidence. It does not replace the "
            "M2 GPU/WAN matrix, NCCL reference, or identity ablation. Every "
            "profile uses the same frozen settings and explicit balanced "
            "common held-out set. The reference independently implements the "
            "outer optimizer but shares the generic inner trainer and "
            "workload; its pair exchange is synchronous in-process."
        ),
        "",
        f"Frozen experiment SHA-256: `{result['experiment_sha256']}`.",
        "",
        (
            "Predeclared gates: common held-out mean accuracy ≥ 0.80; worst-"
            "node accuracy ≥ 0.60 for each codec; compressed mean at most "
            "0.10 below identity; identity/reference parity at every retained"
            " outer step within atol=rtol=1e-6. No M2 compression-ratio or "
            "residual threshold is reused."
        ),
        "",
        (
            "| Profile | Identity mean/min | Compressed mean/min | Reference "
            "mean/min | Local-only mean/min | Parity | Decision |"
        ),
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for profile in result["profiles"]:

        def pair(item: dict[str, Any]) -> str:
            return (
                f"{_number(item.get('common_mean_accuracy'))}/"
                f"{_number(item.get('common_worst_accuracy'))}"
            )

        assessment = profile["assessment"]
        lines.append(
            f"| {profile['profile']} | {pair(profile['identity'])} | "
            f"{pair(profile['bitmap-int8'])} | {pair(profile['reference'])} | "
            f"{pair(profile['local-only'])} | "
            f"{assessment['identity_reference_parity']['passed']} | "
            f"{assessment['decision']} |"
        )
    lines.extend(
        [
            "",
            "| Profile | Identity round wire bytes | Compressed round wire bytes | "
            "Identity/compressed | Compressed max residual L2 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for profile in result["profiles"]:
        identity_bytes = (
            profile["identity"]
            .get("accepted_wire_bytes", {})
            .get("round-protocol", "n/a")
        )
        compressed_bytes = (
            profile["bitmap-int8"]
            .get("accepted_wire_bytes", {})
            .get("round-protocol", "n/a")
        )
        ratio = _number(profile["assessment"]["wire_ratio_identity_to_compressed"])
        residual = _number(
            profile["bitmap-int8"].get("maximum_error_feedback_residual_l2")
        )
        lines.append(
            f"| {profile['profile']} | {identity_bytes} | {compressed_bytes} | "
            f"{ratio} | {residual} |"
        )
    lines.extend(
        [
            "",
            (
                "Per-node common/local loss, accuracy and per-class "
                "support/scores are retained in [results.json](results.json), "
                "along with every runtime round's metrics, residual norms, "
                "retries, sender wire accounting and cross-node weight "
                "dispersion. The full logs and durable run stores remain under "
                "`runs/`; bounded post-COMMITTED slow-weight snapshots remain "
                "under each node's `analysis/`. Failed criteria are retained and "
                "require a new, separately declared experiment for ablations; "
                "they are not adjusted after seeing results."
            ),
            "",
            (
                "Wire accounting counts full serialized Dromeus envelopes "
                "accepted by the transport, including retries and protocol "
                "overhead, with telemetry and formation/control separated. It "
                "excludes AXL/TCP framing. These tiny models can cost more bytes "
                "when compressed because bitmap/scales and metadata dominate; no "
                "speed or 5× compression claim follows. Runtime timing includes "
                "same-host CPU contention and durability; reference/local-only "
                "timing is sequential and is not a network speed comparison."
            ),
            "",
            (
                "Every method receives 400 optimizer steps per node with batch "
                "size 16, retaining partial batches and repeatedly sampling local"
                " datasets as needed. Actual sample presentations are recorded. "
                "The objective gives each node equal influence despite unequal "
                "data counts. Local-only learning has no communication; its local"
                " fit may coexist with poor common-held-out accuracy."
            ),
            "",
            (
                "The initial checkpoint, data-plan hashes, actual sorted public "
                "keys, configuration, source-file hashes, complete source "
                "snapshot and host runtime are retained. The container image "
                "digest is explicitly null because these are host runs. In-memory"
                " results prove the production protocol's local integration; real"
                " local AXL proves the AXL path on one host. Neither establishes "
                "privacy, Byzantine tolerance, WAN resilience, or general non-IID"
                " convergence."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


async def run_suite(
    output_dir: Path,
    *,
    transport: Literal["in-memory", "axl"] = "in-memory",
    axl_binary: Path | None = None,
    profiles: tuple[str, ...] = PROFILES,
) -> dict[str, Any]:
    if (
        not profiles
        or len(set(profiles)) != len(profiles)
        or any(profile not in PROFILES for profile in profiles)
    ):
        raise ValueError(
            "profiles must be a nonempty unique selection from the frozen matrix"
        )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    experiment = load_experiment()
    (output_dir / "experiment.json").write_bytes(experiment.path.read_bytes())
    provenance = source_provenance(output_dir)
    initial = create_initial_checkpoint(
        output_dir / "initial.safetensors",
        model=build_model(experiment.seed),
        model_definition=experiment.definition,
    )
    profile_reports: list[dict[str, Any]] = []
    membership: tuple[str, ...] = ()

    async def execute(bridges: tuple[LocalAXLNode, ...] | None) -> None:
        nonlocal membership
        membership = (
            tuple(node.public_key for node in bridges)
            if bridges
            else tuple(
                sorted(
                    hashlib.sha256(
                        f"federated-{experiment.seed}-rank-{rank}".encode()
                    ).hexdigest()
                    for rank in range(4)
                )
            )
        )
        if membership != tuple(sorted(membership)) or len(set(membership)) != 4:
            raise ValueError("suite requires four unique sorted actual public keys")
        write_json(
            output_dir / "membership.json",
            {"public_keys": membership, "scheduler_seed": experiment.seed},
        )
        for profile in profiles:
            print(f"{profile}: materializing frozen data", flush=True)
            plan = create_data_plan(
                output_dir / "data" / profile, profile=profile, seed=experiment.seed
            )
            controls: dict[str, ControlResult] = {}
            for method in ("reference", "local-only"):
                print(f"{profile}: {method}", flush=True)
                controls[method] = await asyncio.to_thread(
                    run_control,
                    experiment=experiment,
                    plan=plan,
                    initial_checkpoint=initial.path,
                    keys=membership,
                    root=output_dir / "controls" / profile / method,
                    method=method,
                )
            runs: dict[str, RuntimeResult] = {}
            for variant in ("identity", "bitmap-int8"):
                print(f"{profile}: production {variant} over {transport}", flush=True)
                run_root = output_dir / "runs" / profile / variant
                try:
                    runs[variant] = await _runtime_run(
                        root=run_root,
                        experiment=experiment,
                        plan=plan,
                        variant=variant,
                        keys=membership,
                        bridges=bridges,
                        checkpoint_path=initial.path,
                        source_commit=provenance["source_commit"],
                    )
                except Exception:
                    failure = {"status": "failed", "errors": [traceback.format_exc()]}
                    write_json(run_root / "failure.json", failure)
                    runs[variant] = RuntimeResult(failure, None)
            assessment = assess_profile(
                experiment,
                runs["identity"],
                runs["bitmap-int8"],
                controls["reference"].trajectories,
            )
            profile_reports.append(
                {
                    "profile": profile,
                    "data_plan_sha256": plan.sha256,
                    "identity": runs["identity"].report,
                    "bitmap-int8": runs["bitmap-int8"].report,
                    "reference": controls["reference"].report,
                    "local-only": controls["local-only"].report,
                    "assessment": assessment,
                }
            )
            write_json(
                output_dir / "assessments" / f"{profile}.json", profile_reports[-1]
            )
            print(f"{profile}: {assessment['decision']}", flush=True)

    try:
        if transport == "axl":
            binary = axl_binary or await asyncio.to_thread(
                ensure_axl_binary, Path.home() / ".cache" / "dromeus-federated-axl"
            )
            metadata = subprocess.check_output(
                ["go", "version", "-m", str(binary)], text=True
            )
            if (
                f"vcs.revision={AXL_COMMIT}" not in metadata
                or "vcs.modified=false" not in metadata
            ):
                raise ValueError("AXL binary does not identify the pinned clean source")
            write_json(
                output_dir / "axl-binary.json",
                {
                    "source_commit": AXL_COMMIT,
                    "sha256": file_sha256(binary),
                    "go_build_metadata": metadata,
                },
            )
            async with local_axl_cluster(
                binary=binary, log_root=output_dir / "axl", node_count=4
            ) as bridges:
                await execute(bridges)
        else:
            await execute(None)
    except BaseException:
        write_json(
            output_dir / "suite-failure.json",
            {
                "error": traceback.format_exc(),
                "completed_profiles": len(profile_reports),
            },
        )
        raise
    result = {
        "schema_version": 1,
        "transport": transport,
        "scope": "synthetic local CPU development; not M2 acceptance",
        "training_process_topology": "four NodeRuntime instances in one Python process",
        "experiment_sha256": experiment.sha256,
        "initial_checkpoint_sha256": initial.sha256,
        "source_tree_sha256": provenance["source_tree_sha256"],
        "container_image_digest": None,
        "membership": membership,
        "complete_five_profile_matrix": set(profiles) == set(PROFILES),
        "profiles": profile_reports,
        "all_predeclared_gates_passed": all(
            item["assessment"]["decision"] == "pass" for item in profile_reports
        ),
    }
    write_json(output_dir / "results.json", result)
    (output_dir / "report.md").write_text(_markdown(result), encoding="utf-8")
    paths = sorted(path for path in output_dir.rglob("*") if path.is_file())
    (output_dir / "checksums.sha256").write_text(
        "".join(
            f"{file_sha256(path)}  {path.relative_to(output_dir)}\n" for path in paths
        )
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--transport", choices=("in-memory", "axl"), default="in-memory"
    )
    parser.add_argument("--axl-binary", type=Path)
    parser.add_argument("--profiles", default=",".join(PROFILES))
    arguments = parser.parse_args(argv)
    torch.set_num_threads(1)
    result = asyncio.run(
        run_suite(
            arguments.output_dir,
            transport=cast(Literal["in-memory", "axl"], arguments.transport),
            axl_binary=arguments.axl_binary,
            profiles=tuple(arguments.profiles.split(",")),
        )
    )
    print(
        json.dumps(
            {
                "output_dir": str(arguments.output_dir),
                "all_predeclared_gates_passed": result["all_predeclared_gates_passed"],
            }
        ),
        flush=True,
    )
    return 0 if result["all_predeclared_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
