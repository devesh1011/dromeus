"""Command-line helpers for reproducible M1 benchmark execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict

from benchmarks.cifar10.fedavg_reference import run_fedavg
from benchmarks.cifar10.official import (
    FrozenBenchmarkPlan,
    PilotEvidence,
    load_frozen_benchmark_plan,
    prepare_dpsgd_node_configs,
)
from dromeus.manifests.canonical import canonical_hash, parse_draft_yaml
from dromeus.manifests.models import (
    DPSGD_ALGORITHM_ID,
    ConsensusSketchConfig,
    DatasetContract,
    DraftRunSpec,
    EnvironmentFingerprint,
    SealedManifest,
    TrainingPolicy,
    TransportLimits,
)
from dromeus.persistence.archive import RunArchive, RunArchiveError
from dromeus.telemetry.evidence import (
    BenchmarkNodeReadyEvidence,
    EvidenceError,
    EvidenceLog,
    RoundMetricsEvidence,
    RunFailedEvidence,
)
from dromeus.training.cifar10 import (
    DATA_SOURCE,
    DATASET_REVISION,
    DATASET_VERSION,
    PREPROCESSING_HASH,
    create_initial_checkpoint,
    load_cifar10,
)
from dromeus.training.resnet32 import MODEL_DEFINITION_HASH, MODEL_ID, build_model
from dromeus.training.trainer import derive_benchmark_seed

if TYPE_CHECKING:
    from benchmarks.cifar10.report import SeedBenchmarkInput

PINNED_AXL_COMMIT = "628e28ace077f26dfe8d0259009b357216a9d8d4"


class SeedBenchmarkInputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int
    run_roots: tuple[Path, Path, Path, Path]
    event_logs: tuple[Path, Path, Path, Path]
    fedavg_result_path: Path


class ReportInput(BaseModel):
    """Path-only input for deterministic three-seed report generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seeds: tuple[
        SeedBenchmarkInputModel,
        SeedBenchmarkInputModel,
        SeedBenchmarkInputModel,
    ]


class DatasetArtifact(BaseModel):
    """Locally downloaded CIFAR-10 identity recorded by one worker."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data_source: Literal["huggingface-uoft-cs-cifar10"]
    dataset_revision: Literal["0b2714987fa478483af9968de7c934580d0bb9a2"]
    dataset_version: str
    preprocessing_hash: str
    train_sample_count: Literal[50_000]
    test_sample_count: Literal[10_000]


def create_draft(
    *,
    run_id: str,
    benchmark_seed: int,
    dromeus_commit: str,
    image_digest: str,
    pytorch_version: str,
    round_count: int = 400,
    local_steps: int = 40,
    learning_rate: float = 0.1,
) -> DraftRunSpec:
    """Create one production CIFAR draft from measured, explicit values."""
    dataset = DatasetContract(
        dataset_id="cifar10",
        version=DATASET_VERSION,
        preprocessing_hash=PREPROCESSING_HASH,
        iid_partition_seed=7,
        image_shape=(3, 32, 32),
        class_count=10,
        sample_count=50_000,
        partition_sample_counts=(12_500, 12_500, 12_500, 12_500),
        node_index_partitions=(0, 1, 2, 3),
    )
    environment = EnvironmentFingerprint(
        dromeus_version="0.1.0",
        dromeus_commit=dromeus_commit,
        protocol_version=1,
        pytorch_version=pytorch_version,
        axl_version=PINNED_AXL_COMMIT,
        model_definition_hash=MODEL_DEFINITION_HASH,
        container_image_digest=image_digest,
    )
    return DraftRunSpec(
        manifest_version=2,
        protocol_version=1,
        run_id=run_id,
        algorithm_id=DPSGD_ALGORITHM_ID,
        model_id=MODEL_ID,
        model_definition_hash=MODEL_DEFINITION_HASH,
        dataset=dataset,
        environment=environment,
        local_steps=local_steps,
        round_count=round_count,
        optimizer="sgd",
        learning_rate=learning_rate,
        peer_scheduler_seed=benchmark_seed,
        codec_id="safetensors-v1",
        transport=TransportLimits(
            max_payload_bytes=16 * 1024 * 1024,
            max_retries=3,
            retry_timeout_seconds=10.0,
        ),
        consensus_sketch=ConsensusSketchConfig(
            seed=derive_benchmark_seed(benchmark_seed, "consensus-sketch")
        ),
        training=TrainingPolicy(
            batch_size=128,
            momentum=0.9,
            weight_decay=1e-4,
            learning_rate_milestones=(8_000, 12_000),
            learning_rate_gamma=0.1,
            crop_padding=4,
            normalize=True,
            final_consensus_rounds=2,
        ),
    )


def write_draft(draft: DraftRunSpec, output: Path) -> None:
    _write_yaml(output, draft.model_dump(mode="json"))


def _training_policy(draft: DraftRunSpec) -> TrainingPolicy:
    if draft.training is None:
        raise ValueError("benchmark requires an active training policy")
    return draft.training


def write_node_configs(
    *,
    plan_path: Path | None,
    draft_path: Path,
    benchmark_seed: int,
    bootstrap_uri: str,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    """Write four machine-local configs and validate them against the plan."""
    draft = parse_draft_yaml(draft_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index in range(4):
        path = output_dir / f"node-{index}.yaml"
        _write_yaml(
            path,
            {
                "role": "initiator" if index == 0 else "participant",
                "draft_path": "/run/dromeus/draft.yaml",
                "axl_bridge_url": "http://127.0.0.1:9302",
                "run_root": f"/var/lib/dromeus/{draft.run_id}/node-{index}",
                "dataset_cache": "/var/cache/dromeus/cifar10",
                "invitation_path": f"/run/dromeus/{draft.run_id}/invitation.json",
                "bootstrap_uri": bootstrap_uri,
                "benchmark_seed": benchmark_seed,
                "invitation_timeout_seconds": 600.0,
            },
        )
        paths.append(path)
    result = cast(tuple[Path, Path, Path, Path], tuple(paths))
    if plan_path is not None:
        prepare_dpsgd_node_configs(
            plan_path=plan_path,
            draft_path=draft_path,
            seed=benchmark_seed,
            node_config_paths=result,
            deployed_draft_path=Path("/run/dromeus/draft.yaml"),
        )
    return result


def write_pilot_evidence(
    *,
    draft_path: Path,
    run_roots: Sequence[Path],
    event_logs: Sequence[Path],
    data_artifacts: Sequence[Path],
    output: Path,
) -> PilotEvidence:
    """Accept a pilot only when four archived nodes completed the same draft."""
    if len(run_roots) != 4 or len(event_logs) != 4 or len(data_artifacts) != 4:
        raise ValueError("pilot evidence requires four roots, logs, and data artifacts")
    for paths, label in (
        (run_roots, "run roots"),
        (event_logs, "event logs"),
        (data_artifacts, "data artifacts"),
    ):
        if len({path.resolve() for path in paths}) != 4:
            raise ValueError(f"pilot {label} must be distinct")
    draft = parse_draft_yaml(draft_path)
    training = _training_policy(draft)
    manifests: list[SealedManifest] = []
    for root in run_roots:
        store_root = root / "run-store"
        try:
            archive = RunArchive.open(store_root)
        except RunArchiveError as error:
            raise ValueError("invalid pilot run archive") from error
        expected_rounds = draft.round_count + training.final_consensus_rounds
        try:
            archive.require_complete(expected_rounds)
        except RunArchiveError as error:
            raise ValueError("pilot node did not complete every round") from error
        try:
            archive.verify_checkpoint_integrity()
        except RunArchiveError as error:
            raise ValueError("pilot node checkpoint integrity failed") from error
        manifest = archive.manifest
        manifest_draft = DraftRunSpec.model_validate(
            {
                name: getattr(manifest, name)
                for name in DraftRunSpec.model_fields
            }
        )
        if manifest_draft != draft:
            raise ValueError("pilot manifest does not match its draft")
        manifests.append(manifest)
    if len({canonical_hash(manifest) for manifest in manifests}) != 1:
        raise ValueError("pilot nodes do not share one sealed manifest")
    manifest_hash = canonical_hash(manifests[0])
    node_ids: list[str] = []
    final_node_accuracies: list[float] = []
    for path in event_logs:
        try:
            evidence_log = EvidenceLog.open(
                path,
                run_id=draft.run_id,
                manifest_hash=manifest_hash,
            )
        except EvidenceError as error:
            raise ValueError("pilot evidence log is invalid") from error
        if any(
            isinstance(record, RunFailedEvidence)
            for record in evidence_log.records
        ):
            raise ValueError("failed pilot logs cannot be frozen")
        ready = [
            record
            for record in evidence_log.records
            if isinstance(record, BenchmarkNodeReadyEvidence)
        ]
        if len(ready) != 1:
            raise ValueError("each pilot log requires one benchmark node record")
        record = ready[0]
        if (
            record.benchmark_seed != draft.peer_scheduler_seed
            or record.transport != "axl"
        ):
            raise ValueError("pilot node provenance is incomplete")
        node_ids.append(record.node_id)
        final_round = draft.round_count + training.final_consensus_rounds - 1
        final_metrics = [
            evidence
            for evidence in evidence_log.records
            if isinstance(evidence, RoundMetricsEvidence)
            and evidence.round_id == final_round
            and evidence.node_id == record.node_id
            and evidence.evaluation_accuracy is not None
        ]
        if len(final_metrics) == 1:
            accuracy = final_metrics[0].evaluation_accuracy
            assert accuracy is not None
            final_node_accuracies.append(accuracy)
    if len(set(node_ids)) != 4:
        raise ValueError("pilot logs must identify four distinct nodes")
    artifact_hashes: list[str] = []
    for path in data_artifacts:
        artifact = DatasetArtifact.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if (
            artifact.dataset_version != draft.dataset.version
            or artifact.preprocessing_hash != draft.dataset.preprocessing_hash
        ):
            raise ValueError("pilot data artifact does not match the draft")
        artifact_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    evidence = PilotEvidence(
        status="complete",
        model_definition_hash=draft.model_definition_hash,
        dataset=draft.dataset,
        data_source=DATA_SOURCE,
        local_steps=draft.local_steps,
        round_count=draft.round_count,
        learning_rate=draft.learning_rate,
        training=training,
        node_ids=cast(tuple[str, str, str, str], tuple(node_ids)),
        data_artifact_sha256=cast(
            tuple[str, str, str, str], tuple(artifact_hashes)
        ),
        final_node_accuracies=(
            cast(
                tuple[float, float, float, float],
                tuple(final_node_accuracies),
            )
            if len(final_node_accuracies) == 4
            else None
        ),
    )
    _write_json(output, evidence.model_dump(mode="json"))
    return evidence


def write_frozen_plan(
    *,
    draft_path: Path,
    pilot_artifact: Path,
    benchmark_seeds: tuple[int, int, int],
    worker_instance_type: str,
    worker_regions: tuple[str, str, str, str],
    bootstrap_region: str,
    worker_root_volume_gib: int,
    output: Path,
) -> FrozenBenchmarkPlan:
    """Freeze one pilot-backed configuration before official results exist."""
    draft = parse_draft_yaml(draft_path)
    training = _training_policy(draft)
    try:
        portable_pilot_path = pilot_artifact.resolve().relative_to(
            output.parent.resolve()
        )
    except ValueError as error:
        raise ValueError("pilot artifact must be inside the plan directory") from error
    plan = FrozenBenchmarkPlan(
        benchmark_seeds=benchmark_seeds,
        local_steps=draft.local_steps,
        round_count=draft.round_count,
        learning_rate=draft.learning_rate,
        model_id=draft.model_id,
        model_definition_hash=draft.model_definition_hash,
        dataset=draft.dataset,
        environment=draft.environment,
        data_source=DATA_SOURCE,
        weight_decay=training.weight_decay,
        max_payload_bytes=draft.transport.max_payload_bytes,
        max_retries=draft.transport.max_retries,
        retry_timeout_seconds=draft.transport.retry_timeout_seconds,
        pilot_artifact=portable_pilot_path,
        cloud_provider="aws",
        worker_instance_type=worker_instance_type,
        worker_regions=worker_regions,
        bootstrap_region=bootstrap_region,
        worker_root_volume_gib=worker_root_volume_gib,
        parameter_count=sum(
            parameter.numel()
            for parameter in build_model(seed=0).parameters()
        ),
        learning_rate_schedule=(
            "multistep" if training.learning_rate_milestones else "constant"
        ),
        batch_size=training.batch_size,
        training=training,
    )
    _write_yaml(output, plan.model_dump(mode="json"))
    return load_frozen_benchmark_plan(output)


def prepare_cifar_data(
    *,
    dataset_cache: Path,
    output: Path,
) -> DatasetArtifact:
    """Download and validate canonical CIFAR-10 on one worker."""
    train_data = load_cifar10(cache_dir=dataset_cache, train=True)
    test_data = load_cifar10(cache_dir=dataset_cache, train=False)
    if len(train_data) != 50_000 or len(test_data) != 10_000:
        raise ValueError("downloaded CIFAR-10 sample counts do not match")
    artifact = DatasetArtifact(
        data_source=DATA_SOURCE,
        dataset_revision=DATASET_REVISION,
        dataset_version=DATASET_VERSION,
        preprocessing_hash=PREPROCESSING_HASH,
        train_sample_count=50_000,
        test_sample_count=10_000,
    )
    _write_json(output, artifact.model_dump(mode="json"))
    return artifact


def run_fedavg_seed(
    *,
    plan_path: Path,
    benchmark_seed: int,
    dataset_cache: Path,
    output: Path,
) -> None:
    """Run and archive one frozen centralized control."""
    plan = load_frozen_benchmark_plan(plan_path)
    configs = {
        config.trainer_seed: config for config in plan.fedavg_configs()
    }
    try:
        config = configs[benchmark_seed]
    except KeyError as error:
        raise ValueError("FedAvg seed is not frozen") from error
    train_data = load_cifar10(cache_dir=dataset_cache, train=True)
    test_data = load_cifar10(cache_dir=dataset_cache, train=False)
    partitions = train_data.split_iid(
        participant_count=4,
        seed=plan.dataset.iid_partition_seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = create_initial_checkpoint(
        output.with_suffix(".initial.safetensors"),
        seed=derive_benchmark_seed(benchmark_seed, "model-initialization"),
    )
    result = run_fedavg(
        partitions=partitions,
        test_data=test_data,
        initial_checkpoint=checkpoint.path,
        config=config,
    )
    _write_json(output, result.as_dict())


def write_three_seed_report(input_path: Path, output_dir: Path) -> None:
    from benchmarks.cifar10.report import build_three_seed_report

    build_three_seed_report(_report_inputs(input_path)).write_artifacts(output_dir)


def write_three_seed_submission_report(input_path: Path, output_dir: Path) -> None:
    from benchmarks.cifar10.report import build_three_seed_submission_report

    build_three_seed_submission_report(_report_inputs(input_path)).write_artifacts(
        output_dir
    )


def _report_inputs(input_path: Path) -> tuple[SeedBenchmarkInput, ...]:
    from benchmarks.cifar10.report import SeedBenchmarkInput

    value = ReportInput.model_validate_json(input_path.read_text(encoding="utf-8"))
    return tuple(
        SeedBenchmarkInput(
            seed=item.seed,
            run_roots=item.run_roots,
            event_logs=item.event_logs,
            fedavg_result_path=item.fedavg_result_path,
        )
        for item in value.seeds
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    draft = subparsers.add_parser("draft")
    draft.add_argument("--run-id", required=True)
    draft.add_argument("--seed", required=True, type=int)
    draft.add_argument("--dromeus-commit", required=True)
    draft.add_argument("--image-digest", required=True)
    draft.add_argument("--pytorch-version", required=True)
    draft.add_argument("--round-count", type=int, default=400)
    draft.add_argument("--local-steps", type=int, default=40)
    draft.add_argument("--learning-rate", type=float, default=0.1)
    draft.add_argument("--output", required=True, type=Path)

    pilot = subparsers.add_parser("pilot-evidence")
    pilot.add_argument("--draft", required=True, type=Path)
    pilot.add_argument("--run-root", required=True, action="append", type=Path)
    pilot.add_argument("--event-log", required=True, action="append", type=Path)
    pilot.add_argument("--data-artifact", required=True, action="append", type=Path)
    pilot.add_argument("--output", required=True, type=Path)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--draft", required=True, type=Path)
    freeze.add_argument("--pilot-artifact", required=True, type=Path)
    freeze.add_argument("--seed", required=True, action="append", type=int)
    freeze.add_argument("--worker-instance-type", required=True)
    freeze.add_argument("--worker-region", required=True, action="append")
    freeze.add_argument("--bootstrap-region", required=True)
    freeze.add_argument("--worker-root-volume-gib", required=True, type=int)
    freeze.add_argument("--output", required=True, type=Path)

    data = subparsers.add_parser("prepare-data")
    data.add_argument("--dataset-cache", required=True, type=Path)
    data.add_argument("--output", required=True, type=Path)

    nodes = subparsers.add_parser("node-configs")
    nodes.add_argument("--plan", type=Path)
    nodes.add_argument("--draft", required=True, type=Path)
    nodes.add_argument("--seed", required=True, type=int)
    nodes.add_argument("--bootstrap-uri", required=True)
    nodes.add_argument("--output-dir", required=True, type=Path)

    fedavg = subparsers.add_parser("fedavg")
    fedavg.add_argument("--plan", required=True, type=Path)
    fedavg.add_argument("--seed", required=True, type=int)
    fedavg.add_argument("--dataset-cache", required=True, type=Path)
    fedavg.add_argument("--output", required=True, type=Path)

    report = subparsers.add_parser("report")
    report.add_argument("--input", required=True, type=Path)
    report.add_argument("--output-dir", required=True, type=Path)

    submission_report = subparsers.add_parser("submission-report")
    submission_report.add_argument("--input", required=True, type=Path)
    submission_report.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "draft":
        write_draft(
            create_draft(
                run_id=args.run_id,
                benchmark_seed=args.seed,
                dromeus_commit=args.dromeus_commit,
                image_digest=args.image_digest,
                pytorch_version=args.pytorch_version,
                round_count=args.round_count,
                local_steps=args.local_steps,
                learning_rate=args.learning_rate,
            ),
            args.output,
        )
    elif args.command == "pilot-evidence":
        write_pilot_evidence(
            draft_path=args.draft,
            run_roots=args.run_root,
            event_logs=args.event_log,
            data_artifacts=args.data_artifact,
            output=args.output,
        )
    elif args.command == "freeze":
        seeds = tuple(args.seed)
        if len(seeds) != 3:
            raise ValueError("freeze requires exactly three seeds")
        worker_regions = tuple(args.worker_region)
        if len(worker_regions) != 4:
            raise ValueError("freeze requires exactly four worker regions")
        write_frozen_plan(
            draft_path=args.draft,
            pilot_artifact=args.pilot_artifact,
            benchmark_seeds=cast(tuple[int, int, int], seeds),
            worker_instance_type=args.worker_instance_type,
            worker_regions=cast(
                tuple[str, str, str, str],
                worker_regions,
            ),
            bootstrap_region=args.bootstrap_region,
            worker_root_volume_gib=args.worker_root_volume_gib,
            output=args.output,
        )
    elif args.command == "prepare-data":
        prepare_cifar_data(
            dataset_cache=args.dataset_cache,
            output=args.output,
        )
    elif args.command == "node-configs":
        write_node_configs(
            plan_path=args.plan,
            draft_path=args.draft,
            benchmark_seed=args.seed,
            bootstrap_uri=args.bootstrap_uri,
            output_dir=args.output_dir,
        )
    elif args.command == "fedavg":
        run_fedavg_seed(
            plan_path=args.plan,
            benchmark_seed=args.seed,
            dataset_cache=args.dataset_cache,
            output=args.output,
        )
    elif args.command == "report":
        write_three_seed_report(args.input, args.output_dir)
    else:
        write_three_seed_submission_report(args.input, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
