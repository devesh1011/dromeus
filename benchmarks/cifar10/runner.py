"""Command-line helpers for reproducible M1 benchmark execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict

from benchmarks.cifar10.fedavg_reference import run_fedavg
from benchmarks.cifar10.official import (
    FrozenBenchmarkPlan,
    PilotEvidence,
    load_frozen_benchmark_plan,
    prepare_dpsgd_node_configs,
)
from benchmarks.cifar10.report import (
    SeedBenchmarkInput,
    build_three_seed_report,
)
from dromeus.manifests.canonical import canonical_hash, parse_draft_yaml
from dromeus.manifests.models import (
    ConsensusSketchConfig,
    DatasetContract,
    DraftRunSpec,
    EnvironmentFingerprint,
    SealedManifest,
    TransportLimits,
)
from dromeus.training.pytorch import (
    CIFAR10_ARCHIVE_MD5,
    CIFAR10_DATASET_VERSION,
    MODEL_DEFINITION_HASH,
    PREPROCESSING_HASH,
    CIFAR10Data,
    create_initial_checkpoint,
    derive_benchmark_seed,
)

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

    data_source: Literal["torchvision-cifar10"]
    archive_md5: Literal["c58f30108f718f92721af3b95e74349a"]
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
    round_count: int = 100,
    local_steps: int = 5,
    learning_rate: float = 0.1,
) -> DraftRunSpec:
    """Create one production CIFAR draft from measured, explicit values."""
    dataset = DatasetContract(
        dataset_id="cifar10",
        version=CIFAR10_DATASET_VERSION,
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
        manifest_version=1,
        protocol_version=1,
        run_id=run_id,
        algorithm_id="dpsgd-v1",
        model_id="cifar-cnn-v1",
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
    )


def write_draft(draft: DraftRunSpec, output: Path) -> None:
    _write_yaml(output, draft.model_dump(mode="json"))


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
                "cifar_root": "/var/cache/dromeus/cifar10",
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
    manifests: list[SealedManifest] = []
    for root in run_roots:
        store_root = root / "run-store"
        manifest = SealedManifest.model_validate_json(
            (store_root / "manifest.json").read_text(encoding="utf-8")
        )
        state = cast(
            dict[str, object],
            json.loads((store_root / "state.json").read_text(encoding="utf-8")),
        )
        terminal_value = state.get("terminal")
        terminal = (
            cast(dict[str, object], terminal_value)
            if isinstance(terminal_value, dict)
            else None
        )
        if (
            terminal is None
            or terminal.get("result") != "complete"
            or state.get("committed_round") != draft.round_count - 1
        ):
            raise ValueError("pilot node did not complete every round")
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
    for path in event_logs:
        events = [
            cast(dict[str, object], json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        if any(
            event.get("event") in {"run_failed", "formation_failed"}
            for event in events
        ):
            raise ValueError("failed pilot logs cannot be frozen")
        ready = [
            event for event in events if event.get("event") == "benchmark_node_ready"
        ]
        if len(ready) != 1:
            raise ValueError("each pilot log requires one benchmark node record")
        record = ready[0]
        node_id = record.get("node_id")
        if (
            not isinstance(node_id, str)
            or record.get("run_id") != draft.run_id
            or record.get("manifest_hash") != manifest_hash
            or record.get("benchmark_seed") != draft.peer_scheduler_seed
            or record.get("transport") != "axl"
        ):
            raise ValueError("pilot node provenance is incomplete")
        node_ids.append(node_id)
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
        data_source="torchvision-cifar10",
        local_steps=draft.local_steps,
        round_count=draft.round_count,
        learning_rate=draft.learning_rate,
        node_ids=cast(tuple[str, str, str, str], tuple(node_ids)),
        data_artifact_sha256=cast(
            tuple[str, str, str, str], tuple(artifact_hashes)
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
        data_source="torchvision-cifar10",
        max_payload_bytes=draft.transport.max_payload_bytes,
        max_retries=draft.transport.max_retries,
        retry_timeout_seconds=draft.transport.retry_timeout_seconds,
        pilot_artifact=portable_pilot_path,
        cloud_provider="aws",
        worker_instance_type=worker_instance_type,
        worker_regions=worker_regions,
        bootstrap_region=bootstrap_region,
        worker_root_volume_gib=worker_root_volume_gib,
    )
    _write_yaml(output, plan.model_dump(mode="json"))
    return load_frozen_benchmark_plan(output)


def prepare_cifar_data(*, cifar_root: Path, output: Path) -> DatasetArtifact:
    """Download and validate canonical CIFAR-10 on one worker."""
    train_data = CIFAR10Data.from_torchvision(
        root=cifar_root,
        train=True,
        download=True,
    )
    test_data = CIFAR10Data.from_torchvision(
        root=cifar_root,
        train=False,
        download=True,
    )
    archive = cifar_root / "cifar-10-python.tar.gz"
    archive_md5 = hashlib.md5(archive.read_bytes(), usedforsecurity=False).hexdigest()
    if archive_md5 != CIFAR10_ARCHIVE_MD5:
        raise ValueError("downloaded CIFAR-10 archive checksum does not match")
    if len(train_data) != 50_000 or len(test_data) != 10_000:
        raise ValueError("downloaded CIFAR-10 sample counts do not match")
    artifact = DatasetArtifact(
        data_source="torchvision-cifar10",
        archive_md5="c58f30108f718f92721af3b95e74349a",
        dataset_version=CIFAR10_DATASET_VERSION,
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
    cifar_root: Path,
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
    train_data = CIFAR10Data.from_torchvision(
        root=cifar_root,
        train=True,
        download=False,
    )
    test_data = CIFAR10Data.from_torchvision(
        root=cifar_root,
        train=False,
        download=False,
    )
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
    value = ReportInput.model_validate_json(input_path.read_text(encoding="utf-8"))
    inputs = tuple(
        SeedBenchmarkInput(
            seed=item.seed,
            run_roots=item.run_roots,
            event_logs=item.event_logs,
            fedavg_result_path=item.fedavg_result_path,
        )
        for item in value.seeds
    )
    build_three_seed_report(inputs).write_artifacts(output_dir)


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
    draft.add_argument("--round-count", type=int, default=100)
    draft.add_argument("--local-steps", type=int, default=5)
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
    data.add_argument("--cifar-root", required=True, type=Path)
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
    fedavg.add_argument("--cifar-root", required=True, type=Path)
    fedavg.add_argument("--output", required=True, type=Path)

    report = subparsers.add_parser("report")
    report.add_argument("--input", required=True, type=Path)
    report.add_argument("--output-dir", required=True, type=Path)
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
        prepare_cifar_data(cifar_root=args.cifar_root, output=args.output)
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
            cifar_root=args.cifar_root,
            output=args.output,
        )
    else:
        write_three_seed_report(args.input, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
