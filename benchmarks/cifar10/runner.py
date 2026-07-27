"""Command-line helpers for reproducible M1 benchmark execution."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

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
    CIFAR10_DATASET_VERSION,
    MODEL_DEFINITION_HASH,
    PREPROCESSING_HASH,
    CIFAR10Data,
    create_initial_checkpoint,
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
        consensus_sketch=ConsensusSketchConfig(seed=benchmark_seed),
    )


def write_draft(draft: DraftRunSpec, output: Path) -> None:
    _write_yaml(output, draft.model_dump(mode="json"))


def write_node_configs(
    *,
    plan_path: Path,
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
    prepare_dpsgd_node_configs(
        plan_path=plan_path,
        draft_path=draft_path,
        seed=benchmark_seed,
        node_config_paths=result,
    )
    return result


def write_pilot_evidence(
    *,
    draft_path: Path,
    run_roots: Sequence[Path],
    output: Path,
) -> PilotEvidence:
    """Accept a pilot only when four archived nodes completed the same draft."""
    if len(run_roots) != 4:
        raise ValueError("pilot evidence requires four run roots")
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
    evidence = PilotEvidence(
        status="complete",
        model_definition_hash=draft.model_definition_hash,
        dataset=draft.dataset,
        data_source="torchvision-cifar10",
        local_steps=draft.local_steps,
        round_count=draft.round_count,
        learning_rate=draft.learning_rate,
    )
    _write_json(output, evidence.model_dump(mode="json"))
    return evidence


def write_frozen_plan(
    *,
    draft_path: Path,
    pilot_artifact: Path,
    benchmark_seeds: tuple[int, int, int],
    output: Path,
) -> FrozenBenchmarkPlan:
    """Freeze one pilot-backed configuration before official results exist."""
    draft = parse_draft_yaml(draft_path)
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
        pilot_artifact=pilot_artifact.resolve(),
    )
    _write_yaml(output, plan.model_dump(mode="json"))
    return load_frozen_benchmark_plan(output)


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
        seed=benchmark_seed,
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
    pilot.add_argument("--output", required=True, type=Path)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--draft", required=True, type=Path)
    freeze.add_argument("--pilot-artifact", required=True, type=Path)
    freeze.add_argument("--seed", required=True, action="append", type=int)
    freeze.add_argument("--output", required=True, type=Path)

    nodes = subparsers.add_parser("node-configs")
    nodes.add_argument("--plan", required=True, type=Path)
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
            output=args.output,
        )
    elif args.command == "freeze":
        seeds = tuple(args.seed)
        if len(seeds) != 3:
            raise ValueError("freeze requires exactly three seeds")
        write_frozen_plan(
            draft_path=args.draft,
            pilot_artifact=args.pilot_artifact,
            benchmark_seeds=cast(tuple[int, int, int], seeds),
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
            cifar_root=args.cifar_root,
            output=args.output,
        )
    else:
        write_three_seed_report(args.input, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
