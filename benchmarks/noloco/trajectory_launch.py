"""Materialize frozen four-node Dromeus trajectory launch inputs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from benchmarks.noloco.dromeus_adapter import (
    DromeusExperimentAdapter,
    benchmark_transport_limits,
)
from benchmarks.noloco.experiment import ResolvedRun, load_frozen_experiment
from benchmarks.workloads.cifar10.runtime import (
    BenchmarkNodeConfig as NodeConfig,
)
from dromeus.manifests.models import (
    DraftRunSpec,
    EnvironmentFingerprint,
)
from dromeus.node import (
    NodeRole,
)


@dataclass(frozen=True, slots=True)
class TrajectoryLaunch:
    draft: DraftRunSpec
    node_configs: tuple[NodeConfig, ...]


def materialize_trajectory_launch(
    *,
    run: ResolvedRun,
    output_root: Path,
    runtime_root: Path,
    run_id: str,
    environment: EnvironmentFingerprint,
    bootstrap_uri: str,
) -> TrajectoryLaunch:
    """Write one immutable draft and four machine-local node configurations."""
    if (
        run.selector.profile != "trajectory"
        or run.selector.world_size != 4
        or run.run.round_count != 2
    ):
        raise ValueError("launch requires the frozen four-node trajectory profile")
    adapter = DromeusExperimentAdapter(run)
    draft = adapter.build_draft(
        run_id=run_id,
        environment=environment,
        transport=benchmark_transport_limits(run),
        ablation_id="identity",
    )
    expectation = adapter.manifest_expectation(draft=draft)
    draft_runtime_path = runtime_root / "draft.yaml"
    invitation_runtime_path = runtime_root / "invitation.json"
    node_configs = tuple(
        NodeConfig(
            role=NodeRole.INITIATOR if rank == 0 else NodeRole.PARTICIPANT,
            draft_path=draft_runtime_path,
            axl_bridge_url="http://127.0.0.1:9302",
            run_root=runtime_root / f"rank-{rank}" / "run",
            dataset_cache=runtime_root / "cifar-cache",
            invitation_path=invitation_runtime_path,
            bootstrap_uri=bootstrap_uri,
            benchmark_seed=run.selector.benchmark_seed,
            training_device="cuda",
            invitation_timeout_seconds=1200.0,
            manifest_expectation=expectation,
        )
        for rank in range(4)
    )
    output_root.mkdir(parents=True, exist_ok=False)
    _write_yaml(output_root / "draft.yaml", draft.model_dump(mode="json"))
    for rank, config in enumerate(node_configs):
        _write_yaml(
            output_root / f"node-{rank}.yaml",
            config.model_dump(mode="json"),
        )
    return TrajectoryLaunch(draft=draft, node_configs=node_configs)


def _write_yaml(path: Path, value: object) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(value, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bootstrap-uri", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    experiment = load_frozen_experiment(arguments.experiment)
    trajectory_runs = tuple(
        item for item in experiment.runs if item.selector.profile == "trajectory"
    )
    if len(trajectory_runs) != 1:
        raise ValueError("frozen experiment must contain one trajectory profile")
    environment = EnvironmentFingerprint.model_validate_json(
        arguments.environment.read_text(encoding="utf-8")
    )
    materialize_trajectory_launch(
        run=experiment.resolve(trajectory_runs[0].selector),
        output_root=arguments.output_root,
        runtime_root=arguments.runtime_root,
        run_id=arguments.run_id,
        environment=environment,
        bootstrap_uri=arguments.bootstrap_uri,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "TrajectoryLaunch",
    "main",
    "materialize_trajectory_launch",
]
