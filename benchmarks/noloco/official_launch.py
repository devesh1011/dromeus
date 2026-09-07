"""Materialize deterministic Dromeus/NCCL official benchmark launch inputs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from benchmarks.noloco.dromeus_adapter import (
    DromeusExperimentAdapter,
    benchmark_transport_limits,
)
from benchmarks.noloco.experiment import (
    AblationId,
    ResolvedRun,
    RunSelector,
    load_frozen_experiment,
)
from benchmarks.workloads.cifar10.runtime import (
    BenchmarkNodeConfig as NodeConfig,
)
from dromeus.manifests.models import DraftRunSpec, EnvironmentFingerprint
from dromeus.node import (
    NodeRole,
)


@dataclass(frozen=True, slots=True)
class OfficialLaunch:
    draft: DraftRunSpec
    node_configs: tuple[NodeConfig, ...]
    metadata_path: Path


def materialize_official_launch(
    *,
    run: ResolvedRun,
    ablation_id: AblationId,
    output_root: Path,
    runtime_root: Path,
    run_id: str,
    environment: EnvironmentFingerprint,
    bootstrap_uri: str,
    socket_interface: str = "ens5",
) -> OfficialLaunch:
    """Write one frozen draft, N node configs, and explicit NCCL launch contract."""
    if run.selector.profile != "official":
        raise ValueError("official launch requires an official profile")
    if not socket_interface:
        raise ValueError("NCCL socket interface must not be empty")
    adapter = DromeusExperimentAdapter(run)
    draft = adapter.build_draft(
        run_id=run_id,
        environment=environment,
        transport=benchmark_transport_limits(run),
        ablation_id=ablation_id,
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
        for rank in range(run.selector.world_size)
    )
    output_root.mkdir(parents=True, exist_ok=False)
    _write_yaml(output_root / "draft.yaml", draft.model_dump(mode="json"))
    for rank, config in enumerate(node_configs):
        _write_yaml(
            output_root / f"node-{rank}.yaml",
            config.model_dump(mode="json"),
        )
    metadata_path = output_root / "launch.json"
    _write_json(
        metadata_path,
        {
            "schema_version": 1,
            "selector": run.selector.model_dump(mode="json"),
            "ablation_id": ablation_id,
            "run_id": run_id,
            "container_image_digest": run.hardware.container_image_digest,
            "runtime_root": str(runtime_root),
            "dromeus_runner_module": (
                "benchmarks.noloco.dromeus_official_runner"
            ),
            "reference_runner_module": "benchmarks.noloco_reference.runner",
            "analysis_module": "benchmarks.noloco.analysis",
            "docker_shm_size_bytes": 1024 * 1024 * 1024,
            "nccl_socket_interface": socket_interface,
            "gloo_socket_interface": socket_interface,
            "nccl_backend": "nccl",
            "nccl_cross_region_performance_eligible": False,
        },
    )
    return OfficialLaunch(
        draft=draft,
        node_configs=node_configs,
        metadata_path=metadata_path,
    )


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


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--profile", choices=("official",), default="official")
    parser.add_argument("--world-size", type=int, choices=(4, 8, 16), required=True)
    parser.add_argument("--seed", type=int, choices=(17, 29, 41), required=True)
    parser.add_argument("--ablation", choices=("identity", "compressed"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bootstrap-uri", required=True)
    parser.add_argument("--socket-interface", default="ens5")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    experiment = load_frozen_experiment(arguments.experiment)
    run = experiment.resolve(
        RunSelector(
            profile="official",
            world_size=arguments.world_size,
            benchmark_seed=arguments.seed,
        )
    )
    environment = EnvironmentFingerprint.model_validate_json(
        arguments.environment.read_text(encoding="utf-8")
    )
    materialize_official_launch(
        run=run,
        ablation_id=cast(AblationId, arguments.ablation),
        output_root=arguments.output_root,
        runtime_root=arguments.runtime_root,
        run_id=arguments.run_id,
        environment=environment,
        bootstrap_uri=arguments.bootstrap_uri,
        socket_interface=arguments.socket_interface,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["OfficialLaunch", "main", "materialize_official_launch"]
