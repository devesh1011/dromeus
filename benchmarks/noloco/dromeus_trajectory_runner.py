"""Run the bounded Dromeus AXL trajectory profile through the production lifecycle."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from benchmarks.noloco.experiment import ResolvedRun, load_frozen_experiment
from benchmarks.noloco.trajectory import (
    DromeusTrajectoryContext,
    DromeusTrajectoryWriter,
    TrajectoryCapturingAlgorithm,
)
from benchmarks.workloads.cifar10.runtime import (
    load_benchmark_node_config as load_node_config,
)
from benchmarks.workloads.cifar10.runtime import (
    run_benchmark_node as run_node,
)
from dromeus.algorithms.noloco import NoLoCoAlgorithm
from dromeus.manifests.canonical import canonical_hash
from dromeus.membership.formation import FormationResult
from dromeus.node import (
    TrainingDecorator,
)
from dromeus.runtime import TrainingConfig


def create_trajectory_training_decorator(
    *,
    context: DromeusTrajectoryContext,
    output_root: Path,
) -> TrainingDecorator:
    """Return a formed-run decorator that changes only the algorithm value."""

    def decorate(
        result: FormationResult,
        local_public_key: str,
        training: TrainingConfig,
    ) -> TrainingConfig:
        manifest = result.manifest
        participants = tuple(
            item.public_key for item in manifest.participants
        )
        codecs = tuple(
            (item.artifact_name, item.codec_id)
            for item in manifest.artifact_codecs or ()
        )
        if (
            manifest.algorithm_id != "noloco"
            or manifest.round_count != 2
            or participants != context.participants
            or manifest.peer_scheduler_seed != context.scheduler_seed
            or manifest.initial_checkpoint_hash != context.initial_checkpoint_sha256
            or canonical_hash(manifest.tensor_schema) != context.tensor_schema_hash
            or codecs
            != (
                ("outer_gradient", "identity-v1"),
                ("slow_weights", "identity-v1"),
            )
        ):
            raise ValueError("formed run does not match trajectory context")
        if (
            local_public_key != context.node_id
            or participants[context.rank] != local_public_key
        ):
            raise ValueError("local identity does not match trajectory rank")
        if not isinstance(training.algorithm, NoLoCoAlgorithm):
            raise TypeError("trajectory capture requires NoLoCoAlgorithm")
        return replace(
            training,
            algorithm=TrajectoryCapturingAlgorithm(
                delegate=training.algorithm,
                writer=DromeusTrajectoryWriter(
                    root=output_root,
                    context=context,
                ),
            ),
        )

    return decorate


def create_resolved_trajectory_decorator(
    *,
    run: ResolvedRun,
    output_root: Path,
) -> TrainingDecorator:
    """Create the local decorator for one resolved trajectory experiment."""
    if run.selector.profile != "trajectory" or run.selector.world_size != 4:
        raise ValueError("resolved run is not the four-node trajectory profile")
    if run.run.round_count != 2:
        raise ValueError("trajectory profile must contain exactly two rounds")
    participants = tuple(item.public_key for item in run.run.participants)

    def decorate(
        result: FormationResult,
        local_public_key: str,
        training: TrainingConfig,
    ) -> TrainingConfig:
        try:
            rank = participants.index(local_public_key)
        except ValueError as error:
            raise ValueError("local identity is outside frozen trajectory") from error
        return create_trajectory_training_decorator(
            context=DromeusTrajectoryContext(
                experiment_sha256=run.experiment_sha256,
                run_config_sha256=run.run_config_sha256,
                initial_checkpoint_sha256=run.run.checkpoint.sha256,
                tensor_schema_hash=run.model.tensor_schema_hash,
                world_size=run.selector.world_size,
                benchmark_seed=run.selector.benchmark_seed,
                rank=rank,
                node_id=local_public_key,
                participants=participants,
                scheduler_seed=run.run.scheduler_seed,
                total_outer_steps=run.run.round_count,
                source_repository=run.source.repository,
                source_commit=run.source.commit,
                source_path=run.source.files[0],
                outer_gradient_convention=run.source.outer_gradient_convention,
                pairing_convention="dromeus-peer-scheduler-v1",
            ),
            output_root=output_root,
        )(result, local_public_key, training)

    return decorate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-config", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = load_node_config(arguments.node_config)
    experiment = load_frozen_experiment(arguments.experiment)
    trajectory_runs = tuple(
        item for item in experiment.runs if item.selector.profile == "trajectory"
    )
    if len(trajectory_runs) != 1:
        raise ValueError("frozen experiment must contain one trajectory profile")
    run = experiment.resolve(trajectory_runs[0].selector)
    if config.benchmark_seed != run.selector.benchmark_seed:
        raise ValueError("node benchmark seed does not match trajectory profile")
    asyncio.run(
        run_node(
            config,
            training_decorator=create_resolved_trajectory_decorator(
                run=run,
                output_root=arguments.output_dir,
            ),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "create_resolved_trajectory_decorator",
    "create_trajectory_training_decorator",
]
