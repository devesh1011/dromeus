"""Run a frozen Dromeus official profile with periodic analysis checkpoints."""

from __future__ import annotations

import argparse
import asyncio
import shutil
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from benchmarks.noloco.analysis import (
    AnalysisCheckpointCapturingAlgorithm,
    AnalysisCheckpointContext,
    AnalysisCheckpointWriter,
    required_analysis_storage_bytes,
    validate_analysis_storage,
)
from benchmarks.noloco.experiment import (
    ResolvedRun,
    RunSelector,
    load_frozen_experiment,
)
from dromeus.algorithms.noloco import NoLoCoAlgorithm
from dromeus.manifests.canonical import canonical_hash
from dromeus.membership.formation import FormationResult
from dromeus.node import TrainingDecorator, load_node_config, run_node
from dromeus.runtime import TrainingConfig


def create_analysis_training_decorator(
    *,
    run: ResolvedRun,
    output_root: Path,
    available_bytes: int | None = None,
) -> TrainingDecorator:
    """Return the official-run decorator after deterministic storage preflight."""
    if run.selector.profile != "official":
        raise ValueError("analysis capture requires an official profile")
    evidence = run.run.evidence
    required_bytes = required_analysis_storage_bytes(
        raw_parameter_bytes=run.model.parameter_count * 4,
        total_outer_steps=run.run.round_count,
        interval_rounds=evidence.analysis_checkpoint_interval_rounds,
    )
    if available_bytes is None:
        output_root.parent.mkdir(parents=True, exist_ok=True)
        available_bytes = shutil.disk_usage(output_root.parent).free
    validate_analysis_storage(
        required_bytes=required_bytes,
        budget_bytes=evidence.analysis_storage_budget_bytes_per_node,
        available_bytes=available_bytes,
    )
    participants = tuple(item.public_key for item in run.run.participants)

    def decorate(
        result: FormationResult,
        local_public_key: str,
        training: TrainingConfig,
    ) -> TrainingConfig:
        manifest = result.manifest
        formed_participants = tuple(
            item.public_key for item in manifest.participants
        )
        if (
            manifest.algorithm_id != "noloco"
            or manifest.round_count != run.run.round_count
            or formed_participants != participants
            or manifest.initial_checkpoint_hash != run.run.checkpoint.sha256
            or canonical_hash(manifest.tensor_schema) != run.model.tensor_schema_hash
        ):
            raise ValueError("formed run does not match frozen analysis profile")
        try:
            rank = participants.index(local_public_key)
        except ValueError as error:
            raise ValueError("local identity is outside frozen analysis run") from error
        if not isinstance(training.algorithm, NoLoCoAlgorithm):
            raise TypeError("analysis capture requires NoLoCoAlgorithm")
        context = AnalysisCheckpointContext(
            experiment_sha256=run.experiment_sha256,
            run_config_sha256=run.run_config_sha256,
            initial_checkpoint_sha256=run.run.checkpoint.sha256,
            tensor_schema_hash=run.model.tensor_schema_hash,
            world_size=run.selector.world_size,
            benchmark_seed=run.selector.benchmark_seed,
            rank=rank,
            node_id=local_public_key,
            total_outer_steps=run.run.round_count,
            interval_rounds=evidence.analysis_checkpoint_interval_rounds,
            storage_budget_bytes=(
                evidence.analysis_storage_budget_bytes_per_node
            ),
        )
        return replace(
            training,
            algorithm=AnalysisCheckpointCapturingAlgorithm(
                delegate=training.algorithm,
                writer=AnalysisCheckpointWriter(root=output_root, context=context),
            ),
            evaluation_interval=evidence.evaluation_interval_rounds,
        )

    return decorate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-config", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--world-size", type=int, choices=(4, 8, 16), required=True)
    parser.add_argument("--seed", type=int, choices=(17, 29, 41), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = load_node_config(arguments.node_config)
    run = load_frozen_experiment(arguments.experiment).resolve(
        RunSelector(
            profile="official",
            world_size=arguments.world_size,
            benchmark_seed=arguments.seed,
        )
    )
    if config.benchmark_seed != run.selector.benchmark_seed:
        raise ValueError("node benchmark seed does not match official profile")
    asyncio.run(
        run_node(
            config,
            training_decorator=create_analysis_training_decorator(
                run=run,
                output_root=arguments.output_dir,
            ),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["create_analysis_training_decorator", "main"]
