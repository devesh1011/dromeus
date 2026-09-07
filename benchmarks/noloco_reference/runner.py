"""Executable frozen-workload runner for the independent NoLoCo reference."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from benchmarks.noloco.experiment import (
    ResolvedRun,
    RunSelector,
    load_frozen_experiment,
)
from benchmarks.noloco_reference.distributed import (
    PairExchange,
    TorchDistributedPairExchange,
)
from benchmarks.noloco_reference.optimizer import (
    ReferenceNoLoCoConfig,
    apply_outer_step,
    outer_gradient,
)
from benchmarks.noloco_reference.trajectory import (
    TrajectoryContext,
    TrajectoryWriter,
)
from benchmarks.workloads.cifar10.dataset import (
    DATA_SOURCE,
    DATASET_REVISION,
    PREPROCESSING_HASH,
    CIFAR10TrainerSettings,
    create_trainer,
    load_cifar10,
)
from dromeus.adapters.classification.torch_trainer import (
    PyTorchTrainer,
    TrainerSettings,
)
from dromeus.gossip.peer_scheduler import PeerScheduler

Backend = Literal["gloo", "nccl"]


@dataclass(frozen=True, slots=True)
class ReferenceRunResult:
    rank: int
    world_size: int
    backend: Backend
    acceptance_eligible: bool
    completed_outer_steps: int
    final_loss: float
    final_accuracy: float


def scheduled_peer_rank(
    *,
    members: Sequence[str],
    scheduler_seed: int,
    round_count: int,
    round_id: int,
    rank: int,
) -> int:
    """Resolve one rank using Dromeus's canonical public-key scheduler."""
    if not 0 <= rank < len(members):
        raise ValueError("local rank is outside scheduler membership")
    scheduler = PeerScheduler(
        members,
        seed=scheduler_seed,
        training_round_count=round_count,
        final_consensus_rounds=0,
    )
    peer_key = scheduler.schedule(round_id).peer_for(members[rank])
    return tuple(members).index(peer_key)


def run_reference_rank(
    *,
    run: ResolvedRun,
    rank: int,
    backend: Backend,
    trainer: PyTorchTrainer,
    exchange: PairExchange,
    device: torch.device,
    trajectory: TrajectoryWriter | None,
) -> ReferenceRunResult:
    """Execute all inner/outer steps for one distributed reference rank."""
    trainer.load_checkpoint(run.run.checkpoint.path)
    slow_weights = _torch_weights(trainer.weights(), device=device)
    outer_momentum = {
        name: torch.zeros_like(value) for name, value in slow_weights.items()
    }
    if trajectory is not None and trajectory.should_write(completed_outer_steps=0):
        trajectory.write(
            completed_outer_steps=0,
            peer_rank=None,
            slow_weights=slow_weights,
        )
    members = tuple(item.public_key for item in run.run.participants)
    config = ReferenceNoLoCoConfig(
        alpha=run.algorithm.alpha,
        beta=run.algorithm.beta,
        gamma=run.algorithm.gamma,
    )

    for round_id in range(run.run.round_count):
        trainer.load_weights(_numpy_weights(slow_weights))
        trainer.train_local_steps(run.algorithm.inner_steps)
        fast_weights = _torch_weights(trainer.weights(), device=device)
        gradient = outer_gradient(slow_weights, fast_weights)
        peer_rank = scheduled_peer_rank(
            members=members,
            scheduler_seed=run.run.scheduler_seed,
            round_count=run.run.round_count,
            round_id=round_id,
            rank=rank,
        )
        peer = exchange.exchange(
            peer_rank=peer_rank,
            artifacts={
                "outer_gradient": gradient,
                "slow_weights": slow_weights,
            },
        )
        result = apply_outer_step(
            slow_weights=slow_weights,
            outer_momentum=outer_momentum,
            local_outer_gradient=gradient,
            peer_outer_gradient=peer["outer_gradient"],
            peer_slow_weights=peer["slow_weights"],
            config=config,
        )
        slow_weights = result.slow_weights
        outer_momentum = result.outer_momentum
        completed = round_id + 1
        if trajectory is not None and trajectory.should_write(
            completed_outer_steps=completed
        ):
            trajectory.write(
                completed_outer_steps=completed,
                peer_rank=peer_rank,
                slow_weights=slow_weights,
            )
        dist.barrier()  # pyright: ignore[reportUnknownMemberType]

    trainer.load_weights(_numpy_weights(slow_weights))
    final_loss, final_accuracy = trainer.evaluate()
    return ReferenceRunResult(
        rank=rank,
        world_size=run.selector.world_size,
        backend=backend,
        acceptance_eligible=backend == "nccl",
        completed_outer_steps=run.run.round_count,
        final_loss=final_loss,
        final_accuracy=final_accuracy,
    )


def build_reference_trainer(
    *,
    run: ResolvedRun,
    rank: int,
    device: torch.device,
    dataset_cache: Path,
) -> PyTorchTrainer:
    """Materialize the exact shared CIFAR/model/trainer workload for one rank."""
    if (
        run.dataset.source != DATA_SOURCE
        or run.dataset.revision != DATASET_REVISION
        or run.dataset.preprocessing_hash != PREPROCESSING_HASH
    ):
        raise ValueError("frozen dataset identity does not match CIFAR recipe")
    train_data = load_cifar10(cache_dir=dataset_cache, train=True)
    test_data = load_cifar10(cache_dir=dataset_cache, train=False)
    if len(train_data) != run.dataset.sample_count:
        raise ValueError("CIFAR training sample count does not match frozen run")
    partitions = train_data.split_iid(
        participant_count=run.selector.world_size,
        seed=run.dataset.partition_seed,
    )
    hashes = tuple(
        partition.partition_provenance.indices_sha256
        for partition in partitions
        if partition.partition_provenance is not None
    )
    if hashes != run.run.partition_index_hashes:
        raise ValueError("CIFAR partition hashes do not match frozen run")
    seeds = run.run.rank_seeds[rank]
    return create_trainer(
        train_data=partitions[rank],
        test_data=test_data,
        settings=CIFAR10TrainerSettings(
            trainer=TrainerSettings(
                seed=seeds.trainer_seed,
                batch_size=run.workload.batch_size,
                learning_rate=run.algorithm.adam_learning_rate,
                optimizer="adam",
                momentum=0.0,
                weight_decay=run.workload.weight_decay,
                adam_beta1=run.algorithm.adam_beta1,
                adam_beta2=run.algorithm.adam_beta2,
                adam_epsilon=run.algorithm.adam_epsilon,
                gradient_clip_norm=run.algorithm.gradient_clip_norm,
                learning_rate_milestones=(),
                learning_rate_schedule=run.run.schedule,
                device=str(device),
                augment=run.workload.augment,
            ),
            model_id=run.model.model_id,
            model_definition_hash=run.model.definition_hash,
            crop_padding=run.workload.crop_padding,
            normalize=run.workload.normalize,
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    backend = cast(Backend, arguments.backend)
    run = load_frozen_experiment(arguments.experiment).resolve(
        RunSelector(
            profile=arguments.profile,
            world_size=arguments.world_size,
            benchmark_seed=arguments.seed,
        )
    )
    rank, world_size, local_rank = _launcher_environment()
    if world_size != run.selector.world_size:
        raise ValueError("torchrun world size does not match frozen run")
    if not 0 <= rank < world_size:
        raise ValueError("torchrun rank is outside world size")
    _validate_backend_available(backend)
    device = _device_for_backend(
        backend,
        local_rank=local_rank,
        accelerator_class=run.hardware.accelerator_class,
    )
    trainer = build_reference_trainer(
        run=run,
        rank=rank,
        device=device,
        dataset_cache=arguments.dataset_cache,
    )
    dist.init_process_group(backend=backend)
    try:
        trajectory = (
            TrajectoryWriter(
                root=arguments.output_dir,
                context=TrajectoryContext(
                    experiment_sha256=run.experiment_sha256,
                    run_config_sha256=run.run_config_sha256,
                    initial_checkpoint_sha256=run.run.checkpoint.sha256,
                    tensor_schema_hash=run.model.tensor_schema_hash,
                    world_size=world_size,
                    benchmark_seed=run.selector.benchmark_seed,
                    rank=rank,
                    node_id=run.run.participants[rank].public_key,
                    interval=1,
                    total_outer_steps=2,
                    backend=backend,
                    acceptance_eligible=backend == "nccl",
                    source_repository=run.source.repository,
                    source_commit=run.source.commit,
                    source_path=run.source.files[0],
                    outer_gradient_convention=(
                        run.source.outer_gradient_convention
                    ),
                    pairing_convention="dromeus-peer-scheduler-v1",
                ),
            )
            if run.selector.profile == "trajectory"
            else None
        )
        result = run_reference_rank(
            run=run,
            rank=rank,
            backend=backend,
            trainer=trainer,
            exchange=TorchDistributedPairExchange(),
            device=device,
            trajectory=trajectory,
        )
        rank_root = arguments.output_dir / f"rank-{rank}"
        rank_root.mkdir(parents=True, exist_ok=True)
        (rank_root / "summary.json").write_text(
            json.dumps(
                asdict(result),
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        dist.barrier()  # pyright: ignore[reportUnknownMemberType]
        return 0
    finally:
        dist.destroy_process_group()


def _torch_weights(
    values: Mapping[str, np.ndarray],
    *,
    device: torch.device,
) -> dict[str, Tensor]:
    return {
        name: torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
            np.ascontiguousarray(value)
        )
        .to(device=device, dtype=torch.float32)
        .clone()
        for name, value in values.items()
    }


def _numpy_weights(values: Mapping[str, Tensor]) -> dict[str, np.ndarray]:
    return {
        name: value.detach().cpu().numpy().astype(np.float32, copy=True)
        for name, value in values.items()
    }


def _launcher_environment() -> tuple[int, int, int]:
    try:
        return (
            int(os.environ["RANK"]),
            int(os.environ["WORLD_SIZE"]),
            int(os.environ.get("LOCAL_RANK", "0")),
        )
    except (KeyError, ValueError) as error:
        raise ValueError("reference runner must be launched with torchrun") from error


def _validate_backend_available(backend: Backend) -> None:
    if backend == "nccl" and not dist.is_nccl_available():
        raise RuntimeError("installed Torch does not provide NCCL")
    if backend == "gloo" and not dist.is_gloo_available():
        raise RuntimeError("installed Torch does not provide Gloo")


def _device_for_backend(
    backend: Backend,
    *,
    local_rank: int,
    accelerator_class: str,
) -> torch.device:
    torch.use_deterministic_algorithms(True)
    if backend == "gloo":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("NCCL reference requires CUDA")
    torch.cuda.set_device(local_rank)
    actual = torch.cuda.get_device_name(local_rank)
    if _normalized_device_name(accelerator_class) not in _normalized_device_name(
        actual
    ):
        raise RuntimeError("CUDA device does not match frozen accelerator class")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device("cuda", local_rank)


def _normalized_device_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("smoke", "trajectory", "official"),
        required=True,
    )
    parser.add_argument("--world-size", type=int, choices=(4, 8, 16), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ReferenceRunResult",
    "build_reference_trainer",
    "main",
    "run_reference_rank",
    "scheduled_peer_rank",
]
