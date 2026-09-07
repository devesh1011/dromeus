"""Synchronous development controls with independent NoLoCo outer mathematics.

The workload and generic PyTorch inner trainer are shared deliberately. This
module never calls Dromeus's algorithm implementation, codecs, gossip, or runtime.
The reference communicates via in-process copies, not NCCL or AXL.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from safetensors.numpy import (
    load_file as _load_file,  # pyright: ignore[reportUnknownVariableType]
)
from safetensors.numpy import (
    save_file as _save_file,  # pyright: ignore[reportUnknownVariableType]
)

from benchmarks.federated.data_plan import DataPlan
from benchmarks.federated.workload import (
    Experiment,
    build_model,
    evaluate,
    local_data,
    sample_budget,
    summarize_nodes,
    task_for,
    trainer_settings,
    write_json,
)
from benchmarks.noloco_reference.optimizer import (
    PINNED_UPSTREAM_COMMIT,
    ReferenceNoLoCoConfig,
    apply_outer_step,
    outer_gradient,
)
from dromeus.adapters.classification.data import validate_local_data
from dromeus.adapters.classification.torch_trainer import PyTorchTrainer
from dromeus.gossip.peer_scheduler import PeerScheduler

type Weights = dict[str, np.ndarray]
type Trajectories = list[list[Weights]]
save_tensors = cast(Callable[[dict[str, np.ndarray], str], None], _save_file)
load_tensors = cast(Callable[[str], dict[str, np.ndarray]], _load_file)


@dataclass
class ControlResult:
    report: dict[str, object]
    trajectories: Trajectories


def run_control(
    *,
    experiment: Experiment,
    plan: DataPlan,
    initial_checkpoint: Path,
    keys: tuple[str, ...],
    root: Path,
    method: str,
) -> ControlResult:
    if method not in {"reference", "local-only"}:
        raise ValueError("control must be reference or local-only")
    root.mkdir(parents=True, exist_ok=False)
    trainers: list[PyTorchTrainer] = []
    for rank in range(plan.world_size):
        data = validate_local_data(local_data(plan, rank), task_for(plan))
        trainer = PyTorchTrainer(
            model=build_model(experiment.seed),
            model_definition=experiment.definition,
            train_data=data.train_data,
            test_data=data.evaluation_data,
            settings=trainer_settings(experiment, rank),
        )
        trainer.load_checkpoint(initial_checkpoint)
        trainers.append(trainer)
    trajectories: Trajectories = [[trainer.weights()] for trainer in trainers]
    momentum = [
        {name: torch.zeros_like(torch.tensor(value)) for name, value in path[0].items()}
        for path in trajectories
    ]
    scheduler = PeerScheduler(keys, seed=experiment.seed)
    outer = experiment.settings["outer"]
    config = ReferenceNoLoCoConfig(
        alpha=outer["alpha"], beta=outer["beta"], gamma=outer["gamma"]
    )
    started = time.monotonic()
    round_timings: list[float] = []
    for round_id in range(experiment.rounds):
        round_started = time.monotonic()
        slow = [
            {name: torch.tensor(value) for name, value in path[-1].items()}
            for path in trajectories
        ]
        for rank, trainer in enumerate(trainers):
            if method == "reference":
                trainer.load_weights(trajectories[rank][-1])
            trainer.train_local_steps(experiment.settings["inner_steps"])
        if method == "reference":
            gradients = [
                outer_gradient(
                    slow[rank],
                    {
                        name: torch.tensor(value)
                        for name, value in trainer.weights().items()
                    },
                )
                for rank, trainer in enumerate(trainers)
            ]
            pairing = scheduler.schedule(round_id)
            for rank, trainer in enumerate(trainers):
                peer = keys.index(pairing.peer_for(keys[rank]))
                updated = apply_outer_step(
                    slow_weights=slow[rank],
                    outer_momentum=momentum[rank],
                    local_outer_gradient=gradients[rank],
                    peer_outer_gradient=gradients[peer],
                    peer_slow_weights=slow[peer],
                    config=config,
                )
                momentum[rank] = updated.outer_momentum
                trainer.load_weights(
                    {
                        name: value.numpy().copy()
                        for name, value in updated.slow_weights.items()
                    }
                )
        for rank, trainer in enumerate(trainers):
            weights = trainer.weights()
            trajectories[rank].append(weights)
            path = root / f"node-{rank}" / f"round-{round_id:03d}.safetensors"
            path.parent.mkdir(parents=True, exist_ok=True)
            save_tensors(weights, str(path))
        round_timings.append(time.monotonic() - round_started)
    nodes: list[dict[str, object]] = []
    for rank, trainer in enumerate(trainers):
        nodes.append(
            {
                "rank": rank,
                "common_heldout": evaluate(
                    trainer.weights(), plan.common_evaluation.path
                ),
                "local_heldout": evaluate(
                    trainer.weights(), plan.nodes[rank].evaluation.path
                ),
                "optimizer_steps": experiment.rounds
                * experiment.settings["inner_steps"],
                "sample_presentations": sample_budget(
                    plan.nodes[rank].train.sample_count,
                    steps=experiment.rounds * experiment.settings["inner_steps"],
                    batch_size=experiment.settings["batch_size"],
                ),
            }
        )
    report = {
        "method": method,
        "backend": "synchronous-in-process-development",
        "nccl_acceptance_eligible": False,
        "upstream_commit": PINNED_UPSTREAM_COMMIT if method == "reference" else None,
        ("shares"): (
            "model, datasets, initialization, generic inner trainer, seeded "
            "data order, LR schedule, public-key pair scheduler"
        ),
        "independent": "NoLoCo outer optimizer and in-process pair exchange"
        if method == "reference"
        else "no pair exchange or outer optimizer",
        "seconds": time.monotonic() - started,
        "round_seconds": round_timings,
        **summarize_nodes(nodes),
    }
    write_json(root / "result.json", report)
    return ControlResult(report, trajectories)
