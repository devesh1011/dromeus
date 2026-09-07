"""One frozen workload shared by local runtime and independent control runners."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from benchmarks.federated.data_plan import DataPlan
from dromeus.adapters.classification.data import (
    LocalClassificationData,
    load_npz_dataset,
)
from dromeus.adapters.classification.torch_trainer import TrainerSettings
from dromeus.manifests.models import (
    ClassificationTaskContract,
    DraftRunSpec,
    WarmupCosineSchedule,
)

FROZEN_EXPERIMENT_SHA256 = (
    "d3c68077ae9f7d82439844cea4da0450a247997bc88127f945b5681270b9b4d6"
)
PREPROCESSING_HASH = hashlib.sha256(
    b"federated-data-v1;preprocessed-fp32-eight-features"
).hexdigest()


@dataclass(frozen=True)
class Experiment:
    path: Path
    sha256: str
    settings: Mapping[str, Any]

    @property
    def seed(self) -> int:
        return int(self.settings["seed"])

    @property
    def rounds(self) -> int:
        return int(self.settings["round_count"])

    @property
    def definition(self) -> str:
        return str(self.settings["model_definition"])

    @property
    def model_hash(self) -> str:
        return hashlib.sha256(self.definition.encode()).hexdigest()


def load_experiment() -> Experiment:
    """Require the exact predeclared version-1 bytes; no post-run tuning flags."""
    path = Path(__file__).with_name("experiment.json")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != FROZEN_EXPERIMENT_SHA256:
        raise ValueError("frozen development experiment changed; declare a new version")
    return Experiment(path, digest, cast(dict[str, Any], json.loads(payload)))


def build_model(seed: int) -> nn.Module:
    with torch.random.fork_rng(devices=[]):  # pyright: ignore[reportUnknownMemberType]
        torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
        return nn.Linear(8, 4, bias=True).float()


def task_for(plan: DataPlan) -> ClassificationTaskContract:
    return ClassificationTaskContract(
        dataset_id="local-classification-v1",
        input_shape=plan.input_shape,
        input_dtype="float32",
        label_names=plan.label_names,
        preprocessing_hash=PREPROCESSING_HASH,
    )


def local_data(plan: DataPlan, rank: int) -> LocalClassificationData:
    node = plan.nodes[rank]
    return LocalClassificationData(
        train_data=load_npz_dataset(node.train.path),
        evaluation_data=load_npz_dataset(node.evaluation.path),
        label_names=plan.label_names,
        preprocessing_hash=PREPROCESSING_HASH,
        source_id=f"synthetic-v1-{plan.profile}-rank-{rank}",
    )


def trainer_settings(experiment: Experiment, rank: int) -> TrainerSettings:
    config = experiment.settings
    adam = config["adam"]
    return TrainerSettings(
        seed=experiment.seed + rank,
        batch_size=config["batch_size"],
        learning_rate=adam["learning_rate"],
        optimizer="adam",
        weight_decay=config["weight_decay"],
        adam_beta1=adam["beta1"],
        adam_beta2=adam["beta2"],
        adam_epsilon=adam["epsilon"],
        gradient_clip_norm=adam["gradient_clip_norm"],
        learning_rate_schedule=WarmupCosineSchedule.model_validate(
            config["learning_rate_schedule"]
        ),
        device="cpu",
        augment=False,
    )


def build_draft(
    experiment: Experiment,
    plan: DataPlan,
    *,
    variant: str,
    run_id: str,
    source_commit: str,
    axl_version: str,
) -> DraftRunSpec:
    config = experiment.settings
    if variant not in config["variants"]:
        raise ValueError("unsupported frozen codec variant")
    compressed = variant == "bitmap-int8"
    return DraftRunSpec.model_validate(
        {
            "manifest_version": 4,
            "protocol_version": 1,
            "run_id": run_id,
            "algorithm_id": "noloco",
            "model_id": "synthetic-linear-8x4-v1",
            "model_definition_hash": experiment.model_hash,
            "dataset": task_for(plan),
            "expected_participant_count": 4,
            "environment": {
                "dromeus_version": "0.1.0",
                "dromeus_commit": source_commit,
                "protocol_version": 1,
                "pytorch_version": torch.__version__,
                "axl_version": axl_version,
                "model_definition_hash": experiment.model_hash,
                "container_image_digest": None,
            },
            "local_steps": config["inner_steps"],
            "round_count": experiment.rounds,
            "optimizer": "adam",
            "learning_rate": config["adam"]["learning_rate"],
            "peer_scheduler_seed": experiment.seed,
            "codec_id": "safetensors-v1",
            "transport": config["transport"],
            "consensus_sketch": {"size": 4096, "seed": experiment.seed},
            "training": {
                "batch_size": config["batch_size"],
                "momentum": 0.0,
                "weight_decay": config["weight_decay"],
                "learning_rate_schedule": config["learning_rate_schedule"],
                "learning_rate_gamma": 0.1,
                "crop_padding": 0,
                "normalize": False,
                "final_consensus_rounds": 0,
            },
            "algorithm_config": {
                **config["outer"],
                "inner_steps": config["inner_steps"],
                "adam": config["adam"],
            },
            "artifact_codecs": (
                [
                    {
                        "artifact_name": "outer_gradient",
                        "codec_id": "topk-bitmap-int8-v2",
                        "top_k_fraction": config["top_k_fraction"],
                        "lossy_allowed": True,
                    },
                    {
                        "artifact_name": "slow_weights",
                        "codec_id": "dense-int8-v1",
                        "lossy_allowed": True,
                    },
                ]
                if compressed
                else [
                    {"artifact_name": name, "codec_id": "identity-v1"}
                    for name in ("outer_gradient", "slow_weights")
                ]
            ),
        }
    )


def evaluate(weights: Mapping[str, np.ndarray], path: Path) -> dict[str, object]:
    """Evaluate explicit held-out NPZ records without touching training loaders."""
    model = build_model(0)
    state = {
        name: torch.from_numpy(value.copy())  # pyright: ignore[reportUnknownMemberType]
        for name, value in weights.items()
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    with np.load(path, allow_pickle=False) as dataset:
        inputs = torch.tensor(dataset["inputs"], dtype=torch.float32)
        targets = torch.tensor(dataset["labels"], dtype=torch.int64)
    with torch.no_grad():
        logits = model(inputs)
        loss = float(nn.functional.cross_entropy(logits, targets))
        predicted = logits.argmax(dim=1)
        correct = predicted == targets
    per_class: list[dict[str, object]] = []
    for label in range(4):
        mask = targets == label
        count = int(mask.sum())
        per_class.append(
            {
                "label": label,
                "count": count,
                "accuracy": None if not count else float(correct[mask].float().mean()),
            }
        )
    return {
        "loss": loss,
        "accuracy": float(correct.float().mean()),
        "sample_count": len(targets),
        "per_class": per_class,
    }


def summarize_nodes(nodes: list[dict[str, object]]) -> dict[str, object]:
    common = [
        float(cast(dict[str, Any], node["common_heldout"])["accuracy"])
        for node in nodes
    ]
    local = [
        float(cast(dict[str, Any], node["local_heldout"])["accuracy"]) for node in nodes
    ]
    return {
        "nodes": nodes,
        "common_mean_accuracy": float(np.mean(common)),
        "common_worst_accuracy": min(common),
        "local_mean_accuracy": float(np.mean(local)),
        "local_worst_accuracy": min(local),
    }


def sample_budget(sample_count: int, *, steps: int, batch_size: int) -> int:
    batch_sizes = [
        min(batch_size, sample_count - start)
        for start in range(0, sample_count, batch_size)
    ]
    return sum(batch_sizes[index % len(batch_sizes)] for index in range(steps))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(value, allow_nan=False, sort_keys=True, indent=2) + "\n"
        )
