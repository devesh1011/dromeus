"""Materialize M2 from pilot-backed codecs and the paper training horizon."""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import yaml

from benchmarks.noloco.experiment import FrozenExperiment, pairing_digest
from dromeus.manifests.canonical import file_sha256
from dromeus.training.cifar10 import (
    DATASET_REVISION,
    PREPROCESSING_HASH,
    create_initial_checkpoint,
)
from dromeus.training.data import iid_partition_index_hashes
from dromeus.training.models import resolve_model
from dromeus.training.resnet18_groupnorm import MODEL_ID
from dromeus.training.trainer import derive_benchmark_seed

_BENCHMARK_SEEDS = (17, 29, 41)
_WORLD_SIZES = (4, 8, 16)
_UPSTREAM_COMMIT = "a1b4a425bdc4050a356cf9f4bae7c383419703ab"
_AXL_COMMIT = "628e28ace077f26dfe8d0259009b357216a9d8d4"
_PUBLIC_KEY = re.compile(r"^[0-9a-f]{64}$")


def build_frozen_experiment_document(
    *,
    artifact_root: Path,
    identity_public_keys: Sequence[str],
    pilot_report: Path,
    checkpoints: Mapping[int, Path],
) -> dict[str, object]:
    """Build one closed document using a persistent pool of 16 AXL identities."""
    root = artifact_root.resolve()
    members = tuple(sorted(identity_public_keys))
    if (
        len(members) != 16
        or len(set(members)) != 16
        or any(_PUBLIC_KEY.fullmatch(value) is None for value in members)
    ):
        raise ValueError("identity pool must contain 16 unique AXL public keys")
    if set(checkpoints) != set(_BENCHMARK_SEEDS):
        raise ValueError("checkpoint map must contain benchmark seeds 17, 29, and 41")
    pilot_path = _relative_artifact(root, pilot_report, label="pilot report")
    checkpoint_paths = {
        seed: _relative_artifact(
            root,
            checkpoints[seed],
            label=f"checkpoint {seed}",
        )
        for seed in _BENCHMARK_SEEDS
    }
    pilot_values = _load_json(_frozen_root() / "pilot-values.json")
    runtime = _load_json(_frozen_root().parent / "container" / "runtime-lock.json")
    if pilot_values.get("status") != "frozen":
        raise ValueError("pilot values are not frozen")
    pilot_evidence = cast(dict[str, object], pilot_values["evidence"])
    if file_sha256(pilot_report) != pilot_evidence.get("sha256"):
        raise ValueError("pilot report hash does not match frozen pilot evidence")
    if runtime.get("status") != "built-and-gpu-verified":
        raise ValueError("GPU runtime is not built and verified")
    runs = [
        _run_document(
            profile="official",
            world_size=world_size,
            benchmark_seed=seed,
            members=members[:world_size],
            checkpoint_path=checkpoint_paths[seed],
            checkpoint_sha256=file_sha256(checkpoints[seed]),
            round_count=cast(int, pilot_values["outer_round_count"]),
            warmup_inner_steps=cast(
                int,
                cast(dict[str, object], pilot_values["learning_rate_schedule"])[
                    "warmup_inner_steps"
                ],
            ),
            pilot_values=pilot_values,
        )
        for world_size in _WORLD_SIZES
        for seed in _BENCHMARK_SEEDS
    ]
    runs.append(
        _run_document(
            profile="trajectory",
            world_size=4,
            benchmark_seed=17,
            members=members[:4],
            checkpoint_path=checkpoint_paths[17],
            checkpoint_sha256=file_sha256(checkpoints[17]),
            round_count=2,
            warmup_inner_steps=10,
            pilot_values=pilot_values,
        )
    )
    model = resolve_model(MODEL_ID)
    codec = cast(dict[str, object], pilot_values["codec"])
    outer_codec = cast(dict[str, object], codec["outer_gradient"])
    slow_codec = cast(dict[str, object], codec["slow_weights"])
    runtime_image = cast(dict[str, object], runtime["image"])
    pytorch = cast(dict[str, object], runtime["pytorch"])
    document: dict[str, object] = {
        "schema_version": 2,
        "status": "frozen",
        "source": {
            "repository": "gensyn-ai/noloco",
            "commit": _UPSTREAM_COMMIT,
            "files": ["src/noloco/sparse_optimizer_c.py"],
            "outer_gradient_convention": "phi-minus-theta-v1",
        },
        "model": {
            "model_id": model.model_id,
            "definition_hash": model.definition_hash,
            "tensor_schema_hash": model.tensor_schema_hash,
            "parameter_count": model.parameter_count,
            "dtype": "float32",
        },
        "dataset": {
            "dataset_id": "cifar10",
            "source": "huggingface-uoft-cs-cifar10",
            "revision": DATASET_REVISION,
            "preprocessing_hash": PREPROCESSING_HASH,
            "partition_seed": 7,
            "sample_count": 50_000,
        },
        "workload": {
            "batch_size": 128,
            "crop_padding": 4,
            "normalize": True,
            "augment": True,
            "weight_decay": 0.0,
        },
        "algorithm": {
            "alpha": 0.5,
            "beta": 0.7,
            "gamma": 0.7,
            "inner_steps": 50,
            "adam_learning_rate": _schedule_value(
                pilot_values,
                "peak_learning_rate",
            ),
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "gradient_clip_norm": 1.0,
        },
        "ablations": [
            {
                "ablation_id": "identity",
                "artifact_codecs": [
                    {"artifact_name": "outer_gradient", "codec_id": "identity-v1"},
                    {"artifact_name": "slow_weights", "codec_id": "identity-v1"},
                ],
            },
            {
                "ablation_id": "compressed",
                "artifact_codecs": [
                    {
                        "artifact_name": "outer_gradient",
                        "codec_id": outer_codec["codec_id"],
                        "top_k_fraction": outer_codec["top_k_fraction"],
                        "lossy_allowed": True,
                    },
                    {
                        "artifact_name": "slow_weights",
                        "codec_id": slow_codec["codec_id"],
                        "lossy_allowed": True,
                    },
                ],
            },
        ],
        "pilot": {
            "path": pilot_path.as_posix(),
            "sha256": file_sha256(pilot_report),
        },
        "hardware": {
            "accelerator_class": "NVIDIA-A10G",
            "instance_type": "g5.xlarge",
            "container_image_digest": runtime_image["digest"],
            "python_version": runtime["python"],
            "pytorch_version": pytorch["version"],
            "cuda_version": pytorch["cuda"],
            "cudnn_version": pytorch["cudnn"],
            "nccl_version": pytorch["nccl"],
            "driver_version": "595.91.07",
            "axl_commit": _AXL_COMMIT,
        },
        "runs": runs,
    }
    FrozenExperiment.model_validate(document)
    return document


def freeze_experiment(
    *,
    output_root: Path,
    identity_public_keys: Sequence[str],
    pilot_report: Path,
) -> Path:
    """Create deterministic checkpoints and write a validated frozen artifact."""
    import torch

    validate_freeze_runtime(
        machine=platform.machine(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    local_pilot = output_root / "pilot-report.json"
    shutil.copyfile(pilot_report, local_pilot)
    model = resolve_model(MODEL_ID)
    checkpoints: dict[int, Path] = {}
    for seed in _BENCHMARK_SEEDS:
        path = output_root / f"checkpoint-{seed}.safetensors"
        create_initial_checkpoint(
            path,
            seed=derive_benchmark_seed(seed, "model-initialization"),
            model_id=model.model_id,
            model_definition_hash=model.definition_hash,
        )
        checkpoints[seed] = path
    document = build_frozen_experiment_document(
        artifact_root=output_root,
        identity_public_keys=identity_public_keys,
        pilot_report=local_pilot,
        checkpoints=checkpoints,
    )
    output = output_root / "experiment.yaml"
    output.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
    return output


def validate_freeze_runtime(
    *,
    machine: str,
    python_version: str,
    torch_version: str,
) -> None:
    """Require checkpoint generation in the same frozen x86 container runtime."""
    if machine != "x86_64":
        raise RuntimeError("frozen checkpoints must be generated on x86_64")
    if python_version != "3.12.11" or torch_version != "2.12.1+cu130":
        raise RuntimeError("checkpoint generation runtime is not frozen")


def _run_document(
    *,
    profile: str,
    world_size: int,
    benchmark_seed: int,
    members: tuple[str, ...],
    checkpoint_path: Path,
    checkpoint_sha256: str,
    round_count: int,
    warmup_inner_steps: int,
    pilot_values: dict[str, object],
) -> dict[str, object]:
    trainer_seed = derive_benchmark_seed(benchmark_seed, "local-training")
    return {
        "selector": {
            "profile": profile,
            "world_size": world_size,
            "benchmark_seed": benchmark_seed,
        },
        "checkpoint": {
            "path": checkpoint_path.as_posix(),
            "sha256": checkpoint_sha256,
        },
        "model_seed": derive_benchmark_seed(
            benchmark_seed,
            "model-initialization",
        ),
        "scheduler_seed": benchmark_seed,
        "rank_seeds": [
            {
                "rank": rank,
                "trainer_seed": trainer_seed + rank,
                "augmentation_seed": trainer_seed + rank + 1,
                "loader_seed": trainer_seed + rank + 2,
            }
            for rank in range(world_size)
        ],
        "participants": [
            {"rank": rank, "node_index": rank, "public_key": member}
            for rank, member in enumerate(members)
        ],
        "partition_index_hashes": iid_partition_index_hashes(
            source_sample_count=50_000,
            participant_count=world_size,
            seed=7,
        ),
        "round_count": round_count,
        "transfer": {
            "chunk_size_bytes": pilot_values["chunk_size_bytes"],
            "window_size": pilot_values["window_size"],
        },
        "schedule": {
            "schedule_id": "linear-warmup-cosine-v1",
            "total_inner_steps": round_count * 50,
            "warmup_inner_steps": warmup_inner_steps,
            "start_learning_rate": _schedule_value(
                pilot_values,
                "start_learning_rate",
            ),
            "peak_learning_rate": _schedule_value(
                pilot_values,
                "peak_learning_rate",
            ),
            "final_learning_rate": _schedule_value(
                pilot_values,
                "final_learning_rate",
            ),
        },
        "pairing_digest": pairing_digest(
            members,
            seed=benchmark_seed,
            rounds=round_count,
        ),
        "evidence": {
            "trajectory_interval": 1,
            "absolute_tolerance": 1e-6,
            "relative_tolerance": 1e-6,
            "evaluation_interval_rounds": pilot_values[
                "evaluation_interval_rounds"
            ],
            "countsketch_interval_rounds": pilot_values[
                "countsketch_interval_rounds"
            ],
            "analysis_checkpoint_interval_rounds": pilot_values[
                "analysis_checkpoint_interval_rounds"
            ],
            "analysis_storage_budget_bytes_per_node": pilot_values[
                "analysis_storage_budget_bytes_per_node"
            ],
            "residual_l2_bound": cast(
                dict[str, object], pilot_values["residual_bounds"]
            )["l2_norm"],
            "residual_to_signal_ratio_bound": cast(
                dict[str, object], pilot_values["residual_bounds"]
            )["residual_to_signal_ratio"],
            "overlap_enabled": pilot_values["overlap"],
        },
    }


def _schedule_value(values: dict[str, object], field: str) -> float:
    schedule = cast(dict[str, object], values["learning_rate_schedule"])
    return cast(float, schedule[field])


def _relative_artifact(root: Path, path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"{label} must be a file inside the artifact root")
    return resolved.relative_to(root)


def _frozen_root() -> Path:
    return Path(__file__).resolve().parent / "frozen"


def _load_json(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identities", type=Path, required=True)
    parser.add_argument("--pilot-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    identities = cast(
        dict[str, object],
        json.loads(arguments.identities.read_text(encoding="utf-8")),
    )
    public_keys = cast(list[str], identities.get("public_keys"))
    output = freeze_experiment(
        output_root=arguments.output_root,
        identity_public_keys=public_keys,
        pilot_report=arguments.pilot_report,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_frozen_experiment_document",
    "freeze_experiment",
    "main",
    "validate_freeze_runtime",
]
