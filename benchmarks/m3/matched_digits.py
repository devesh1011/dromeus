"""Prepare/run a public-digit baseline from a private reference draft.

Uses the existing four-source CNN/task, with the private draft's optimizer,
Poisson batch request, schedule, horizon, outer parameters and codecs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from benchmarks.m3.matched_baseline import (
    BaselineRecipe,
    MatchedBaseline,
    matched_draft,
)
from dromeus.application import PreparedApplication, definition_hash
from dromeus.manifests.canonical import (
    canonical_hash,
    canonical_json,
    file_sha256,
    save_safetensors,
)
from dromeus.manifests.models import ApplicationTaskContract, DraftRunSpec
from dromeus.node import NodeConfig, run_node
from dromeus.persistence.archive import RunArchive
from dromeus.training.private_trainer import build_private_optimizer

SOURCES = ("mnist", "usps", "svhn", "optdigits")
TASK = (
    "four-real-digit-sources-v1:labels0..9;gray32x32;bilinear;"
    "source-intensity-scaled01;common-minus0.5-div0.5;crossentropy"
)


def model_definition(seed: int) -> str:
    return (
        "digits-v1:Conv1-16k3GN4ReLUPool2,Conv16-32k3GN8ReLUPool2,Conv32-6"
        f"4k3GN8ReLUAdaptivePool4,Flatten,Linear1024-10;FP32;seed{seed}"
    )


def model_factory(device: str, seed: int) -> nn.Sequential:
    with torch.random.fork_rng(devices=[]):  # pyright: ignore[reportUnknownMemberType]
        torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
        model = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(1024, 10),
        )
    return model.to(device)


def load_arrays(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path, allow_pickle=False) as arrays:
        x, y = arrays["inputs"], arrays["labels"]
        if (
            x.dtype != np.float32
            or y.dtype != np.int64
            or not len(y)
            or x.shape != (len(y), 1, 32, 32)
            or not np.isfinite(x).all()
            or x.min() < 0
            or x.max() > 1
            or y.min() < 0
            or y.max() > 9
        ):
            raise ValueError("invalid public digit arrays")
        return torch.tensor(x) * 2 - 1, torch.tensor(y)


def recipe_for(
    reference: DraftRunSpec, *, evaluation_interval: int = 25
) -> BaselineRecipe:
    spec = reference.private_training
    if (
        spec is None
        or reference.privacy is None
        or spec.loss != "cross_entropy"
        or spec.evaluation_provenance != "public_fixed_holdout"
    ):
        raise ValueError("a private cross-entropy reference draft is required")
    return BaselineRecipe(
        batch_size=spec.batch_size,
        logical_steps=reference.round_count * reference.local_steps,
        loss=spec.loss,
        schedule=spec.schedule,
        optimizer_definition=canonical_json(spec.optimizer).decode(),
        evaluation_interval_rounds=evaluation_interval,
    )


def make_baseline(
    reference: DraftRunSpec,
    *,
    data: Path,
    rank: int,
    device: str,
    seed: int,
    physical: int,
    evaluation_interval: int = 25,
) -> MatchedBaseline:
    if not 0 <= rank < 4 or reference.expected_participant_count != 4:
        raise ValueError("digit baseline needs four nodes and rank 0..3")
    if (
        reference.model_definition_hash != definition_hash(model_definition(seed))
        or not isinstance(reference.dataset, ApplicationTaskContract)
        or reference.dataset.definition_hash != definition_hash(TASK)
    ):
        raise ValueError("reference differs from the public digit model/task")
    recipe = recipe_for(reference, evaluation_interval=evaluation_interval)
    spec = reference.private_training
    assert spec is not None
    if spec.optimizer.torch_version != torch.__version__:
        raise ValueError("reference optimizer Torch version mismatch")
    training_path = data / f"{SOURCES[rank]}-train.npz"
    x, y = load_arrays(training_path)
    tests = {name: load_arrays(data / f"{name}-test.npz") for name in SOURCES}

    def evaluate(model: nn.Module) -> dict[str, float]:
        metrics: dict[str, float] = {}
        with torch.no_grad():
            for name, (inputs, targets) in tests.items():
                correct = sum(
                    int(
                        (
                            model(inputs[i : i + 256].to(device)).argmax(1).cpu()
                            == targets[i : i + 256]
                        ).sum()
                    )
                    for i in range(0, len(targets), 256)
                )
                metrics[f"accuracy_{name}"] = correct / len(targets)
        metrics["accuracy_macro_source"] = sum(metrics.values()) / 4
        metrics["accuracy_worst_source"] = min(
            metrics[f"accuracy_{n}"] for n in SOURCES
        )
        return metrics

    def optimizer(model: nn.Module) -> torch.optim.Optimizer:
        if (
            tuple(name for name, _ in model.named_parameters())
            != spec.optimizer.parameter_names
        ):
            raise ValueError("reference optimizer parameter names mismatch")
        return build_private_optimizer(model, spec.optimizer)

    return MatchedBaseline(
        model_factory=lambda: model_factory(device, seed),
        optimizer_factory=optimizer,
        inputs=x,
        targets=y,
        recipe=recipe,
        sampling_seed=seed + rank,
        local_data_identity=file_sha256(training_path),
        max_physical_batch_size=physical,
        evaluate=evaluate,
    )


def execute(
    *,
    mode: str,
    reference: DraftRunSpec,
    data: Path,
    root: Path,
    rank: int,
    device: str,
    seed: int,
    physical: int,
    run_id: str,
    evaluation_interval: int,
) -> None:
    if evaluation_interval <= 0:
        raise ValueError("evaluation interval must be positive")
    if (root / "baseline.json").exists() or (root / "initial.safetensors").exists():
        raise ValueError("use a fresh output directory for each baseline run")
    if not 0 <= rank < 4:
        raise ValueError("digit rank must be 0..3")
    input_paths = [data / f"{SOURCES[rank]}-train.npz"] + [
        data / f"{n}-test.npz" for n in SOURCES
    ]
    input_hashes = {path: file_sha256(path) for path in input_paths}
    if mode == "node":
        config = NodeConfig.model_validate_json((root / "node.json").read_bytes())
        if (
            config.run_root.resolve() != root.resolve()
            or config.draft_path.resolve() != (root / "draft.json").resolve()
        ):
            raise ValueError("node config must use this baseline root and draft.json")
    baseline = make_baseline(
        reference,
        data=data,
        rank=rank,
        device=device,
        seed=seed,
        physical=physical,
        evaluation_interval=evaluation_interval,
    )
    if any(file_sha256(p) != digest for p, digest in input_hashes.items()):
        raise ValueError("public digit inputs changed during preparation")
    draft = matched_draft(reference, baseline.recipe, run_id=run_id)
    prepared = baseline.prepare(
        draft,
        model_definition=model_definition(seed),
        task_definition=TASK,
        evaluation_interval=evaluation_interval,
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "draft.json").write_bytes(canonical_json(draft))
    initial = prepared.create_initial_checkpoint(root / "initial.safetensors")
    history: list[dict[str, object]] = []
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()

    def evaluation() -> None:
        result = baseline.trainer.evaluate()
        assert result is not None
        history.append(
            {"logical_steps": baseline.trainer.completed_steps, **result.metrics}
        )

    evaluation()
    if mode == "local":
        interval = evaluation_interval * draft.local_steps
        while baseline.trainer.completed_steps < baseline.recipe.logical_steps:
            baseline.trainer.train_local_steps(
                min(
                    interval,
                    baseline.recipe.logical_steps - baseline.trainer.completed_steps,
                )
            )
            evaluation()
    elif mode == "node":
        config = NodeConfig.model_validate_json((root / "node.json").read_bytes())

        def training(actual: DraftRunSpec) -> PreparedApplication:
            prepared.validate_draft(actual)
            return prepared

        asyncio.run(run_node(config, prepare_training=training))
        archive = RunArchive.open(root / "run-store")
        if (
            archive.state.terminal is None
            or archive.state.terminal.result != "complete"
            or archive.state.committed_round != draft.round_count - 1
        ):
            raise ValueError("baseline node did not complete the reference horizon")
        evaluation()
    else:
        raise ValueError("baseline mode must be local or node")
    if baseline.trainer.completed_steps != baseline.recipe.logical_steps:
        raise ValueError("baseline logical step count mismatch")
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    if any(file_sha256(p) != digest for p, digest in input_hashes.items()):
        raise ValueError("public digit inputs changed during training")
    save_safetensors(
        baseline.trainer.weights(), str(root / "baseline-model.safetensors")
    )
    (root / "baseline.json").write_text(
        json.dumps(
            {
                **baseline.statistics,
                "mode": mode,
                "elapsed_training_and_evaluation_seconds": elapsed,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
                if device.startswith("cuda")
                else None,
                "source": SOURCES[rank],
                "rank": rank,
                "reference_draft_hash": canonical_hash(reference),
                "baseline_draft_hash": canonical_hash(draft),
                "initial_checkpoint_sha256": initial.sha256,
                "model_sha256": file_sha256(root / "baseline-model.safetensors"),
                "training_data_sha256": file_sha256(
                    data / f"{SOURCES[rank]}-train.npz"
                ),
                "public_holdout_sha256": {
                    n: file_sha256(data / f"{n}-test.npz") for n in SOURCES
                },
                "model_seed": seed,
                "sampling_seed": seed + rank,
                "physical_batch_limit": physical,
                "evaluation_interval_rounds": evaluation_interval,
                "recipe": baseline.recipe.model_dump(mode="json"),
                "history": history,
                "device": device,
                "torch_version": torch.__version__,
                "opacus_version": "1.6.0",
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["local", "node"])
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--physical-batch", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--evaluation-interval", type=int, default=25)
    args = parser.parse_args()
    torch.set_num_threads(1)
    execute(
        mode=args.mode,
        reference=DraftRunSpec.model_validate_json(args.reference.read_bytes()),
        data=args.data,
        root=args.root,
        rank=args.rank,
        device=args.device,
        seed=args.seed,
        physical=args.physical_batch,
        run_id=args.run_id,
        evaluation_interval=args.evaluation_interval,
    )


if __name__ == "__main__":
    main()
