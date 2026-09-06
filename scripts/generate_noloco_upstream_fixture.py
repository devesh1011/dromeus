#!/usr/bin/env python3
"""Generate the deterministic NoLoCo outer-step fixture from pinned upstream code."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

PINNED_REPOSITORY = "https://github.com/gensyn-ai/noloco"
PINNED_COMMIT = "a1b4a425bdc4050a356cf9f4bae7c383419703ab"
UPSTREAM_SOURCE = Path("src/noloco/sparse_optimizer_c.py")
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "tests/golden/noloco_upstream_outer_step_v1.json"
)
OUTER_LR = 0.7
OUTER_MOMENTUM = 0.5
NODE_INPUTS = (
    {
        "rank": 0,
        "slow_weights": [1.0, -2.0, 0.5],
        "fast_weights": [0.6, -1.0, 0.25],
        "outer_momentum_before": [0.2, -0.4, 0.1],
    },
    {
        "rank": 1,
        "slow_weights": [3.0, 0.0, -1.5],
        "fast_weights": [2.4, -0.5, -0.75],
        "outer_momentum_before": [-0.3, 0.6, -0.2],
    },
)


class _FixtureModel(nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(values, dtype=torch.float32))


def _git_output(upstream_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(upstream_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_upstream(upstream_root: Path) -> tuple[Path, str]:
    commit = _git_output(upstream_root, "rev-parse", "HEAD")
    if commit != PINNED_COMMIT:
        raise ValueError(f"upstream HEAD must be {PINNED_COMMIT}, got {commit}")
    source = upstream_root / UPSTREAM_SOURCE
    if not source.is_file():
        raise FileNotFoundError(source)
    changed = subprocess.run(
        [
            "git",
            "-C",
            str(upstream_root),
            "diff",
            "--quiet",
            "HEAD",
            "--",
            str(UPSTREAM_SOURCE),
        ],
        check=False,
    )
    if changed.returncode != 0:
        raise ValueError(f"upstream source has local changes: {UPSTREAM_SOURCE}")
    blob = _git_output(upstream_root, "rev-parse", f"HEAD:{UPSTREAM_SOURCE}")
    return source, blob


def _load_upstream(source: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_pinned_noloco_optimizer", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load upstream source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _worker(rank: int, init_path: str, source_path: str, result_root: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
    )
    try:
        node = NODE_INPUTS[rank]
        model = _FixtureModel(cast(list[float], node["fast_weights"]))
        slow = [
            torch.tensor(
                cast(list[float], node["slow_weights"]), dtype=torch.float32
            )
        ]
        momentum = [
            torch.tensor(
                cast(list[float], node["outer_momentum_before"]),
                dtype=torch.float32,
            )
        ]
        module = _load_upstream(Path(source_path))
        outer_step = cast(Callable[..., Any], module.outer_step)
        outer_step(
            [0, 1],
            model,
            slow,
            momentum,
            OUTER_LR,
            OUTER_MOMENTUM,
        )
        result = {
            "rank": rank,
            "outer_momentum_after": momentum[0].detach().cpu().tolist(),
            "slow_weights_after": slow[0].detach().cpu().tolist(),
        }
        destination = Path(result_root) / f"rank-{rank}.json"
        destination.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def generate(upstream_root: Path, output: Path) -> None:
    source, source_blob = _validate_upstream(upstream_root)
    with tempfile.TemporaryDirectory(prefix="noloco-upstream-fixture-") as temporary:
        temporary_root = Path(temporary)
        init_path = temporary_root / "gloo-rendezvous"
        mp.spawn(
            _worker,
            args=(str(init_path), str(source), str(temporary_root)),
            nprocs=2,
            join=True,
        )
        expected = {
            int(result["rank"]): result
            for result in (
                json.loads(
                    (temporary_root / f"rank-{rank}.json").read_text(
                        encoding="utf-8"
                    )
                )
                for rank in range(2)
            )
        }

    nodes = []
    for node in NODE_INPUTS:
        rank = cast(int, node["rank"])
        slow = np.asarray(node["slow_weights"], dtype=np.float32)
        fast = np.asarray(node["fast_weights"], dtype=np.float32)
        nodes.append(
            {
                **node,
                "outer_gradient": (slow - fast).astype(np.float32).tolist(),
                "outer_momentum_after": expected[rank]["outer_momentum_after"],
                "slow_weights_after": expected[rank]["slow_weights_after"],
            }
        )

    fixture = {
        "fixture_version": 1,
        "authority": {
            "repository": PINNED_REPOSITORY,
            "commit": PINNED_COMMIT,
            "source": str(UPSTREAM_SOURCE),
            "source_blob": source_blob,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "function": "outer_step",
        },
        "execution": {
            "backend": "gloo",
            "device": "cpu",
            "dtype": "float32",
            "torch_version": torch.__version__,
            "world_size": 2,
        },
        "convention": {
            "alpha": OUTER_MOMENTUM,
            "beta": OUTER_LR,
            "effective_gamma": OUTER_LR,
            "dromeus_outer_gradient": "slow_weights - fast_weights",
        },
        "comparison_tolerance": {"absolute": 1e-6, "relative": 0.0},
        "tensor_name": "weight",
        "nodes": nodes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream-root",
        type=Path,
        required=True,
        help="clean checkout of the pinned gensyn-ai/noloco repository",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    generate(arguments.upstream_root.resolve(), arguments.output.resolve())


if __name__ == "__main__":
    main()
