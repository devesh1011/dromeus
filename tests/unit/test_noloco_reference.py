from __future__ import annotations

import ast
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import (
    load_file as _load_file,  # pyright: ignore[reportUnknownVariableType]
)

from benchmarks.noloco_reference.distributed import TorchDistributedPairExchange
from benchmarks.noloco_reference.optimizer import (
    PINNED_UPSTREAM_COMMIT,
    PINNED_UPSTREAM_SOURCE,
    ReferenceNoLoCoConfig,
    apply_outer_step,
    outer_gradient,
)
from benchmarks.noloco_reference.runner import scheduled_peer_rank
from benchmarks.noloco_reference.trajectory import (
    TrajectoryContext,
    TrajectoryWriter,
)

FIXTURE = Path(__file__).parents[1] / "golden" / "noloco_upstream_outer_step_v1.json"
REFERENCE_ROOT = Path(__file__).parents[2] / "benchmarks" / "noloco_reference"
load_safetensors = cast(Callable[[str], dict[str, torch.Tensor]], _load_file)


def _fixture() -> dict[str, object]:
    return cast(dict[str, object], json.loads(FIXTURE.read_text(encoding="utf-8")))


def _node(fixture: dict[str, object], rank: int) -> dict[str, object]:
    nodes = cast(list[object], fixture["nodes"])
    return cast(dict[str, object], nodes[rank])


def _tensor(node: dict[str, object], name: str) -> torch.Tensor:
    return torch.tensor(cast(list[float], node[name]), dtype=torch.float32)


def test_reference_outer_step_matches_pinned_upstream_fixture() -> None:
    fixture = _fixture()
    authority = cast(dict[str, object], fixture["authority"])
    convention = cast(dict[str, object], fixture["convention"])
    config = ReferenceNoLoCoConfig(
        alpha=cast(float, convention["alpha"]),
        beta=cast(float, convention["beta"]),
        gamma=cast(float, convention["effective_gamma"]),
    )

    assert authority["commit"] == PINNED_UPSTREAM_COMMIT
    assert authority["source"] == PINNED_UPSTREAM_SOURCE
    for rank in (0, 1):
        local = _node(fixture, rank)
        peer = _node(fixture, 1 - rank)
        slow = {"weight": _tensor(local, "slow_weights")}
        fast = {"weight": _tensor(local, "fast_weights")}
        momentum = {"weight": _tensor(local, "outer_momentum_before")}
        peer_slow = {"weight": _tensor(peer, "slow_weights")}
        peer_gradient = {
            "weight": outer_gradient(
                {"weight": _tensor(peer, "slow_weights")},
                {"weight": _tensor(peer, "fast_weights")},
            )["weight"]
        }

        local_gradient = outer_gradient(slow, fast)
        result = apply_outer_step(
            slow_weights=slow,
            outer_momentum=momentum,
            local_outer_gradient=local_gradient,
            peer_outer_gradient=peer_gradient,
            peer_slow_weights=peer_slow,
            config=config,
        )

        torch.testing.assert_close(
            local_gradient["weight"],
            _tensor(local, "outer_gradient"),
            rtol=0.0,
            atol=1e-6,
        )
        torch.testing.assert_close(
            result.outer_momentum["weight"],
            _tensor(local, "outer_momentum_after"),
            rtol=0.0,
            atol=1e-6,
        )
        torch.testing.assert_close(
            result.slow_weights["weight"],
            _tensor(local, "slow_weights_after"),
            rtol=0.0,
            atol=1e-6,
        )


def test_reference_outer_step_is_non_mutating_and_validates_schema() -> None:
    slow = {"weight": torch.tensor([1.0], dtype=torch.float32)}
    fast = {"weight": torch.tensor([0.5], dtype=torch.float32)}
    momentum = {"weight": torch.tensor([0.25], dtype=torch.float32)}
    originals = tuple(value.clone() for value in (*slow.values(), *momentum.values()))

    result = apply_outer_step(
        slow_weights=slow,
        outer_momentum=momentum,
        local_outer_gradient=outer_gradient(slow, fast),
        peer_outer_gradient={"weight": torch.tensor([0.25])},
        peer_slow_weights={"weight": torch.tensor([2.0])},
        config=ReferenceNoLoCoConfig(alpha=0.5, beta=0.7, gamma=0.7),
    )

    assert all(
        torch.equal(value, original)
        for value, original in zip(
            (*slow.values(), *momentum.values()),
            originals,
            strict=True,
        )
    )
    assert result.slow_weights["weight"] is not slow["weight"]
    with pytest.raises(ValueError, match="names"):
        apply_outer_step(
            slow_weights=slow,
            outer_momentum=momentum,
            local_outer_gradient={"other": torch.tensor([0.5])},
            peer_outer_gradient={"weight": torch.tensor([0.25])},
            peer_slow_weights={"weight": torch.tensor([2.0])},
            config=ReferenceNoLoCoConfig(alpha=0.5, beta=0.7, gamma=0.7),
        )
    with pytest.raises(ValueError, match="finite"):
        outer_gradient(
            {"weight": torch.tensor([float("nan")])},
            fast,
        )


def _gloo_worker(rank: int, init_path: str, output_root: str) -> None:
    fixture = _fixture()
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
    )
    try:
        node = _node(fixture, rank)
        slow = {"weight": _tensor(node, "slow_weights")}
        fast = {"weight": _tensor(node, "fast_weights")}
        gradient = outer_gradient(slow, fast)
        exchange = TorchDistributedPairExchange()
        peer = exchange.exchange(
            peer_rank=1 - rank,
            artifacts={
                "outer_gradient": gradient,
                "slow_weights": slow,
            },
        )
        result = apply_outer_step(
            slow_weights=slow,
            outer_momentum={
                "weight": _tensor(node, "outer_momentum_before")
            },
            local_outer_gradient=gradient,
            peer_outer_gradient=peer["outer_gradient"],
            peer_slow_weights=peer["slow_weights"],
            config=ReferenceNoLoCoConfig(alpha=0.5, beta=0.7, gamma=0.7),
        )
        Path(output_root, f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "momentum": result.outer_momentum[
                        "weight"
                    ].tolist(),  # pyright: ignore[reportUnknownMemberType]
                    "slow": result.slow_weights[
                        "weight"
                    ].tolist(),  # pyright: ignore[reportUnknownMemberType]
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        dist.barrier()  # pyright: ignore[reportUnknownMemberType]
    finally:
        dist.destroy_process_group()


def test_gloo_pair_exchange_matches_fixture(tmp_path: Path) -> None:
    rendezvous = tmp_path / "gloo-rendezvous"
    mp.spawn(  # pyright: ignore[reportUnknownMemberType,reportPrivateImportUsage]
        _gloo_worker,
        args=(str(rendezvous), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    fixture = _fixture()

    for rank in (0, 1):
        node = _node(fixture, rank)
        result = cast(
            dict[str, list[float]],
            json.loads(
                (tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8")
            ),
        )
        torch.testing.assert_close(
            torch.tensor(result["momentum"]),
            _tensor(node, "outer_momentum_after"),
            rtol=0.0,
            atol=1e-6,
        )
        torch.testing.assert_close(
            torch.tensor(result["slow"]),
            _tensor(node, "slow_weights_after"),
            rtol=0.0,
            atol=1e-6,
        )


def test_reference_package_does_not_import_dromeus_runtime_math() -> None:
    forbidden = (
        "dromeus.algorithms",
        "dromeus.runtime",
        "dromeus.transport",
        "dromeus.membership",
        "dromeus.persistence",
        "dromeus.telemetry",
    )
    for path in REFERENCE_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        imports.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        disallowed_gossip = {
            imported
            for imported in imports
            if (
                imported == "dromeus.gossip"
                or imported.startswith("dromeus.gossip.")
            )
            and imported != "dromeus.gossip.peer_scheduler"
        }
        assert not disallowed_gossip, path
        assert not any(
            imported == prefix or imported.startswith(f"{prefix}.")
            for imported in imports
            for prefix in forbidden
        ), path


def test_runner_uses_frozen_public_key_scheduler() -> None:
    members = ("key-0", "key-1", "key-2", "key-3")

    peers = tuple(
        scheduled_peer_rank(
            members=members,
            scheduler_seed=17,
            round_count=3,
            round_id=1,
            rank=rank,
        )
        for rank in range(4)
    )

    assert all(peers[peer] == rank for rank, peer in enumerate(peers))
    assert all(peer != rank for rank, peer in enumerate(peers))


def test_trajectory_writer_records_fp32_snapshot_and_index(tmp_path: Path) -> None:
    context = TrajectoryContext(
        experiment_sha256="1" * 64,
        run_config_sha256="2" * 64,
        initial_checkpoint_sha256="3" * 64,
        tensor_schema_hash="4" * 64,
        world_size=4,
        benchmark_seed=17,
        rank=2,
        node_id="key-2",
        interval=1,
        total_outer_steps=2,
    )
    writer = TrajectoryWriter(root=tmp_path, context=context)

    record = writer.write(
        completed_outer_steps=1,
        peer_rank=3,
        slow_weights={"weight": torch.tensor([1.0, -2.0])},
    )

    assert record.completed_outer_steps == 1
    assert record.peer_rank == 3
    assert record.snapshot_path.is_file()
    assert torch.equal(
        load_safetensors(str(record.snapshot_path))["weight"],
        torch.tensor([1.0, -2.0]),
    )
    index = [
        json.loads(line)
        for line in (tmp_path / "rank-2" / "trajectory.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert index[0]["run_config_sha256"] == "2" * 64
    assert index[0]["completed_outer_steps"] == 1

    assert writer.should_write(completed_outer_steps=0)
    assert writer.should_write(completed_outer_steps=2)
    with pytest.raises(ValueError, match="float32"):
        writer.write(
            completed_outer_steps=2,
            peer_rank=3,
            slow_weights={"weight": torch.tensor([1.0], dtype=torch.float64)},
        )
