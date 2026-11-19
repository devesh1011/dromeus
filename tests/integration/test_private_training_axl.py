"""Run four private Python task processes over real AXL (opt-in, public synthetic data).

The subprocess task setup lives in this module, so each node imports it as
``test_private_training_axl:setup`` with this directory on PYTHONPATH.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

import pytest
import torch
from support.application_fixture import FIXTURE_TRANSPORT
from support.local_axl import AXL_SKIP_REASON, axl_binary, local_axl_cluster
from support.paths import REPO_ROOT
from torch import nn
from torch.utils.data import TensorDataset

from dromeus.application import (
    LocalData,
    NoLoCo,
    Privacy,
    draft_for_task,
)
from dromeus.node import NodeConfig, NodeRole
from dromeus.persistence.archive import RunArchive
from dromeus.persistence.privacy_ledger import PrivacyLedger
from dromeus.telemetry.evidence import EvidenceLog, RoundMetricsEvidence
from dromeus.training.task import Optimizer, TensorBatchAdapter, TrainingTask

MODEL_DEFINITION = "w1-smoke-linear-2-1-fp32-public-init-v1"
TASK_DEFINITION = "w1-smoke-fixed-public-synthetic-mse-v1"

pytestmark = pytest.mark.skipif(axl_binary() is None, reason=AXL_SKIP_REASON)


def public_model() -> nn.Module:
    model = nn.Linear(2, 1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.1)
    return model


def squared_error(model: nn.Module, batch: Any) -> torch.Tensor:
    inputs, targets = batch
    return (model(inputs) - targets).square().flatten(1).mean(1)


def synthetic_pairs(rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Public synthetic records; every node's offset is fixed by its rank."""
    inputs = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 16 + rank * 0.01
    return inputs, inputs[:, :1] * 0.3


def synthetic_task(rank: int, optimizer: Optimizer) -> TrainingTask:
    inputs, targets = synthetic_pairs(rank)
    return TrainingTask(
        model_factory=public_model,
        batch_adapter=TensorBatchAdapter(
            example=(inputs[:2], targets[:2]),
            definition="w1-public-two-record-pairs-v1",
        ),
        loss_per_record=squared_error,
        model_definition=MODEL_DEFINITION,
        task_definition=TASK_DEFINITION,
        loss_definition="w1-smoke-mean-squared-error-per-record-v1",
        implementation_hash="5" * 64,
        optimizer=optimizer,
        batch_size=4,
        schedule="linear_decay",
        record_unit="example",
    )


def optimizer_for(name: str) -> Optimizer:
    if name == "sgd":
        return Optimizer.sgd(0.01, momentum=0.9, nesterov=True)
    if name == "adam":
        return Optimizer.adam(0.01)
    if name == "adamw":
        return Optimizer.adamw(0.01, weight_decay=0.0)
    raise ValueError(f"unknown optimizer: {name}")


def setup() -> tuple[TrainingTask, LocalData]:
    """Entry point for `dromeus run --task` in each private process."""
    root = Path(os.environ["DROMEUS_PRIVATE_NODE_ROOT"])
    rank = int(os.environ["DROMEUS_PRIVATE_NODE_RANK"])
    optimizer = optimizer_for(os.environ["DROMEUS_PRIVATE_OPTIMIZER"])
    # All records are public synthetic data. This bootstrap must never be copied
    # into a deployment that recreates a missing existing private-data ledger.
    ledger = PrivacyLedger.create(
        root / "privacy-ledger.json", lineage=f"w1-public-synthetic-{rank}"
    )
    inputs, targets = synthetic_pairs(rank)
    return synthetic_task(rank, optimizer), LocalData(
        dataset=TensorDataset(inputs, targets),
        data_identity=f"w1-public-tensors-rank-{rank}-v1",
        ledger=ledger,
        physical_batch_size=2,
        research_seed=101 + rank,
    )


@pytest.mark.parametrize("optimizer_name", ["sgd", "adam", "adamw"])
def test_four_private_python_processes_over_axl(
    tmp_path: Path, optimizer_name: Literal["sgd", "adam", "adamw"]
) -> None:
    binary = axl_binary()
    assert binary is not None
    root = REPO_ROOT
    draft = draft_for_task(
        synthetic_task(0, optimizer_for(optimizer_name)),
        run_id="run-001",
        participants=4,
        rounds=3,
        local_steps=3,
        axl_version="1.0.0",
        noloco=NoLoCo(alpha=0.4, beta=0.3, gamma=0.2, top_k_fraction=0.5),
        privacy=Privacy(
            max_grad_norm=1,
            noise_multiplier=1,
            delta=1e-5,
            randomness="reproducible_public_benchmark",
        ),
        transport=FIXTURE_TRANSPORT,
        seed=8,
    )
    draft_path = tmp_path / "draft.json"
    # Exercise the actual CLI default in each independent Python process.
    draft_path.write_text(
        draft.model_dump_json(exclude={"algorithm_id"}, exclude_none=True)
    )
    # Each node imports the factory above from this file by module path.
    integration_root = Path(__file__).resolve().parent
    python_path = os.pathsep.join(
        part
        for part in (
            str(integration_root),
            str(root / "tests"),
            os.environ.get("PYTHONPATH", ""),
        )
        if part
    )

    async def run() -> None:
        async with local_axl_cluster(binary=binary, log_root=tmp_path / "axl") as nodes:
            processes: list[asyncio.subprocess.Process] = []
            try:
                for rank, node in enumerate(nodes):
                    config = NodeConfig(
                        role=NodeRole.INITIATOR if rank == 0 else NodeRole.PARTICIPANT,
                        draft_path=draft_path,
                        axl_bridge_url=node.bridge_url,
                        run_root=tmp_path / f"node-{rank}",
                        invitation_path=tmp_path / "invitation.json",
                        bootstrap_uri="axl://custom-process-smoke",
                    )
                    config_path = tmp_path / f"node-{rank}.json"
                    config_path.write_text(config.model_dump_json())
                    env = dict(
                        os.environ,
                        DROMEUS_PRIVATE_NODE_ROOT=str(tmp_path / f"node-{rank}"),
                        DROMEUS_PRIVATE_NODE_RANK=str(rank),
                        DROMEUS_PRIVATE_OPTIMIZER=optimizer_name,
                        PYTHONPATH=python_path,
                        OMP_NUM_THREADS="1",
                        MKL_NUM_THREADS="1",
                    )
                    with (tmp_path / f"python-{rank}.log").open("wb") as log:
                        processes.append(
                            await asyncio.create_subprocess_exec(
                                sys.executable,
                                "-m",
                                "dromeus",
                                "run",
                                "--config",
                                str(config_path),
                                "--task",
                                "test_private_training_axl:setup",
                                cwd=root,
                                env=env,
                                stdout=log,
                                stderr=asyncio.subprocess.STDOUT,
                            )
                        )
                codes = await asyncio.wait_for(
                    asyncio.gather(*(process.wait() for process in processes)),
                    timeout=60,
                )
                assert codes == [0] * 4, {
                    rank: (tmp_path / f"python-{rank}.log").read_text()
                    for rank, code in enumerate(codes)
                    if code
                }
            finally:
                for process in processes:
                    if process.returncode is None:
                        process.kill()
                await asyncio.gather(*(process.wait() for process in processes))

    asyncio.run(run())
    hashes = set[str]()
    node_summaries: list[dict[str, object]] = []
    for rank in range(4):
        run_root = tmp_path / f"node-{rank}"
        archive = RunArchive.open(run_root / "run-store")
        assert archive.manifest.algorithm_id == "noloco"
        hashes.add(archive.manifest_hash)
        assert archive.state.terminal is not None
        assert archive.state.terminal.result == "complete"
        assert archive.state.committed_round == 2
        assert archive.algorithm_state is not None
        state = archive.algorithm_state.load_tensors()
        assert int(state["noloco.completed_outer_steps"][0]) == 3
        assert any("application." in key for key in state)
        assert int(state["noloco.trainer.meta.steps"][0]) == 9
        checkpoint_files = list(
            (run_root / "run-store/checkpoints").glob("*.safetensors")
        )
        assert checkpoint_files and all(
            p.stat().st_mode & 0o777 == 0o600 for p in checkpoint_files
        )
        assert (run_root / "run-store").stat().st_mode & 0o777 == 0o700
        assert state["noloco.trainer.meta.loss"].size == 0
        reservations = PrivacyLedger(
            run_root / "privacy-ledger.json", lineage=f"w1-public-synthetic-{rank}"
        ).reservations()
        assert len(reservations) == 1 and reservations[0].logical_steps == 9
        log = EvidenceLog.open(
            run_root / "logs/dromeus.jsonl",
            run_id=draft.run_id,
            manifest_hash=archive.manifest_hash,
        )
        metrics = [
            record for record in log.records if isinstance(record, RoundMetricsEvidence)
        ]
        assert len(metrics) == 3
        assert all(record.local_loss is None for record in metrics)
        node_summaries.append(
            {
                "node": rank,
                "manifest_hash": archive.manifest_hash,
                "logical_steps": 9,
                "reserved_steps": reservations[0].logical_steps,
                "sample_rate": reservations[0].sample_rate,
                "completed_outer_rounds": 3,
                "raw_losses_suppressed": all(r.local_loss is None for r in metrics),
                "archive_state_sha256": hashlib.sha256(
                    b"".join(k.encode() + v.tobytes() for k, v in sorted(state.items()))
                ).hexdigest(),
            }
        )
    assert len(hashes) == 1
    evidence_root = os.environ.get("DROMEUS_PRIVATE_EVIDENCE_ROOT")
    if evidence_root:
        destination = Path(evidence_root)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"axl-{optimizer_name}.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "profile": "public_synthetic_research",
                    "optimizer": optimizer_name,
                    "torch_version": torch.__version__,
                    "opacus_version": "1.6.0",
                    "axl_binary_sha256": hashlib.sha256(
                        binary.read_bytes()
                    ).hexdigest(),
                    "independent_python_processes": 4,
                    "nodes": node_summaries,
                    "status": "passed",
                },
                indent=2,
            )
            + "\n"
        )
