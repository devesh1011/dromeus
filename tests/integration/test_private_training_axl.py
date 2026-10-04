"""Run an ordinary user Python factory in four separate processes over real AXL."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Literal

import pytest
import torch
from support.application_fixture import application_draft
from support.paths import REPO_ROOT

from benchmarks.federated.local_axl import local_axl_cluster
from benchmarks.m3.private_smoke import MODEL_DEFINITION, TASK_DEFINITION
from dromeus.application import definition_hash
from dromeus.manifests.models import (
    DraftRunSpec,
    PrivacyPolicy,
    PrivateOptimizerSpec,
    PrivateTrainingSpec,
)
from dromeus.node import NodeConfig, NodeRole
from dromeus.persistence.archive import RunArchive
from dromeus.persistence.privacy_ledger import PrivacyLedger
from dromeus.telemetry.evidence import EvidenceLog, RoundMetricsEvidence
from dromeus.training.private_trainer import private_training_definition

pytestmark = pytest.mark.skipif(
    not os.environ.get("DROMEUS_PRIVATE_AXL_BINARY"),
    reason="set DROMEUS_PRIVATE_AXL_BINARY to run the real custom training test",
)


@pytest.mark.parametrize("optimizer_name", ["sgd", "adam", "adamw"])
def test_four_private_python_processes_over_axl(
    tmp_path: Path, optimizer_name: Literal["sgd", "adam", "adamw"]
) -> None:
    binary = Path(os.environ["DROMEUS_PRIVATE_AXL_BINARY"])
    root = REPO_ROOT
    base = application_draft(compressed=True).model_dump(mode="python")
    spec = PrivateTrainingSpec(
        optimizer=PrivateOptimizerSpec(
            name=optimizer_name,
            torch_version=torch.__version__,
            parameter_names=("weight", "bias"),
            learning_rate=0.01,
            momentum=0.9 if optimizer_name == "sgd" else 0,
            nesterov=optimizer_name == "sgd",
        ),
        loss="mse",
        batch_size=4,
        schedule="linear_decay",
    )
    base.update(
        manifest_version=5,
        private_training=spec,
        privacy=PrivacyPolicy(
            max_grad_norm=1,
            noise_multiplier=1,
            delta=1e-5,
            max_logical_steps=9,
            randomness_profile="reproducible_public_benchmark",
        ),
        model_definition_hash=definition_hash(MODEL_DEFINITION),
    )
    base["environment"]["model_definition_hash"] = base["model_definition_hash"]
    base["dataset"]["definition_hash"] = definition_hash(TASK_DEFINITION)
    base["application_training"]["definition_hash"] = definition_hash(
        private_training_definition(spec)
    )
    draft = DraftRunSpec.model_validate(base)
    draft = draft.model_copy(
        update={
            "environment": draft.environment.model_copy(
                update={"pytorch_version": torch.__version__}
            )
        }
    )
    draft_path = tmp_path / "draft.json"
    # Exercise the actual CLI default in each independent Python process.
    draft_path.write_text(draft.model_dump_json(exclude={"algorithm_id"}))

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
                        DROMEUS_W1_LOCAL_ROOT=str(tmp_path / f"node-{rank}"),
                        DROMEUS_W1_RANK=str(rank),
                        OMP_NUM_THREADS="1",
                        MKL_NUM_THREADS="1",
                    )
                    with (tmp_path / f"python-{rank}.log").open("wb") as log:
                        processes.append(
                            await asyncio.create_subprocess_exec(
                                sys.executable,
                                "-m",
                                "dromeus.node",
                                "--config",
                                str(config_path),
                                "--factory",
                                "benchmarks.m3.private_smoke:prepare_training",
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
        assert int(state["noloco.v1.completed_outer_steps"][0]) == 3
        assert any("application." in key for key in state)
        assert int(state["noloco.v1.trainer.meta.steps"][0]) == 9
        checkpoint_files = list(
            (run_root / "run-store/checkpoints").glob("*.safetensors")
        )
        assert checkpoint_files and all(
            p.stat().st_mode & 0o777 == 0o600 for p in checkpoint_files
        )
        assert (run_root / "run-store").stat().st_mode & 0o777 == 0o700
        assert state["noloco.v1.trainer.meta.loss"].size == 0
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
        assert all(
            record.local_loss is None and record.evaluation_loss is None
            for record in metrics
        )
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
    evidence_root = os.environ.get("DROMEUS_W1_EVIDENCE_ROOT")
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
