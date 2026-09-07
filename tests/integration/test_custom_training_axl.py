"""Run an ordinary user Python factory in four separate processes over real AXL."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest
import torch
from support.application_fixture import application_draft, local_file
from support.paths import REPO_ROOT

from benchmarks.federated.local_axl import local_axl_cluster
from dromeus.node import NodeConfig, NodeRole
from dromeus.persistence.archive import RunArchive
from dromeus.telemetry.evidence import EvidenceLog, TaskRoundMetricsEvidence

pytestmark = pytest.mark.skipif(
    not os.environ.get("DROMEUS_CUSTOM_AXL_BINARY"),
    reason="set DROMEUS_CUSTOM_AXL_BINARY to run the real custom training test",
)


def test_four_independent_custom_python_processes_over_axl(tmp_path: Path) -> None:
    binary = Path(os.environ["DROMEUS_CUSTOM_AXL_BINARY"])
    root = REPO_ROOT
    draft = application_draft(compressed=True)
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
                        DROMEUS_LOCAL_DATA=str(
                            local_file(tmp_path / f"data-{rank}.npz", rank)
                        ),
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
                                "examples.custom_training:prepare_training",
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
        log = EvidenceLog.open(
            run_root / "logs/dromeus.jsonl",
            run_id=draft.run_id,
            manifest_hash=archive.manifest_hash,
        )
        metrics = [
            record
            for record in log.records
            if isinstance(record, TaskRoundMetricsEvidence)
        ]
        assert len(metrics) == 3
        assert all(
            set(record.evaluation_metrics) == {"mae", "rmse"} for record in metrics
        )
    assert len(hashes) == 1
