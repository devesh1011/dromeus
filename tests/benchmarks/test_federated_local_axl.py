from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

import pytest

from benchmarks.federated.local_axl import local_axl_cluster


@pytest.mark.parametrize("node_count", (0, 1, 3, 5, True))
def test_local_axl_rejects_invalid_node_count(tmp_path: Path, node_count: int) -> None:
    async def run() -> None:
        with pytest.raises(ValueError, match="node_count"):
            async with local_axl_cluster(
                binary=Path(sys.executable), log_root=tmp_path, node_count=node_count
            ):
                pytest.fail("invalid cluster reached readiness")

    asyncio.run(run())
    assert not list(tmp_path.iterdir())


def test_local_axl_preserves_existing_logs(tmp_path: Path) -> None:
    retained = tmp_path / "axl-node-0.log"
    retained.write_bytes(b"original execution record")

    async def run() -> None:
        with pytest.raises(ValueError, match="already exists"):
            async with local_axl_cluster(
                binary=Path(sys.executable), log_root=tmp_path
            ):
                pytest.fail("existing evidence was overwritten")

    asyncio.run(run())
    assert retained.read_bytes() == b"original execution record"


@pytest.mark.skipif(
    not os.environ.get("DROMEUS_FEDERATED_AXL_BINARY"),
    reason="set DROMEUS_FEDERATED_AXL_BINARY for real local AXL lifecycle checks",
)
@pytest.mark.parametrize("exit_kind", ("exception", "cancellation"))
def test_real_local_axl_full_mesh_and_cleanup(
    tmp_path: Path, exit_kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = Path(os.environ["DROMEUS_FEDERATED_AXL_BINARY"])
    bridges: list[str] = []
    private_roots: list[Path] = []
    temporary_directory = tempfile.TemporaryDirectory

    def tracked_directory(*, prefix: str) -> tempfile.TemporaryDirectory[str]:
        directory = temporary_directory(prefix=prefix)
        private_roots.append(Path(directory.name))
        return directory

    monkeypatch.setattr(
        "benchmarks.federated.local_axl.tempfile.TemporaryDirectory", tracked_directory
    )

    async def run_cluster(ready: asyncio.Event) -> None:
        async with local_axl_cluster(binary=binary, log_root=tmp_path) as nodes:
            keys = tuple(node.public_key for node in nodes)
            assert keys == tuple(sorted(set(keys)))
            assert len(keys) == 4
            for rank, node in enumerate(nodes):
                bridges.append(node.bridge_url)
                with urlopen(f"{node.bridge_url}/topology", timeout=1.0) as response:
                    topology = json.load(response)
                assert topology["our_public_key"] == node.public_key
                assert {
                    item["public_key"] for item in topology["peers"] if item["up"]
                } == set(keys) - {node.public_key}
                retained = json.loads(
                    (tmp_path / f"topology-ready-{rank}.json").read_bytes()
                )
                assert retained["our_public_key"] == node.public_key
            ready.set()
            if exit_kind == "exception":
                raise RuntimeError("experiment failed deliberately")
            await asyncio.Event().wait()

    async def run() -> None:
        ready = asyncio.Event()
        task = asyncio.create_task(run_cluster(ready))
        if exit_kind == "exception":
            with pytest.raises(RuntimeError, match="deliberately"):
                await asyncio.wait_for(task, timeout=40)
        else:
            await asyncio.wait_for(ready.wait(), timeout=40)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
    assert len(bridges) == 4
    assert private_roots and all(not root.exists() for root in private_roots)
    for bridge in bridges:
        with pytest.raises(OSError):
            urlopen(f"{bridge}/topology", timeout=0.2)
    assert len(list(tmp_path.glob("axl-node-*.log"))) == 4
    assert not list(tmp_path.rglob("*.pem"))
    for log in tmp_path.glob("*.log"):
        assert b"BEGIN PRIVATE KEY" not in log.read_bytes()
