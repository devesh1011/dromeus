from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import cast
from urllib.request import urlopen

import msgpack  # pyright: ignore[reportMissingTypeStubs]
import pytest
from support.sample_manifest import manifest_data, write_checkpoint

from dromeus.manifests.models import DraftRunSpec, SealedManifest
from dromeus.membership.protocol import create_invitation
from dromeus.runtime import NodeRuntime, NodeState
from dromeus.transport.axl import AXLBridgeConfig, AXLTransport
from dromeus.transport.base import AsyncTransport, ReceivedBytes
from dromeus.transport.envelope import MessageType
from dromeus.transport.transfer import ArtifactStore

pytestmark = pytest.mark.skipif(
    os.environ.get("DROMEUS_RUN_AXL_TESTS") != "1",
    reason="set DROMEUS_RUN_AXL_TESTS=1 to run real AXL integration tests",
)

LOG_ROOT = Path(__file__).resolve().parents[2] / "logs"


class FaultInjectingTransport:
    def __init__(
        self,
        transport: AsyncTransport,
        *,
        drop_first_ack: bool = False,
        duplicate_first_chunk: bool = False,
    ) -> None:
        self._transport = transport
        self._drop_first_ack = drop_first_ack
        self._duplicate_first_chunk = duplicate_first_chunk

    async def local_public_key(self) -> str:
        return await self._transport.local_public_key()

    async def send(self, destination: str, payload: bytes) -> None:
        unpacked = cast(
            dict[str, object],
            msgpack.unpackb(payload, raw=False),  # pyright: ignore[reportUnknownMemberType]
        )
        message_type = unpacked.get("message_type")
        if self._drop_first_ack and message_type == MessageType.CHUNK_ACK:
            self._drop_first_ack = False
            return
        await self._transport.send(destination, payload)
        if self._duplicate_first_chunk and message_type == MessageType.CHUNK:
            self._duplicate_first_chunk = False
            await self._transport.send(destination, payload)

    async def recv(self, timeout_seconds: float) -> ReceivedBytes | None:
        return await self._transport.recv(timeout_seconds)

    @property
    def faults_applied(self) -> bool:
        return not self._drop_first_ack and not self._duplicate_first_chunk


def test_four_local_axl_nodes_form_and_transfer_8mib() -> None:
    asyncio.run(_test_four_local_axl_nodes_form_and_transfer_8mib())


async def _test_four_local_axl_nodes_form_and_transfer_8mib() -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    draft_data = manifest.model_dump(mode="python")
    for field in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del draft_data[field]
    draft_data["transport"]["max_payload_bytes"] = 16 * 1024 * 1024
    draft_data["transport"]["max_retries"] = 2
    draft_data["transport"]["retry_timeout_seconds"] = 1.0
    draft = DraftRunSpec.model_validate(draft_data)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        binary = build_axl_binary(root)
        processes = start_nodes(binary, root, LOG_ROOT)
        try:
            await wait_for_topology(9302)
            await wait_for_topology(9303)
            await wait_for_topology(9304)
            await wait_for_topology(9305)
            await wait_for_peer_count(9302, 3)
            await wait_for_peer_count(9303, 1)
            await wait_for_peer_count(9304, 1)
            await wait_for_peer_count(9305, 1)
            axl_transports = [
                AXLTransport(AXLBridgeConfig(base_url=f"http://127.0.0.1:{port}"))
                for port in (9302, 9303, 9304, 9305)
            ]
            duplicate_transport = FaultInjectingTransport(
                axl_transports[0], duplicate_first_chunk=True
            )
            ack_loss_transport = FaultInjectingTransport(
                axl_transports[1], drop_first_ack=True
            )
            transports: list[AsyncTransport] = [
                duplicate_transport,
                ack_loss_transport,
                axl_transports[2],
                axl_transports[3],
            ]
            nodes = [
                NodeRuntime(
                    transport=transport,
                    draft=draft,
                    environment=manifest.environment,
                    dataset=manifest.dataset,
                    artifact_store=ArtifactStore(root / f"artifacts-{index}"),
                )
                for index, transport in enumerate(transports)
            ]
            checkpoint = root / "checkpoint.safetensors"
            element_count = (8 * 1024 * 1024 - 128) // 4
            write_checkpoint(checkpoint, shape=(element_count,))
            tensor_schema = manifest.tensor_schema.model_copy(
                update={
                    "tensors": (
                        manifest.tensor_schema.tensors[0].model_copy(
                            update={"shape": (element_count,)}
                        ),
                    )
                }
            )
            initiator_key = await transports[0].local_public_key()
            invitation = create_invitation(
                draft=draft,
                initiator_public_key=initiator_key,
                bootstrap_uri="tls://127.0.0.1:9300",
            )
            tasks = [
                asyncio.create_task(
                    nodes[0].initiate(
                        bootstrap_uri="tls://127.0.0.1:9300",
                        checkpoint_path=checkpoint,
                        tensor_schema=tensor_schema,
                    )
                )
            ]
            tasks.extend(
                asyncio.create_task(node.join(invitation=invitation))
                for node in nodes[1:]
            )
            results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=60.0)
            assert duplicate_transport.faults_applied
            assert ack_loss_transport.faults_applied
            assert len({result.manifest_hash for result in results}) == 1
            assert all(node.state is NodeState.READY for node in nodes)
            for result in results[1:]:
                assert result.checkpoint_path.stat().st_size >= 8 * 1024 * 1024 - 256
                assert result.checkpoint_path.stat().st_size <= 8 * 1024 * 1024
            for node in nodes:
                await node.stop()
        finally:
            for process in processes:
                process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()


def build_axl_binary(root: Path) -> Path:
    repo = root / "axl"
    subprocess.run(
        ["git", "clone", "https://github.com/gensyn-ai/axl.git", str(repo)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        ["git", "checkout", "628e28ace077f26dfe8d0259009b357216a9d8d4"],
        check=True,
        cwd=repo,
        stdout=subprocess.DEVNULL,
    )
    binary = root / "node"
    subprocess.run(
        ["go", "build", "-o", str(binary), "./cmd/node"],
        check=True,
        cwd=repo,
        stdout=subprocess.DEVNULL,
    )
    return binary


def start_nodes(
    binary: Path, root: Path, log_root: Path
) -> list[subprocess.Popen[str]]:
    processes: list[subprocess.Popen[str]] = []
    log_root.mkdir(exist_ok=True)
    openssl = find_openssl()
    configs: list[dict[str, object]] = [
        {
            "PrivateKeyPath": str(root / "private-0.pem"),
            "Peers": [],
            "Listen": ["tls://127.0.0.1:9300"],
            "api_port": 9302,
            "bridge_addr": "127.0.0.1",
            "max_message_size": 16777216,
        },
        {
            "PrivateKeyPath": str(root / "private-1.pem"),
            "Peers": ["tls://127.0.0.1:9300"],
            "Listen": [],
            "api_port": 9303,
            "bridge_addr": "127.0.0.1",
            "max_message_size": 16777216,
        },
        {
            "PrivateKeyPath": str(root / "private-2.pem"),
            "Peers": ["tls://127.0.0.1:9300"],
            "Listen": [],
            "api_port": 9304,
            "bridge_addr": "127.0.0.1",
            "max_message_size": 16777216,
        },
        {
            "PrivateKeyPath": str(root / "private-3.pem"),
            "Peers": ["tls://127.0.0.1:9300"],
            "Listen": [],
            "api_port": 9305,
            "bridge_addr": "127.0.0.1",
            "max_message_size": 16777216,
        },
    ]
    for index, config in enumerate(configs):
        subprocess.run(
            [
                openssl,
                "genpkey",
                "-algorithm",
                "ed25519",
                "-out",
                str(config["PrivateKeyPath"]),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        config_path = root / f"node-{index}.json"
        config_path.write_text(json.dumps(config))
        log_path = log_root / f"node-{index}.log"
        processes.append(
            subprocess.Popen(
                [str(binary), "-config", str(config_path)],
                stdout=log_path.open("w"),
                stderr=subprocess.STDOUT,
                text=True,
            )
        )
    return processes


def find_openssl() -> str:
    for candidate in (
        "/opt/homebrew/opt/openssl/bin/openssl",
        "/usr/local/opt/openssl/bin/openssl",
        shutil.which("openssl"),
    ):
        if candidate:
            return candidate
    raise RuntimeError("openssl binary not found")


async def wait_for_topology(port: int) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{port}/topology", timeout=1.0) as response:
                if response.status == 200:
                    return
        except Exception:
            await asyncio.sleep(0.2)
            continue
        await asyncio.sleep(0.2)
    raise TimeoutError(f"AXL topology did not become ready on port {port}")


async def wait_for_peer_count(port: int, expected_count: int) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{port}/topology", timeout=1.0) as response:
                payload = cast(dict[str, object], json.load(response))
        except Exception:
            await asyncio.sleep(0.2)
            continue
        peers = payload.get("peers", [])
        if isinstance(peers, list):
            peer_list = cast(list[object], peers)
            if len(peer_list) >= expected_count:
                return
        await asyncio.sleep(0.2)
    raise TimeoutError(
        f"AXL topology on port {port} never reached {expected_count} peers"
    )
