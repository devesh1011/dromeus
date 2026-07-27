from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from importlib.metadata import version as package_version
from pathlib import Path
from typing import cast
from urllib.request import urlopen

import msgpack  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pytest
from support.sample_manifest import manifest_data, write_checkpoint

from dromeus.algorithms.dpsgd import DPSGDAdapter
from dromeus.manifests.canonical import canonical_hash
from dromeus.manifests.models import DraftRunSpec, SealedManifest
from dromeus.membership.protocol import create_invitation, seal_manifest
from dromeus.persistence.run_store import RunStore
from dromeus.runtime import NodeRuntime, NodeState, TrainingConfig
from dromeus.telemetry.events import JsonlEventSink
from dromeus.telemetry.metrics import JsonlMetricsPublisher
from dromeus.training.pytorch import (
    MODEL_DEFINITION_HASH,
    CIFAR10Data,
    CIFAR10Trainer,
    create_initial_checkpoint,
)
from dromeus.transport.axl import AXLBridgeConfig, AXLTransport
from dromeus.transport.base import AsyncTransport, ReceivedBytes
from dromeus.transport.envelope import MessageType
from dromeus.transport.transfer import ArtifactStore

pytestmark = pytest.mark.skipif(
    os.environ.get("DROMEUS_RUN_AXL_TESTS") != "1",
    reason="set DROMEUS_RUN_AXL_TESTS=1 to run real AXL integration tests",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = Path(os.environ.get("DROMEUS_AXL_LOG_ROOT", REPO_ROOT / "logs"))
AXL_COMMIT = "628e28ace077f26dfe8d0259009b357216a9d8d4"


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


def test_four_local_axl_nodes_train_cifar10() -> None:
    asyncio.run(_test_four_local_axl_nodes_train_cifar10())


async def _test_four_local_axl_nodes_train_cifar10() -> None:
    pilot_data_source_override = os.environ.get("DROMEUS_PILOT_DATA_SOURCE")
    if pilot_data_source_override is None and any(
        name in os.environ
        for name in (
            "DROMEUS_PILOT_ROUNDS",
            "DROMEUS_PILOT_LOCAL_STEPS",
            "DROMEUS_PILOT_LEARNING_RATE",
            "DROMEUS_PILOT_RETRY_TIMEOUT_SECONDS",
            "DROMEUS_PILOT_TIMEOUT_SECONDS",
        )
    ):
        raise ValueError("pilot runs must declare DROMEUS_PILOT_DATA_SOURCE")
    draft_data = manifest_data()
    draft_data["model_definition_hash"] = MODEL_DEFINITION_HASH
    environment = cast(dict[str, object], draft_data["environment"])
    dromeus_version, pytorch_version, dromeus_commit = await asyncio.gather(
        asyncio.to_thread(package_version, "dromeus"),
        asyncio.to_thread(package_version, "torch"),
        asyncio.to_thread(
            subprocess.check_output,
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ),
    )
    environment.update(
        {
            "dromeus_version": dromeus_version,
            "dromeus_commit": dromeus_commit.strip(),
            "pytorch_version": pytorch_version,
            "axl_version": AXL_COMMIT,
            "model_definition_hash": MODEL_DEFINITION_HASH,
            "container_image_digest": os.environ.get(
                "DROMEUS_CONTAINER_IMAGE_DIGEST", f"sha256:{'0' * 64}"
            ),
        }
    )
    for field in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del draft_data[field]
    draft_data["round_count"] = int(os.environ.get("DROMEUS_PILOT_ROUNDS", "2"))
    draft_data["local_steps"] = int(os.environ.get("DROMEUS_PILOT_LOCAL_STEPS", "1"))
    draft_data["learning_rate"] = float(
        os.environ.get("DROMEUS_PILOT_LEARNING_RATE", "0.01")
    )
    pilot_retry_timeout = os.environ.get("DROMEUS_PILOT_RETRY_TIMEOUT_SECONDS")
    if pilot_retry_timeout is not None:
        transport = cast(dict[str, object], draft_data["transport"])
        transport["retry_timeout_seconds"] = float(pilot_retry_timeout)
    draft = DraftRunSpec.model_validate(draft_data)
    cache_dir = Path(
        os.environ.get(
            "DROMEUS_CIFAR_CACHE",
            Path.home() / ".cache" / "dromeus" / "cifar10",
        )
    )
    pilot_data_source = (
        pilot_data_source_override or "huggingface-uoft-cs-cifar10"
    )
    if pilot_data_source == "torchvision-cifar10":
        train_data, test_data = await asyncio.gather(
            asyncio.to_thread(
                CIFAR10Data.from_torchvision,
                root=cache_dir,
                train=True,
                download=False,
            ),
            asyncio.to_thread(
                CIFAR10Data.from_torchvision,
                root=cache_dir,
                train=False,
                download=False,
            ),
        )
    else:
        train_data, test_data = await asyncio.gather(
            asyncio.to_thread(
                CIFAR10Data.from_huggingface,
                cache_dir=cache_dir,
                train=True,
            ),
            asyncio.to_thread(
                CIFAR10Data.from_huggingface,
                cache_dir=cache_dir,
                train=False,
            ),
        )
    assert train_data.matches_source(source=pilot_data_source, split="train")
    assert test_data.matches_source(source=pilot_data_source, split="test")
    partitions = await asyncio.to_thread(
        train_data.split_iid,
        participant_count=len(draft.dataset.partition_sample_counts),
        seed=draft.dataset.iid_partition_seed,
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        binary = await asyncio.to_thread(build_axl_binary, root)
        processes = await asyncio.to_thread(start_nodes, binary, root, LOG_ROOT)
        nodes: list[NodeRuntime] = []
        try:
            for port in (9302, 9303, 9304, 9305):
                await wait_for_topology(port)
            await wait_for_peer_count(9302, 3)
            for port in (9303, 9304, 9305):
                await wait_for_peer_count(port, 1)
            transports: list[AsyncTransport] = [
                AXLTransport(AXLBridgeConfig(base_url=f"http://127.0.0.1:{port}"))
                for port in (9302, 9303, 9304, 9305)
            ]
            checkpoint = root / "checkpoint.safetensors"
            prepared = await asyncio.to_thread(
                create_initial_checkpoint, checkpoint, seed=17
            )
            local_keys = await asyncio.gather(
                *(transport.local_public_key() for transport in transports)
            )
            expected_manifest = seal_manifest(
                draft=draft,
                participant_keys=set(local_keys),
                initial_checkpoint_hash=prepared.sha256,
                tensor_schema=prepared.tensor_schema,
            )
            expected_manifest_hash = canonical_hash(expected_manifest)
            node_indices = {
                participant.public_key: participant.node_index
                for participant in expected_manifest.participants
            }
            trainers: list[CIFAR10Trainer] = []
            for index, transport in enumerate(transports):
                node_index = node_indices[local_keys[index]]
                partition_index = draft.dataset.node_index_partitions[node_index]
                trainer = await asyncio.to_thread(
                    CIFAR10Trainer,
                    train_data=partitions[partition_index],
                    test_data=test_data,
                    seed=17 + node_index,
                    learning_rate=draft.learning_rate,
                )
                trainers.append(trainer)
                event_sink = JsonlEventSink(
                    LOG_ROOT / f"dromeus-training-node-{index}.log"
                )
                nodes.append(
                    NodeRuntime(
                        transport=transport,
                        draft=draft,
                        environment=draft.environment,
                        dataset=draft.dataset,
                        artifact_store=ArtifactStore(root / f"artifacts-{index}"),
                        event_sink=event_sink,
                        training=TrainingConfig(
                            algorithm=DPSGDAdapter(
                                trainer=trainer,
                                tensor_schema=prepared.tensor_schema,
                                local_steps=draft.local_steps,
                                learning_rate=draft.learning_rate,
                            ),
                            load_checkpoint=trainer.load_checkpoint,
                            run_store=RunStore(root / f"run-{index}"),
                            artifact_root=root / f"rounds-{index}",
                            metrics_publisher=JsonlMetricsPublisher(
                                sink=event_sink,
                                run_id=draft.run_id,
                                manifest_hash=expected_manifest_hash,
                                node_id=local_keys[index],
                            ),
                        ),
                    )
                )
            initiator_key = await transports[0].local_public_key()
            invitation = create_invitation(
                draft=draft,
                initiator_public_key=initiator_key,
                bootstrap_uri="tls://127.0.0.1:9300",
            )
            formation_tasks = [
                asyncio.create_task(
                    nodes[0].initiate(
                        bootstrap_uri="tls://127.0.0.1:9300",
                        checkpoint_path=checkpoint,
                        tensor_schema=prepared.tensor_schema,
                    )
                )
            ]
            formation_tasks.extend(
                asyncio.create_task(node.join(invitation=invitation))
                for node in nodes[1:]
            )
            results = await asyncio.wait_for(
                asyncio.gather(*formation_tasks), timeout=60.0
            )
            assert {result.manifest_hash for result in results} == {
                expected_manifest_hash
            }
            commits = await asyncio.wait_for(
                asyncio.gather(*(node.run() for node in nodes)),
                timeout=float(os.environ.get("DROMEUS_PILOT_TIMEOUT_SECONDS", "180")),
            )
            assert all(len(records) == draft.round_count for records in commits)
            assert all(node.state is NodeState.COMPLETE for node in nodes)
            for records in commits:
                for commit in records:
                    assert any(
                        not np.array_equal(
                            commit.pre_local.weights[name],
                            commit.local_bundle.tensors[name],
                        )
                        for name in commit.pre_local.weights
                    )
                    for name, local_weights in commit.local_bundle.tensors.items():
                        np.testing.assert_allclose(
                            commit.post_mix.weights[name],
                            (local_weights + commit.peer_bundle.tensors[name]) * 0.5,
                        )
            evaluations = await asyncio.gather(
                *(asyncio.to_thread(trainer.evaluate) for trainer in trainers)
            )
            assert all(math.isfinite(loss) for loss, _ in evaluations)
            assert all(0.0 <= accuracy <= 1.0 for _, accuracy in evaluations)
            for index, ((loss, accuracy), node_id) in enumerate(
                zip(evaluations, local_keys, strict=True)
            ):
                log_path = LOG_ROOT / f"dromeus-training-node-{index}.log"
                log_text = await asyncio.to_thread(log_path.read_text, encoding="utf-8")
                records = [
                    cast(dict[str, object], json.loads(line))
                    for line in log_text.splitlines()
                ]
                metrics = [
                    record
                    for record in records
                    if record.get("event") == "round_metrics"
                    and record.get("manifest_hash") == expected_manifest_hash
                    and record.get("node_id") == node_id
                ]
                assert [record.get("round_id") for record in metrics] == list(
                    range(draft.round_count)
                )
                consensus = [
                    record
                    for record in records
                    if record.get("event") == "consensus_distance"
                    and record.get("manifest_hash") == expected_manifest_hash
                    and record.get("node_id") == node_id
                ]
                assert [record.get("round_id") for record in consensus] == list(
                    range(draft.round_count)
                )
                assert all(record.get("sketch_count") == 4 for record in consensus)
                assert all(
                    isinstance(record.get("local_loss"), float) for record in metrics
                )
                assert cast(float, metrics[-1]["evaluation_loss"]) == pytest.approx(
                    loss
                )
                assert cast(float, metrics[-1]["evaluation_accuracy"]) == pytest.approx(
                    accuracy
                )
        finally:
            for node in nodes:
                await node.stop()
            for process in processes:
                process.terminate()
            for process in processes:
                try:
                    await asyncio.to_thread(process.wait, timeout=5)
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
        ["git", "checkout", AXL_COMMIT],
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
