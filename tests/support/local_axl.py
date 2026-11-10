"""Pinned real-AXL subprocesses for local CPU development experiments.

This harness is deliberately separate from AWS and the frozen M2 GPU/WAN runs.
Private identities and live node configurations exist only in a temporary
directory. Retained logs and topology records use ranks sorted by the public keys
reported by the running AXL APIs; startup failures may retain launch-order logs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
from collections.abc import AsyncGenerator
from contextlib import ExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.request import urlopen

AXL_COMMIT = "628e28ace077f26dfe8d0259009b357216a9d8d4"
AXL_REPOSITORY = "https://github.com/gensyn-ai/axl.git"
_PUBLIC_KEY = re.compile(r"^[0-9a-f]{64}$")
_STARTUP_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class LocalAXLNode:
    public_key: str
    bridge_url: str


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _output(*command: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        command, cwd=cwd, check=True, capture_output=True, text=True, timeout=30
    ).stdout.strip()


def _validate_binary_cache(root: Path, go: str, go_version: str) -> Path:
    binary = root / "node"
    source = root / "source"
    try:
        record = json.loads((root / "build.json").read_bytes())
        if record != {
            "schema_version": 1,
            "repository": AXL_REPOSITORY,
            "commit": AXL_COMMIT,
            "go_version": go_version,
            "binary_sha256": _file_sha256(binary),
            "build_flags": ["-mod=readonly", "-trimpath", "-buildvcs=true"],
        }:
            raise ValueError("build record or binary hash differs")
        if not os.access(binary, os.X_OK):
            raise ValueError("binary is not executable")
        if _output("git", "rev-parse", "HEAD", cwd=source) != AXL_COMMIT:
            raise ValueError("source revision differs")
        if _output("git", "status", "--porcelain", cwd=source):
            raise ValueError("source checkout is dirty")
        if _output("git", "remote", "get-url", "origin", cwd=source) != AXL_REPOSITORY:
            raise ValueError("source remote differs")
        metadata = _output(go, "version", "-m", str(binary))
        for required in (
            f"vcs.revision={AXL_COMMIT}",
            "vcs.modified=false",
            "-trimpath=true",
        ):
            if required not in metadata:
                raise ValueError(f"binary lacks verified build setting {required}")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            f"AXL cache failed verification at {root}: {error}"
        ) from error
    return binary


def ensure_axl_binary(cache_root: Path) -> Path:
    """Build the pinned source with local Go, or revalidate a matching cache.

    A mismatching or incomplete existing cache is rejected, never silently reused.
    Go's embedded VCS revision, clean source checkout, toolchain identity, and the
    recorded binary checksum must all agree. Failed build output is retained in
    the cache root for diagnosis.
    """
    go = shutil.which("go")
    if go is None or shutil.which("git") is None:
        raise RuntimeError("local Go and Git are required to build pinned AXL")
    go_version = _output(go, "version")
    cache_root = cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    toolchain_id = hashlib.sha256(go_version.encode()).hexdigest()[:12]
    target = cache_root / f"axl-{AXL_COMMIT}-{toolchain_id}"
    if target.exists():
        return _validate_binary_cache(target, go, go_version)
    with tempfile.TemporaryDirectory(prefix=".axl-build-", dir=cache_root) as directory:
        staging = Path(directory) / "build"
        source = staging / "source"
        source.mkdir(parents=True)
        build_log = staging / "build.log"
        commands = (
            ("git", "init", "-q"),
            ("git", "remote", "add", "origin", AXL_REPOSITORY),
            ("git", "fetch", "--depth", "1", "origin", AXL_COMMIT),
            ("git", "checkout", "--detach", "FETCH_HEAD"),
            (
                go,
                "build",
                "-mod=readonly",
                "-trimpath",
                "-buildvcs=true",
                "-o",
                str(staging / "node"),
                "./cmd/node",
            ),
        )
        try:
            with build_log.open("w") as output:
                for command in commands:
                    subprocess.run(
                        command,
                        cwd=source,
                        check=True,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        timeout=600,
                        env={**os.environ, "GOTOOLCHAIN": "local"},
                    )
            (staging / "build.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "repository": AXL_REPOSITORY,
                        "commit": AXL_COMMIT,
                        "go_version": go_version,
                        "binary_sha256": _file_sha256(staging / "node"),
                        "build_flags": ["-mod=readonly", "-trimpath", "-buildvcs=true"],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            _validate_binary_cache(staging, go, go_version)
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            retained = cache_root / f"failed-{time.time_ns()}.log"
            if build_log.exists():
                shutil.copyfile(build_log, retained)
            raise RuntimeError(f"pinned AXL build failed; see {retained}") from error
        if target.exists():
            return _validate_binary_cache(target, go, go_version)
        staging.rename(target)
    return _validate_binary_cache(target, go, go_version)


def _openssl() -> str:
    for candidate in (
        "/opt/homebrew/opt/openssl/bin/openssl",
        "/usr/local/opt/openssl/bin/openssl",
        shutil.which("openssl"),
    ):
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError("OpenSSL with Ed25519 support is required for local AXL")


def _topology(bridge_url: str) -> dict[str, object]:
    with urlopen(f"{bridge_url}/topology", timeout=1.0) as response:
        payload = cast(object, json.loads(response.read()))
    if not isinstance(payload, dict):
        raise ValueError("AXL topology must be an object")
    return cast(dict[str, object], payload)


def _public_key(topology: dict[str, object]) -> str:
    key = topology.get("our_public_key")
    if not isinstance(key, str) or _PUBLIC_KEY.fullmatch(key) is None:
        raise ValueError("AXL API did not return a valid public key")
    return key


def _live_peers(topology: dict[str, object]) -> set[str]:
    peers = topology.get("peers")
    if not isinstance(peers, list):
        return set()
    result: set[str] = set()
    for raw in cast(list[object], peers):
        if isinstance(raw, dict):
            item = cast(dict[str, object], raw)
            key = item.get("public_key")
            if item.get("up") is True and isinstance(key, str):
                result.add(key)
    return result


async def _wait_for_cluster(
    bridges: tuple[str, ...], processes: list[subprocess.Popen[bytes]]
) -> tuple[dict[str, object], ...]:
    deadline = time.monotonic() + _STARTUP_SECONDS
    while time.monotonic() < deadline:
        if any(process.poll() is not None for process in processes):
            raise RuntimeError("AXL exited during startup; inspect retained node logs")
        snapshots = await asyncio.gather(
            *(asyncio.to_thread(_topology, bridge) for bridge in bridges),
            return_exceptions=True,
        )
        if not any(isinstance(snapshot, BaseException) for snapshot in snapshots):
            topologies = cast(tuple[dict[str, object], ...], tuple(snapshots))
            try:
                keys = {_public_key(topology) for topology in topologies}
                if len(keys) != len(bridges):
                    raise RuntimeError("running AXL APIs returned duplicate identities")
                if all(
                    _live_peers(topology) == keys - {_public_key(topology)}
                    for topology in topologies
                ):
                    return topologies
            except ValueError:
                pass
        await asyncio.sleep(0.1)
    raise TimeoutError("AXL full mesh did not become ready before the startup deadline")


def _stop_processes(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 5.0
    for process in processes:
        try:
            process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for process in processes:
        if process.poll() is None:
            process.kill()
    for process in processes:
        process.wait(timeout=5.0)


@asynccontextmanager
async def local_axl_cluster(
    *, binary: Path, log_root: Path, node_count: int = 4
) -> AsyncGenerator[tuple[LocalAXLNode, ...]]:
    """Start isolated full-mesh AXL processes and always remove their private keys."""
    if type(node_count) is not int or node_count not in (4, 8, 16):
        raise ValueError("node_count must be 4, 8, or 16")
    binary = binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError("binary must be an executable AXL node")
    log_root = log_root.resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    outputs = [log_root / f"axl-node-{rank}.log" for rank in range(node_count)]
    outputs.extend(
        log_root / f"topology-ready-{rank}.json" for rank in range(node_count)
    )
    outputs.append(log_root / "axl-cluster.json")
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise ValueError("AXL output already exists; use a fresh log directory")
    processes: list[subprocess.Popen[bytes]] = []
    with tempfile.TemporaryDirectory(prefix="dromeus-local-axl-") as directory:
        private_root = Path(directory)
        private_root.chmod(0o700)
        try:
            with ExitStack() as stack:
                reservations = [
                    stack.enter_context(socket.socket()) for _ in range(node_count * 2)
                ]
                for reservation in reservations:
                    reservation.bind(("127.0.0.1", 0))
                ports = [reservation.getsockname()[1] for reservation in reservations]
                endpoints = tuple(f"tls://127.0.0.1:{port}" for port in ports[::2])
                bridges = tuple(f"http://127.0.0.1:{port}" for port in ports[1::2])
                for index in range(node_count):
                    private_key = private_root / f"private-{index}.pem"
                    subprocess.run(
                        [
                            _openssl(),
                            "genpkey",
                            "-algorithm",
                            "ed25519",
                            "-out",
                            str(private_key),
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=10,
                    )
                    private_key.chmod(0o600)
                    config = private_root / f"node-{index}.json"
                    config.write_text(
                        json.dumps(
                            {
                                "PrivateKeyPath": str(private_key),
                                "Peers": [
                                    peer
                                    for peer in endpoints
                                    if peer != endpoints[index]
                                ],
                                "Listen": [endpoints[index]],
                                "api_port": ports[2 * index + 1],
                                "bridge_addr": "127.0.0.1",
                                "max_message_size": 16 * 1024 * 1024,
                            }
                        )
                    )
                    config.chmod(0o600)
                    reservations[2 * index].close()
                    reservations[2 * index + 1].close()
                    with (log_root / f"axl-node-{index}.log").open("xb") as log:
                        processes.append(
                            subprocess.Popen(
                                [str(binary), "-config", str(config)],
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                cwd=private_root,
                            )
                        )
            topologies = await _wait_for_cluster(bridges, processes)
            order = sorted(
                range(node_count), key=lambda index: _public_key(topologies[index])
            )
            staging_logs = tuple(
                log_root / f".sorting-{private_root.name}-{index}.log"
                for index in range(node_count)
            )
            for index in range(node_count):
                (log_root / f"axl-node-{index}.log").rename(staging_logs[index])
            nodes: list[LocalAXLNode] = []
            for rank, index in enumerate(order):
                staging_logs[index].rename(log_root / f"axl-node-{rank}.log")
                (log_root / f"topology-ready-{rank}.json").write_text(
                    json.dumps(topologies[index], indent=2, sort_keys=True) + "\n"
                )
                nodes.append(
                    LocalAXLNode(_public_key(topologies[index]), bridges[index])
                )
            (log_root / "axl-cluster.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "evidence_kind": "local-cpu-development",
                        "axl_commit": AXL_COMMIT,
                        "binary_sha256": _file_sha256(binary),
                        "nodes": [
                            {
                                "rank": rank,
                                "public_key": node.public_key,
                                "bridge_url": node.bridge_url,
                                "log": f"axl-node-{rank}.log",
                            }
                            for rank, node in enumerate(nodes)
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            yield tuple(nodes)
        finally:
            cleanup = asyncio.create_task(asyncio.to_thread(_stop_processes, processes))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise
