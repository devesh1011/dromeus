"""Bounded synthetic evidence collected from actual public runtime operations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import subprocess
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmarks.federated.controls import Trajectories, Weights, save_tensors
from benchmarks.federated.workload import write_json
from dromeus.manifests.canonical import file_sha256
from dromeus.persistence.archive import ArchiveState, RunArchive
from dromeus.persistence.run_store import RunStore
from dromeus.protocol.codec import decode_envelope
from dromeus.protocol.models import MessageType
from dromeus.transport.interface import AsyncTransport, ReceivedBytes


class InMemoryTransport:
    """Benchmark-local byte adapter; it does not bypass formation or transfer."""

    def __init__(
        self, key: str, queues: dict[str, asyncio.Queue[ReceivedBytes]]
    ) -> None:
        self.key = key
        self.queues = queues

    async def local_public_key(self) -> str:
        return self.key

    async def send(self, destination: str, payload: bytes) -> None:
        await self.queues[destination].put(ReceivedBytes(self.key, payload))

    async def recv(self, timeout_seconds: float) -> ReceivedBytes | None:
        try:
            return await asyncio.wait_for(self.queues[self.key].get(), timeout_seconds)
        except TimeoutError:
            return None


class MeasuredTransport:
    def __init__(self, underlying: AsyncTransport) -> None:
        self.underlying = underlying
        self.records: list[dict[str, object]] = []
        self._public_key: str | None = None

    async def local_public_key(self) -> str:
        if self._public_key is None:
            self._public_key = await self.underlying.local_public_key()
        return self._public_key

    async def send(self, destination: str, payload: bytes) -> None:
        envelope = decode_envelope(
            payload,
            authenticated_sender=await self.local_public_key(),
            participant_keys=None,
        )
        category = (
            "telemetry"
            if envelope.message_type is MessageType.CONSENSUS_SKETCH
            else "round-protocol"
            if envelope.round_id is not None
            else "formation-control"
        )
        record: dict[str, object] = {
            "run_id": envelope.run_id,
            "round_id": envelope.round_id,
            "message_type": envelope.message_type.value,
            "category": category,
            "destination": destination,
            "bytes": len(payload),
            "accepted_by_transport": False,
        }
        self.records.append(record)
        await self.underlying.send(destination, payload)
        record["accepted_by_transport"] = True

    async def recv(self, timeout_seconds: float) -> ReceivedBytes | None:
        return await self.underlying.recv(timeout_seconds)

    def summary(self) -> dict[str, object]:
        attempted: dict[str, int] = defaultdict(int)
        accepted: dict[str, int] = defaultdict(int)
        by_type: dict[str, int] = defaultdict(int)
        by_round: dict[str, dict[str, int]] = {}
        for record in self.records:
            category = str(record["category"])
            size = int(str(record["bytes"]))
            attempted[category] += size
            if record["accepted_by_transport"]:
                accepted[category] += size
                by_type[str(record["message_type"])] += size
                round_id = str(record["round_id"])
                counts = by_round.setdefault(round_id, {})
                counts[category] = counts.get(category, 0) + size
        return {
            ("boundary"): (
                "full serialized Dromeus envelopes, including retries; excludes "
                "AXL transport/TCP framing; sender totals are not doubled by "
                "receives"
            ),
            "attempted_bytes": dict(attempted),
            "accepted_bytes": dict(accepted),
            "accepted_bytes_by_message_type": dict(by_type),
            "accepted_bytes_by_round": by_round,
        }


class TrajectoryStore(RunStore):
    """Retain only small synthetic slow weights after durable peer confirmation."""

    def __init__(self, root: Path, round_count: int) -> None:
        super().__init__(root)
        self.root = root
        self.round_count = round_count
        self.snapshots: dict[int, Weights] = {}
        self.commit_times: dict[int, float] = {}
        self.training_started = time.monotonic()

    def confirm_prepared_commit(
        self, *, committed_round: int, state_checksum: str
    ) -> ArchiveState:
        state = super().confirm_prepared_commit(
            committed_round=committed_round, state_checksum=state_checksum
        )
        if not 0 <= committed_round < self.round_count:
            raise ValueError("synthetic trajectory exceeded its frozen bounded horizon")
        if committed_round not in self.snapshots:
            archive = RunArchive.open(self.root)
            assert archive.algorithm_state is not None
            tensors = archive.algorithm_state.load_tensors()
            prefix = "noloco.v1.slow_weights."
            slow = {
                name.removeprefix(prefix): value
                for name, value in tensors.items()
                if name.startswith(prefix)
            }
            if not slow:
                raise ValueError("committed NoLoCo slow weights are missing")
            self.snapshots[committed_round] = slow
            self.commit_times[committed_round] = time.monotonic()
            path = (
                self.root.parent
                / "analysis"
                / f"round-{committed_round:03d}.safetensors"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            save_tensors(slow, str(path))
        return state

    def commit_intervals(self) -> list[float]:
        boundaries = [
            self.training_started,
            *(self.commit_times[index] for index in sorted(self.commit_times)),
        ]
        return [right - left for left, right in zip(boundaries, boundaries[1:])]


def dispersion(trajectories: Trajectories) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for step in range(len(trajectories[0])):
        states = np.stack(
            [
                np.concatenate(
                    [path[step][name].reshape(-1) for name in sorted(path[step])]
                ).astype(np.float64)
                for path in trajectories
            ]
        )
        variance = np.var(states, axis=0)
        records.append(
            {
                "completed_outer_steps": step,
                "rms_coordinate_std": float(np.sqrt(np.mean(variance))),
                "mean_coordinate_std": float(np.sqrt(variance).mean()),
            }
        )
    return records


def compare_trajectories(
    observed: Trajectories, reference: Trajectories, *, atol: float, rtol: float
) -> dict[str, object]:
    if (
        not observed
        or not reference
        or any(not path for path in (*observed, *reference))
    ):
        return {"passed": False, "reason": "trajectory is empty"}
    if len(observed) != len(reference) or any(
        len(a) != len(b) for a, b in zip(observed, reference, strict=True)
    ):
        return {"passed": False, "reason": "trajectory ranks or step counts differ"}
    records: list[dict[str, object]] = []
    for rank, (actual_steps, expected_steps) in enumerate(
        zip(observed, reference, strict=True)
    ):
        for step, (actual, expected) in enumerate(
            zip(actual_steps, expected_steps, strict=True)
        ):
            if set(actual) != set(expected):
                return {"passed": False, "reason": "trajectory tensor schemas differ"}
            maximum = 0.0
            normalized = 0.0
            for name in expected:
                if (
                    not np.isfinite(actual[name]).all()
                    or not np.isfinite(expected[name]).all()
                ):
                    return {
                        "passed": False,
                        "reason": "trajectory contains non-finite tensors",
                    }
                if actual[name].shape != expected[name].shape:
                    return {
                        "passed": False,
                        "reason": "trajectory tensor shapes differ",
                    }
                delta = np.abs(
                    actual[name].astype(np.float64) - expected[name].astype(np.float64)
                )
                tolerance = atol + rtol * np.abs(expected[name].astype(np.float64))
                maximum = max(maximum, float(delta.max()))
                normalized = max(normalized, float((delta / tolerance).max()))
            records.append(
                {
                    "rank": rank,
                    "completed_outer_steps": step,
                    "max_absolute_error": maximum,
                    "max_tolerance_fraction": normalized,
                    "passed": normalized <= 1.0,
                }
            )
    return {
        "passed": all(record["passed"] for record in records),
        "atol": atol,
        "rtol": rtol,
        "max_absolute_error": max(
            float(str(record["max_absolute_error"])) for record in records
        ),
        "per_step": records,
    }


def source_provenance(root: Path) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    paths = list((repository / "src").rglob("*.py"))
    paths.extend((repository / "benchmarks" / "federated").glob("*.py"))
    paths.extend((repository / "benchmarks" / "noloco_reference").glob("*.py"))
    paths.extend(
        repository / name
        for name in (
            "pyproject.toml",
            "uv.lock",
            "benchmarks/federated/experiment.json",
        )
    )
    hashes = {
        str(path.relative_to(repository)): file_sha256(path)
        for path in sorted(set(paths))
    }
    archive = root / "source-snapshot.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for relative in hashes:
            output.add(repository / relative, arcname=relative)
    record: dict[str, Any] = {
        "source_commit": commit,
        ("source_identity"): (
            "working-tree content snapshot; HEAD alone does not identify "
            "uncommitted implementation"
        ),
        "source_tree_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
        "source_files": hashes,
        "source_snapshot_sha256": file_sha256(archive),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
        "container_image_digest": None,
    }
    write_json(root / "provenance.json", record)
    return record
