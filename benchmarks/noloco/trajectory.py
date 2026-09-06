"""Temporary Dromeus trajectory capture and cross-implementation comparison."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
from safetensors.numpy import (
    load_file as _load_file,  # pyright: ignore[reportUnknownVariableType]
)
from safetensors.numpy import (
    save_file as _save_file,  # pyright: ignore[reportUnknownVariableType]
)

from dromeus.algorithms.base import (
    AlgorithmEvaluation,
    AlgorithmObservations,
    AlgorithmSnapshot,
    AlgorithmUpdate,
    UpdateBundle,
)
from dromeus.algorithms.noloco import NoLoCoAlgorithm
from dromeus.gossip.peer_scheduler import PeerScheduler


@dataclass(frozen=True, slots=True)
class DromeusTrajectoryContext:
    experiment_sha256: str
    run_config_sha256: str
    initial_checkpoint_sha256: str
    tensor_schema_hash: str
    world_size: int
    benchmark_seed: int
    rank: int
    node_id: str
    participants: tuple[str, ...]
    scheduler_seed: int
    total_outer_steps: int
    source_repository: str
    source_commit: str
    source_path: str
    outer_gradient_convention: str
    pairing_convention: str

    def __post_init__(self) -> None:
        hashes = (
            self.experiment_sha256,
            self.run_config_sha256,
            self.initial_checkpoint_sha256,
            self.tensor_schema_hash,
        )
        if any(len(value) != 64 for value in hashes):
            raise ValueError("trajectory context hashes must be SHA-256 values")
        if self.world_size != 4 or self.total_outer_steps != 2:
            raise ValueError(
                "Dromeus trajectory capture requires four nodes and two steps"
            )
        if len(self.participants) != self.world_size:
            raise ValueError("trajectory participants must match world size")
        if len(set(self.participants)) != self.world_size:
            raise ValueError("trajectory participants must be unique")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("trajectory rank is outside world size")
        if self.participants[self.rank] != self.node_id:
            raise ValueError("trajectory node identity does not match rank")
        _validate_source_provenance(
            source_repository=self.source_repository,
            source_commit=self.source_commit,
            source_path=self.source_path,
            outer_gradient_convention=self.outer_gradient_convention,
            pairing_convention=self.pairing_convention,
        )


@dataclass(frozen=True, slots=True)
class DromeusTrajectoryRecord:
    completed_outer_steps: int
    peer_rank: int | None
    snapshot_path: Path
    snapshot_sha256: str


@dataclass(frozen=True, slots=True)
class DromeusTrajectoryWriter:
    root: Path
    context: DromeusTrajectoryContext

    def write(
        self,
        *,
        completed_outer_steps: int,
        peer_rank: int | None,
        slow_weights: Mapping[str, np.ndarray],
    ) -> DromeusTrajectoryRecord:
        if not 0 <= completed_outer_steps <= self.context.total_outer_steps:
            raise ValueError("completed outer steps are outside trajectory")
        if completed_outer_steps == 0 and peer_rank is not None:
            raise ValueError("initial trajectory snapshot cannot have a peer")
        if completed_outer_steps > 0 and (
            peer_rank is None
            or not 0 <= peer_rank < self.context.world_size
            or peer_rank == self.context.rank
        ):
            raise ValueError("post-outer trajectory snapshot requires a valid peer")
        tensors = _validated_numpy_tensors(slow_weights)
        rank_root = self.root / f"rank-{self.context.rank}"
        snapshot_root = rank_root / "snapshots"
        snapshot_root.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_root / (
            f"outer-{completed_outer_steps:06d}.safetensors"
        )
        temporary = snapshot_path.with_suffix(".tmp")
        try:
            _save_file(tensors, str(temporary))
            temporary.replace(snapshot_path)
        finally:
            temporary.unlink(missing_ok=True)
        record = DromeusTrajectoryRecord(
            completed_outer_steps=completed_outer_steps,
            peer_rank=peer_rank,
            snapshot_path=snapshot_path,
            snapshot_sha256=_file_sha256(snapshot_path),
        )
        metadata = {
            **asdict(self.context),
            "implementation": "dromeus",
            "backend": "axl",
            "acceptance_eligible": True,
            "interval": 1,
            "phase": (
                "initialization" if completed_outer_steps == 0 else "post_outer"
            ),
            "completed_outer_steps": completed_outer_steps,
            "peer_rank": peer_rank,
            "snapshot_path": str(snapshot_path.relative_to(rank_root)),
            "snapshot_sha256": record.snapshot_sha256,
        }
        rank_root.mkdir(parents=True, exist_ok=True)
        with (rank_root / "trajectory.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    metadata,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
        return record


@dataclass(slots=True)
class TrajectoryCapturingAlgorithm:
    """Benchmark-only NoLoCo wrapper that captures the bounded parity profile."""

    delegate: NoLoCoAlgorithm
    writer: DromeusTrajectoryWriter
    _current_round: int | None = field(default=None, init=False, repr=False)
    _initial_written: bool = field(default=False, init=False, repr=False)
    _scheduler: PeerScheduler = field(init=False, repr=False)

    def __post_init__(self) -> None:
        context = self.writer.context
        self._scheduler = PeerScheduler(
            context.participants,
            seed=context.scheduler_seed,
            training_round_count=context.total_outer_steps,
            final_consensus_rounds=0,
        )

    def configure_bundle_codec(
        self,
        *,
        artifact_root: Path,
        run_id: str,
        manifest_hash: str,
        sender_public_key: str,
        algorithm_id: str,
    ) -> None:
        self.delegate.configure_bundle_codec(
            artifact_root=artifact_root,
            run_id=run_id,
            manifest_hash=manifest_hash,
            sender_public_key=sender_public_key,
            algorithm_id=algorithm_id,
        )

    def pre_local(self, round_id: int) -> None:
        if not 0 <= round_id < self.writer.context.total_outer_steps:
            raise ValueError("trajectory round is outside bounded profile")
        self.delegate.pre_local(round_id)
        self._current_round = round_id
        if round_id == 0 and not self._initial_written:
            self.writer.write(
                completed_outer_steps=0,
                peer_rank=None,
                slow_weights=self.delegate.snapshot().weights,
            )
            self._initial_written = True

    def local_training(self) -> None:
        self.delegate.local_training()

    def post_local_bundle(self) -> UpdateBundle:
        return self.delegate.post_local_bundle()

    def validate_peer(self, peer_bundle: UpdateBundle) -> AlgorithmUpdate:
        return self.delegate.validate_peer(peer_bundle)

    def peer_apply(self, peer_update: AlgorithmUpdate) -> AlgorithmSnapshot:
        if self._current_round is None:
            raise RuntimeError("trajectory round has not started")
        snapshot = self.delegate.peer_apply(peer_update)
        context = self.writer.context
        peer_id = self._scheduler.schedule(self._current_round).peer_for(
            context.node_id
        )
        self.writer.write(
            completed_outer_steps=self._current_round + 1,
            peer_rank=context.participants.index(peer_id),
            slow_weights=snapshot.weights,
        )
        return snapshot

    def release_bundle(self, bundle: UpdateBundle) -> None:
        self.delegate.release_bundle(bundle)

    def checkpoint_tensors(self) -> dict[str, np.ndarray]:
        return self.delegate.checkpoint_tensors()

    def observations(self) -> AlgorithmObservations:
        return self.delegate.observations()

    def evaluate(self) -> AlgorithmEvaluation | None:
        return self.delegate.evaluate()

    def snapshot(self) -> AlgorithmSnapshot:
        return self.delegate.snapshot()


@dataclass(frozen=True, slots=True)
class SnapshotComparison:
    rank: int
    completed_outer_steps: int
    peer_rank: int | None
    dromeus_sha256: str
    reference_sha256: str
    maximum_absolute_error: float
    maximum_relative_error: float
    passed: bool


@dataclass(frozen=True, slots=True)
class TrajectoryComparisonReport:
    passed: bool
    absolute_tolerance: float
    relative_tolerance: float
    source_commit: str
    source_path: str
    outer_gradient_convention: str
    pairing_convention: str
    snapshots: tuple[SnapshotComparison, ...]


def compare_trajectories(
    *,
    dromeus_root: Path,
    reference_root: Path,
    report_path: Path,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> TrajectoryComparisonReport:
    """Compare all temporary snapshots, retain hashes/report, then delete tensors."""
    if absolute_tolerance <= 0 or relative_tolerance <= 0:
        raise ValueError("trajectory tolerances must be positive")
    snapshot_paths: set[Path] = set()
    try:
        dromeus = _read_trajectory(dromeus_root, snapshot_paths=snapshot_paths)
        reference = _read_trajectory(reference_root, snapshot_paths=snapshot_paths)
        expected = {(rank, step) for rank in range(4) for step in range(3)}
        if set(dromeus) != expected or set(reference) != expected:
            raise ValueError(
                "trajectory evidence must cover four ranks and three steps"
            )
        comparisons: list[SnapshotComparison] = []
        for key in sorted(expected):
            left = dromeus[key]
            right = reference[key]
            _validate_matching_context(left, right)
            left_path = cast(Path, left["resolved_snapshot_path"])
            right_path = cast(Path, right["resolved_snapshot_path"])
            left_tensors = _load_file(str(left_path))
            right_tensors = _load_file(str(right_path))
            if set(left_tensors) != set(right_tensors):
                raise ValueError("trajectory tensor names do not match")
            maximum_absolute = 0.0
            maximum_relative = 0.0
            passed = True
            for name in sorted(left_tensors):
                left_value = np.asarray(left_tensors[name])
                right_value = np.asarray(right_tensors[name])
                if (
                    left_value.dtype != np.float32
                    or right_value.dtype != np.float32
                    or left_value.shape != right_value.shape
                    or not np.isfinite(left_value).all()
                    or not np.isfinite(right_value).all()
                ):
                    raise ValueError(f"trajectory tensor {name} is incompatible")
                difference = np.abs(left_value - right_value)
                maximum_absolute = max(
                    maximum_absolute,
                    float(np.max(difference)) if difference.size else 0.0,
                )
                denominator = np.maximum(
                    np.abs(right_value),
                    np.finfo(np.float32).tiny,
                )
                maximum_relative = max(
                    maximum_relative,
                    float(np.max(difference / denominator))
                    if difference.size
                    else 0.0,
                )
                passed = passed and bool(
                    np.allclose(
                        left_value,
                        right_value,
                        atol=absolute_tolerance,
                        rtol=relative_tolerance,
                    )
                )
            comparisons.append(
                SnapshotComparison(
                    rank=key[0],
                    completed_outer_steps=key[1],
                    peer_rank=cast(int | None, left["peer_rank"]),
                    dromeus_sha256=cast(str, left["snapshot_sha256"]),
                    reference_sha256=cast(str, right["snapshot_sha256"]),
                    maximum_absolute_error=maximum_absolute,
                    maximum_relative_error=maximum_relative,
                    passed=passed,
                )
            )
        report = TrajectoryComparisonReport(
            passed=all(item.passed for item in comparisons),
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            source_commit=cast(str, dromeus[(0, 0)]["source_commit"]),
            source_path=cast(str, dromeus[(0, 0)]["source_path"]),
            outer_gradient_convention=cast(
                str,
                dromeus[(0, 0)]["outer_gradient_convention"],
            ),
            pairing_convention=cast(
                str,
                dromeus[(0, 0)]["pairing_convention"],
            ),
            snapshots=tuple(comparisons),
        )
        _write_report(report_path, report)
        return report
    finally:
        for path in snapshot_paths:
            path.unlink(missing_ok=True)


def _read_trajectory(
    root: Path,
    *,
    snapshot_paths: set[Path],
) -> dict[tuple[int, int], dict[str, object]]:
    records: dict[tuple[int, int], dict[str, object]] = {}
    for rank_root in sorted(root.glob("rank-*")):
        index_path = rank_root / "trajectory.jsonl"
        if not index_path.is_file():
            continue
        for line in index_path.read_text(encoding="utf-8").splitlines():
            value = cast(dict[str, object], json.loads(line))
            rank = value.get("rank")
            step = value.get("completed_outer_steps")
            relative = value.get("snapshot_path")
            if not isinstance(rank, int) or not isinstance(step, int):
                raise ValueError("trajectory rank and step must be integers")
            if not isinstance(relative, str):
                raise ValueError("trajectory snapshot path is missing")
            resolved = (rank_root / relative).resolve()
            if (
                not resolved.is_relative_to(rank_root.resolve())
                or not resolved.is_file()
            ):
                raise ValueError("trajectory snapshot path is invalid")
            snapshot_paths.add(resolved)
            expected_hash = value.get("snapshot_sha256")
            if expected_hash != _file_sha256(resolved):
                raise ValueError("trajectory snapshot hash does not match")
            key = (rank, step)
            if key in records:
                raise ValueError("trajectory rank and step are duplicated")
            records[key] = {**value, "resolved_snapshot_path": resolved}
    return records


def _validate_matching_context(
    dromeus: Mapping[str, object],
    reference: Mapping[str, object],
) -> None:
    fields = (
        "experiment_sha256",
        "run_config_sha256",
        "initial_checkpoint_sha256",
        "tensor_schema_hash",
        "world_size",
        "benchmark_seed",
        "rank",
        "node_id",
        "completed_outer_steps",
        "peer_rank",
        "source_repository",
        "source_commit",
        "source_path",
        "outer_gradient_convention",
        "pairing_convention",
    )
    if any(dromeus.get(field) != reference.get(field) for field in fields):
        raise ValueError("trajectory contexts do not match")
    if dromeus.get("backend") != "axl" or reference.get("backend") != "nccl":
        raise ValueError("trajectory backends are not acceptance eligible")
    if not dromeus.get("acceptance_eligible") or not reference.get(
        "acceptance_eligible"
    ):
        raise ValueError("trajectory evidence is not acceptance eligible")


def _write_report(path: Path, report: TrajectoryComparisonReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "passed": report.passed,
        "absolute_tolerance": report.absolute_tolerance,
        "relative_tolerance": report.relative_tolerance,
        "source_commit": report.source_commit,
        "source_path": report.source_path,
        "outer_gradient_convention": report.outer_gradient_convention,
        "pairing_convention": report.pairing_convention,
        "snapshots": [asdict(item) for item in report.snapshots],
    }
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validated_numpy_tensors(
    values: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if not values:
        raise ValueError("trajectory slow weights must not be empty")
    result: dict[str, np.ndarray] = {}
    for name in sorted(values):
        value = np.ascontiguousarray(values[name])
        if value.dtype != np.float32:
            raise ValueError(f"trajectory tensor {name} must use float32")
        if not np.isfinite(value).all():
            raise ValueError(f"trajectory tensor {name} must be finite")
        result[name] = value.copy()
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source_provenance(
    *,
    source_repository: str,
    source_commit: str,
    source_path: str,
    outer_gradient_convention: str,
    pairing_convention: str,
) -> None:
    if (
        source_repository,
        source_commit,
        source_path,
        outer_gradient_convention,
        pairing_convention,
    ) != (
        "gensyn-ai/noloco",
        "a1b4a425bdc4050a356cf9f4bae7c383419703ab",
        "src/noloco/sparse_optimizer_c.py",
        "phi-minus-theta-v1",
        "dromeus-peer-scheduler-v1",
    ):
        raise ValueError("trajectory source provenance does not match frozen NoLoCo")


__all__ = [
    "DromeusTrajectoryContext",
    "DromeusTrajectoryRecord",
    "DromeusTrajectoryWriter",
    "SnapshotComparison",
    "TrajectoryCapturingAlgorithm",
    "TrajectoryComparisonReport",
    "compare_trajectories",
]
