"""Periodic official-run checkpoints and exact NoLoCo divergence analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
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
from dromeus.manifests.models import WarmupCosineSchedule

_SNAPSHOT_OVERHEAD_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class AnalysisCheckpointContext:
    experiment_sha256: str
    run_config_sha256: str
    initial_checkpoint_sha256: str
    tensor_schema_hash: str
    world_size: int
    benchmark_seed: int
    rank: int
    node_id: str
    total_outer_steps: int
    interval_rounds: int
    storage_budget_bytes: int

    def __post_init__(self) -> None:
        hashes = (
            self.experiment_sha256,
            self.run_config_sha256,
            self.initial_checkpoint_sha256,
            self.tensor_schema_hash,
        )
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        ):
            raise ValueError("analysis context hashes must be SHA-256 values")
        if self.world_size not in {4, 8, 16}:
            raise ValueError("analysis world size must be 4, 8, or 16")
        if not 0 <= self.rank < self.world_size or not self.node_id:
            raise ValueError("analysis rank and node identity are invalid")
        if self.total_outer_steps <= 0 or self.interval_rounds <= 0:
            raise ValueError("analysis round count and interval must be positive")
        if self.storage_budget_bytes <= 0:
            raise ValueError("analysis storage budget must be positive")


@dataclass(frozen=True, slots=True)
class AnalysisCheckpointRecord:
    completed_outer_steps: int
    snapshot_path: Path
    snapshot_sha256: str
    size_bytes: int


@dataclass(slots=True)
class AnalysisCheckpointWriter:
    root: Path
    context: AnalysisCheckpointContext
    _written_steps: set[int] = field(
        default_factory=lambda: set[int](),
        init=False,
        repr=False,
    )
    _bytes_written: int = field(default=0, init=False, repr=False)

    def should_write(self, *, completed_outer_steps: int) -> bool:
        if not 1 <= completed_outer_steps <= self.context.total_outer_steps:
            raise ValueError("completed outer steps are outside analysis run")
        return (
            completed_outer_steps == self.context.total_outer_steps
            or completed_outer_steps % self.context.interval_rounds == 0
        )

    def write(
        self,
        *,
        completed_outer_steps: int,
        slow_weights: Mapping[str, np.ndarray],
    ) -> AnalysisCheckpointRecord:
        if not self.should_write(completed_outer_steps=completed_outer_steps):
            raise ValueError("analysis checkpoint is outside frozen cadence")
        if completed_outer_steps in self._written_steps:
            raise ValueError("analysis checkpoint step is duplicated")
        tensors = _validated_tensors(slow_weights)
        rank_root = self.root / f"rank-{self.context.rank}"
        snapshot_root = rank_root / "snapshots"
        snapshot_root.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_root / (
            f"outer-{completed_outer_steps:06d}.safetensors"
        )
        temporary = snapshot_path.with_suffix(".tmp")
        try:
            _save_file(tensors, str(temporary))
            size_bytes = temporary.stat().st_size
            if self._bytes_written + size_bytes > self.context.storage_budget_bytes:
                raise ValueError("analysis checkpoints exceed frozen storage budget")
            temporary.replace(snapshot_path)
        finally:
            temporary.unlink(missing_ok=True)
        record = AnalysisCheckpointRecord(
            completed_outer_steps=completed_outer_steps,
            snapshot_path=snapshot_path,
            snapshot_sha256=_file_sha256(snapshot_path),
            size_bytes=snapshot_path.stat().st_size,
        )
        metadata = {
            **asdict(self.context),
            "completed_outer_steps": completed_outer_steps,
            "snapshot_path": str(snapshot_path.relative_to(rank_root)),
            "snapshot_sha256": record.snapshot_sha256,
            "size_bytes": record.size_bytes,
        }
        with (rank_root / "analysis.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    metadata,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
        self._written_steps.add(completed_outer_steps)
        self._bytes_written += record.size_bytes
        return record


@dataclass(slots=True)
class AnalysisCheckpointCapturingAlgorithm:
    """Benchmark-only NoLoCo wrapper for frozen periodic analysis snapshots."""

    delegate: NoLoCoAlgorithm
    writer: AnalysisCheckpointWriter
    _current_round: int | None = field(default=None, init=False, repr=False)

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
            raise ValueError("analysis round is outside frozen run")
        self.delegate.pre_local(round_id)
        self._current_round = round_id

    def local_training(self) -> None:
        self.delegate.local_training()

    def post_local_bundle(self) -> UpdateBundle:
        return self.delegate.post_local_bundle()

    def validate_peer(self, peer_bundle: UpdateBundle) -> AlgorithmUpdate:
        return self.delegate.validate_peer(peer_bundle)

    def peer_apply(self, peer_update: AlgorithmUpdate) -> AlgorithmSnapshot:
        if self._current_round is None:
            raise RuntimeError("analysis round has not started")
        snapshot = self.delegate.peer_apply(peer_update)
        completed = self._current_round + 1
        if self.writer.should_write(completed_outer_steps=completed):
            self.writer.write(
                completed_outer_steps=completed,
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
class AnalysisPoint:
    completed_outer_steps: int
    completed_inner_steps: int
    learning_rate: float
    weight_std_l2: float
    normalized_rms: float
    snapshot_sha256_by_rank: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AnalysisReport:
    passed: bool
    smoothing_window: int
    trend_slope_per_inner_step: float | None
    final_below_post_warmup_peak: bool
    points: tuple[AnalysisPoint, ...]


def analyze_analysis_checkpoints(
    *,
    root: Path,
    report_path: Path,
    learning_rate_schedule: WarmupCosineSchedule,
    inner_steps_per_round: int,
) -> AnalysisReport:
    """Compute source-style cross-worker weight std and delete full tensors."""
    if inner_steps_per_round <= 0:
        raise ValueError("inner steps per round must be positive")
    snapshot_paths: set[Path] = set()
    try:
        records = _read_records(root, snapshot_paths=snapshot_paths)
        if not records:
            raise ValueError("analysis checkpoint evidence is empty")
        first = next(iter(records.values()))
        world_size = cast(int, first["world_size"])
        total_outer_steps = cast(int, first["total_outer_steps"])
        interval_rounds = cast(int, first["interval_rounds"])
        expected_steps = _analysis_steps(total_outer_steps, interval_rounds)
        expected = {
            (rank, completed)
            for rank in range(world_size)
            for completed in expected_steps
        }
        if set(records) != expected:
            raise ValueError("analysis checkpoints do not cover frozen cadence")
        _validate_record_contexts(records)
        points = tuple(
            _analysis_point(
                completed_outer_steps=completed,
                records=records,
                world_size=world_size,
                schedule=learning_rate_schedule,
                inner_steps_per_round=inner_steps_per_round,
            )
            for completed in expected_steps
        )
        post_warmup = tuple(
            point
            for point in points
            if point.completed_inner_steps
            > learning_rate_schedule.warmup_inner_steps
        )
        smoothing_window = min(3, len(post_warmup))
        smoothed = _smoothed_points(post_warmup, window=smoothing_window)
        slope = _trend_slope(smoothed)
        final_below_peak = bool(
            post_warmup
            and post_warmup[-1].weight_std_l2
            < max(point.weight_std_l2 for point in post_warmup)
        )
        report = AnalysisReport(
            passed=(slope is not None and slope < 0.0 and final_below_peak),
            smoothing_window=smoothing_window,
            trend_slope_per_inner_step=slope,
            final_below_post_warmup_peak=final_below_peak,
            points=points,
        )
        _write_report(report_path, report)
        return report
    finally:
        for path in snapshot_paths:
            path.unlink(missing_ok=True)


def required_analysis_storage_bytes(
    *,
    raw_parameter_bytes: int,
    total_outer_steps: int,
    interval_rounds: int,
) -> int:
    if raw_parameter_bytes <= 0:
        raise ValueError("raw parameter bytes must be positive")
    count = len(_analysis_steps(total_outer_steps, interval_rounds))
    return count * (raw_parameter_bytes + _SNAPSHOT_OVERHEAD_BYTES)


def validate_analysis_storage(
    *,
    required_bytes: int,
    budget_bytes: int,
    available_bytes: int,
) -> None:
    if required_bytes <= 0 or budget_bytes <= 0 or available_bytes < 0:
        raise ValueError("analysis storage values are invalid")
    if required_bytes > budget_bytes:
        raise ValueError("analysis storage requirement exceeds frozen budget")
    if required_bytes > available_bytes:
        raise ValueError("analysis storage requirement exceeds available bytes")


def _analysis_steps(total_outer_steps: int, interval_rounds: int) -> tuple[int, ...]:
    if total_outer_steps <= 0 or interval_rounds <= 0:
        raise ValueError("analysis cadence values must be positive")
    steps = list(range(interval_rounds, total_outer_steps + 1, interval_rounds))
    if not steps or steps[-1] != total_outer_steps:
        steps.append(total_outer_steps)
    return tuple(steps)


def _read_records(
    root: Path,
    *,
    snapshot_paths: set[Path],
) -> dict[tuple[int, int], dict[str, object]]:
    records: dict[tuple[int, int], dict[str, object]] = {}
    for rank_root in sorted(root.glob("rank-*")):
        index = rank_root / "analysis.jsonl"
        if not index.is_file():
            continue
        for line in index.read_text(encoding="utf-8").splitlines():
            value = cast(dict[str, object], json.loads(line))
            rank = value.get("rank")
            completed = value.get("completed_outer_steps")
            relative = value.get("snapshot_path")
            if not isinstance(rank, int) or not isinstance(completed, int):
                raise ValueError("analysis rank and step must be integers")
            if not isinstance(relative, str):
                raise ValueError("analysis snapshot path is missing")
            path = (rank_root / relative).resolve()
            if not path.is_relative_to(rank_root.resolve()) or not path.is_file():
                raise ValueError("analysis snapshot path is invalid")
            snapshot_paths.add(path)
            if value.get("snapshot_sha256") != _file_sha256(path):
                raise ValueError("analysis snapshot hash does not match")
            key = (rank, completed)
            if key in records:
                raise ValueError("analysis rank and step are duplicated")
            records[key] = {**value, "resolved_snapshot_path": path}
    return records


def _validate_record_contexts(
    records: Mapping[tuple[int, int], Mapping[str, object]],
) -> None:
    shared_fields = (
        "experiment_sha256",
        "run_config_sha256",
        "initial_checkpoint_sha256",
        "tensor_schema_hash",
        "world_size",
        "benchmark_seed",
        "total_outer_steps",
        "interval_rounds",
        "storage_budget_bytes",
    )
    first = next(iter(records.values()))
    if any(
        record.get(field) != first.get(field)
        for record in records.values()
        for field in shared_fields
    ):
        raise ValueError("analysis checkpoint contexts do not match")
    rank_nodes: dict[int, object] = {}
    for (rank, _), record in records.items():
        node_id = record.get("node_id")
        existing = rank_nodes.setdefault(rank, node_id)
        if existing != node_id:
            raise ValueError("analysis rank changed node identity")
    if len(set(rank_nodes.values())) != len(rank_nodes):
        raise ValueError("analysis node identities are not unique")


def _analysis_point(
    *,
    completed_outer_steps: int,
    records: Mapping[tuple[int, int], Mapping[str, object]],
    world_size: int,
    schedule: WarmupCosineSchedule,
    inner_steps_per_round: int,
) -> AnalysisPoint:
    mean: dict[str, np.ndarray] = {}
    sum_squared_deviation: dict[str, np.ndarray] = {}
    names: tuple[str, ...] | None = None
    for rank in range(world_size):
        path = cast(
            Path,
            records[(rank, completed_outer_steps)]["resolved_snapshot_path"],
        )
        tensors = _load_file(str(path))
        current_names = tuple(sorted(tensors))
        if names is None:
            names = current_names
            mean = {
                name: np.zeros_like(tensors[name], dtype=np.float64) for name in names
            }
            sum_squared_deviation = {
                name: np.zeros_like(tensors[name], dtype=np.float64) for name in names
            }
        elif current_names != names:
            raise ValueError("analysis tensor names do not match")
        count = rank + 1
        for name in names:
            value = np.asarray(tensors[name])
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"analysis tensor {name} is incompatible")
            if value.shape != mean[name].shape:
                raise ValueError(f"analysis tensor {name} shape does not match")
            value64 = value.astype(np.float64)
            delta = value64 - mean[name]
            mean[name] += delta / count
            sum_squared_deviation[name] += delta * (value64 - mean[name])
    if names is None:
        raise ValueError("analysis snapshot has no tensors")
    variance_l2 = sum(
        float(np.sum(sum_squared_deviation[name])) for name in names
    ) / world_size
    weight_std_l2 = math.sqrt(max(0.0, variance_l2))
    mean_norm = math.sqrt(sum(float(np.sum(mean[name] ** 2)) for name in names))
    completed_inner_steps = completed_outer_steps * inner_steps_per_round
    return AnalysisPoint(
        completed_outer_steps=completed_outer_steps,
        completed_inner_steps=completed_inner_steps,
        learning_rate=schedule.learning_rate(completed_inner_steps - 1),
        weight_std_l2=weight_std_l2,
        normalized_rms=weight_std_l2 / (mean_norm + 1e-12),
        snapshot_sha256_by_rank=tuple(
            cast(str, records[(rank, completed_outer_steps)]["snapshot_sha256"])
            for rank in range(world_size)
        ),
    )


def _smoothed_points(
    points: tuple[AnalysisPoint, ...],
    *,
    window: int,
) -> tuple[tuple[float, float], ...]:
    if window <= 0:
        return ()
    return tuple(
        (
            float(points[end].completed_inner_steps),
            float(
                np.mean(
                    [
                        point.weight_std_l2
                        for point in points[end - window + 1 : end + 1]
                    ]
                )
            ),
        )
        for end in range(window - 1, len(points))
    )


def _trend_slope(points: tuple[tuple[float, float], ...]) -> float | None:
    if len(points) < 2:
        return None
    x = np.array([point[0] for point in points], dtype=np.float64)
    y = np.array([point[1] for point in points], dtype=np.float64)
    centered = x - x.mean()
    denominator = float(np.sum(centered**2))
    if denominator == 0.0:
        return None
    return float(np.sum(centered * (y - y.mean())) / denominator)


def _write_report(path: Path, report: AnalysisReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "passed": report.passed,
        "smoothing_window": report.smoothing_window,
        "trend_slope_per_inner_step": report.trend_slope_per_inner_step,
        "final_below_post_warmup_peak": report.final_below_post_warmup_peak,
        "points": [asdict(point) for point in report.points],
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


def _validated_tensors(
    values: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if not values:
        raise ValueError("analysis slow weights must not be empty")
    tensors: dict[str, np.ndarray] = {}
    for name in sorted(values):
        value = np.ascontiguousarray(values[name])
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"analysis tensor {name} must be finite FP32")
        tensors[name] = value.copy()
    return tensors


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--world-size", type=int, choices=(4, 8, 16), required=True)
    parser.add_argument("--seed", type=int, choices=(17, 29, 41), required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from benchmarks.noloco.experiment import (
        RunSelector,
        load_frozen_experiment,
    )

    arguments = _parser().parse_args(argv)
    run = load_frozen_experiment(arguments.experiment).resolve(
        RunSelector(
            profile="official",
            world_size=arguments.world_size,
            benchmark_seed=arguments.seed,
        )
    )
    report = analyze_analysis_checkpoints(
        root=arguments.analysis_root,
        report_path=arguments.report,
        learning_rate_schedule=run.run.schedule,
        inner_steps_per_round=run.algorithm.inner_steps,
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AnalysisCheckpointContext",
    "AnalysisCheckpointCapturingAlgorithm",
    "AnalysisCheckpointRecord",
    "AnalysisCheckpointWriter",
    "AnalysisPoint",
    "AnalysisReport",
    "analyze_analysis_checkpoints",
    "main",
    "required_analysis_storage_bytes",
    "validate_analysis_storage",
]
