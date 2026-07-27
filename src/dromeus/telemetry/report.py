"""Exact offline consensus reports from archived RunStore checkpoints."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from safetensors.numpy import (
    load_file as _load_file,  # pyright: ignore[reportUnknownVariableType]
)

from dromeus.manifests.canonical import canonical_hash
from dromeus.manifests.models import SealedManifest
from dromeus.telemetry.consensus import exact_normalized_rms_distance

_load_safetensors = cast(
    Callable[[str], dict[str, np.ndarray]],
    _load_file,
)


class ConsensusReportError(ValueError):
    """Archived run inputs cannot produce one trustworthy report."""


@dataclass(frozen=True, slots=True)
class ConsensusRoundReport:
    """Exact consensus values around one completed gossip mix."""

    round_id: int
    pre_mix_distance: float
    post_mix_distance: float
    smoothed_post_mix_distance: float

    def as_dict(self) -> dict[str, object]:
        return {
            "round_id": self.round_id,
            "pre_mix_distance": self.pre_mix_distance,
            "post_mix_distance": self.post_mix_distance,
            "smoothed_post_mix_distance": self.smoothed_post_mix_distance,
        }


@dataclass(frozen=True, slots=True)
class ExactConsensusReport:
    """Machine-readable exact consensus report and verification result."""

    manifest_hash: str
    node_count: int
    smoothing_window: int
    rounds: tuple[ConsensusRoundReport, ...]
    mixing_non_increasing: bool
    mixing_violations: tuple[int, ...]
    final_normalized_distance: float

    def as_dict(self) -> dict[str, object]:
        return {
            "manifest_hash": self.manifest_hash,
            "node_count": self.node_count,
            "smoothing_window": self.smoothing_window,
            "rounds": [round_report.as_dict() for round_report in self.rounds],
            "mixing_non_increasing": self.mixing_non_increasing,
            "mixing_violations": list(self.mixing_violations),
            "final_normalized_distance": self.final_normalized_distance,
        }

    def write_json(self, path: Path) -> None:
        """Write deterministic machine-readable report output."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_dict(), allow_nan=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )

def build_exact_consensus_report(
    run_roots: Sequence[Path],
    *,
    smoothing_window: int = 3,
    epsilon: float = 1e-12,
    tolerance: float = 1e-12,
) -> ExactConsensusReport:
    """Compute exact pre/post-mix consensus from one archive per node."""
    if len(run_roots) < 2:
        raise ConsensusReportError("at least two node run stores are required")
    if smoothing_window <= 0:
        raise ValueError("smoothing_window must be positive")
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be positive and finite")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")

    archives = [_read_archive(root) for root in run_roots]
    first = archives[0]
    if len(run_roots) != len(first.manifest.participants):
        raise ConsensusReportError(
            "run store count does not match sealed participant count"
        )
    if any(archive.manifest_hash != first.manifest_hash for archive in archives[1:]):
        raise ConsensusReportError("run stores do not share one manifest hash")

    round_ids = set(first.pre_paths)
    if not round_ids or round_ids != set(first.post_paths):
        raise ConsensusReportError("first run store has incomplete checkpoint archives")
    for archive in archives[1:]:
        if set(archive.pre_paths) != round_ids or set(archive.post_paths) != round_ids:
            raise ConsensusReportError("run stores do not share checkpoint rounds")

    report_rounds: list[ConsensusRoundReport] = []
    post_distances: list[float] = []
    violations: list[int] = []
    for round_id in sorted(round_ids):
        pre_models = [
            _load_checkpoint(archive.root, archive.pre_paths[round_id])
            for archive in archives
        ]
        post_models = [
            _load_checkpoint(archive.root, archive.post_paths[round_id])
            for archive in archives
        ]
        _validate_model_set(pre_models)
        _validate_model_set(post_models, expected=pre_models[0])
        pre_distance = exact_normalized_rms_distance(pre_models, epsilon=epsilon)
        post_distance = exact_normalized_rms_distance(post_models, epsilon=epsilon)
        post_distances.append(post_distance)
        if post_distance > pre_distance + tolerance:
            violations.append(round_id)
        start = max(0, len(post_distances) - smoothing_window)
        smoothed = sum(post_distances[start:]) / len(post_distances[start:])
        report_rounds.append(
            ConsensusRoundReport(
                round_id=round_id,
                pre_mix_distance=pre_distance,
                post_mix_distance=post_distance,
                smoothed_post_mix_distance=smoothed,
            )
        )

    return ExactConsensusReport(
        manifest_hash=first.manifest_hash,
        node_count=len(archives),
        smoothing_window=smoothing_window,
        rounds=tuple(report_rounds),
        mixing_non_increasing=not violations,
        mixing_violations=tuple(violations),
        final_normalized_distance=post_distances[-1],
    )


@dataclass(frozen=True, slots=True)
class _Archive:
    root: Path
    manifest: SealedManifest
    manifest_hash: str
    pre_paths: dict[int, str]
    post_paths: dict[int, str]


def _read_archive(root: Path) -> _Archive:
    try:
        manifest = SealedManifest.model_validate(
            json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        )
        state = cast(dict[str, Any], json.loads((root / "state.json").read_text()))
    except (OSError, ValueError, TypeError) as error:
        raise ConsensusReportError(f"invalid run store at {root}") from error
    manifest_hash = canonical_hash(manifest)
    if state.get("manifest_hash") != manifest_hash:
        raise ConsensusReportError(f"manifest hash mismatch in {root}")
    return _Archive(
        root=root,
        manifest=manifest,
        manifest_hash=manifest_hash,
        pre_paths=_checkpoint_map(state, "pre_mix_checkpoints", root),
        post_paths=_checkpoint_map(state, "post_mix_checkpoints", root),
    )


def _checkpoint_map(
    state: Mapping[str, object], key: str, root: Path
) -> dict[int, str]:
    value = state.get(key)
    if not isinstance(value, Mapping):
        raise ConsensusReportError(f"{key} missing from {root}")
    result: dict[int, str] = {}
    entries = cast(Mapping[object, object], value)
    for round_value, path_value in entries.items():
        if not isinstance(round_value, str) or not round_value.isdecimal():
            raise ConsensusReportError(f"invalid {key} round in {root}")
        if not isinstance(path_value, str):
            raise ConsensusReportError(f"invalid {key} path in {root}")
        result[int(round_value)] = path_value
    return result


def _load_checkpoint(root: Path, relative_path: str) -> dict[str, np.ndarray]:
    root_resolved = root.resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root_resolved) or not path.is_file():
        raise ConsensusReportError(
            f"checkpoint is missing or outside run store: {path}"
        )
    try:
        value = _load_safetensors(str(path))
    except (OSError, ValueError, TypeError) as error:
        raise ConsensusReportError(f"invalid checkpoint: {path}") from error
    return value


def _validate_model_set(
    models: Sequence[Mapping[str, np.ndarray]],
    *,
    expected: Mapping[str, np.ndarray] | None = None,
) -> None:
    if not models:
        raise ConsensusReportError("checkpoint set is empty")
    reference = expected or models[0]
    for model in models:
        if set(model) != set(reference):
            raise ConsensusReportError("checkpoint tensor names do not match")
        for name, value in model.items():
            reference_value = reference[name]
            if (
                value.shape != reference_value.shape
                or value.dtype != reference_value.dtype
            ):
                raise ConsensusReportError("checkpoint tensor schemas do not match")
            if (
                not np.issubdtype(value.dtype, np.floating)
                or not np.isfinite(value).all()
            ):
                raise ConsensusReportError("checkpoint contains invalid tensor values")


__all__ = [
    "ConsensusReportError",
    "ConsensusRoundReport",
    "ExactConsensusReport",
    "build_exact_consensus_report",
]
