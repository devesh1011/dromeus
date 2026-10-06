"""Reject invented curves and reconcile individual distances with group evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from support.sample_manifest import manifest_data

from benchmarks.m3.plot_node_divergence import load_series, write_plot
from dromeus.manifests.canonical import canonical_hash
from dromeus.manifests.models import DraftRunSpec, SealedManifest
from dromeus.telemetry.evidence import EvidenceError


def measurement_fixture(root: Path) -> tuple[SealedManifest, list[Path]]:
    data = manifest_data()
    data.update(manifest_version=5, round_count=3)
    draft = DraftRunSpec.model_validate(
        {k: v for k, v in data.items() if k in DraftRunSpec.model_fields}
    )
    data["draft_hash"] = canonical_hash(draft)
    manifest = SealedManifest.model_validate(data)
    paths: list[Path] = []
    # Scalar sketches [0, 0, 4, 8]: mean 3, distances [3, 3, 1, 5], RMS sqrt(11).
    for rank, distance in enumerate([3, 3, 1, 5]):
        path = root / f"node-{rank}.jsonl"
        records: list[dict[str, object]] = []
        for round_id in range(3):
            if rank == 0 and round_id == 1:
                continue
            records.append(
                {
                    "timestamp": "2026-10-06T00:00:00Z",
                    "evidence_version": 2,
                    "event": "node_consensus_distance",
                    "run_id": manifest.run_id,
                    "manifest_hash": canonical_hash(manifest),
                    "node_id": f"peer-{rank}",
                    "message_id": f"distance-{round_id}",
                    "round_id": round_id,
                    "absolute_distance": distance,
                    "normalized_distance": distance / (3 + 1e-12),
                    "group_normalized_rms": math.sqrt(11) / (3 + 1e-12),
                    "mean_sketch_norm": 3,
                    "denominator_degenerate": False,
                    "sketch_count": 4,
                }
            )
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        paths.append(path)
    return manifest, paths


def test_plot_preserves_gaps_identity_and_reconciles_rms(tmp_path: Path) -> None:
    manifest, paths = measurement_fixture(tmp_path)
    series = load_series(manifest, list(reversed(paths)))
    assert [s.node_index for s in series] == [0, 1, 2, 3]
    assert series[0].records[1] is None
    assert [
        s.records[0].absolute_distance for s in series if s.records[0] is not None
    ] == [3, 3, 1, 5]
    output = tmp_path / "synthetic.png"
    write_plot(
        manifest, series, output, title="Synthetic local measurement verification"
    )
    assert output.read_bytes().startswith(b"\x89PNG")
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["complete_group_rounds"] == 2
    assert report["nodes"][0]["missing_round_ids"] == [1]
    assert report["nodes"][0]["points"][1] is None
    assert all(n["sha256"] and n["public_key"] for n in report["nodes"])


@pytest.mark.parametrize(
    "fault",
    [
        "identity",
        "hash",
        "duplicate",
        "round",
        "coverage",
        "scale",
        "rms",
        "legacy",
        "missing_node",
    ],
)
def test_plot_rejects_invalid_or_legacy_only_evidence(
    tmp_path: Path, fault: str
) -> None:
    manifest, paths = measurement_fixture(tmp_path)
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if fault == "legacy":
                for key in (
                    "absolute_distance",
                    "normalized_distance",
                    "mean_sketch_norm",
                    "denominator_degenerate",
                ):
                    del row[key]
                row.update(
                    event="consensus_distance",
                    evidence_version=1,
                    normalized_rms=row.pop("group_normalized_rms"),
                )
            elif path == paths[0]:
                if fault == "identity":
                    row["node_id"] = "unknown-peer"
                elif fault == "hash":
                    row["manifest_hash"] = "f" * 64
                elif fault == "round":
                    row["round_id"] += 3
                elif fault == "coverage":
                    row["sketch_count"] = 3
                elif fault == "scale":
                    row["mean_sketch_norm"] = 2
                    row["normalized_distance"] = row["absolute_distance"] / (2 + 1e-12)
            if fault == "rms":
                row["group_normalized_rms"] = 2
        if fault == "duplicate" and path == paths[0]:
            rows.append(rows[0])
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    if fault == "missing_node":
        paths.pop()
    with pytest.raises(EvidenceError):
        load_series(manifest, paths)
