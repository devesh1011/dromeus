from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from support.sample_manifest import manifest_data

from dromeus.manifests.models import SealedManifest
from dromeus.persistence.run_store import RunStore
from dromeus.telemetry.report import (
    ConsensusReportError,
    build_exact_consensus_report,
)


def _write_round_stores(
    root: Path,
    models: list[float],
    post_models: list[float],
) -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    for index, (pre, post) in enumerate(zip(models, post_models, strict=True)):
        store = RunStore(root / f"node-{index}")
        store.initialize(manifest)
        store.persist_commit(
            committed_round=0,
            algorithm_state={"weight": np.array([post], dtype=np.float32)},
            pre_mix_state={"weight": np.array([pre], dtype=np.float32)},
            post_mix_state={"weight": np.array([post], dtype=np.float32)},
            state_checksum=f"{index + 1:064x}",
            schedule={"round_id": 0, "peer": "peer-1"},
        )


def test_exact_report_computes_before_after_smoothed_and_final_distance(
    tmp_path: Path,
) -> None:
    _write_round_stores(tmp_path, [0, 2, 4, 6], [1, 1, 5, 5])

    report = build_exact_consensus_report(
        [tmp_path / f"node-{index}" for index in range(4)], smoothing_window=3
    )

    assert report.rounds[0].pre_mix_distance == pytest.approx(0.7453559925)
    assert report.rounds[0].post_mix_distance == pytest.approx(0.6666666667)
    assert report.rounds[0].smoothed_post_mix_distance == pytest.approx(
        report.rounds[0].post_mix_distance
    )
    assert report.mixing_non_increasing
    assert report.final_normalized_distance == pytest.approx(0.6666666667)

    json_path = tmp_path / "consensus.json"
    svg_path = tmp_path / "consensus.svg"
    report.write_json(json_path)
    report.write_svg(svg_path)
    payload = json.loads(json_path.read_text())
    assert payload["final_normalized_distance"] == pytest.approx(
        report.final_normalized_distance
    )
    assert "pre-mix" in svg_path.read_text()
    assert svg_path.read_text().startswith("<svg")


def test_exact_report_surfaces_non_monotonic_mixing(tmp_path: Path) -> None:
    _write_round_stores(tmp_path, [0, 0, 0, 0], [1, -1, 0, 0])

    report = build_exact_consensus_report(
        [tmp_path / f"node-{index}" for index in range(4)]
    )

    assert not report.mixing_non_increasing
    assert report.mixing_violations == (0,)


def test_exact_report_rejects_manifest_mismatch(tmp_path: Path) -> None:
    _write_round_stores(tmp_path, [0, 2, 4, 6], [1, 1, 5, 5])
    manifest = SealedManifest.model_validate(manifest_data())
    other_data = manifest.model_dump(mode="python")
    other_data["run_id"] = "other-run"
    other = SealedManifest.model_validate(other_data)
    other_store = RunStore(tmp_path / "other")
    other_store.initialize(other)

    with pytest.raises(ConsensusReportError, match="manifest hash"):
        build_exact_consensus_report(
            [tmp_path / f"node-{index}" for index in range(3)] + [tmp_path / "other"]
        )
