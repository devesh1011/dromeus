"""Plots preserve node identities, missing measurements and seed-level uncertainty."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from benchmarks.m3.plot_detector_validation import noise_summary


def test_noise_bands_use_three_seed_curves_and_leave_incomplete_rounds_missing() -> (
    None
):
    runs = [
        {
            "sigma": sigma,
            "seed": seed,
            "group_normalized_rms": [value, None if seed == 29 else value],
        }
        for sigma in (0.5, 1.0, 2.0, 4.0)
        for seed, value in ((17, 0.1), (29, 0.2), (41, 0.3))
    ]
    result = noise_summary(runs)
    assert result[0]["mean"] == pytest.approx([0.2, None])
    assert result[0]["sample_sd"] == pytest.approx([0.1, None])
    assert result[0]["seed_coverage"] == [3, 2]
    with pytest.raises(ValueError, match="seed"):
        noise_summary(runs[:-1])


def test_source_labels_follow_keys_and_plot_retains_gap_and_live_detection_round(
    tmp_path: Path,
) -> None:
    import json
    import math

    from support.sample_manifest import manifest_data

    from benchmarks.m3.plot_detector_validation import collect_node_curves, render_run
    from dromeus.manifests.canonical import canonical_hash
    from dromeus.manifests.models import DivergencePolicy, DraftRunSpec, SealedManifest
    from dromeus.telemetry.evidence import DivergenceStatusEvidence

    policy = DivergencePolicy(
        threshold_set_id="synthetic-plot-test",
        warmup_rounds=0,
        window_rounds=4,
        patience_windows=2,
        distance_floor=0.001,
        growth_ratio=1.5,
        min_log_slope=0.02,
        severe_distance=0.2,
        severe_patience=2,
        recovery_windows=3,
        recovery_distance=0.1,
        max_lag_rounds=8,
    )
    data = manifest_data()
    data.update(
        manifest_version=5, round_count=3, divergence_detection=policy.model_dump()
    )
    draft = DraftRunSpec.model_validate(
        {k: v for k, v in data.items() if k in DraftRunSpec.model_fields}
    )
    data["draft_hash"] = canonical_hash(draft)
    manifest = SealedManifest.model_validate(data)
    logs: list[Path] = []
    summaries: list[Path] = []
    for index, distance in enumerate((3, 3, 1, 5)):
        key = f"peer-{index}"
        rows: list[dict[str, Any]] = []
        for round_id in range(3):
            if index == 0 and round_id == 1:
                continue
            rows.append(
                {
                    "timestamp": "2026-10-21T00:00:00Z",
                    "event": "node_consensus_distance",
                    "evidence_version": 2,
                    "run_id": manifest.run_id,
                    "manifest_hash": canonical_hash(manifest),
                    "node_id": key,
                    "message_id": f"measurement-{round_id}",
                    "round_id": round_id,
                    "absolute_distance": distance,
                    "normalized_distance": distance / (3 + 1e-12),
                    "group_normalized_rms": math.sqrt(11) / (3 + 1e-12),
                    "mean_sketch_norm": 3,
                    "denominator_degenerate": False,
                    "sketch_count": 4,
                }
            )
        status = DivergenceStatusEvidence(
            run_id=manifest.run_id,
            manifest_hash=canonical_hash(manifest),
            node_id=key,
            policy_hash=canonical_hash(policy),
            threshold_set_id=policy.threshold_set_id,
            transition_id=1,
            previous_state="warming_up",
            state="diverging",
            warning_active=True,
            scale_warning=False,
            reason="severe_distance",
            source_round=0,
            detection_round=2,
            window_start=0,
            window_end=0,
            window_count=1,
            expected_participants=4,
            observed_participants=4,
            age_rounds=2,
            missing_rounds=0,
            measurement_channel="normalized",
        )
        path = tmp_path / f"log-{index}.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
            + status.model_dump_json()
            + "\n"
        )
        logs.append(path)
        summary = tmp_path / f"summary-{index}.json"
        source_rank = (index + 1) % 4
        summary.write_text(
            json.dumps(
                {
                    "node_public_key": key,
                    "rank": source_rank,
                    "source": ("mnist", "usps", "svhn", "optdigits")[source_rank],
                    "draft_hash": manifest.draft_hash,
                    "event_delivery_dropped": 0,
                }
            )
        )
        summaries.append(summary)
    result = collect_node_curves(manifest, list(reversed(logs)), summaries)
    assert result["nodes"][0]["node_id"] == "peer-3"
    assert result["nodes"][0]["source"] == "mnist"
    assert result["nodes"][1]["normalized_distance"][1] is None
    assert result["group_normalized_rms"][1] is None
    assert result["nodes"][0]["live_alerts"][0]["source_round"] == 0
    assert result["nodes"][0]["live_alerts"][0]["detection_round"] == 2
    output = tmp_path / "synthetic-live-warning.png"
    render_run(
        {**result, "fault_start_round": 0}, output, title="Synthetic local verification"
    )
    assert output.read_bytes().startswith(b"\x89PNG")
    saved = json.loads(output.with_suffix(".json").read_text())
    assert saved["nodes"][1]["missing_round_ids"] == [1]
    corrupt = json.loads(summaries[0].read_text())
    corrupt["node_public_key"] = "unknown-peer"
    summaries[0].write_text(json.dumps(corrupt))
    with pytest.raises(ValueError, match="identity"):
        collect_node_curves(manifest, logs, summaries)
