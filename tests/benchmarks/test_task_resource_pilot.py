"""Bounded resource measurements use the real generic private training path."""

import json
from pathlib import Path

import pytest

from benchmarks.m3.task_resource_pilot import run_pilot

pytest.importorskip("opacus")


def test_tiny_resource_probe_records_real_steps_and_codec_sizes(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    report = run_pilot(
        root,
        device="cpu",
        model_size="tiny",
        steps=2,
        secure=False,
        measure_codecs=True,
    )
    assert report["status"] == "complete"
    # 16x8 token + 4x8 position + one 872-parameter block + final LN(16).
    assert report["unique_parameter_count"] == 1048
    assert report["model_config"]["block_size"] == 4
    assert report["logical_steps"] == 2
    assert report["sample_rate"] == 0.25
    assert report["accounting_history"] == [[1.0, 0.25, 2]]
    assert report["expected_training_tokens"] == 16
    assert report["actual_training_tokens"] == report["actual_sampled_records"] * 4
    assert report["preparation_dataset_reads"] == 0
    assert report["host_peak_rss_bytes"] > 0
    assert report["gpu_peak_allocated_bytes"] is None
    codecs = report["codec_measurements"]
    assert {row["codec"] for row in codecs} == {
        "identity-v1",
        "dense-int8-v1",
        "topk-bitmap-int8-v2",
    }
    assert all(row["serialized_bytes"] > 0 for row in codecs)
    assert not list(root.glob("*.safetensors"))
    assert not (root / "failure.json").exists()
    assert (root / "resource.json").exists()
    with pytest.raises(FileExistsError):
        run_pilot(root, device="cpu", model_size="tiny", steps=2, secure=False)


def test_invalid_resource_profile_retains_only_bounded_failure_metadata(
    tmp_path: Path,
) -> None:
    root = tmp_path / "invalid-profile"
    with pytest.raises(ValueError, match="sequence length"):
        run_pilot(root, model_size="tiny", sequence_length=99)
    failure = json.loads((root / "failure.json").read_text())
    assert failure["stage"] == "configuration"
    assert failure["exception_type"] == "ValueError"
    assert failure["automatic_retry"] is False
    assert set(failure) == {
        "status",
        "stage",
        "exception_type",
        "elapsed_seconds",
        "automatic_retry",
    }
    assert not (root / "resource.json").exists()
