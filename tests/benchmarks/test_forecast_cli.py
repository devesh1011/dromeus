"""Forecast CLI boundaries: immutable inputs and complete published reports."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _config(path: Path) -> bytes:
    source = json.dumps(
        {
            "nodes": [{"node_id": "node-a", "sample_count": 16, "prior_history": []}],
            "batch_size": 4,
            "noise_multipliers": [1.0],
            "logical_step_horizons": [4],
            "repetitions": 1,
            "delta": 0.00001,
            "delta_rationale": "Illustrative public workload only.",
            "randomness_profile": "reproducible_public_benchmark",
        },
        indent=2,
    ).encode()
    path.write_bytes(source)
    return source


def _run(config: Path, output: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.m3.forecast_cli",
            "--config",
            str(config),
            "--output",
            str(output),
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_cli_publishes_complete_report_without_changing_input(tmp_path: Path) -> None:
    pytest.importorskip("opacus")
    config = tmp_path / "plan.json"
    original = _config(config)
    output = tmp_path / "result.json"
    result = _run(config, output)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["input_file_sha256"] == hashlib.sha256(original).hexdigest()
    assert len(report["input_hash"]) == 64
    assert report["generated_at"].endswith("+00:00")
    assert report["versions"]["opacus"] == "1.6.0"
    assert report["source_sha256"]
    assert report["nodes"][0]["candidates"][0]["one_run"]["prv_epsilon"] > 0
    assert config.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["plan.json", "result.json"]


def test_cli_refuses_existing_output_and_input_alias(tmp_path: Path) -> None:
    config = tmp_path / "plan.json"
    original = _config(config)
    output = tmp_path / "result.json"
    output.write_bytes(b"earlier evidence")
    assert _run(config, output).returncode != 0
    assert _run(config, config).returncode != 0
    assert config.read_bytes() == original
    assert output.read_bytes() == b"earlier evidence"
    output.unlink()
    output.symlink_to(config)
    assert _run(config, output).returncode != 0
    assert output.is_symlink()
    assert config.read_bytes() == original


def test_cli_invalid_config_and_oversize_input_publish_nothing(tmp_path: Path) -> None:
    config = tmp_path / "plan.json"
    output = tmp_path / "result.json"
    for source in (b"{}", b"{" * (1024 * 1024 + 1)):
        config.write_bytes(source)
        assert _run(config, output).returncode != 0
        assert config.read_bytes() == source
        assert list(tmp_path.iterdir()) == [config]


def test_cli_deadline_failure_leaves_no_partial_artifact(tmp_path: Path) -> None:
    config = tmp_path / "plan.json"
    original = _config(config)
    output = tmp_path / "result.json"
    result = _run(config, output, "--timeout-seconds", "0.000000001")
    assert result.returncode != 0
    assert "deadline" in result.stderr
    assert config.read_bytes() == original
    assert list(tmp_path.iterdir()) == [config]


def test_cli_rejects_unbounded_deadlines(tmp_path: Path) -> None:
    config = tmp_path / "plan.json"
    _config(config)
    output = tmp_path / "result.json"
    for value in ("0", "-1", "nan", "inf", "3601"):
        assert _run(config, output, "--timeout-seconds", value).returncode != 0
        assert list(tmp_path.iterdir()) == [config]


def test_cli_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    config = tmp_path / "plan.json"
    source = _config(config)
    output = tmp_path / "result.json"
    for payload in (
        source.replace(b'"repetitions": 1', b'"repetitions": 3, "repetitions": 1'),
        source.replace(
            b'"sample_count": 16', b'"sample_count": 32, "sample_count": 16'
        ),
    ):
        config.write_bytes(payload)
        result = _run(config, output)
        assert result.returncode != 0
        assert "duplicate" in result.stderr
        assert list(tmp_path.iterdir()) == [config]
