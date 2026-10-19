"""Actual public-digit resource harness, including its report boundaries."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from benchmarks.m3.dp_resource_pilot import execute
from dromeus.manifests.canonical import file_sha256

pytest.importorskip("opacus")


def test_cpu_resource_measurement_reconciles_arms(tmp_path: Path) -> None:
    # Only this node's source exists: resource probing must not load other data.
    data = tmp_path / "data"
    data.mkdir()
    path = data / "usps-train.npz"
    rng = np.random.default_rng(4)
    np.savez(
        path,
        inputs=rng.random((65, 1, 32, 32), dtype=np.float32),
        labels=np.arange(65, dtype=np.int64) % 10,
    )
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        report = execute(
            data=data,
            root=tmp_path / "result",
            rank=1,
            device="cpu",
            horizon=4,
            warmup=1,
            steps=1,
            physical_batches=(16, 32),
        )
    finally:
        torch.set_num_threads(old_threads)
    saved = json.loads((tmp_path / "result/resource-pilot.json").read_text())
    assert saved["training_data_sha256"] == file_sha256(path)
    assert saved["sample_count"] == 65
    assert saved["source"] == "usps"
    arms: list[dict[str, Any]] = report["arms"]
    assert len(arms) == 4
    assert len({a["initial_weights_sha256"] for a in arms}) == 1
    assert {a["sample_rate"] for a in arms} == {0.5}
    assert {a["expected_batch_size"] for a in arms} == {32}
    for arm in arms:
        assert arm["logical_steps"] == 2
        assert arm["measured_logical_steps"] == 1
        assert arm["measured_physical_steps"] >= 1
        assert arm["measured_seconds"] > 0
        assert arm["measured_cuda_peak"] == {
            "allocated_bytes": None,
            "reserved_bytes": None,
        }
        if arm["arm"] == "public_seeded_dp_diagnostic":
            assert arm["accounting"]["logical_steps"] == 2
            assert arm["accounting"]["reserved_steps"] == 4
            assert arm["accounting"]["privacy_epsilon"] > 0
            assert arm["accounting_history"] == ((1.0, 0.5, 2),)
            assert len(arm["reservations"]) == 1
            assert arm["reservations"][0]["logical_steps"] == 4
        else:
            assert arm["epsilon"] is None
            assert arm["accounting"] is None
            assert arm["reservations"] == []
    assert not list((tmp_path / "result").rglob("*.safetensors"))
    assert not list((tmp_path / "result").rglob("*.pt"))
    with pytest.raises(ValueError, match="fresh"):
        execute(data=data, root=tmp_path / "result", rank=1, device="cpu")


@pytest.mark.parametrize(
    "options",
    [
        {"rank": 4},
        {"steps": 0},
        {"warmup": -1},
        {"horizon": 1},
        {"physical_batches": ()},
        {"physical_batches": (16, 16)},
        {"physical_batches": (0,)},
        {"device": "mps"},
    ],
)
def test_invalid_plan_does_not_create_output(
    tmp_path: Path, options: dict[str, Any]
) -> None:
    kwargs: dict[str, Any] = {
        "data": tmp_path,
        "root": tmp_path / "out",
        "rank": 0,
        "device": "cpu",
        **options,
    }
    with pytest.raises(ValueError):
        execute(**kwargs)
    assert not (tmp_path / "out").exists()


def test_cuda_unavailable_fails_without_cpu_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA requested but unavailable"):
        execute(data=tmp_path, root=tmp_path / "out", rank=0)
    assert not (tmp_path / "out").exists()
