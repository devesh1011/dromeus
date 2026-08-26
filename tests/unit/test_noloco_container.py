from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from benchmarks.noloco.container_preflight import RuntimeProbe, validate_probe

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_gpu_container_is_digest_pinned_and_records_noloco_source() -> None:
    dockerfile = (
        REPO_ROOT / "benchmarks" / "noloco" / "container" / "Dockerfile"
    ).read_text(encoding="utf-8")
    lock = json.loads(
        (
            REPO_ROOT
            / "benchmarks"
            / "noloco"
            / "container"
            / "runtime-lock.json"
        ).read_text(encoding="utf-8")
    )
    dockerignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert (
        "@sha256:78a2f9fd8f605301464fc61c42be3acb5a65cd92b0c91f481c3dcdd411dff810"
        in dockerfile
    )
    assert "torch==2.12.1" in dockerfile
    assert "https://download.pytorch.org/whl/cu130" in dockerfile
    assert lock["upstream_noloco"]["commit"] == (
        "a1b4a425bdc4050a356cf9f4bae7c383419703ab"
    )
    assert lock["upstream_noloco"]["source"] == (
        "src/noloco/sparse_optimizer_c.py"
    )
    assert lock["base_image"]["amd64_digest"] == (
        "sha256:78a2f9fd8f605301464fc61c42be3acb5a65cd92b0c91f481c3dcdd411dff810"
    )
    assert lock["status"] == "built-and-gpu-verified"
    assert lock["image"]["digest"] == (
        "sha256:da86b19c764a0f6ccd36d55b532b23f9416766177fe6a444a4d42c019979be1c"
    )
    assert "benchmarks/results/" in dockerignore
    assert "benchmarks/noloco/container/runtime-lock.json" in dockerignore
    assert "benchmarks/noloco/container/build-evidence.json" in dockerignore


def test_gpu_container_preflight_accepts_frozen_runtime() -> None:
    probe = RuntimeProbe(
        python="3.12.11",
        torch="2.12.1+cu130",
        cuda="13.0",
        cudnn=92000,
        nccl="2.29.7",
        cuda_available=True,
        device_name="NVIDIA A10G",
    )

    assert validate_probe(probe) is probe


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("torch", "2.12.0+cu130", "Torch"),
        ("cuda", "12.8", "CUDA"),
        ("cuda_available", False, "CUDA"),
        ("device_name", "NVIDIA T4", "A10G"),
    ),
)
def test_gpu_container_preflight_rejects_runtime_drift(
    field: str,
    value: str | bool,
    message: str,
) -> None:
    probe = RuntimeProbe(
        python="3.12.11",
        torch="2.12.1+cu130",
        cuda="13.0",
        cudnn=92000,
        nccl="2.29.7",
        cuda_available=True,
        device_name="NVIDIA A10G",
    )
    if field == "torch":
        probe = replace(probe, torch=cast(str, value))
    elif field == "cuda":
        probe = replace(probe, cuda=cast(str, value))
    elif field == "cuda_available":
        probe = replace(probe, cuda_available=cast(bool, value))
    elif field == "device_name":
        probe = replace(probe, device_name=cast(str, value))
    else:
        raise AssertionError(f"unsupported drift field: {field}")

    with pytest.raises(RuntimeError, match=message):
        validate_probe(probe)
