"""Validate the immutable GPU runtime before NoLoCo benchmark execution."""

from __future__ import annotations

import json
import platform
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib import import_module
from typing import cast


@dataclass(frozen=True, slots=True)
class RuntimeProbe:
    python: str
    torch: str
    cuda: str | None
    cudnn: int | None
    nccl: str | None
    cuda_available: bool
    device_name: str


def validate_probe(probe: RuntimeProbe) -> RuntimeProbe:
    """Reject any runtime that differs from the frozen GPU container contract."""
    if probe.python != "3.12.11":
        raise RuntimeError("Python version does not match frozen container")
    if probe.torch != "2.12.1+cu130":
        raise RuntimeError("Torch version does not match frozen container")
    if not probe.cuda_available or probe.cuda != "13.0":
        raise RuntimeError("CUDA runtime is unavailable or differs from frozen value")
    if probe.cudnn != 92000:
        raise RuntimeError("cuDNN version does not match frozen container")
    if probe.nccl != "2.29.7":
        raise RuntimeError("NCCL version does not match frozen container")
    if "A10G" not in probe.device_name.upper():
        raise RuntimeError("GPU is not the frozen NVIDIA A10G class")
    return probe


def probe_runtime() -> RuntimeProbe:
    import torch

    nccl_module = import_module("torch.cuda.nccl")
    nccl_version = cast(
        Callable[[], tuple[int, ...] | None],
        getattr(nccl_module, "version"),
    )()
    return RuntimeProbe(
        python=platform.python_version(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        nccl=(
            ".".join(str(part) for part in nccl_version)
            if nccl_version is not None
            else None
        ),
        cuda_available=torch.cuda.is_available(),
        device_name=(
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        ),
    )


def main() -> int:
    probe = validate_probe(probe_runtime())
    print(json.dumps(asdict(probe), allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["RuntimeProbe", "probe_runtime", "validate_probe"]
