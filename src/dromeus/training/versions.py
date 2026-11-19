"""The PyTorch version a group agrees on."""

from __future__ import annotations

import torch


def torch_release() -> str:
    """Installed PyTorch release without its build label.

    Wheels for one release differ only in their local label: `2.13.0+cpu` on
    Linux CPU, `2.13.0+cu130` on CUDA, `2.13.0` on macOS. Nodes on different
    builds of one release train together, so drafts and checks use the release.
    """
    return torch.__version__.split("+", 1)[0]
