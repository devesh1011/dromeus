"""The AXL node binary, pinned and built for this platform.

AXL is Gensyn's peer-to-peer network node (https://github.com/gensyn-ai/axl).
Dromeus nodes talk to it over HTTP on loopback. This package carries one pinned
build of it, redistributed with Gensyn's permission, so that `pip install
dromeus` brings a working AXL without Go or a source checkout.
"""

from __future__ import annotations

import json
import os
import sys
from importlib.resources import files
from pathlib import Path
from typing import cast

AXL_REPOSITORY = "https://github.com/gensyn-ai/axl"
AXL_COMMIT = "628e28ace077f26dfe8d0259009b357216a9d8d4"

__all__ = ["AXL_COMMIT", "AXL_REPOSITORY", "binary", "build_info", "main"]


def _packaged(name: str) -> Path:
    path = Path(str(files(__name__).joinpath(name)))
    if not path.is_file():
        raise FileNotFoundError(
            f"dromeus-axl has no {name}: it was installed from source rather than "
            "a platform wheel. Reinstall it from PyPI on a supported platform."
        )
    return path


def binary() -> Path:
    """Path of the bundled AXL node executable."""
    return _packaged("bin/axl")


def build_info() -> dict[str, str]:
    """How the bundled binary was built: AXL commit, Go version, target, sha256."""
    record = json.loads(_packaged("build.json").read_text())
    return cast(dict[str, str], record)


def main() -> None:
    """Run the bundled AXL node with this command's arguments (`-config FILE`)."""
    path = binary()
    os.execv(path, [str(path), *sys.argv[1:]])
