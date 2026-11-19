"""Put one staged AXL build into a platform wheel.

`scripts/build_axl_wheels.py` cross-compiles AXL into a staging directory and
builds this package with `DROMEUS_AXL_STAGE` pointing at it. The staging
directory's `build.json` names the wheel's platform tag. There is no sdist: a
source install would have no binary.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class AXLBinaryHook(BuildHookInterface):  # type: ignore[type-arg]
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if self.target_name != "wheel":
            raise RuntimeError("dromeus-axl is published as platform wheels only")
        stage_env = os.environ.get("DROMEUS_AXL_STAGE")
        if not stage_env:
            raise RuntimeError(
                "build dromeus-axl with scripts/build_axl_wheels.py "
                "(DROMEUS_AXL_STAGE is not set)"
            )
        stage = Path(stage_env).resolve()
        record = json.loads((stage / "build.json").read_text())
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{record['platform_tag']}"
        build_data["force_include"][str(stage / "axl")] = "dromeus_axl/bin/axl"
        build_data["force_include"][str(stage / "build.json")] = (
            "dromeus_axl/build.json"
        )
        build_data["force_include"][str(stage / "THIRD_PARTY_LICENSES.txt")] = (
            "dromeus_axl/THIRD_PARTY_LICENSES.txt"
        )
