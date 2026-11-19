"""Run the quickstart guide's commands as written, the way a reader would.

The guide's first block (clone and bootstrap) is replaced by a copy of
`examples/` and a link to this test's virtual environment. Every later bash
block runs unchanged, in order, in one shell.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from support.local_axl import AXL_SKIP_REASON, axl_binary
from support.paths import REPO_ROOT

pytestmark = pytest.mark.skipif(axl_binary() is None, reason=AXL_SKIP_REASON)

GUIDE = REPO_ROOT / "guides" / "quickstart.md"
_BASH_BLOCK = re.compile(r"^```bash\n(.*?)^```$", re.MULTILINE | re.DOTALL)
_COMPLETE = re.compile(
    r"^run quickstart complete: 20 rounds committed, manifest ([0-9a-f]{64})$",
    re.MULTILINE,
)


def quickstart_blocks() -> list[str]:
    blocks = _BASH_BLOCK.findall(GUIDE.read_text())
    assert "./scripts/bootstrap" in blocks[0], "the first block should install"
    return blocks[1:]


def test_quickstart_commands_train_four_nodes(tmp_path: Path) -> None:
    shutil.copytree(
        REPO_ROOT / "examples",
        tmp_path / "examples",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (tmp_path / ".venv").symlink_to(Path(sys.prefix), target_is_directory=True)
    script = "\n".join(quickstart_blocks())
    environment = {
        key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"
    }
    process = subprocess.Popen(
        ["bash", "-c", script],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=600)
    finally:
        # The guide's own last step stops AXL; this also covers a failed run.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):  # gone, or only zombies (macOS)
            pass
        process.wait()

    logs = "\n".join(
        f"--- {path.name}\n{path.read_text()[-2000:]}"
        for path in sorted((tmp_path / "runs" / "quickstart").glob("*.log"))
    )
    context = f"{output}\n{logs}"
    assert process.returncode == 0, context
    assert "connected to 3 peer(s)" in output, context
    manifests = _COMPLETE.findall(output)
    assert len(manifests) == 4, context
    assert len(set(manifests)) == 1, context
