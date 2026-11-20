from __future__ import annotations

from pathlib import Path

import pytest

from dromeus.errors import ConfigError
from dromeus.telemetry.liveness import hold_node_lock, node_is_running


def test_node_is_running_only_while_it_holds_the_lock(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    assert not node_is_running(run_root)

    with hold_node_lock(run_root):
        assert node_is_running(run_root)

    assert not node_is_running(run_root)


def test_a_second_node_cannot_use_a_running_nodes_run_root(tmp_path: Path) -> None:
    with hold_node_lock(tmp_path), pytest.raises(ConfigError, match="already running"):
        with hold_node_lock(tmp_path):
            pass

    with hold_node_lock(tmp_path):  # free again once the first node stops
        pass
