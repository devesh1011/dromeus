"""The published example draft must stay valid and match the example task."""

from __future__ import annotations

from support.paths import REPO_ROOT

from dromeus.application import load_draft, require_task_matches
from dromeus.node import load_node_config
from examples.custom_training import task

EXAMPLES = REPO_ROOT / "examples"


def test_example_draft_matches_example_task() -> None:
    draft = load_draft(EXAMPLES / "custom-regression-draft.yaml")
    require_task_matches(draft, task())


def test_example_node_config_loads_and_points_at_example_draft() -> None:
    config = load_node_config(EXAMPLES / "node.yaml")

    assert config.draft_path.as_posix() == "examples/custom-regression-draft.yaml"
