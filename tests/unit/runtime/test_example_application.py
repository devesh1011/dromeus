"""The published example draft must stay valid and match the example application."""

from __future__ import annotations

from support.paths import REPO_ROOT

from dromeus.application import definition_hash
from dromeus.manifests.canonical import parse_draft_yaml
from dromeus.node import load_node_config
from examples.custom_training import (
    MODEL_DEFINITION,
    TASK_DEFINITION,
    TRAINING_DEFINITION,
)

EXAMPLES = REPO_ROOT / "examples"


def test_example_draft_matches_example_application_definitions() -> None:
    draft = parse_draft_yaml(EXAMPLES / "custom-regression-draft.yaml")

    assert draft.model_definition_hash == definition_hash(MODEL_DEFINITION)
    assert draft.dataset.definition_hash == definition_hash(TASK_DEFINITION)
    assert draft.application_training.definition_hash == definition_hash(
        TRAINING_DEFINITION
    )


def test_example_node_config_loads_and_points_at_example_draft() -> None:
    config = load_node_config(EXAMPLES / "node.yaml")

    assert config.draft_path.as_posix() == "examples/custom-regression-draft.yaml"
