"""Custom nanoGPT scripts train on separate records through four actual AXL nodes."""

import asyncio
import json
import os
from pathlib import Path

import pytest
from support.application_fixture import application_draft

from benchmarks.m3.generic_task_smoke import audit_smoke, run_smoke
from dromeus.manifests.models import DraftRunSpec, TaskTrainingSpec

pytestmark = pytest.mark.skipif(
    not os.environ.get("DROMEUS_PRIVATE_AXL_BINARY"),
    reason="set DROMEUS_PRIVATE_AXL_BINARY for real generic-task AXL conformance",
)


@pytest.mark.parametrize("private,compressed", [(False, False), (True, True)])
def test_four_generic_script_processes_over_real_axl(
    tmp_path: Path, private: bool, compressed: bool
) -> None:
    evidence = os.environ.get("DROMEUS_GENERIC_EVIDENCE_ROOT")
    receipt = asyncio.run(
        run_smoke(
            root=tmp_path,
            binary=Path(os.environ["DROMEUS_PRIVATE_AXL_BINARY"]),
            template=application_draft(compressed=compressed),
            private=private,
            evidence_root=Path(evidence) if evidence else None,
        )
    )
    assert receipt["status"] == "passed"
    assert receipt["independent_python_processes"] == 4
    assert receipt["private"] is private
    assert receipt["logical_steps_per_node"] == 6
    assert receipt["shared_manifest"] is True
    assert receipt["tied_weights_preserved"] is True
    assert receipt["distinct_local_datasets"] is True

    summary_path = tmp_path / "node-summary-0.json"
    original = summary_path.read_text()
    summary = json.loads(original)
    summary["event_delivery_dropped"] = 1
    summary_path.write_text(json.dumps(summary))
    try:
        with pytest.raises(ValueError, match="dropped"):
            audit_smoke(
                tmp_path,
                Path(os.environ["DROMEUS_PRIVATE_AXL_BINARY"]),
                DraftRunSpec.model_validate_json((tmp_path / "draft.json").read_text()),
                TaskTrainingSpec.model_validate(
                    json.loads((tmp_path / "task.json").read_text())["specification"]
                ),
                private=private,
            )
    finally:
        summary_path.write_text(original)
