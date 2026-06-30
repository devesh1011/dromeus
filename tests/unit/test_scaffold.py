import json

import pytest

import dromeus
from dromeus.telemetry.events import emit_event


def test_import_and_structured_event(capsys: pytest.CaptureFixture[str]) -> None:
    assert dromeus.__version__ == "0.1.0"
    emit_event("test", run_id="run-1", round_id=2)

    output = capsys.readouterr().out
    assert json.loads(output)["run_id"] == "run-1"
