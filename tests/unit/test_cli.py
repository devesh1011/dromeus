"""The `dromeus` command: arguments, output and exit codes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from support.paths import REPO_ROOT

from dromeus import cli
from dromeus.cli import ExitCode, exit_code_for, load_task, main
from dromeus.errors import (
    ConfigError,
    DraftError,
    DromeusError,
    FormationError,
    PrivacyError,
    RunError,
    TaskError,
)
from dromeus.telemetry.events import JsonlEventSink, emit_event, event_log_path

EXAMPLE_TASK = "examples.custom_training:task"


@pytest.fixture
def in_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Task references resolve from the current directory, as for users."""
    monkeypatch.chdir(REPO_ROOT)


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ConfigError("x"), ExitCode.INVALID_INPUT),
        (DraftError("x"), ExitCode.INVALID_INPUT),
        (TaskError("x"), ExitCode.INVALID_INPUT),
        (PrivacyError("x"), ExitCode.INVALID_INPUT),
        (FormationError("x"), ExitCode.FORMATION_FAILED),
        (RunError("x"), ExitCode.RUN_FAILED),
        (DromeusError("x"), ExitCode.ERROR),
        (KeyError("x"), ExitCode.ERROR),
    ],
)
def test_errors_map_to_stable_exit_codes(error: Exception, code: ExitCode) -> None:
    assert exit_code_for(error) is code


def test_exit_code_values_are_stable() -> None:
    assert {code.name: code.value for code in ExitCode} == {
        "OK": 0,
        "ERROR": 1,
        "USAGE": 2,
        "INVALID_INPUT": 3,
        "FORMATION_FAILED": 4,
        "RUN_FAILED": 5,
        "NOT_READY": 6,
        "INTERRUPTED": 130,
    }


def _node_config(tmp_path: Path, **overrides: object) -> Path:
    values: dict[str, object] = {
        "role": "initiator",
        "draft_path": str(tmp_path / "draft.yaml"),
        "axl_bridge_url": "http://127.0.0.1:9002",
        "run_root": str(tmp_path / "run"),
        "invitation_path": str(tmp_path / "invitation.json"),
        "bootstrap_uri": "tls://bootstrap.example:9000",
    }
    values.update(overrides)
    path = tmp_path / "node.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


def test_input_errors_print_one_line_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _node_config(tmp_path, extra="rejected")

    code = main(["run", "--config", str(config), "--task", EXAMPLE_TASK])

    err = capsys.readouterr().err
    assert code == ExitCode.INVALID_INPUT
    assert err.startswith("dromeus: invalid node config")
    assert err.count("\n") == 1
    assert "Traceback" not in err


@pytest.mark.parametrize("position", ["before", "after"])
def test_debug_adds_the_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], position: str
) -> None:
    config = _node_config(tmp_path, extra="rejected")
    command = ["run", "--config", str(config), "--task", EXAMPLE_TASK]
    argv = ["--debug", *command] if position == "before" else [*command, "--debug"]

    code = main(argv)

    assert code == ExitCode.INVALID_INPUT
    assert "Traceback" in capsys.readouterr().err


def test_unexpected_errors_name_their_type_and_point_at_debug(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def broken(reference: str) -> object:
        raise KeyError("missing column")

    monkeypatch.setattr(cli, "load_task", broken)

    code = main(["check", "--draft", str(tmp_path / "d.yaml"), "--task", "m:f"])

    err = capsys.readouterr().err
    assert code == ExitCode.ERROR
    assert "unexpected KeyError" in err
    assert "--debug" in err


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ("no_separator", "module:callable"),
        ("missing_module_xyz:setup", "cannot import"),
        ("examples.custom_training:MODEL_DEFINITION", "not a callable"),
        ("examples.custom_training:build_model", "must return a TrainingTask"),
    ],
)
def test_bad_task_references_are_input_errors(
    in_repo: None, reference: str, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        load_task(reference)


def test_task_reference_may_return_task_or_task_and_data(
    in_repo: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    from dromeus import LocalData, TrainingTask

    task, local = load_task(EXAMPLE_TASK)
    assert isinstance(task, TrainingTask)
    assert local is None

    data = tmp_path / "local.npz"
    np.savez(
        data,
        inputs=np.zeros((4, 3), dtype=np.float32),
        targets=np.zeros((4, 1), dtype=np.float32),
    )
    monkeypatch.setenv("DROMEUS_LOCAL_DATA", str(data))
    task, local = load_task("examples.custom_training:setup")
    assert isinstance(task, TrainingTask)
    assert isinstance(local, LocalData)


def test_run_requires_local_data(
    in_repo: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _node_config(tmp_path)

    code = main(["run", "--config", str(config), "--task", EXAMPLE_TASK])

    assert code == ExitCode.INVALID_INPUT
    assert "(TrainingTask, LocalData)" in capsys.readouterr().err


def test_draft_then_check_round_trip(
    in_repo: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    draft = tmp_path / "draft.yaml"

    code = main(
        [
            "draft",
            str(draft),
            "--task",
            EXAMPLE_TASK,
            "--run-id",
            "cli-test",
            "--participants",
            "4",
            "--rounds",
            "3",
            "--local-steps",
            "2",
            "--axl-version",
            "test-axl",
        ]
    )
    assert code == ExitCode.OK
    drafted = capsys.readouterr().out
    assert yaml.safe_load(draft.read_text())["run_id"] == "cli-test"

    code = main(["check", "--draft", str(draft), "--task", EXAMPLE_TASK])

    out = capsys.readouterr().out
    assert code == ExitCode.OK
    assert "matches this task and machine" in out
    digest = drafted.rsplit("draft hash ", 1)[1].strip()
    assert f"draft hash: {digest}" in out


def test_check_names_what_differs(
    in_repo: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    draft = tmp_path / "draft.yaml"
    from examples.custom_training import task

    from dromeus import make_draft

    make_draft(
        task(),
        draft,
        run_id="cli-test",
        participants=4,
        rounds=3,
        local_steps=2,
        axl_version="test-axl",
    )
    values = yaml.safe_load(draft.read_text())
    values["training"]["loss_definition_hash"] = "0" * 64
    draft.write_text(yaml.safe_dump(values))

    code = main(["check", "--draft", str(draft), "--task", EXAMPLE_TASK])

    assert code == ExitCode.INVALID_INPUT
    assert "loss_definition_hash" in capsys.readouterr().err


def _write_log(run_root: Path, *, fail: bool) -> None:
    sink = JsonlEventSink(event_log_path(run_root))
    emit_event("node_start", run_id="r", role="initiator", sink=sink)
    emit_event("node_ready", run_id="r", manifest_hash="a" * 64, sink=sink)
    emit_event(
        "round_metrics",
        run_id="r",
        round_id=0,
        peer_id="b" * 64,
        local_loss=0.25,
        evaluation_metrics={"rmse": 0.5},
        local_compute_seconds=1.0,
        peer_wait_seconds=0.5,
        transfer_seconds=0.25,
        mixing_seconds=0.0,
        evaluation_seconds=0.25,
        sink=sink,
    )
    emit_event("consensus_observation", run_id="r", round_id=0, normalized_rms=0.01,
               sink=sink)
    if fail:
        emit_event("run_failed", run_id="r", round_id=1, error="peer lost", sink=sink)
    else:
        emit_event("node_complete", run_id="r", committed_rounds=1, sink=sink)


def test_watch_prints_rounds_and_completion(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_log(tmp_path, fail=False)

    code = main(["watch", "--run-root", str(tmp_path), "--once"])

    assert code == ExitCode.OK
    assert capsys.readouterr().out.splitlines() == [
        "node started: run r as initiator",
        f"group formed: manifest {'a' * 16}",
        f"round     0  loss 0.25  rmse 0.5  2.0s  peer {'b' * 8}",
        "round     0  consensus distance 0.01",
        "run complete: 1 rounds committed",
    ]


def test_watch_exits_with_run_failed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_log(tmp_path, fail=True)

    code = main(["watch", "--run-root", str(tmp_path), "--once"])

    assert code == ExitCode.RUN_FAILED
    assert capsys.readouterr().out.splitlines()[-1] == (
        "run failed in round 1: peer lost"
    )


def test_watch_without_a_log_is_an_input_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["watch", "--run-root", str(tmp_path), "--once"])

    assert code == ExitCode.INVALID_INPUT
    assert "event log not found" in capsys.readouterr().err


def test_doctor_prints_each_finding_and_fails_when_one_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dromeus import diagnostics
    from dromeus.diagnostics import Finding, Status

    async def fake_diagnose(config_path: Path, **kwargs: object) -> list[Finding]:
        return [
            Finding("config", Status.OK, "loaded"),
            Finding("peers", Status.FAIL, "0 peer links up", fix="check Peers"),
        ]

    monkeypatch.setattr(diagnostics, "diagnose", fake_diagnose)

    code = main(["doctor", "--config", str(tmp_path / "node.yaml")])

    assert code == ExitCode.NOT_READY
    assert capsys.readouterr().out.splitlines() == [
        "ok    config  loaded",
        "fail  peers   0 peer links up",
        "              fix: check Peers",
    ]


def test_python_dash_m_dromeus_runs_the_command() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dromeus", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.startswith("dromeus ")
