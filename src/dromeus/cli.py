"""The `dromeus` command: a thin shell over the library.

Each command parses its arguments, makes one library call and prints the
result. Errors print as one line and map to a stable exit code (`ExitCode`);
`--debug` adds the traceback.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import sys
import traceback
from collections.abc import Callable, Sequence
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, cast

from dromeus.errors import (
    ConfigError,
    DraftError,
    FormationError,
    PrivacyError,
    RunError,
    TaskError,
    one_line,
)

if TYPE_CHECKING:
    from dromeus.application import LocalData
    from dromeus.telemetry.progress import Progress
    from dromeus.training.task import TrainingTask


class ExitCode(IntEnum):
    """Process exit status of every `dromeus` command."""

    OK = 0
    ERROR = 1
    """Unexpected failure: a bug, or an exception from the task's own code."""
    USAGE = 2
    """The command line is malformed."""
    INVALID_INPUT = 3
    """A node config, draft, task or privacy setting is wrong."""
    FORMATION_FAILED = 4
    """The group did not form: invitation, enrollment or readiness failed."""
    RUN_FAILED = 5
    """The run failed after formation."""
    NOT_READY = 6
    """`dromeus doctor` found at least one failing check."""
    INTERRUPTED = 130
    """Stopped with Ctrl-C."""


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    command = cast(Callable[[argparse.Namespace], ExitCode], args.command)
    debug = bool(getattr(args, "debug", False))
    try:
        return int(command(args))
    except KeyboardInterrupt:
        _print_error("interrupted")
        return ExitCode.INTERRUPTED
    except Exception as error:
        if debug:
            traceback.print_exc()
        code = exit_code_for(error)
        message = one_line(str(error)) or type(error).__name__
        if code is ExitCode.ERROR:
            message = f"unexpected {type(error).__name__}: {message}"
            if not debug:
                message += " (rerun with --debug for the traceback)"
        _print_error(message)
        return code


def exit_code_for(error: BaseException) -> ExitCode:
    """Map a failure to its documented exit code."""
    if isinstance(error, FormationError):
        return ExitCode.FORMATION_FAILED
    if isinstance(error, RunError):
        return ExitCode.RUN_FAILED
    if isinstance(error, ConfigError | DraftError | TaskError | PrivacyError):
        return ExitCode.INVALID_INPUT
    return ExitCode.ERROR


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--debug",
        action="store_true",
        default=argparse.SUPPRESS,  # so `dromeus --debug run` and `run --debug` agree
        help="print full tracebacks on errors",
    )
    parser = argparse.ArgumentParser(
        prog="dromeus",
        description="Decentralized federated learning over Gensyn AXL.",
        parents=[common],
    )
    parser.add_argument("--version", action="version", version=_version())
    commands = parser.add_subparsers(title="commands", required=True)
    task_help = (
        "module:callable returning the TrainingTask, or (TrainingTask, LocalData); "
        "the module is imported from the current directory"
    )

    run = commands.add_parser(
        "run", parents=[common], help="join the group and train every round"
    )
    run.add_argument("--config", required=True, type=Path, help="node config YAML")
    run.add_argument(
        "--task",
        required=True,
        help="module:callable returning (TrainingTask, LocalData) for this node",
    )
    run.set_defaults(command=_run)

    check = commands.add_parser(
        "check",
        parents=[common],
        help="check a draft against the task and this machine, offline",
    )
    check.add_argument("--draft", required=True, type=Path, help="shared draft YAML")
    check.add_argument("--task", required=True, help=task_help)
    check.set_defaults(command=_check)

    draft = commands.add_parser(
        "draft",
        parents=[common],
        help="write the group's shared draft for a task",
        description=(
            "Write the shared draft with default NoLoCo, transport and no privacy. "
            "For privacy, transport or divergence settings call make_draft from "
            "Python."
        ),
    )
    draft.add_argument("output", type=Path, help="draft YAML to write")
    draft.add_argument("--task", required=True, help=task_help)
    draft.add_argument("--run-id", required=True)
    draft.add_argument("--participants", required=True, type=int)
    draft.add_argument("--rounds", required=True, type=int)
    draft.add_argument("--local-steps", required=True, type=int)
    draft.add_argument(
        "--axl-version", required=True, help="AXL revision every node runs"
    )
    draft.add_argument("--seed", type=int, default=0)
    draft.set_defaults(command=_draft)

    doctor = commands.add_parser(
        "doctor",
        parents=[common],
        help="check that this machine is ready to run a node",
    )
    doctor.add_argument("--config", required=True, type=Path, help="node config YAML")
    doctor.add_argument("--task", help=f"also check the draft against it; {task_help}")
    doctor.set_defaults(command=_doctor)

    watch = commands.add_parser(
        "watch",
        parents=[common],
        help="follow a node's rounds, loss and consensus distance",
    )
    watch.add_argument(
        "--run-root", required=True, type=Path, help="run_root from the node config"
    )
    watch.add_argument(
        "--once", action="store_true", help="print what is logged so far and exit"
    )
    watch.set_defaults(command=_watch)
    return parser


def _run(args: argparse.Namespace) -> ExitCode:
    from dromeus.node import Node, load_node_config

    config = load_node_config(cast(Path, args.config))
    task, local = load_task(cast(str, args.task))
    if local is None:
        raise ConfigError(
            f"{args.task} must return (TrainingTask, LocalData) to run a node"
        )
    _print_error(f"running; follow with: dromeus watch --run-root {config.run_root}")
    result = asyncio.run(Node(config).run(task, local=local))
    print(
        f"run {result.run_id} complete: {result.committed_rounds} rounds committed, "
        f"manifest {result.manifest_hash}"
    )
    return ExitCode.OK


def _check(args: argparse.Namespace) -> ExitCode:
    from dromeus.application import check_draft, load_draft
    from dromeus.manifests.canonical import canonical_hash

    path = cast(Path, args.draft)
    task, _ = load_task(cast(str, args.task))
    check_draft(path, task)
    draft = load_draft(path)
    print(f"draft {path} matches this task and machine")
    print("  model, task and loss definitions: match")
    print(
        f"  Dromeus {draft.environment.dromeus_version}, "
        f"PyTorch {draft.environment.pytorch_version}: installed"
    )
    print(f"  draft hash: {canonical_hash(draft)}")
    return ExitCode.OK


def _draft(args: argparse.Namespace) -> ExitCode:
    from dromeus.application import make_draft

    task, _ = load_task(cast(str, args.task))
    path = cast(Path, args.output)
    digest = make_draft(
        task,
        path,
        run_id=cast(str, args.run_id),
        participants=cast(int, args.participants),
        rounds=cast(int, args.rounds),
        local_steps=cast(int, args.local_steps),
        axl_version=cast(str, args.axl_version),
        seed=cast(int, args.seed),
    )
    print(f"wrote {path}; draft hash {digest}")
    return ExitCode.OK


def _doctor(args: argparse.Namespace) -> ExitCode:
    from dromeus.diagnostics import Status, diagnose

    reference = cast(str | None, args.task)
    task = None if reference is None else load_task(reference)[0]
    findings = asyncio.run(diagnose(cast(Path, args.config), task=task))
    width = max(len(finding.check) for finding in findings)
    for finding in findings:
        print(f"{finding.status.value:<5} {finding.check:<{width}}  {finding.detail}")
        if finding.fix is not None:
            print(f"{'':<5} {'':<{width}}  fix: {finding.fix}")
    if any(finding.status is Status.FAIL for finding in findings):
        return ExitCode.NOT_READY
    return ExitCode.OK


def _watch(args: argparse.Namespace) -> ExitCode:
    from dromeus.telemetry.events import event_log_path
    from dromeus.telemetry.progress import RunFailed, read_progress

    path = event_log_path(cast(Path, args.run_root))
    code = ExitCode.OK
    for item in read_progress(path, follow=not args.once):
        print(describe_progress(item), flush=True)
        if isinstance(item, RunFailed):
            code = ExitCode.RUN_FAILED
    return code


def describe_progress(item: Progress) -> str:
    """One terminal line for one progress item."""
    from dromeus.telemetry.progress import (
        ConsensusMeasured,
        NodeReady,
        NodeStarted,
        RoundCommitted,
        RunCompleted,
    )

    match item:
        case NodeStarted():
            return f"node started: run {item.run_id} as {item.role}"
        case NodeReady():
            return f"group formed: manifest {item.manifest_hash[:16]}"
        case RoundCommitted():
            loss = "-" if item.local_loss is None else f"{item.local_loss:.4g}"
            metrics = "".join(
                f"  {name} {value:.4g}" for name, value in sorted(item.metrics.items())
            )
            return (
                f"round {item.round_id:>5}  loss {loss}{metrics}  "
                f"{item.round_seconds:.1f}s  peer {item.peer_id[:8]}"
            )
        case ConsensusMeasured():
            return (
                f"round {item.round_id:>5}  consensus distance "
                f"{item.normalized_rms:.4g}"
            )
        case RunCompleted():
            return f"run complete: {item.committed_rounds} rounds committed"
        case _:
            where = "" if item.round_id is None else f" in round {item.round_id}"
            return f"run failed{where}: {item.error}"


def load_task(reference: str) -> tuple[TrainingTask, LocalData | None]:
    """Call a `module:callable` that returns a task, or a task and local data.

    The current directory is importable, as with `python -m`.
    """
    from dromeus.application import LocalData
    from dromeus.training.task import TrainingTask

    module_name, separator, name = reference.partition(":")
    if not separator or not module_name or not name:
        raise ConfigError(f"task must be given as module:callable, got {reference!r}")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name is not None and module_name.startswith(error.name):
            raise ConfigError(
                f"cannot import {module_name!r} from {os.getcwd()}: {error}"
            ) from error
        raise
    setup = getattr(module, name, None)
    if not callable(setup):
        raise ConfigError(f"{reference} is not a callable in {module_name}")
    value: object = setup()
    if isinstance(value, TrainingTask):
        return value, None
    match value:
        case (TrainingTask() as task, LocalData() as local):
            return task, local
        case _:
            raise ConfigError(
                f"{reference} must return a TrainingTask or "
                f"(TrainingTask, LocalData), not {type(value).__name__}"
            )


def _print_error(message: str) -> None:
    print(f"dromeus: {message}", file=sys.stderr, flush=True)


def _version() -> str:
    from importlib.metadata import version

    return f"dromeus {version('dromeus')}"
