"""Internal Dromeus node entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dromeus.telemetry.events import emit_event


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Dromeus node")
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config: Path = args.config
    if not config.is_file():
        _parser().error(f"config file does not exist: {config}")

    emit_event("node_start", config=str(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
