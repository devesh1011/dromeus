"""Compare bounded Dromeus/NoLoCo trajectories and delete full tensors."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from benchmarks.noloco.trajectory import compare_trajectories


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dromeus-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--absolute-tolerance", type=float, required=True)
    parser.add_argument("--relative-tolerance", type=float, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = compare_trajectories(
        dromeus_root=arguments.dromeus_root,
        reference_root=arguments.reference_root,
        report_path=arguments.report,
        absolute_tolerance=arguments.absolute_tolerance,
        relative_tolerance=arguments.relative_tolerance,
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
