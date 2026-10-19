"""Bounded local privacy forecasting with immutable inputs and atomic reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from benchmarks.m3 import privacy_forecast


def _compute(source: bytes, staged: str) -> None:
    """Child owns numerical computation, never the final report path."""
    plan = privacy_forecast.parse_plan(source)
    report = privacy_forecast.forecast(plan)
    report["input_file_sha256"] = hashlib.sha256(source).hexdigest()
    report["generated_at"] = datetime.now(UTC).isoformat()
    root = Path(__file__).resolve().parents[2]
    report["source_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "benchmarks/m3/forecast_cli.py",
            "benchmarks/m3/privacy_forecast.py",
            "src/dromeus/training/privacy.py",
        )
    }
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with Path(staged).open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def run_forecast(config: Path, output: Path, timeout_seconds: float = 120) -> None:
    """Publish a complete report once, or raise with no final output on failure.

    The deadline bounds the spawned accounting process, including its startup.
    Input validation precedes it. Existing outputs (including symlinks) are refused.
    """
    if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 3600:
        raise ValueError("timeout must be finite and in (0, 3600] seconds")
    if output.exists() or output.is_symlink() or output.resolve() == config.resolve():
        raise FileExistsError("forecast output already exists or aliases the input")
    with config.open("rb") as handle:
        source = handle.read(1024 * 1024 + 1)
    if len(source) > 1024 * 1024:
        raise ValueError("forecast input exceeds 1 MiB")
    privacy_forecast.parse_plan(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged = tempfile.mkstemp(
        prefix=".privacy-forecast-", dir=output.parent
    )
    os.close(descriptor)
    process = multiprocessing.get_context("spawn").Process(
        target=_compute,
        args=(source, staged),
    )
    started = False
    try:
        process.start()
        started = True
        process.join(timeout_seconds)
        if process.is_alive():
            raise TimeoutError("privacy forecast exceeded its deadline")
        if process.exitcode != 0:
            raise ValueError("privacy forecast failed; no complete report published")
        # The hardlink publishes complete bytes atomically without replacing output.
        os.link(staged, output)
    finally:
        if started and process.is_alive():
            process.terminate()
            process.join(1)
            if process.is_alive():
                process.kill()
                process.join()
        process.close()
        Path(staged).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    args = parser.parse_args()
    try:
        run_forecast(args.config, args.output, args.timeout_seconds)
    except (ValueError, OSError) as error:
        parser.exit(2, f"privacy forecast: {error}\n")
    print(f"Wrote {args.output}; prospective mechanism estimates only.")


if __name__ == "__main__":
    main()
