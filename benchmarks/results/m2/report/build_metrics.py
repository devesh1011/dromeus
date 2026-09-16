"""Build review tables from the accepted-run index without changing evidence."""

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, TypedDict, cast

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent


class MetricRow(TypedDict):
    method: str
    workers: int
    seed: int
    run_id: str
    acceptance_report: str
    mean_accuracy_percent: float
    worker_population_sd_pp: float
    minimum_accuracy_percent: float
    maximum_accuracy_percent: float


def build() -> None:
    sources: dict[str, str] = {}

    def read(relative: str) -> dict[str, Any]:
        path = ROOT / relative
        raw = path.read_bytes()
        sources[relative] = hashlib.sha256(raw).hexdigest()
        return cast(dict[str, Any], json.loads(raw))

    index = read("run-matrix.json")
    rows: list[MetricRow] = []
    for entry in index["runs"]:
        report = read(entry["acceptance_report"])
        assert sources[entry["acceptance_report"]] == entry["acceptance_report_sha256"]
        assert report["run_id"] == entry["run_id"]
        if entry["method"] == "nccl" and entry["workers"] == 8 and entry["seed"] == 17:
            accuracy = read("nccl-reference/8-nodes/seed-17/reports/summary.json")[
                "aggregate"
            ]
        else:
            accuracy = report["accuracy"]

        def value(*names: str) -> float:
            result = next(accuracy[name] for name in names if name in accuracy)
            assert isinstance(result, (int, float)) and math.isfinite(result)
            return result * 100

        row: MetricRow = {
            "method": entry["method"],
            "workers": entry["workers"],
            "seed": entry["seed"],
            "run_id": entry["run_id"],
            "acceptance_report": entry["acceptance_report"],
            "mean_accuracy_percent": value("mean", "mean_final_accuracy"),
            "worker_population_sd_pp": value("stddev", "stddev_final_accuracy"),
            "minimum_accuracy_percent": value(
                "minimum", "min", "minimum_final_accuracy"
            ),
            "maximum_accuracy_percent": value(
                "maximum", "max", "maximum_final_accuracy"
            ),
        }
        assert (
            row["minimum_accuracy_percent"]
            <= row["mean_accuracy_percent"]
            <= row["maximum_accuracy_percent"]
        )
        rows.append(row)
    assert len(rows) == 16
    comparisons: list[dict[str, int | float]] = []
    aggregates: list[dict[str, str | int | float | list[int]]] = []
    for workers in (4, 8, 16):
        for method in ("compressed_axl", "nccl"):
            selected = [
                r for r in rows if r["method"] == method and r["workers"] == workers
            ]
            values = [r["mean_accuracy_percent"] for r in selected]
            aggregates.append(
                {
                    "method": method,
                    "workers": workers,
                    "seeds": [r["seed"] for r in selected],
                    "mean_accuracy_percent": statistics.mean(values),
                    "seed_sample_sd_pp": statistics.stdev(values),
                    "minimum_seed_mean_percent": min(values),
                    "maximum_seed_mean_percent": max(values),
                }
            )
        for seed in (17, 29):
            matched = {
                r["method"]: r
                for r in rows
                if r["workers"] == workers and r["seed"] == seed
            }
            delta = (
                matched["compressed_axl"]["mean_accuracy_percent"]
                - matched["nccl"]["mean_accuracy_percent"]
            )
            assert abs(delta) < 0.38
            comparisons.append(
                {"workers": workers, "seed": seed, "axl_minus_nccl_pp": delta}
            )
    result = {
        "schema_version": 1,
        "runs": rows,
        "matched_comparisons": comparisons,
        "across_seed_summaries": aggregates,
        "sources_sha256": sources,
        "notes": [
            "Per-run SD is population SD across workers; "
            "workers are not independent repetitions.",
            "Across-seed SD is sample SD of worker means (n-1 denominator).",
            "AXL uses seeds 17/29/41; NCCL uses 17/29. No missing result is imputed.",
            "The identity ablation has one seed; no across-seed SD is estimated.",
        ],
    }
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (OUT / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Verified and exported {len(rows)} accepted runs "
        f"and {len(comparisons)} matched comparisons."
    )


if __name__ == "__main__":
    build()
