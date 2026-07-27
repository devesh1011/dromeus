from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from support.sample_manifest import manifest_data

from benchmarks.cifar10.official import (
    OfficialBenchmarkError,
    load_frozen_benchmark_plan,
)
from dromeus.manifests.models import DraftRunSpec


def _write_plan(root: Path) -> Path:
    pilot = root / "pilot.json"
    pilot.write_text("{}\n", encoding="utf-8")
    path = root / "benchmark.yaml"
    path.write_text(
        "\n".join(
            (
                "benchmark_seeds: [17, 23, 29]",
                "partition_seed: 7",
                "local_steps: 5",
                "round_count: 100",
                "learning_rate: 0.1",
                f"pilot_artifact: {pilot}",
            )
        ),
        encoding="utf-8",
    )
    return path


def test_frozen_plan_requires_pilot_and_builds_three_fedavg_configs(
    tmp_path: Path,
) -> None:
    path = _write_plan(tmp_path)
    plan = load_frozen_benchmark_plan(path)

    assert plan.benchmark_seeds == (17, 23, 29)
    assert [config.trainer_seed for config in plan.fedavg_configs()] == [17, 23, 29]
    assert all(
        config.training_signature() == (5, 100, 0.1, 32, "cpu", True)
        for config in plan.fedavg_configs()
    )

    plan.pilot_artifact.unlink()
    with pytest.raises(OfficialBenchmarkError, match="pilot artifact"):
        load_frozen_benchmark_plan(path)


def test_frozen_plan_rejects_duplicate_seeds(tmp_path: Path) -> None:
    path = _write_plan(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace("[17, 23, 29]", "[17, 17, 29]"),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="distinct"):
        load_frozen_benchmark_plan(path)


def test_frozen_plan_rejects_mismatched_dpsgd_draft(tmp_path: Path) -> None:
    plan = load_frozen_benchmark_plan(_write_plan(tmp_path))
    data = manifest_data()
    for field in (
        "draft_hash",
        "participants",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del data[field]
    data["peer_scheduler_seed"] = 17
    draft = DraftRunSpec.model_validate(data)

    plan.validate_dpsgd_draft(draft, seed=17)

    mismatched = draft.model_copy(update={"round_count": 99})
    with pytest.raises(OfficialBenchmarkError, match="frozen configuration"):
        plan.validate_dpsgd_draft(mismatched, seed=17)
