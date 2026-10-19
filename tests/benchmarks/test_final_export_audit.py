"""Independent CPU model exports are scored without changing GPU evidence."""

from pathlib import Path

import numpy as np
import torch

from benchmarks.m3.final_export_audit import evaluate_export
from benchmarks.m3.heterogeneous_data import SOURCES
from benchmarks.m3.matched_digits import model_factory
from dromeus.manifests.canonical import save_safetensors


def test_fresh_cpu_export_returns_exact_correct_counts(tmp_path: Path) -> None:
    model = model_factory("cpu", 17)
    weights = {
        name: np.zeros(tuple(value.shape), dtype=np.float32)
        for name, value in model.state_dict().items()
    }
    path = tmp_path / "model.safetensors"
    save_safetensors(weights, str(path))
    holdouts = {
        source: (torch.zeros(10, 1, 32, 32), torch.arange(10)) for source in SOURCES
    }
    # All-zero classifier logits predict class 0; exactly one label is 0.
    assert evaluate_export(path, holdouts=holdouts, seed=17) == {
        source: 1 for source in SOURCES
    }


def test_saved_accuracy_must_be_an_integer_count() -> None:
    import pytest

    from benchmarks.m3.final_export_audit import accuracy_count

    assert accuracy_count(235 / 470, 470) == 235
    for value in (0.33333, float("nan"), -0.1, 1.1, True):
        with pytest.raises(ValueError):
            accuracy_count(value, 470)


def test_cpu_export_rejects_nonfinite_and_unexpected_model_state(
    tmp_path: Path,
) -> None:
    import pytest

    model = model_factory("cpu", 17)
    weights = {
        name: np.zeros(tuple(value.shape), dtype=np.float32)
        for name, value in model.state_dict().items()
    }
    holdouts = {
        source: (torch.zeros(10, 1, 32, 32), torch.arange(10)) for source in SOURCES
    }
    path = tmp_path / "model.safetensors"
    weights[next(iter(weights))].flat[0] = np.nan
    save_safetensors(weights, str(path))
    with pytest.raises(ValueError, match="invalid model"):
        evaluate_export(path, holdouts=holdouts, seed=17)
    weights[next(iter(weights))].flat[0] = 0
    weights["optimizer.secret"] = np.zeros(1, dtype=np.float32)
    save_safetensors(weights, str(path))
    with pytest.raises(RuntimeError, match="Unexpected key"):
        evaluate_export(path, holdouts=holdouts, seed=17)


def test_summary_json_accepts_integer_train_counts_and_rejects_coercion() -> None:
    import json

    import pytest

    from benchmarks.m3.final_export_audit import validate_export_summary
    from benchmarks.m3.final_runner import FrozenRunSpec
    from dromeus.manifests.canonical import canonical_hash

    spec = FrozenRunSpec(
        run_id="robustness-json",
        source_commit="a" * 40,
        torch_version=str(torch.__version__),
        arm="private",
        split_manifest_sha256="b" * 64,
        evaluation_partition="final",
        evaluation_count=470,
        train_counts=(3000, 2400, 1800, 1200),
        input_sha256={
            f"{source}-{split}.npz": "c" * 64
            for source in SOURCES
            for split in ("train", "final")
        },
    )
    summary = json.loads(
        json.dumps(
            {
                "rank": 1,
                "source": "usps",
                "run_config": spec.model_dump(mode="json"),
                "run_config_hash": canonical_hash(spec),
                "model_sha256": "d" * 64,
                "history": [
                    {
                        "logical_steps": 750,
                        **{f"accuracy_{source}": 0.5 for source in SOURCES},
                    }
                ],
            }
        )
    )
    assert validate_export_summary(
        summary, spec, rank=1, model_sha256="d" * 64
    ) == dict.fromkeys(SOURCES, 235)
    summary["run_config"]["train_counts"][1] = "2400"
    with pytest.raises(ValueError):
        validate_export_summary(summary, spec, rank=1, model_sha256="d" * 64)
