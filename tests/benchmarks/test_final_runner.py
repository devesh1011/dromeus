import json
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.m3.final_runner import FrozenRunSpec, RunWorkload
from benchmarks.m3.matched_digits import SOURCES
from dromeus.manifests.canonical import file_sha256

pytest.importorskip("opacus")


def _fixture(tmp_path: Path, arm: str = "baseline") -> FrozenRunSpec:
    for name in SOURCES:
        for partition in ("train", "calibration"):
            np.savez(
                tmp_path / f"{name}-{partition}.npz",
                inputs=np.zeros((4, 1, 32, 32), dtype=np.float32),
                labels=np.arange(4, dtype=np.int64),
            )
    manifest = tmp_path / "dataset-provenance.json"
    manifest.write_text(
        json.dumps(
            {
                "subsets": {
                    source: {
                        partition: {
                            "file": f"{source}-{partition}.npz",
                            "sha256": file_sha256(
                                tmp_path / f"{source}-{partition}.npz"
                            ),
                        }
                        for partition in ("train", "calibration")
                    }
                    for source in SOURCES
                }
            }
        )
    )
    return FrozenRunSpec.model_validate(
        {
            "run_id": "final-test-" + arm,
            "source_commit": "a" * 40,
            "torch_version": torch.__version__,
            "arm": arm,
            "split_manifest_sha256": file_sha256(manifest),
            "input_sha256": {p.name: file_sha256(p) for p in tmp_path.glob("*.npz")},
            "steps": 2,
            "local_steps": 1,
            "batch_size": 4,
            "train_count": 4,
            "evaluation_count": 4,
            "evaluation_partition": "calibration",
        }
    )


def test_baseline_uses_only_declared_partition_and_exact_horizon(
    tmp_path: Path,
) -> None:
    spec = _fixture(tmp_path)
    workload = RunWorkload(spec, data=tmp_path, rank=0, device="cpu")
    assert workload.trainer.completed_steps == 0
    workload.trainer.train_local_steps(2)
    assert workload.trainer.completed_steps == 2
    assert workload.trainer.evaluate() is not None
    assert not (tmp_path / "mnist-final.npz").exists()
    with pytest.raises(ValueError, match="horizon"):
        workload.trainer.train_local_steps(1)


def test_rejects_changed_inputs_before_training(tmp_path: Path) -> None:
    spec = _fixture(tmp_path)
    (tmp_path / "mnist-calibration.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        RunWorkload(spec, data=tmp_path, rank=0, device="cpu")


def test_private_fault_changes_only_sanitized_weights_and_not_accounting(
    tmp_path: Path,
) -> None:
    spec = _fixture(tmp_path, "private")
    spec = FrozenRunSpec.model_validate(
        {
            **spec.model_dump(),
            "fault": {"start_round": 0, "scale": 0.05, "growth": 1.15},
        }
    )
    workload = RunWorkload(
        spec,
        data=tmp_path,
        rank=0,
        device="cpu",
        ledger_path=tmp_path / "ledger.json",
        create_ledger=True,
    )
    trainer = workload.prepared.private_trainer
    assert trainer is not None
    trainer.train_local_steps(1)
    before = trainer.weights()
    history = trainer.accounting_history
    workload.apply_fault()
    assert trainer.accounting_history == history
    assert trainer.completed_steps == 1
    for name, value in trainer.weights().items():
        np.testing.assert_allclose(value, before[name] + np.float32(0.05))
    assert workload.injections == [{"round_id": 0, "amplitude": 0.05}]
    healthy = FrozenRunSpec.model_validate({**spec.model_dump(), "fault": None})
    assert spec.task_definition != healthy.task_definition


@pytest.mark.parametrize("arm", ["baseline", "clipped", "private"])
def test_local_execution_exports_model_and_complete_counts(
    tmp_path: Path, arm: str
) -> None:
    from benchmarks.m3.final_runner import execute

    spec = _fixture(tmp_path, arm)
    result = execute(
        spec,
        data=tmp_path,
        root=tmp_path / "run",
        rank=0,
        mode="local",
        ledger_path=tmp_path / "ledger.json",
        create_ledger=True,
    )
    assert result["logical_steps"] == 2
    assert result["event_delivery_dropped"] is None
    assert result["node_public_key"] is None
    assert (tmp_path / "run" / "model.safetensors").is_file()
    assert (tmp_path / "run" / "summary.json").is_file()
    if arm == "private":
        assert result["accounting_history"] == ((1.0, 1.0, 2),)


def test_input_hashes_must_match_frozen_manifest(tmp_path: Path) -> None:
    spec = _fixture(tmp_path)
    hashes = dict(spec.input_sha256)
    hashes["usps-train.npz"] = "b" * 64
    spec = FrozenRunSpec.model_validate({**spec.model_dump(), "input_sha256": hashes})
    with pytest.raises(ValueError, match="manifest input bindings"):
        RunWorkload(spec, data=tmp_path, rank=0, device="cpu")


def test_lossy_profile_uses_accepted_bitmap_pair(tmp_path: Path) -> None:
    from benchmarks.m3.final_runner import reference_draft

    spec = _fixture(tmp_path)
    spec = FrozenRunSpec.model_validate({**spec.model_dump(), "codecs": "lossy"})
    draft = reference_draft(spec)
    assert draft.artifact_codecs is not None
    assert [codec.codec_id for codec in draft.artifact_codecs] == [
        "topk-bitmap-int8-v2",
        "dense-int8-v1",
    ]
    assert draft.artifact_codecs[0].top_k_fraction == 0.45


def test_unequal_training_counts_preserve_model_and_use_local_sampling(
    tmp_path: Path,
) -> None:
    from benchmarks.m3.final_runner import execute

    spec = _fixture(tmp_path, "private")
    counts = (4, 6, 8, 10)
    for source, count in zip(SOURCES, counts, strict=True):
        np.savez(
            tmp_path / f"{source}-train.npz",
            inputs=np.zeros((count, 1, 32, 32), dtype=np.float32),
            labels=np.arange(count, dtype=np.int64) % 4,
        )
    manifest_path = tmp_path / "dataset-provenance.json"
    provenance = json.loads(manifest_path.read_text())
    hashes = {p.name: file_sha256(p) for p in tmp_path.glob("*.npz")}
    for source in SOURCES:
        provenance["subsets"][source]["train"]["sha256"] = hashes[f"{source}-train.npz"]
    manifest_path.write_text(json.dumps(provenance))
    spec = FrozenRunSpec.model_validate(
        {
            **spec.model_dump(),
            "train_counts": counts,
            "input_sha256": hashes,
            "split_manifest_sha256": file_sha256(manifest_path),
        }
    )
    left = RunWorkload(
        spec,
        data=tmp_path,
        rank=0,
        device="cpu",
        ledger_path=tmp_path / "left-ledger.json",
        create_ledger=True,
    )
    right = RunWorkload(
        spec,
        data=tmp_path,
        rank=1,
        device="cpu",
        ledger_path=tmp_path / "right-ledger.json",
        create_ledger=True,
    )
    for name, value in left.trainer.weights().items():
        np.testing.assert_array_equal(value, right.trainer.weights()[name])
    result = execute(
        spec,
        data=tmp_path,
        root=tmp_path / "run",
        rank=1,
        mode="local",
        ledger_path=tmp_path / "run-ledger.json",
        create_ledger=True,
    )
    assert result["training_records"] == 6
    assert result["sample_rate"] == 0.5
    assert result["expected_batch_size"] == 3
    assert result["accounting_history"] == ((1.0, 0.5, 2),)
    assert (
        spec.task_definition
        != FrozenRunSpec.model_validate(
            {**spec.model_dump(), "train_counts": None}
        ).task_definition
    )
