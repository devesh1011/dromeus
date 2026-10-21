"""Offline public-token evaluation binds every score to retained model bytes."""

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.m3.task_export_audit import audit_exports
from dromeus.application import describe_task
from dromeus.manifests.canonical import load_safetensors, save_safetensors
from dromeus.manifests.models import PrivateOptimizerSpec
from examples.nanogpt_training import GPTConfig, MMapTokenDataset, build_task


def fixture(root: Path) -> tuple[Path, Path, list[Path]]:
    config = GPTConfig(block_size=2, vocab_size=3, n_layer=1, n_head=1, n_embd=4)
    holdout = root / "public-holdout.npy"
    np.save(holdout, np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int64))
    task = build_task(
        MMapTokenDataset(holdout, sequence_length=2, vocab_size=3),
        config,
        tokenizer_definition="public-three-token-fixture",
    )
    model = task.model_factory()
    spec = describe_task(
        task,
        batch_size=2,
        optimizer=PrivateOptimizerSpec(
            name="sgd",
            torch_version=torch.__version__,
            parameter_names=tuple(name for name, _ in model.named_parameters()),
            learning_rate=0.01,
        ),
    )
    task_config = root / "task.json"
    task_config.write_text(
        json.dumps(
            {
                "model": asdict(config),
                "tokenizer_definition": "public-three-token-fixture",
                "specification": spec.model_dump(mode="json"),
                "initialization_seed": 17,
                "device": "cpu",
            }
        )
    )
    export = root / "model.safetensors"
    save_safetensors(
        {
            name: np.zeros_like(value.detach().numpy())
            for name, value in model.state_dict().items()
        },
        str(export),
    )
    return task_config, holdout, [export]


def test_zero_model_has_uniform_cross_entropy_and_perplexity(tmp_path: Path) -> None:
    task_config, holdout, models = fixture(tmp_path)
    report = audit_exports(
        task_config=task_config,
        holdout=holdout,
        models=models,
        output=tmp_path / "audit.json",
        public_holdout=True,
    )
    assert report["status"] == "complete"
    assert report["records"] == 2
    assert report["predicted_tokens"] == 4
    row = report["models"][0]
    assert math.isclose(row["mean_token_cross_entropy"], math.log(3), abs_tol=1e-6)
    assert math.isclose(row["perplexity"], 3, abs_tol=1e-6)
    assert row["aliases_verified"] is True


def test_public_holdout_opt_in_is_required_before_reading_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicitly public"):
        audit_exports(
            task_config=tmp_path / "missing-task.json",
            holdout=tmp_path / "missing-data.npy",
            models=[tmp_path / "missing-model.safetensors"],
            output=tmp_path / "audit.json",
        )
    assert not (tmp_path / "audit.json").exists()


def test_conflicting_tied_weights_reject_without_an_audit(tmp_path: Path) -> None:
    task_config, holdout, models = fixture(tmp_path)
    values = load_safetensors(str(models[0]))
    values["lm_head.weight"][:] = 1
    save_safetensors(values, str(models[0]))
    with pytest.raises(ValueError, match="conflicting tied weights"):
        audit_exports(
            task_config=task_config,
            holdout=holdout,
            models=models,
            output=tmp_path / "audit.json",
            public_holdout=True,
        )
    assert not (tmp_path / "audit.json").exists()


@pytest.mark.parametrize("change", ["shape", "state-key", "source"])
def test_changed_architecture_or_source_rejects(tmp_path: Path, change: str) -> None:
    task_config, holdout, models = fixture(tmp_path)
    if change == "source":
        values = json.loads(task_config.read_text())
        values["specification"]["implementation_hash"] = "f" * 64
        task_config.write_text(json.dumps(values))
    else:
        tensors = load_safetensors(str(models[0]))
        if change == "shape":
            tensors["transformer.wpe.weight"] = np.zeros((2, 5), dtype=np.float32)
        else:
            tensors["unexpected_state"] = np.ones(1, dtype=np.float32)
        save_safetensors(tensors, str(models[0]))
    with pytest.raises(ValueError, match="implementation|architecture|keys"):
        audit_exports(
            task_config=task_config,
            holdout=holdout,
            models=models,
            output=tmp_path / "audit.json",
            public_holdout=True,
        )
    assert not (tmp_path / "audit.json").exists()


def test_input_changed_during_evaluation_never_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_config, holdout, models = fixture(tmp_path)
    original = MMapTokenDataset.__getitem__

    def changed(dataset: MMapTokenDataset, index: int) -> dict[str, torch.Tensor]:
        value = original(dataset, index)
        task_config.write_text(task_config.read_text() + "\n")
        return value

    monkeypatch.setattr(MMapTokenDataset, "__getitem__", changed)
    with pytest.raises(ValueError, match="input changed"):
        audit_exports(
            task_config=task_config,
            holdout=holdout,
            models=models,
            output=tmp_path / "audit.json",
            public_holdout=True,
        )
    assert not (tmp_path / "audit.json").exists()


def test_existing_audit_is_never_overwritten(tmp_path: Path) -> None:
    task_config, holdout, models = fixture(tmp_path)
    output = tmp_path / "audit.json"
    output.write_text("retained original")
    with pytest.raises(FileExistsError):
        audit_exports(
            task_config=task_config,
            holdout=holdout,
            models=models,
            output=output,
            public_holdout=True,
        )
    assert output.read_text() == "retained original"
