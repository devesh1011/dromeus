"""The names exported from `dromeus` are the public API; change them deliberately."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn

import dromeus
from dromeus import application
from dromeus.application import installed_versions, load_draft
from dromeus.training.versions import torch_release

PUBLIC_API = (
    "ConfigError",
    "DivergencePolicy",
    "DraftError",
    "DromeusError",
    "FormationError",
    "LocalData",
    "NoLoCo",
    "Node",
    "Optimizer",
    "OptimizerRegistration",
    "Privacy",
    "PrivacyError",
    "PrivacyLedger",
    "RunError",
    "RunResult",
    "TaskError",
    "TensorBatchAdapter",
    "TrainingTask",
    "TransportLimits",
    "__version__",
    "check_draft",
    "code_hash",
    "make_draft",
)


def test_exported_names_match_the_public_api() -> None:
    # Adding or removing a name here is an API change: record it in the changelog.
    assert tuple(sorted(dromeus.__all__)) == PUBLIC_API
    for name in PUBLIC_API:
        assert hasattr(dromeus, name), name
    lazy = set(dromeus._EXPORTS)  # pyright: ignore[reportPrivateUsage]
    assert lazy | {"__version__"} == set(PUBLIC_API)


def _task(learning_rate: float = 0.01) -> dromeus.TrainingTask:
    return dromeus.TrainingTask(
        model_factory=lambda: nn.Linear(3, 1),
        batch_adapter=dromeus.TensorBatchAdapter(
            example=(torch.zeros(2, 3), torch.zeros(2, 1)),
            definition="inputs [B,3], targets [B,1]",
        ),
        loss_per_record=lambda model, batch: (model(batch[0]) - batch[1]).square()[
            :, 0
        ],
        model_definition="Linear(3,1)",
        task_definition="three features, one target",
        loss_definition="squared error per record",
        implementation_hash="a" * 64,
        optimizer=dromeus.Optimizer.sgd(learning_rate),
        batch_size=4,
    )


def _draft(path: Path, task: dromeus.TrainingTask, **settings: object) -> str:
    values: dict[str, object] = {
        "run_id": "api-test",
        "participants": 4,
        "rounds": 3,
        "local_steps": 2,
        "axl_version": "1.0.0",
    }
    values.update(settings)
    return dromeus.make_draft(task, path, **values)  # pyright: ignore[reportArgumentType]


def test_make_draft_writes_every_hash_and_check_draft_accepts_same_task(
    tmp_path: Path,
) -> None:
    path = tmp_path / "draft.yaml"
    digest = _draft(path, _task())

    draft = load_draft(path)
    assert len(digest) == 64
    assert draft.model_definition_hash == hashlib.sha256(b"Linear(3,1)").hexdigest()
    assert draft.training.sampling == "shuffle"
    assert draft.training.trainable_parameter_names == ("weight", "bias")
    assert (draft.environment.dromeus_version, draft.environment.pytorch_version) == (
        installed_versions()
    )
    assert draft.algorithm_config.inner_steps == 2
    dromeus.check_draft(path, _task())


def test_check_draft_names_what_differs_from_the_task(tmp_path: Path) -> None:
    path = tmp_path / "draft.yaml"
    _draft(path, _task(learning_rate=0.01))

    with pytest.raises(dromeus.DraftError, match="optimizer"):
        dromeus.check_draft(path, _task(learning_rate=0.02))


def test_private_draft_derives_horizon_and_poisson_sampling(tmp_path: Path) -> None:
    path = tmp_path / "draft.yaml"
    _draft(
        path,
        _task(),
        privacy=dromeus.Privacy(max_grad_norm=1.0, noise_multiplier=1.0, delta=1e-5),
        noloco=dromeus.NoLoCo(top_k_fraction=0.25),
    )

    draft = load_draft(path)
    assert draft.privacy is not None
    assert draft.privacy.max_logical_steps == 6
    assert draft.privacy.randomness_profile == "secure"
    assert draft.training.sampling == "poisson"
    assert [codec.codec_id for codec in draft.artifact_codecs] == [
        "topk-bitmap-int8",
        "dense-int8",
    ]


def test_check_draft_refuses_other_installed_versions(tmp_path: Path) -> None:
    path = tmp_path / "draft.yaml"
    _draft(path, _task())
    text = path.read_text().replace(
        f"pytorch_version: {torch_release()}", "pytorch_version: 0.0.1"
    )
    path.write_text(text)

    with pytest.raises(dromeus.DraftError, match="PyTorch 0.0.1"):
        dromeus.check_draft(path, _task())


def test_invalid_settings_raise_public_errors(tmp_path: Path) -> None:
    with pytest.raises(dromeus.DraftError):
        _draft(tmp_path / "draft.yaml", _task(), participants=3)
    with pytest.raises(dromeus.TaskError):
        dromeus.Optimizer.sgd(-1.0).specification(("weight",))
    with pytest.raises(dromeus.ConfigError):
        dromeus.Node.from_file(tmp_path / "missing.yaml")


def test_code_hash_covers_file_bytes(tmp_path: Path) -> None:
    first, second = tmp_path / "a.py", tmp_path / "b.py"
    first.write_text("x = 1\n")
    second.write_text("x = 2\n")

    assert dromeus.code_hash(first) != dromeus.code_hash(second)
    assert dromeus.code_hash(first, second) != dromeus.code_hash(second, first)
    assert len(dromeus.code_hash(application)) == 64
