from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from benchmarks.cifar10.runner import (
    ReportInput,
    create_draft,
    write_frozen_plan,
    write_pilot_evidence,
)
from dromeus.manifests.canonical import canonical_hash
from dromeus.manifests.models import SealedManifest
from dromeus.persistence.run_store import RunStore
from dromeus.training.pytorch import (
    CIFAR10_DATASET_VERSION,
    MODEL_DEFINITION_HASH,
    PREPROCESSING_HASH,
)


def _draft():
    return create_draft(
        run_id="pilot-001",
        benchmark_seed=17,
        dromeus_commit="a" * 40,
        image_digest=f"sha256:{'b' * 64}",
        pytorch_version="2.13.0+cpu",
        round_count=2,
        local_steps=1,
    )


def _write_completed_roots(root: Path) -> tuple[Path, Path, Path, Path]:
    draft = _draft()
    sealed = SealedManifest.model_validate(
        {
            **draft.model_dump(mode="json"),
            "draft_hash": canonical_hash(draft),
            "participants": [
                {"public_key": f"peer-{index}", "node_index": index}
                for index in range(4)
            ],
            "initial_checkpoint_hash": "c" * 64,
            "tensor_schema": {
                "tensors": [
                    {"name": "weight", "dtype": "float32", "shape": [1]}
                ]
            },
        }
    )
    roots: list[Path] = []
    for index in range(4):
        run_root = root / f"node-{index}"
        store = RunStore(run_root / "run-store")
        store.initialize(sealed)
        for round_id in range(draft.round_count):
            store.persist_commit(
                committed_round=round_id,
                algorithm_state={"weight": np.array([1], dtype=np.float32)},
                pre_mix_state={"weight": np.array([0], dtype=np.float32)},
                post_mix_state={"weight": np.array([1], dtype=np.float32)},
                state_checksum=f"{index * 2 + round_id + 1:064x}",
                schedule={"round_id": round_id, "peer": f"peer-{index}"},
            )
        store.record_terminal("complete", {"committed_rounds": draft.round_count})
        roots.append(run_root)
    return roots[0], roots[1], roots[2], roots[3]


def test_create_draft_uses_canonical_production_values() -> None:
    draft = _draft()

    assert draft.model_definition_hash == MODEL_DEFINITION_HASH
    assert draft.dataset.version == CIFAR10_DATASET_VERSION
    assert draft.dataset.preprocessing_hash == PREPROCESSING_HASH
    assert draft.environment.pytorch_version == "2.13.0+cpu"
    assert draft.transport.max_payload_bytes == 16 * 1024 * 1024
    assert draft.peer_scheduler_seed == 17


def test_completed_pilot_can_freeze_one_plan(tmp_path: Path) -> None:
    draft_path = tmp_path / "draft.yaml"
    draft_path.write_text(
        json.dumps(_draft().model_dump(mode="json")),
        encoding="utf-8",
    )
    evidence_path = tmp_path / "pilot.json"
    evidence = write_pilot_evidence(
        draft_path=draft_path,
        run_roots=_write_completed_roots(tmp_path),
        output=evidence_path,
    )

    plan = write_frozen_plan(
        draft_path=draft_path,
        pilot_artifact=evidence_path,
        benchmark_seeds=(17, 29, 41),
        output=tmp_path / "plan.yaml",
    )

    assert evidence.status == "complete"
    assert plan.benchmark_seeds == (17, 29, 41)
    assert plan.dataset.preprocessing_hash == PREPROCESSING_HASH


def test_report_input_requires_three_seed_archives() -> None:
    with pytest.raises(ValidationError):
        ReportInput.model_validate({"seeds": []})
