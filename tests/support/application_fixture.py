"""Shared custom application inputs for unit and real AXL integration tests."""

from pathlib import Path

import numpy as np

from dromeus.application import definition_hash
from dromeus.manifests.models import DraftRunSpec
from examples.custom_training import (
    MODEL_DEFINITION,
    TASK_DEFINITION,
    TRAINING_DEFINITION,
)
from support.sample_manifest import manifest_data


def application_draft(
    algorithm: str = "noloco", compressed: bool = False
) -> DraftRunSpec:
    value = manifest_data()
    for key in (
        "participants",
        "draft_hash",
        "initial_checkpoint_hash",
        "tensor_schema",
    ):
        del value[key]
    value.update(
        manifest_version=4,
        expected_participant_count=4,
        model_id="custom-regression",
        model_definition_hash=definition_hash(MODEL_DEFINITION),
        dataset={
            "dataset_id": "application-v1",
            "task_id": "regression",
            "definition_hash": definition_hash(TASK_DEFINITION),
        },
        application_training={
            "policy_id": "rmsprop-huber",
            "definition_hash": definition_hash(TRAINING_DEFINITION),
        },
        optimizer="application",
        training=None,
        learning_rate=None,
        algorithm_id=algorithm,
        local_steps=3,
        round_count=3,
    )
    value["environment"].update(
        model_definition_hash=value["model_definition_hash"],
        container_image_digest=None,
    )
    if algorithm == "noloco":
        value.update(
            algorithm_config={
                "alpha": 0.4,
                "beta": 0.3,
                "gamma": 0.2,
                "inner_steps": 3,
            },
            artifact_codecs=[
                {
                    "artifact_name": "outer_gradient",
                    "codec_id": "topk-bitmap-int8-v2",
                    "top_k_fraction": 0.5,
                    "lossy_allowed": True,
                }
                if compressed
                else {"artifact_name": "outer_gradient", "codec_id": "identity-v1"},
                {
                    "artifact_name": "slow_weights",
                    "codec_id": "dense-int8-v1",
                    "lossy_allowed": True,
                }
                if compressed
                else {"artifact_name": "slow_weights", "codec_id": "identity-v1"},
            ],
        )
        value["transport"].update(chunk_size_bytes=4096, window_size=2)
    return DraftRunSpec.model_validate(value)


def local_file(path: Path, rank: int = 0) -> Path:
    rng = np.random.default_rng(23 + rank)
    inputs = rng.normal(loc=rank / 2, size=(7 + rank * 3, 3)).astype(np.float32)
    held_out = rng.normal(size=(4, 3)).astype(np.float32)
    np.savez(
        path,
        inputs=inputs,
        targets=(inputs[:, :1] * 0.3).astype(np.float32),
        evaluation_inputs=held_out,
        evaluation_targets=(held_out[:, :1] * 0.3).astype(np.float32),
    )
    return path
