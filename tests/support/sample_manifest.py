from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import save_file  # pyright: ignore[reportUnknownVariableType]

from dromeus.manifests.canonical import canonical_hash
from dromeus.manifests.models import DraftRunSpec


def manifest_data() -> dict[str, Any]:
    zero = "0" * 64
    one = "1" * 64
    data: dict[str, Any] = {
        "manifest_version": 7,
        "protocol_version": 1,
        "run_id": "run-001",
        "algorithm_id": "noloco",
        "model_id": "sample-linear",
        "model_definition_hash": zero,
        "dataset": {
            "dataset_id": "application",
            "task_id": "sample-task",
            "definition_hash": one,
        },
        "expected_participant_count": 4,
        "environment": {
            "dromeus_version": "0.1.0",
            "dromeus_commit": "abcdef0",
            "protocol_version": 1,
            "pytorch_version": "2.7.0",
            "axl_version": "1.0.0",
            "model_definition_hash": zero,
        },
        "local_steps": 5,
        "round_count": 100,
        "peer_scheduler_seed": 8,
        "codec_id": "safetensors",
        "transport": {
            "max_payload_bytes": 8388608,
            "max_retries": 3,
            "retry_timeout_seconds": 5.0,
            "chunk_size_bytes": 1048576,
            "window_size": 4,
            "max_message_payload_bytes": 2097152,
            "max_inflight_bytes": 4194304,
        },
        "consensus_sketch": {"size": 4096, "seed": 9},
        "application_training": {
            "policy_id": "sample-training",
            "definition_hash": "3" * 64,
        },
        "algorithm_config": {
            "alpha": 0.5,
            "beta": 0.7,
            "gamma": 0.7,
            "inner_steps": 5,
        },
        "artifact_codecs": [
            {"artifact_name": "outer_gradient", "codec_id": "identity"},
            {"artifact_name": "slow_weights", "codec_id": "identity"},
        ],
        "participants": [
            {"public_key": f"peer-{index}", "node_index": index}
            for index in range(4)
        ],
        "initial_checkpoint_hash": "2" * 64,
        "tensor_schema": {
            "tensors": [
                {"name": "layer.weight", "dtype": "float32", "shape": [2, 2]}
            ]
        },
    }
    draft_data = data.copy()
    del draft_data["participants"]
    del draft_data["initial_checkpoint_hash"]
    del draft_data["tensor_schema"]
    data["draft_hash"] = canonical_hash(DraftRunSpec.model_validate(draft_data))
    return data


def write_checkpoint(path: Path, *, shape: tuple[int, ...] = (2, 2)) -> None:
    save_file({"layer.weight": np.zeros(shape, dtype=np.float32)}, str(path))
