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
        "manifest_version": 1,
        "protocol_version": 1,
        "run_id": "run-001",
        "algorithm_id": "dpsgd-v1",
        "model_id": "cifar-cnn-v1",
        "model_definition_hash": zero,
        "dataset": {
            "dataset_id": "cifar10",
            "version": "1",
            "preprocessing_hash": one,
            "iid_partition_seed": 7,
            "image_shape": [3, 32, 32],
            "class_count": 10,
            "sample_count": 50000,
            "partition_sample_counts": [12500, 12500, 12500, 12500],
            "node_index_partitions": [0, 1, 2, 3],
        },
        "environment": {
            "dromeus_version": "0.1.0",
            "dromeus_commit": "abcdef0",
            "protocol_version": 1,
            "pytorch_version": "2.7.0",
            "axl_version": "1.0.0",
            "model_definition_hash": zero,
            "container_image_digest": f"sha256:{one}",
        },
        "local_steps": 5,
        "round_count": 100,
        "optimizer": "sgd",
        "learning_rate": 0.1,
        "peer_scheduler_seed": 8,
        "codec_id": "safetensors-v1",
        "transport": {
            "max_payload_bytes": 8388608,
            "max_retries": 3,
            "retry_timeout_seconds": 5.0,
        },
        "consensus_sketch": {"size": 4096, "seed": 9},
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
