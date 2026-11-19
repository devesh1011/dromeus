"""Shared custom application inputs for unit and real AXL integration tests."""

from pathlib import Path

import numpy as np

from dromeus.application import NoLoCo, draft_for_task
from dromeus.manifests.models import DraftRunSpec, TransportLimits
from dromeus.training.task import TrainingTask
from examples.custom_training import task as example_task

FIXTURE_TRANSPORT = TransportLimits(
    max_payload_bytes=8388608,
    max_retries=3,
    retry_timeout_seconds=5.0,
    chunk_size_bytes=4096,
    window_size=2,
    max_message_payload_bytes=2097152,
    max_inflight_bytes=4194304,
)


def application_draft(
    compressed: bool = False, task: TrainingTask | None = None
) -> DraftRunSpec:
    """A small four-node ordinary draft for `task` (default: the example)."""
    return draft_for_task(
        task if task is not None else example_task(),
        run_id="run-001",
        participants=4,
        rounds=3,
        local_steps=3,
        axl_version="1.0.0",
        noloco=NoLoCo(
            alpha=0.4, beta=0.3, gamma=0.2, top_k_fraction=0.5 if compressed else None
        ),
        transport=FIXTURE_TRANSPORT,
        seed=8,
    )


def local_file(path: Path, rank: int = 0) -> Path:
    rng = np.random.default_rng(23 + rank)
    inputs = rng.normal(loc=rank / 2, size=(7 + rank * 3, 3)).astype(np.float32)
    np.savez(path, inputs=inputs, targets=(inputs[:, :1] * 0.3).astype(np.float32))
    return path
