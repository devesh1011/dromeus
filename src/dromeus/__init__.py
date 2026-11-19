"""Decentralized federated learning over Gensyn AXL.

Every node trains the same `TrainingTask` on its own records and exchanges NoLoCo
updates with one peer per round. Typical use:

1. Define a `TrainingTask` that every node builds identically.
2. One person writes the shared draft with `make_draft(task, "draft.yaml", ...)`.
3. Each node runs `await Node.from_file("node.yaml").run(task, local=LocalData(...))`.

Only the names exported here are the public, stable API.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import version as _version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dromeus.application import (
        LocalData,
        NoLoCo,
        Privacy,
        check_draft,
        code_hash,
        make_draft,
    )
    from dromeus.errors import (
        ConfigError,
        DraftError,
        DromeusError,
        FormationError,
        PrivacyError,
        RunError,
        TaskError,
    )
    from dromeus.manifests.models import DivergencePolicy, TransportLimits
    from dromeus.node import Node, RunResult
    from dromeus.persistence.privacy_ledger import PrivacyLedger
    from dromeus.training.optimizer_factory import OptimizerRegistration
    from dromeus.training.task import Optimizer, TensorBatchAdapter, TrainingTask

__version__ = _version("dromeus")

# Imported on first use, so `import dromeus` stays light (no PyTorch until needed).
_EXPORTS = {
    "LocalData": "dromeus.application",
    "NoLoCo": "dromeus.application",
    "Privacy": "dromeus.application",
    "check_draft": "dromeus.application",
    "code_hash": "dromeus.application",
    "make_draft": "dromeus.application",
    "ConfigError": "dromeus.errors",
    "DraftError": "dromeus.errors",
    "DromeusError": "dromeus.errors",
    "FormationError": "dromeus.errors",
    "PrivacyError": "dromeus.errors",
    "RunError": "dromeus.errors",
    "TaskError": "dromeus.errors",
    "DivergencePolicy": "dromeus.manifests.models",
    "TransportLimits": "dromeus.manifests.models",
    "Node": "dromeus.node",
    "RunResult": "dromeus.node",
    "PrivacyLedger": "dromeus.persistence.privacy_ledger",
    "OptimizerRegistration": "dromeus.training.optimizer_factory",
    "Optimizer": "dromeus.training.task",
    "TensorBatchAdapter": "dromeus.training.task",
    "TrainingTask": "dromeus.training.task",
}


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'dromeus' has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
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
]
