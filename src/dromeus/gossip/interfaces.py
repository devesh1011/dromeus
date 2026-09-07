"""Transport-independent contracts for gossip orchestration and adapters."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from dromeus.algorithms.base import (
    AlgorithmEvaluation,
    AlgorithmObservations,
    AlgorithmSnapshot,
    AlgorithmUpdate,
    UpdateBundle,
)
from dromeus.manifests.models import PublicKey, RoundId


class PairCommitError(RuntimeError):
    """A peer update or pair commit could not be completed safely."""


@dataclass(frozen=True, slots=True)
class RunFailure:
    """Terminal failure evidence for persistence and control-plane reporting."""

    round_id: RoundId
    error_type: str
    reason: str


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """One node's deterministic evaluation result for a committed round."""

    round_id: RoundId
    loss: float | None
    accuracy: float | None
    metrics: Mapping[str, float] | None = None


@dataclass(frozen=True, slots=True)
class PairExchangeResult:
    """Peer bundle plus immutable diagnostics for its transfer."""

    bundle: UpdateBundle
    transfer_id: str | None = None
    retry_count: int = 0

    def __post_init__(self) -> None:
        if self.transfer_id is not None and not self.transfer_id:
            raise ValueError("transfer_id must be non-empty when present")
        if self.retry_count < 0:
            raise ValueError("retry_count must be non-negative")


@dataclass(frozen=True, slots=True)
class ConsensusBroadcastResult:
    """Best-effort consensus telemetry broadcast accounting."""

    payload_bytes: int
    recipient_count: int
    successful_recipient_count: int
    retry_count: int


class FailureBroadcaster(Protocol):
    async def broadcast_run_failed(self, failure: RunFailure) -> None: ...


class ConsensusPublisher(Protocol):
    def submit(
        self, *, round_id: RoundId, weights: Mapping[str, np.ndarray]
    ) -> bool: ...


EvaluationCallback = Callable[[EvaluationMetrics], None | Awaitable[None]]


class GossipAlgorithm(Protocol):
    def configure_bundle_codec(
        self,
        *,
        artifact_root: Path,
        run_id: str,
        manifest_hash: str,
        sender_public_key: str,
        algorithm_id: str,
    ) -> None: ...

    def pre_local(self, round_id: RoundId) -> None: ...

    def local_training(self) -> None: ...

    def post_local_bundle(self) -> UpdateBundle: ...

    def validate_peer(self, peer_bundle: UpdateBundle) -> AlgorithmUpdate: ...

    def peer_apply(self, peer_update: AlgorithmUpdate) -> AlgorithmSnapshot: ...

    def release_bundle(self, bundle: UpdateBundle) -> None: ...

    def checkpoint_tensors(self) -> dict[str, np.ndarray]: ...

    def observations(self) -> AlgorithmObservations: ...

    def evaluate(self) -> AlgorithmEvaluation | None: ...


class PairTransport(Protocol):
    """Transport seam for one peer's update and commit handshake."""

    async def exchange_update(
        self,
        *,
        peer: PublicKey,
        round_id: RoundId,
        bundle: UpdateBundle,
    ) -> PairExchangeResult: ...

    async def exchange_update_ready(
        self,
        *,
        peer: PublicKey,
        round_id: RoundId,
        bundle_checksum: str,
    ) -> str: ...

    async def exchange_round_committed(
        self,
        *,
        peer: PublicKey,
        round_id: RoundId,
        state_checksum: str,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class RoundCommit:
    """Evidence passed to the atomic persistence seam after peer validation."""

    round_id: RoundId
    peer_public_key: PublicKey
    local_bundle_digest: str
    peer_bundle_digest: str
    state_checksum: str
    phase: Literal["training", "final-consensus"] = "training"
    local_loss: float | None = None
    error_feedback_residual_l2_norm: float | None = None
    error_feedback_signal_l2_norm: float | None = None
    error_feedback_residual_to_signal_ratio: float | None = None
    transfer_id: str | None = None
    transfer_retries: int = 0
    encoded_artifact_bytes: int | None = None


CommitCallback = Callable[[RoundCommit], None | Awaitable[None]]
