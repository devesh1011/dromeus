"""Event-driven local training and pairwise commit orchestration."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import msgpack  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from safetensors.numpy import (
    load_file as _load_file,  # pyright: ignore[reportUnknownVariableType]
)
from safetensors.numpy import (
    save_file as _save_file,  # pyright: ignore[reportUnknownVariableType]
)

from dromeus.algorithms.base import (
    AlgorithmSnapshot,
    TrainedWeightsBundle,
    checksum_tensors,
)
from dromeus.gossip.scheduler import PeerScheduler
from dromeus.manifests.models import (
    AlgorithmId,
    DomainModel,
    MessageId,
    PublicKey,
    RoundId,
    RunId,
    Sha256,
    TensorSchema,
    TransportLimits,
)
from dromeus.transport.envelope import (
    Envelope,
    MessageType,
    create_envelope,
    encode_envelope,
)
from dromeus.transport.receiver import MessageChannel, Receiver
from dromeus.transport.sender import OutboundScheduler, Priority
from dromeus.transport.transfer import TransferError, TransferManager

_load_safetensors = Callable[[str], dict[str, np.ndarray]]
_save_safetensors = Callable[[dict[str, np.ndarray], str], None]
load_file = cast(_load_safetensors, _load_file)
save_file = cast(_save_safetensors, _save_file)


class PairCommitError(RuntimeError):
    """A peer update or pair commit could not be completed safely."""


class GossipAlgorithm(Protocol):
    def pre_local(self, round_id: RoundId) -> AlgorithmSnapshot: ...

    def local_training(self) -> AlgorithmSnapshot: ...

    def post_local_bundle(self) -> TrainedWeightsBundle: ...

    def validate_peer(self, peer_bundle: TrainedWeightsBundle) -> None: ...

    def peer_apply(self, peer_bundle: TrainedWeightsBundle) -> AlgorithmSnapshot: ...


class PairTransport(Protocol):
    """Transport seam for one peer's update and commit handshake."""

    async def exchange_update(
        self,
        *,
        peer: PublicKey,
        round_id: RoundId,
        bundle: TrainedWeightsBundle,
    ) -> TrainedWeightsBundle: ...

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


class PairCommitMessage(DomainModel):
    """Validated metadata carried by pair-commit envelopes."""

    round_id: RoundId
    checksum: Sha256


class AXLPairTransport:
    """Pair transport backed by reliable safetensors transfer and AXL envelopes."""

    def __init__(
        self,
        *,
        local_public_key: PublicKey,
        run_id: RunId,
        manifest_hash: Sha256,
        algorithm_id: AlgorithmId,
        tensor_schema: TensorSchema,
        transport_limits: TransportLimits,
        receiver: Receiver,
        sender: OutboundScheduler,
        transfer_manager: TransferManager,
        artifact_root: Path,
    ) -> None:
        self._local_public_key = local_public_key
        self._run_id = run_id
        self._manifest_hash = manifest_hash
        self._algorithm_id = algorithm_id
        self._tensor_schema = tensor_schema
        self._transport_limits = transport_limits
        self._receiver = receiver
        self._sender = sender
        self._transfer_manager = transfer_manager
        self._artifact_root = artifact_root
        self._artifact_root.mkdir(parents=True, exist_ok=True)
        self._ready_cache: dict[RoundId, str] = {}
        self._committed_rounds: dict[RoundId, str] = {}

    async def exchange_update(
        self,
        *,
        peer: PublicKey,
        round_id: RoundId,
        bundle: TrainedWeightsBundle,
    ) -> TrainedWeightsBundle:
        if bundle.round_id != round_id:
            raise PairCommitError("local update round does not match current round")
        artifact_path = (
            self._artifact_root / f"round-{round_id}-trained-weights.safetensors"
        )
        await asyncio.to_thread(
            save_file,
            {
                name: np.ascontiguousarray(value)
                for name, value in bundle.tensors.items()
            },
            str(artifact_path),
        )
        try:
            await self._transfer_manager.send_artifact(
                destination=peer,
                artifact_name=f"round-{round_id}-trained-weights",
                artifact_path=artifact_path,
                codec_id="safetensors-v1",
                tensor_schema=self._tensor_schema,
                round_id=round_id,
            )
            receipt = await self._transfer_manager.next_artifact(
                timeout_seconds=self._timeout_seconds
            )
            if (
                receipt.sender_public_key != peer
                or receipt.round_id != round_id
                or receipt.artifact_name != f"round-{round_id}-trained-weights"
            ):
                raise PairCommitError("received unexpected peer update artifact")
            values = await asyncio.to_thread(load_file, str(receipt.path))
            tensors = {
                name: np.ascontiguousarray(value) for name, value in values.items()
            }
            checksum = checksum_tensors(tensors)
            return TrainedWeightsBundle(
                round_id=round_id,
                tensors=tensors,
                checksum=checksum,
            )
        except (
            OSError,
            ValueError,
            TypeError,
            TransferError,
            msgpack.UnpackException,
        ) as error:
            raise PairCommitError("peer update transfer failed") from error
        finally:
            artifact_path.unlink(missing_ok=True)

    async def exchange_update_ready(
        self,
        *,
        peer: PublicKey,
        round_id: RoundId,
        bundle_checksum: str,
    ) -> str:
        cached = self._ready_cache.get(round_id)
        if cached is not None:
            return cached
        await self._send_pair_message(
            destination=peer,
            message_type=MessageType.UPDATE_READY,
            message_id=f"update-ready-{round_id}",
            payload=PairCommitMessage(round_id=round_id, checksum=bundle_checksum),
            round_id=round_id,
        )
        envelope = await self._receive_pair_message(
            peer=peer,
            message_type=MessageType.UPDATE_READY,
            round_id=round_id,
        )
        message = PairCommitMessage.model_validate(_unpack(envelope.payload))
        if message.round_id != round_id:
            raise PairCommitError("peer UPDATE_READY round mismatch")
        self._ready_cache[round_id] = message.checksum
        return message.checksum

    async def exchange_round_committed(
        self,
        *,
        peer: PublicKey,
        round_id: RoundId,
        state_checksum: str,
    ) -> None:
        committed_checksum = self._committed_rounds.get(round_id)
        if committed_checksum is not None:
            if committed_checksum != state_checksum:
                raise PairCommitError("duplicate ROUND_COMMITTED checksum mismatch")
            return
        await self._send_pair_message(
            destination=peer,
            message_type=MessageType.ROUND_COMMITTED,
            message_id=f"round-committed-{round_id}",
            payload=PairCommitMessage(round_id=round_id, checksum=state_checksum),
            round_id=round_id,
        )
        envelope = await self._receive_pair_message(
            peer=peer,
            message_type=MessageType.ROUND_COMMITTED,
            round_id=round_id,
        )
        PairCommitMessage.model_validate(_unpack(envelope.payload))
        self._receiver.set_current_round(round_id + 1)
        await self._receiver.advance_round(round_id + 1)
        self._committed_rounds[round_id] = state_checksum

    @property
    def _timeout_seconds(self) -> float:
        return self._transport_limits.retry_timeout_seconds * (
            self._transport_limits.max_retries + 4
        )

    async def _send_pair_message(
        self,
        *,
        destination: PublicKey,
        message_type: MessageType,
        message_id: MessageId,
        payload: PairCommitMessage,
        round_id: RoundId,
    ) -> None:
        envelope = create_envelope(
            message_type=message_type,
            message_id=message_id,
            run_id=self._run_id,
            manifest_hash=self._manifest_hash,
            sender_public_key=self._local_public_key,
            algorithm_id=self._algorithm_id,
            round_id=round_id,
            correlation_id=f"pair-round-{round_id}",
            payload=_pack(payload),
        )
        await self._sender.send(
            destination,
            encode_envelope(envelope),
            priority=Priority.CONTROL,
            retries=self._transport_limits.max_retries,
            retry_delay_seconds=self._transport_limits.retry_timeout_seconds,
        )

    async def _receive_pair_message(
        self,
        *,
        peer: PublicKey,
        message_type: MessageType,
        round_id: RoundId,
    ) -> Envelope:
        try:
            envelope = await self._receiver.receive(
                MessageChannel.PAIR_COMMIT,
                timeout_seconds=self._timeout_seconds,
            )
        except TimeoutError as error:
            raise PairCommitError("pair commit deadline exceeded") from error
        if (
            envelope.sender_public_key != peer
            or envelope.message_type is not message_type
            or envelope.round_id != round_id
        ):
            raise PairCommitError("received unexpected pair commit message")
        return envelope


def _pack(model: DomainModel) -> bytes:
    return cast(
        bytes,
        msgpack.packb(  # pyright: ignore[reportUnknownMemberType]
            model.model_dump(mode="python")
        ),
    )


def _unpack(data: bytes) -> object:
    return cast(
        object,
        msgpack.unpackb(  # pyright: ignore[reportUnknownMemberType]
            data, raw=False, strict_map_key=True
        ),
    )


@dataclass(frozen=True, slots=True)
class RoundCommit:
    """Evidence passed to the atomic persistence seam after peer validation."""

    round_id: RoundId
    peer_public_key: PublicKey
    pre_local: AlgorithmSnapshot
    local_bundle: TrainedWeightsBundle
    peer_bundle: TrainedWeightsBundle
    post_mix: AlgorithmSnapshot
    state_checksum: str


CommitCallback = Callable[[RoundCommit], None | Awaitable[None]]


class GossipEngine:
    """Run fixed-round local training without a group-wide barrier."""

    def __init__(
        self,
        *,
        local_public_key: PublicKey,
        round_count: int,
        scheduler: PeerScheduler,
        algorithm: GossipAlgorithm,
        transport: PairTransport,
        commit_callback: CommitCallback,
    ) -> None:
        if round_count <= 0:
            raise ValueError("round_count must be positive")
        self._local_public_key = local_public_key
        self._round_count = round_count
        self._scheduler = scheduler
        self._algorithm = algorithm
        self._transport = transport
        self._commit_callback = commit_callback
        self._current_round = 0
        self._commits: list[RoundCommit] = []

    @property
    def current_round(self) -> RoundId:
        return self._current_round

    @property
    def commits(self) -> tuple[RoundCommit, ...]:
        return tuple(self._commits)

    async def run(self) -> tuple[RoundCommit, ...]:
        """Train and commit every manifest round in order."""
        while self._current_round < self._round_count:
            await self.run_round(self._current_round)
        return self.commits

    async def run_round(self, round_id: RoundId) -> RoundCommit:
        """Complete one scheduled pair exchange and commit."""
        if round_id != self._current_round:
            raise PairCommitError(
                f"round {round_id} is not current; expected {self._current_round}"
            )
        pairing = self._scheduler.schedule(round_id)
        try:
            peer = pairing.peer_for(self._local_public_key)
        except KeyError as error:
            raise PairCommitError(str(error)) from error

        pre_local = await asyncio.to_thread(self._algorithm.pre_local, round_id)
        await asyncio.to_thread(self._algorithm.local_training)
        local_bundle = await asyncio.to_thread(self._algorithm.post_local_bundle)
        local_checksum = await asyncio.to_thread(checksum_tensors, local_bundle.tensors)
        if local_checksum != local_bundle.checksum:
            raise PairCommitError("local update checksum mismatch")
        peer_bundle = await self._transport.exchange_update(
            peer=peer,
            round_id=round_id,
            bundle=local_bundle,
        )
        if peer_bundle.round_id != round_id:
            raise PairCommitError("peer update round does not match current round")
        peer_checksum = await asyncio.to_thread(checksum_tensors, peer_bundle.tensors)
        if peer_checksum != peer_bundle.checksum:
            raise PairCommitError("peer update checksum mismatch")
        try:
            await asyncio.to_thread(self._algorithm.validate_peer, peer_bundle)
        except (ValueError, TypeError) as error:
            raise PairCommitError("peer update validation failed") from error
        remote_checksum = await self._transport.exchange_update_ready(
            peer=peer,
            round_id=round_id,
            bundle_checksum=local_bundle.checksum,
        )
        if remote_checksum != peer_bundle.checksum:
            raise PairCommitError("peer UPDATE_READY checksum mismatch")

        try:
            post_mix = await asyncio.to_thread(
                self._algorithm.peer_apply,
                peer_bundle,
            )
        except (ValueError, TypeError) as error:
            raise PairCommitError("peer update application failed") from error
        state_checksum = await asyncio.to_thread(checksum_tensors, post_mix.weights)
        commit = RoundCommit(
            round_id=round_id,
            peer_public_key=peer,
            pre_local=pre_local,
            local_bundle=local_bundle,
            peer_bundle=peer_bundle,
            post_mix=post_mix,
            state_checksum=state_checksum,
        )
        result = await asyncio.to_thread(self._commit_callback, commit)
        if inspect.isawaitable(result):
            await result
        await self._transport.exchange_round_committed(
            peer=peer,
            round_id=round_id,
            state_checksum=state_checksum,
        )
        self._commits.append(commit)
        self._current_round += 1
        return commit


__all__ = [
    "AXLPairTransport",
    "GossipAlgorithm",
    "GossipEngine",
    "PairCommitError",
    "PairTransport",
    "RoundCommit",
]
