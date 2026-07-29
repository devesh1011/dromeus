"""Membership formation protocol."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from dromeus.manifests.canonical import (
    canonical_hash,
    canonical_json,
    parse_sealed_json,
)
from dromeus.manifests.models import (
    DatasetContract,
    DraftRunSpec,
    EnvironmentFingerprint,
    Invitation,
    Participant,
    PublicKey,
    SealedManifest,
    Sha256,
    TensorSchema,
    TransportLimits,
)
from dromeus.protocol.codec import decode_message, encode_envelope, encode_message
from dromeus.protocol.models import (
    Envelope,
    JoinAccepted,
    JoinRequest,
    MessageId,
    MessageType,
    ReadyMessage,
    StartMessage,
    create_envelope,
)
from dromeus.telemetry.events import EventSink, emit_event
from dromeus.transport.base import AsyncTransport
from dromeus.transport.receiver import MessageChannel, Receiver, ReceiverPolicy
from dromeus.transport.sender import OutboundScheduler, Priority
from dromeus.transport.transfer import ArtifactStore, TransferManager


class ReadyValidationError(ValueError):
    """The local node cannot enter the READY barrier."""


class FormationError(RuntimeError):
    """Fixed-membership startup failed."""


def _file_sha256(path: Path) -> Sha256:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_ready(
    *,
    manifest: SealedManifest,
    local_public_key: PublicKey,
    environment: EnvironmentFingerprint,
    dataset: DatasetContract,
    checkpoint_hash: Sha256,
) -> None:
    if local_public_key not in {
        participant.public_key for participant in manifest.participants
    }:
        raise ReadyValidationError("local public key is not a sealed participant")
    if environment != manifest.environment:
        raise ReadyValidationError("environment fingerprint does not match manifest")
    if dataset != manifest.dataset:
        raise ReadyValidationError("dataset contract does not match manifest")
    if checkpoint_hash != manifest.initial_checkpoint_hash:
        raise ReadyValidationError("checkpoint hash does not match manifest")


def create_invitation(
    *,
    draft: DraftRunSpec,
    initiator_public_key: PublicKey,
    bootstrap_uri: str,
    enrollment_expires_at: datetime | None = None,
) -> Invitation:
    return Invitation(
        run_id=draft.run_id,
        initiator_public_key=initiator_public_key,
        bootstrap_uri=bootstrap_uri,
        draft_hash=canonical_hash(draft),
        enrollment_expires_at=enrollment_expires_at,
    )


def seal_manifest(
    *,
    draft: DraftRunSpec,
    participant_keys: set[PublicKey],
    initial_checkpoint_hash: Sha256,
    tensor_schema: TensorSchema,
) -> SealedManifest:
    ordered_keys = tuple(sorted(participant_keys))
    participants = tuple(
        Participant(public_key=public_key, node_index=node_index)
        for node_index, public_key in enumerate(ordered_keys)
    )
    data = draft.model_dump(mode="python")
    data.update(
        {
            "draft_hash": canonical_hash(draft),
            "participants": participants,
            "initial_checkpoint_hash": initial_checkpoint_hash,
            "tensor_schema": tensor_schema,
        }
    )
    return SealedManifest.model_validate(data)


@dataclass(frozen=True)
class FormationResult:
    manifest: SealedManifest
    manifest_hash: Sha256
    checkpoint_path: Path


@dataclass(frozen=True, slots=True)
class FormationServices:
    """Validated services retained for post-formation runtime work."""

    receiver: Receiver
    sender: OutboundScheduler
    transfer_manager: TransferManager


class FormationProtocol:
    """End-to-end fixed-membership formation over receiver/sender/transfer."""

    def __init__(
        self,
        *,
        transport: AsyncTransport,
        draft: DraftRunSpec,
        environment: EnvironmentFingerprint,
        dataset: DatasetContract,
        transport_limits: TransportLimits,
        artifact_store: ArtifactStore,
        event_sink: EventSink | None = None,
    ) -> None:
        self._transport = transport
        self._draft = draft
        self._environment = environment
        self._dataset = dataset
        self._transport_limits = transport_limits
        self._artifact_store = artifact_store
        self._event_sink = event_sink
        self._receiver = Receiver(
            transport,
            ReceiverPolicy(
                run_id=draft.run_id,
                algorithm_id=draft.algorithm_id,
                max_payload_bytes=transport_limits.max_payload_bytes,
            ),
            event_sink=event_sink,
        )
        self._sender = OutboundScheduler(transport)
        self._transfer_manager: TransferManager | None = None
        self._formation_complete = False

    @property
    def services(self) -> FormationServices:
        """Return transport services after sealed formation completes."""
        if not self._formation_complete or self._transfer_manager is None:
            raise FormationError("formation has not completed")
        return FormationServices(
            receiver=self._receiver,
            sender=self._sender,
            transfer_manager=self._transfer_manager,
        )

    async def start(self) -> None:
        await self._receiver.start()
        await self._sender.start()

    async def stop(self) -> None:
        self._formation_complete = False
        if self._transfer_manager is not None:
            await self._transfer_manager.stop()
        await self._sender.stop()
        await self._receiver.stop()

    async def initiate(
        self,
        *,
        bootstrap_uri: str,
        checkpoint_path: Path,
        tensor_schema: TensorSchema,
    ) -> FormationResult:
        local_key = await self._transport.local_public_key()
        emit_event(
            "formation_started",
            run_id=self._draft.run_id,
            peer_id=local_key,
            role="initiator",
            sink=self._event_sink,
        )
        invitation = create_invitation(
            draft=self._draft,
            initiator_public_key=local_key,
            bootstrap_uri=bootstrap_uri,
        )
        participant_keys = {local_key}
        sealed = False
        while len(participant_keys) < 4:
            envelope = await self._next_control()
            if envelope.message_type is not MessageType.JOIN_REQUEST:
                continue
            join = decode_message(
                envelope.payload,
                JoinRequest,
                max_bytes=self._transport_limits.max_payload_bytes,
            )
            if join.draft_hash != invitation.draft_hash:
                continue
            if envelope.sender_public_key in participant_keys:
                continue
            if sealed:
                continue
            participant_keys.add(envelope.sender_public_key)
            await self._send_control(
                destination=envelope.sender_public_key,
                message_type=MessageType.JOIN_ACCEPTED,
                message_id=f"join-accepted-{len(participant_keys)}",
                payload=encode_message(
                    JoinAccepted(draft_hash=invitation.draft_hash)
                ),
            )
        sealed = True
        checkpoint_hash = _file_sha256(checkpoint_path)
        manifest = seal_manifest(
            draft=self._draft,
            participant_keys=participant_keys,
            initial_checkpoint_hash=checkpoint_hash,
            tensor_schema=tensor_schema,
        )
        manifest_hash = canonical_hash(manifest)
        self._receiver.configure_sealed_run(
            participant_keys=frozenset(participant_keys),
            manifest_hash=manifest_hash,
        )
        await self._broadcast_manifest(manifest, participant_keys - {local_key})
        transfer = await self._create_transfer_manager(
            local_key, manifest, manifest_hash
        )
        for peer in participant_keys - {local_key}:
            await transfer.send_artifact(
                destination=peer,
                artifact_name="initial-checkpoint",
                artifact_path=checkpoint_path,
                codec_id=self._draft.codec_id,
                tensor_schema=tensor_schema,
            )
        validate_ready(
            manifest=manifest,
            local_public_key=local_key,
            environment=self._environment,
            dataset=self._dataset,
            checkpoint_hash=checkpoint_hash,
        )
        ready_keys = {local_key}
        while len(ready_keys) < 4:
            envelope = await self._next_control()
            if envelope.message_type is not MessageType.READY:
                continue
            ready = decode_message(
                envelope.payload,
                ReadyMessage,
                max_bytes=self._transport_limits.max_payload_bytes,
            )
            if ready.manifest_hash != manifest_hash:
                continue
            ready_keys.add(envelope.sender_public_key)
        for peer in participant_keys - {local_key}:
            await self._send_control(
                destination=peer,
                message_type=MessageType.START,
                message_id=f"start-{peer[:8]}",
                payload=encode_message(StartMessage(manifest_hash=manifest_hash)),
                manifest_hash=manifest_hash,
            )
        started_keys = {local_key}
        while len(started_keys) < 4:
            envelope = await self._next_control()
            if envelope.message_type is not MessageType.START_ACK:
                continue
            start = decode_message(
                envelope.payload,
                StartMessage,
                max_bytes=self._transport_limits.max_payload_bytes,
            )
            if start.manifest_hash != manifest_hash:
                continue
            started_keys.add(envelope.sender_public_key)
        result = FormationResult(
            manifest=manifest,
            manifest_hash=manifest_hash,
            checkpoint_path=checkpoint_path,
        )
        emit_event(
            "formation_completed",
            run_id=manifest.run_id,
            peer_id=local_key,
            manifest_hash=manifest_hash,
            sink=self._event_sink,
        )
        self._formation_complete = True
        return result

    async def join(
        self,
        *,
        invitation: Invitation,
    ) -> FormationResult:
        if (
            invitation.enrollment_expires_at is not None
            and invitation.enrollment_expires_at <= datetime.now(UTC)
        ):
            raise FormationError("invitation has expired")
        local_key = await self._transport.local_public_key()
        emit_event(
            "formation_started",
            run_id=self._draft.run_id,
            peer_id=local_key,
            role="participant",
            sink=self._event_sink,
        )
        await self._send_control(
            destination=invitation.initiator_public_key,
            message_type=MessageType.JOIN_REQUEST,
            message_id=f"join-{local_key[:8]}",
            payload=encode_message(JoinRequest(draft_hash=invitation.draft_hash)),
        )
        accepted = False
        manifest: SealedManifest | None = None
        manifest_hash: Sha256 | None = None
        while manifest is None:
            envelope = await self._next_control()
            if envelope.message_type is MessageType.JOIN_ACCEPTED:
                join = decode_message(
                    envelope.payload,
                    JoinAccepted,
                    max_bytes=self._transport_limits.max_payload_bytes,
                )
                if join.draft_hash == invitation.draft_hash:
                    accepted = True
                continue
            if envelope.message_type is not MessageType.MANIFEST_SEALED:
                continue
            manifest = parse_sealed_json(envelope.payload)
            if manifest.draft_hash != canonical_hash(self._draft):
                raise FormationError("sealed manifest draft hash mismatch")
            manifest_hash = canonical_hash(manifest)
        if not accepted:
            raise FormationError("initiator never accepted join request")
        if local_key not in {
            participant.public_key for participant in manifest.participants
        }:
            raise FormationError("local node was not sealed into the manifest")
        assert manifest_hash is not None
        self._receiver.configure_sealed_run(
            participant_keys=frozenset(
                participant.public_key for participant in manifest.participants
            ),
            manifest_hash=manifest_hash,
        )
        transfer = await self._create_transfer_manager(
            local_key, manifest, manifest_hash
        )
        try:
            checkpoint = await transfer.next_artifact(
                timeout_seconds=self._formation_timeout_seconds
            )
        except TimeoutError as error:
            raise FormationError("timed out waiting for initial checkpoint") from error
        validate_ready(
            manifest=manifest,
            local_public_key=local_key,
            environment=self._environment,
            dataset=self._dataset,
            checkpoint_hash=checkpoint.sha256,
        )
        await self._send_control(
            destination=invitation.initiator_public_key,
            message_type=MessageType.READY,
            message_id=f"ready-{local_key[:8]}",
            payload=encode_message(ReadyMessage(manifest_hash=manifest_hash)),
            manifest_hash=manifest_hash,
        )
        while True:
            envelope = await self._next_control()
            if envelope.message_type is not MessageType.START:
                continue
            start = decode_message(
                envelope.payload,
                StartMessage,
                max_bytes=self._transport_limits.max_payload_bytes,
            )
            if start.manifest_hash != manifest_hash:
                continue
            break
        await self._send_control(
            destination=invitation.initiator_public_key,
            message_type=MessageType.START_ACK,
            message_id=f"start-ack-{local_key[:8]}",
            payload=encode_message(StartMessage(manifest_hash=manifest_hash)),
            manifest_hash=manifest_hash,
        )
        result = FormationResult(
            manifest=manifest,
            manifest_hash=manifest_hash,
            checkpoint_path=checkpoint.path,
        )
        emit_event(
            "formation_completed",
            run_id=manifest.run_id,
            peer_id=local_key,
            manifest_hash=manifest_hash,
            sink=self._event_sink,
        )
        self._formation_complete = True
        return result

    async def _broadcast_manifest(
        self, manifest: SealedManifest, peers: set[PublicKey]
    ) -> None:
        payload = canonical_json(manifest)
        manifest_hash = canonical_hash(manifest)
        for peer in peers:
            await self._send_control(
                destination=peer,
                message_type=MessageType.MANIFEST_SEALED,
                message_id=f"manifest-{peer[:8]}",
                payload=payload,
                manifest_hash=manifest_hash,
            )

    async def _create_transfer_manager(
        self,
        local_public_key: PublicKey,
        manifest: SealedManifest,
        manifest_hash: Sha256,
    ) -> TransferManager:
        manager = TransferManager(
            local_public_key=local_public_key,
            run_id=manifest.run_id,
            manifest_hash=manifest_hash,
            algorithm_id=manifest.algorithm_id,
            transport_limits=manifest.transport,
            receiver=self._receiver,
            sender=self._sender,
            artifact_store=self._artifact_store,
            event_sink=self._event_sink,
        )
        await manager.start()
        self._transfer_manager = manager
        return manager

    async def _next_control(self) -> Envelope:
        try:
            return await self._receiver.receive(
                MessageChannel.CONTROL,
                timeout_seconds=self._formation_timeout_seconds,
            )
        except TimeoutError as error:
            emit_event(
                "formation_failed",
                run_id=self._draft.run_id,
                error="formation control deadline exceeded",
                sink=self._event_sink,
            )
            raise FormationError("formation control deadline exceeded") from error

    @property
    def _formation_timeout_seconds(self) -> float:
        return self._transport_limits.retry_timeout_seconds * (
            self._transport_limits.max_retries + 4
        )

    async def _send_control(
        self,
        *,
        destination: PublicKey,
        message_type: MessageType,
        message_id: MessageId,
        payload: bytes,
        manifest_hash: Sha256 | None = None,
    ) -> None:
        envelope = create_envelope(
            message_type=message_type,
            message_id=message_id,
            run_id=self._draft.run_id,
            manifest_hash=manifest_hash or cast(Sha256, "0" * 64),
            sender_public_key=await self._transport.local_public_key(),
            algorithm_id=self._draft.algorithm_id,
            payload=payload,
        )
        await self._sender.send(
            destination,
            encode_envelope(envelope),
            priority=Priority.CONTROL,
            retries=self._transport_limits.max_retries,
            retry_delay_seconds=self._transport_limits.retry_timeout_seconds,
        )
