"""Distributed all-pairs AXL baseline with one exclusive bridge per worker."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from dromeus.manifests.canonical import canonical_hash, file_sha256
from dromeus.manifests.models import SealedManifest
from dromeus.protocol.codec import decode_envelope, encode_envelope
from dromeus.protocol.models import MessageType, create_envelope
from dromeus.transport.axl import AXLBridgeConfig, AXLTransport
from dromeus.transport.interface import AsyncTransport
from dromeus.transport.outbound_scheduler import OutboundScheduler
from dromeus.transport.receiver import Receiver, ReceiverPolicy
from dromeus.transport.transfer import TransferManager

from .axl_baseline import (
    DEFAULT_PAYLOAD_SIZES,
    ArtifactCase,
    ControlRTTSample,
    MeasuredTransport,
    RecordingEventSink,
    TransferSample,
    create_payload_cases,
)


@dataclass(frozen=True, slots=True)
class DistributedNodeReport:
    phase: str
    local_public_key: str
    participant_keys: tuple[str, ...]
    manifest_hash: str
    topology: Mapping[str, object]
    control_rtt_samples: tuple[ControlRTTSample, ...] = ()
    transfer_samples: tuple[TransferSample, ...] = ()
    received_artifacts: int = 0
    checksum_failure_count: int = 0

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


async def _wait_until(start_at_epoch: float) -> None:
    delay = start_at_epoch - time.time()
    if delay > 0:
        await asyncio.sleep(delay)


def _control_probe(
    *,
    manifest: SealedManifest,
    kind: str,
    token: str,
    sender: str,
) -> bytes:
    payload = json.dumps(
        {
            "benchmark": "dromeus-axl-baseline-v1",
            "kind": kind,
            "token": token,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return encode_envelope(
        create_envelope(
            message_type=MessageType.JOIN_REQUEST,
            message_id=f"{kind}-{token}",
            run_id=manifest.run_id,
            manifest_hash=canonical_hash(manifest),
            sender_public_key=sender,
            algorithm_id=manifest.algorithm_id,
            correlation_id=token,
            payload=payload,
        )
    )


async def run_rtt_phase(
    *,
    bridge_url: str,
    manifest: SealedManifest,
    output_path: Path,
    samples_per_pair: int,
    start_at_epoch: float,
) -> DistributedNodeReport:
    transport = MeasuredTransport(
        AXLTransport(AXLBridgeConfig(base_url=bridge_url.rstrip("/")))
    )
    local_key = await transport.local_public_key()
    participant_keys = tuple(
        participant.public_key for participant in manifest.participants
    )
    if local_key not in participant_keys:
        raise ValueError("local AXL key is not in the sealed manifest")
    peers = tuple(key for key in participant_keys if key != local_key)
    pending: dict[str, tuple[str, float]] = {}
    for peer in peers:
        for sample_index in range(samples_per_pair):
            token = f"{local_key[:8]}-{peer[:8]}-{sample_index}"
            pending[token] = (peer, 0.0)
    await _wait_until(start_at_epoch)
    for token, (peer, _) in tuple(pending.items()):
        started = time.perf_counter()
        await transport.send(
            peer,
            _control_probe(
                manifest=manifest,
                kind="ping",
                token=token,
                sender=local_key,
            ),
        )
        pending[token] = (peer, started)

    samples: list[ControlRTTSample] = []
    inbound_pings: set[str] = set()
    expected_inbound = len(peers) * samples_per_pair
    deadline = time.monotonic() + manifest.transport.retry_timeout_seconds * 4
    while pending or len(inbound_pings) < expected_inbound:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("distributed RTT phase timed out")
        inbound = await transport.recv(min(1.0, remaining))
        if inbound is None:
            continue
        try:
            envelope = decode_envelope(
                inbound.payload,
                authenticated_sender=inbound.sender_public_key,
                participant_keys=frozenset(participant_keys),
                max_payload_bytes=manifest.transport.max_payload_bytes,
            )
            record = cast(object, json.loads(envelope.payload))
        except (UnicodeDecodeError, ValueError):
            continue
        if not isinstance(record, dict):
            continue
        value = cast(dict[object, object], record)
        if value.get("benchmark") != "dromeus-axl-baseline-v1":
            continue
        token = value.get("token")
        if not isinstance(token, str):
            continue
        kind = value.get("kind")
        if kind == "ping":
            inbound_pings.add(f"{inbound.sender_public_key}:{token}")
            await transport.send(
                inbound.sender_public_key,
                _control_probe(
                    manifest=manifest,
                    kind="ack",
                    token=token,
                    sender=local_key,
                ),
            )
        elif kind == "ack" and token in pending:
            destination, started = pending.pop(token)
            if inbound.sender_public_key != destination:
                raise ValueError("RTT acknowledgment sender mismatch")
            samples.append(
                ControlRTTSample(
                    source=local_key,
                    destination=destination,
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
    report = DistributedNodeReport(
        phase="rtt",
        local_public_key=local_key,
        participant_keys=participant_keys,
        manifest_hash=canonical_hash(manifest),
        topology=await transport.topology(),
        control_rtt_samples=tuple(samples),
    )
    await asyncio.to_thread(report.write, output_path)
    return report


async def run_transfer_phase(
    *,
    bridge_url: str,
    manifest: SealedManifest,
    checkpoint_path: Path,
    output_path: Path,
    samples_per_pair: int,
    start_at_epoch: float,
) -> DistributedNodeReport:
    measured = MeasuredTransport(
        AXLTransport(AXLBridgeConfig(base_url=bridge_url.rstrip("/")))
    )
    local_key = await measured.local_public_key()
    participant_keys = tuple(
        participant.public_key for participant in manifest.participants
    )
    if local_key not in participant_keys:
        raise ValueError("local AXL key is not in the sealed manifest")
    peers = tuple(key for key in participant_keys if key != local_key)
    payload_cases = await asyncio.to_thread(
        create_payload_cases,
        output_path.parent / "payloads",
        DEFAULT_PAYLOAD_SIZES,
    )
    checkpoint_case = ArtifactCase(
        name="checkpoint",
        path=checkpoint_path,
        tensor_schema=manifest.tensor_schema,
    )
    artifacts = (*payload_cases, checkpoint_case)
    artifact_hashes = {
        artifact.name: await asyncio.to_thread(file_sha256, artifact.path)
        for artifact in artifacts
    }
    events = RecordingEventSink()
    receiver = Receiver(
        cast(AsyncTransport, measured),
        ReceiverPolicy(
            run_id=manifest.run_id,
            manifest_hash=canonical_hash(manifest),
            algorithm_id=manifest.algorithm_id,
            participant_keys=frozenset(participant_keys),
            max_payload_bytes=manifest.transport.max_payload_bytes,
        ),
        event_sink=events,
    )
    sender = OutboundScheduler(cast(AsyncTransport, measured))
    manager = TransferManager(
        local_public_key=local_key,
        run_id=manifest.run_id,
        manifest_hash=canonical_hash(manifest),
        algorithm_id=manifest.algorithm_id,
        transport_limits=manifest.transport,
        receiver=receiver,
        sender=sender,
        artifact_root=output_path.parent / "received",
        event_sink=events,
    )
    await sender.start()
    await receiver.start()
    await manager.start()
    expected_received = len(peers) * len(artifacts) * samples_per_pair

    async def consume() -> int:
        received = 0
        while received < expected_received:
            receipt = await manager.next_artifact(
                timeout_seconds=manifest.transport.retry_timeout_seconds
                * (manifest.transport.max_retries + 8)
            )
            expected_hash = artifact_hashes.get(receipt.artifact_name)
            if expected_hash is None or receipt.sha256 != expected_hash:
                raise ValueError("received baseline artifact hash mismatch")
            await manager.release_receipt(receipt)
            received += 1
        return received

    async def send_to_peer(peer: str) -> list[TransferSample]:
        peer_samples: list[TransferSample] = []
        for artifact in artifacts:
            payload_bytes = artifact.path.stat().st_size
            for _ in range(samples_per_pair):
                started = time.perf_counter()
                transfer_id = await manager.send_artifact(
                    destination=peer,
                    artifact_name=artifact.name,
                    artifact_path=artifact.path,
                    codec_id="safetensors-v1",
                    tensor_schema=artifact.tensor_schema,
                )
                elapsed = time.perf_counter() - started
                retry_count = sum(
                    value
                    for record in events.records
                    if record.get("transfer_id") == transfer_id
                    and record.get("event") == "transfer_message_sent"
                    and isinstance((value := record.get("retry_count")), int)
                )
                peer_samples.append(
                    TransferSample(
                        source=local_key,
                        destination=peer,
                        artifact=artifact.name,
                        payload_bytes=payload_bytes,
                        elapsed_seconds=elapsed,
                        retry_count=retry_count,
                        checksum_failure_count=0,
                        dromeus_wire_bytes=payload_bytes,
                    )
                )
        return peer_samples

    try:
        await _wait_until(start_at_epoch)
        consumer_task = asyncio.create_task(consume())
        try:
            sent_groups = await asyncio.gather(*(send_to_peer(peer) for peer in peers))
            received_count = await consumer_task
        except BaseException:
            consumer_task.cancel()
            await asyncio.gather(consumer_task, return_exceptions=True)
            raise
        samples = tuple(sample for group in sent_groups for sample in group)
    finally:
        await manager.stop()
        await receiver.stop()
        await sender.stop()
    report = DistributedNodeReport(
        phase="transfer",
        local_public_key=local_key,
        participant_keys=participant_keys,
        manifest_hash=canonical_hash(manifest),
        topology=await measured.topology(),
        transfer_samples=samples,
        received_artifacts=received_count,
        checksum_failure_count=events.checksum_failure_count,
    )
    await asyncio.to_thread(report.write, output_path)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("rtt", "transfer"))
    parser.add_argument("--bridge-url", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-pair", type=int, default=5)
    parser.add_argument("--start-at-epoch", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = SealedManifest.model_validate_json(
        args.manifest.read_text(encoding="utf-8")
    )
    if args.phase == "rtt":
        asyncio.run(
            run_rtt_phase(
                bridge_url=args.bridge_url,
                manifest=manifest,
                output_path=args.output,
                samples_per_pair=args.samples_per_pair,
                start_at_epoch=args.start_at_epoch,
            )
        )
        return
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for transfer phase")
    asyncio.run(
        run_transfer_phase(
            bridge_url=args.bridge_url,
            manifest=manifest,
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            samples_per_pair=args.samples_per_pair,
            start_at_epoch=args.start_at_epoch,
        )
    )


if __name__ == "__main__":
    main()
