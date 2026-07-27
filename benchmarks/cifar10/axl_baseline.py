"""All-pairs real-AXL baseline over four exclusively held loopback bridges."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

import msgpack  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from safetensors.numpy import (
    load_file,  # pyright: ignore[reportUnknownVariableType]
    save_file,  # pyright: ignore[reportUnknownVariableType]
)

from dromeus.manifests.canonical import canonical_hash
from dromeus.manifests.models import SealedManifest, Tensor, TensorSchema
from dromeus.telemetry.events import EventSink
from dromeus.transport.axl import AXLBridgeConfig, AXLTransport
from dromeus.transport.base import AsyncTransport, ReceivedBytes
from dromeus.transport.envelope import MessageType, create_envelope, encode_envelope
from dromeus.transport.receiver import Receiver, ReceiverPolicy
from dromeus.transport.sender import OutboundScheduler
from dromeus.transport.transfer import ArtifactStore, Chunk, TransferManager

MIB = 1024 * 1024
DEFAULT_PAYLOAD_SIZES = (1 * MIB, 4 * MIB, 8 * MIB, 12 * MIB)
PINNED_AXL_COMMIT = "628e28ace077f26dfe8d0259009b357216a9d8d4"


@dataclass(frozen=True, slots=True)
class ControlRTTSample:
    source: str
    destination: str
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class TransferSample:
    source: str
    destination: str
    artifact: str
    payload_bytes: int
    elapsed_seconds: float
    retry_count: int
    checksum_failure_count: int
    dromeus_wire_bytes: int


@dataclass(frozen=True, slots=True)
class TransferSummary:
    source: str
    destination: str
    artifact: str
    payload_bytes: int
    sample_count: int
    p50_seconds: float
    p95_seconds: float
    p99_seconds: float
    mean_goodput_mib_per_second: float
    retry_rate: float
    checksum_failure_rate: float
    mean_dromeus_overhead_bytes: float


@dataclass(frozen=True, slots=True)
class ArtifactCase:
    name: str
    path: Path
    tensor_schema: TensorSchema


@dataclass(frozen=True, slots=True)
class AXLBaselineReport:
    axl_version: str
    axl_binary_sha256: str
    manifest_hash: str
    rtt_layer: str
    transfer_layer: str
    artifact_sha256: Mapping[str, str]
    largest_dromeus_payload_bytes: int
    largest_axl_message_bytes: int
    axl_max_message_bytes: int
    participant_keys: tuple[str, ...]
    topology: Mapping[str, Mapping[str, object]]
    control_rtt_samples: tuple[ControlRTTSample, ...]
    transfer_samples: tuple[TransferSample, ...]
    transfer_summaries: tuple[TransferSummary, ...]

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class _MeasuredTransport:
    def __init__(self, transport: AXLTransport) -> None:
        self._transport = transport
        self.sent_bytes = 0

    async def local_public_key(self) -> str:
        return await self._transport.local_public_key()

    async def send(self, destination: str, payload: bytes) -> None:
        await self._transport.send(destination, payload)
        self.sent_bytes += len(payload)

    async def recv(self, timeout_seconds: float) -> ReceivedBytes | None:
        return await self._transport.recv(timeout_seconds)

    async def topology(self) -> dict[str, object]:
        return await self._transport.topology()


class _RecordingEventSink(EventSink):
    def __init__(self) -> None:
        self.records: list[Mapping[str, object]] = []

    def append(self, record: Mapping[str, object]) -> None:
        self.records.append(dict(record))

    @property
    def checksum_failure_count(self) -> int:
        return sum(
            record.get("event") == "transfer_message_rejected"
            and "checksum" in str(record.get("error", "")).lower()
            for record in self.records
        )


@dataclass(slots=True)
class _TransferServices:
    transport: _MeasuredTransport
    receiver: Receiver
    sender: OutboundScheduler
    manager: TransferManager
    events: _RecordingEventSink

    async def stop(self) -> None:
        await self.manager.stop()
        await self.receiver.stop()
        await self.sender.stop()


def summarize_transfer_samples(
    samples: Sequence[TransferSample],
    *,
    participant_keys: Sequence[str],
    expected_payload_bytes: Sequence[int],
) -> tuple[TransferSummary, ...]:
    """Summarize samples only after every directed pair and size is present."""
    groups: dict[tuple[str, str, str, int], list[TransferSample]] = {}
    for sample in samples:
        groups.setdefault(
            (
                sample.source,
                sample.destination,
                sample.artifact,
                sample.payload_bytes,
            ),
            [],
        ).append(sample)
    missing = [
        (source, destination, payload_bytes)
        for source in participant_keys
        for destination in participant_keys
        if source != destination
        for payload_bytes in expected_payload_bytes
        if not any(
            key[0] == source and key[1] == destination and key[3] == payload_bytes
            for key in groups
        )
    ]
    if missing:
        raise ValueError(f"missing transfer samples: {missing}")

    summaries: list[TransferSummary] = []
    for (source, destination, artifact, payload_bytes), group in sorted(groups.items()):
        elapsed = [sample.elapsed_seconds for sample in group]
        if any(value <= 0 for value in elapsed):
            raise ValueError("transfer elapsed time must be positive")
        summaries.append(
            TransferSummary(
                source=source,
                destination=destination,
                artifact=artifact,
                payload_bytes=payload_bytes,
                sample_count=len(group),
                p50_seconds=_nearest_rank(elapsed, 0.50),
                p95_seconds=_nearest_rank(elapsed, 0.95),
                p99_seconds=_nearest_rank(elapsed, 0.99),
                mean_goodput_mib_per_second=sum(
                    sample.payload_bytes / sample.elapsed_seconds / MIB
                    for sample in group
                )
                / len(group),
                retry_rate=sum(sample.retry_count > 0 for sample in group) / len(group),
                checksum_failure_rate=sum(
                    sample.checksum_failure_count for sample in group
                )
                / len(group),
                mean_dromeus_overhead_bytes=sum(
                    sample.dromeus_wire_bytes - sample.payload_bytes for sample in group
                )
                / len(group),
            )
        )
    return tuple(summaries)


async def run_axl_baseline(
    *,
    bridge_urls: Sequence[str],
    manifest: SealedManifest,
    checkpoint_path: Path,
    output_path: Path,
    axl_binary_path: Path,
    axl_build_record_path: Path,
    axl_max_message_bytes: int,
    samples_per_pair: int,
) -> AXLBaselineReport:
    """Measure raw RTT and reliable transfers with exclusive bridge ownership."""
    if len(bridge_urls) != 4:
        raise ValueError("AXL baseline requires exactly four bridge URLs")
    _validate_loopback_bridges(bridge_urls)
    if samples_per_pair <= 0:
        raise ValueError("samples_per_pair must be positive")
    axl_binary_sha256 = await asyncio.to_thread(_file_sha256, axl_binary_path)
    await asyncio.to_thread(
        _validate_axl_build_record,
        axl_build_record_path,
        axl_binary_sha256,
    )
    payload_root = output_path.parent / "payloads"
    payload_cases = await asyncio.to_thread(
        _create_payload_cases,
        payload_root,
        DEFAULT_PAYLOAD_SIZES,
    )
    checkpoint_case = ArtifactCase(
        name="checkpoint",
        path=checkpoint_path,
        tensor_schema=manifest.tensor_schema,
    )
    artifact_cases = (*payload_cases, checkpoint_case)
    (
        artifact_sha256,
        largest_dromeus_payload_bytes,
        largest_axl_message_bytes,
    ) = await asyncio.to_thread(
        preflight_artifacts,
        artifact_cases,
        manifest,
        axl_max_message_bytes,
    )
    transports = tuple(
        _MeasuredTransport(AXLTransport(AXLBridgeConfig(base_url=url.rstrip("/"))))
        for url in bridge_urls
    )
    participant_keys = tuple(
        await asyncio.gather(
            *(transport.local_public_key() for transport in transports)
        )
    )
    sealed_keys = tuple(
        sorted(participant.public_key for participant in manifest.participants)
    )
    if tuple(sorted(participant_keys)) != sealed_keys:
        raise ValueError("live AXL keys do not match sealed manifest participants")
    topology_values = await asyncio.gather(
        *(transport.topology() for transport in transports)
    )
    topology = {
        public_key: value
        for public_key, value in zip(participant_keys, topology_values, strict=True)
    }
    rtt_samples = await _measure_control_rtt(
        transports,
        participant_keys=participant_keys,
        samples_per_pair=samples_per_pair,
        timeout_seconds=manifest.transport.retry_timeout_seconds,
    )
    services = await _start_transfer_services(
        transports,
        participant_keys=participant_keys,
        manifest=manifest,
        artifact_root=output_path.parent / "received",
    )
    try:
        transfer_samples = await _measure_transfers(
            services,
            participant_keys=participant_keys,
            artifact_cases=artifact_cases,
            samples_per_pair=samples_per_pair,
            timeout_seconds=manifest.transport.retry_timeout_seconds
            * (manifest.transport.max_retries + 4),
        )
    finally:
        await asyncio.gather(*(service.stop() for service in services))
    payload_sizes = await asyncio.to_thread(
        _artifact_sizes,
        payload_cases,
    )
    report = AXLBaselineReport(
        axl_version=PINNED_AXL_COMMIT,
        axl_binary_sha256=axl_binary_sha256,
        manifest_hash=canonical_hash(manifest),
        rtt_layer="raw-axl-bridge",
        transfer_layer="dromeus-reliable-transfer",
        artifact_sha256=artifact_sha256,
        largest_dromeus_payload_bytes=largest_dromeus_payload_bytes,
        largest_axl_message_bytes=largest_axl_message_bytes,
        axl_max_message_bytes=axl_max_message_bytes,
        participant_keys=participant_keys,
        topology=topology,
        control_rtt_samples=rtt_samples,
        transfer_samples=transfer_samples,
        transfer_summaries=summarize_transfer_samples(
            transfer_samples,
            participant_keys=participant_keys,
            expected_payload_bytes=payload_sizes,
        ),
    )
    await asyncio.to_thread(report.write, output_path)
    return report


async def _measure_control_rtt(
    transports: Sequence[_MeasuredTransport],
    *,
    participant_keys: Sequence[str],
    samples_per_pair: int,
    timeout_seconds: float,
) -> tuple[ControlRTTSample, ...]:
    samples: list[ControlRTTSample] = []
    for source_index, source in enumerate(transports):
        for destination_index, destination in enumerate(transports):
            if source_index == destination_index:
                continue
            for _ in range(samples_per_pair):
                token = uuid.uuid4().hex
                ping = _probe_payload(
                    kind="ping",
                    token=token,
                    sender=participant_keys[source_index],
                )
                started = time.perf_counter()
                await source.send(participant_keys[destination_index], ping)
                await _receive_probe(
                    destination,
                    kind="ping",
                    token=token,
                    sender=participant_keys[source_index],
                    timeout_seconds=timeout_seconds,
                )
                await destination.send(
                    participant_keys[source_index],
                    _probe_payload(
                        kind="ack",
                        token=token,
                        sender=participant_keys[destination_index],
                    ),
                )
                await _receive_probe(
                    source,
                    kind="ack",
                    token=token,
                    sender=participant_keys[destination_index],
                    timeout_seconds=timeout_seconds,
                )
                samples.append(
                    ControlRTTSample(
                        source=participant_keys[source_index],
                        destination=participant_keys[destination_index],
                        elapsed_seconds=time.perf_counter() - started,
                    )
                )
    return tuple(samples)


async def _start_transfer_services(
    transports: Sequence[_MeasuredTransport],
    *,
    participant_keys: Sequence[str],
    manifest: SealedManifest,
    artifact_root: Path,
) -> tuple[_TransferServices, ...]:
    manifest_hash = canonical_hash(manifest)
    allowed = frozenset(participant_keys)
    artifact_stores = await asyncio.gather(
        *(
            asyncio.to_thread(
                ArtifactStore,
                artifact_root / f"node-{index}",
            )
            for index in range(len(transports))
        )
    )
    services: list[_TransferServices] = []
    for public_key, transport, artifact_store in zip(
        participant_keys, transports, artifact_stores, strict=True
    ):
        events = _RecordingEventSink()
        receiver = Receiver(
            cast(AsyncTransport, transport),
            ReceiverPolicy(
                run_id=manifest.run_id,
                manifest_hash=manifest_hash,
                algorithm_id=manifest.algorithm_id,
                participant_keys=allowed,
                max_payload_bytes=manifest.transport.max_payload_bytes,
            ),
            event_sink=events,
        )
        sender = OutboundScheduler(cast(AsyncTransport, transport))
        manager = TransferManager(
            local_public_key=public_key,
            run_id=manifest.run_id,
            manifest_hash=manifest_hash,
            algorithm_id=manifest.algorithm_id,
            transport_limits=manifest.transport,
            receiver=receiver,
            sender=sender,
            artifact_store=artifact_store,
            event_sink=events,
        )
        services.append(
            _TransferServices(
                transport=transport,
                receiver=receiver,
                sender=sender,
                manager=manager,
                events=events,
            )
        )
    await asyncio.gather(*(service.sender.start() for service in services))
    await asyncio.gather(*(service.receiver.start() for service in services))
    await asyncio.gather(*(service.manager.start() for service in services))
    return tuple(services)


def _artifact_sizes(artifacts: Sequence[ArtifactCase]) -> tuple[int, ...]:
    return tuple(artifact.path.stat().st_size for artifact in artifacts)


async def _measure_transfers(
    services: Sequence[_TransferServices],
    *,
    participant_keys: Sequence[str],
    artifact_cases: Sequence[ArtifactCase],
    samples_per_pair: int,
    timeout_seconds: float,
) -> tuple[TransferSample, ...]:
    samples: list[TransferSample] = []
    for source_index, source in enumerate(services):
        for destination_index, destination in enumerate(services):
            if source_index == destination_index:
                continue
            for artifact in artifact_cases:
                for _ in range(samples_per_pair):
                    before_wire_bytes = (
                        source.transport.sent_bytes + destination.transport.sent_bytes
                    )
                    before_checksum_failures = destination.events.checksum_failure_count
                    started = time.perf_counter()
                    transfer_id = await source.manager.send_artifact(
                        destination=participant_keys[destination_index],
                        artifact_name=artifact.name,
                        artifact_path=artifact.path,
                        codec_id="safetensors-v1",
                        tensor_schema=artifact.tensor_schema,
                    )
                    receipt = await destination.manager.wait_for_artifact(
                        transfer_id,
                        timeout_seconds=timeout_seconds,
                    )
                    elapsed_seconds = time.perf_counter() - started
                    timing = source.manager.last_timing
                    if timing is None:
                        raise RuntimeError("transfer timing was not recorded")
                    samples.append(
                        TransferSample(
                            source=participant_keys[source_index],
                            destination=participant_keys[destination_index],
                            artifact=artifact.name,
                            payload_bytes=receipt.size_bytes,
                            elapsed_seconds=elapsed_seconds,
                            retry_count=timing.retry_count,
                            checksum_failure_count=(
                                destination.events.checksum_failure_count
                                - before_checksum_failures
                            ),
                            dromeus_wire_bytes=(
                                source.transport.sent_bytes
                                + destination.transport.sent_bytes
                                - before_wire_bytes
                            ),
                        )
                    )
    return tuple(samples)


def _create_payload_cases(
    root: Path, target_sizes: Sequence[int]
) -> tuple[ArtifactCase, ...]:
    root.mkdir(parents=True, exist_ok=True)
    cases: list[ArtifactCase] = []
    for target_size in target_sizes:
        element_count = max(1, (target_size - 128) // 4)
        path = root / f"payload-{target_size}.safetensors"
        save_file(
            {"payload": np.zeros((element_count,), dtype=np.float32)},
            str(path),
        )
        cases.append(
            ArtifactCase(
                name=f"payload-{target_size}",
                path=path,
                tensor_schema=TensorSchema(
                    tensors=(
                        Tensor(
                            name="payload",
                            dtype="float32",
                            shape=(element_count,),
                        ),
                    )
                ),
            )
        )
    return tuple(cases)


def _probe_payload(*, kind: str, token: str, sender: str) -> bytes:
    return cast(
        bytes,
        msgpack.packb(  # pyright: ignore[reportUnknownMemberType]
            {
                "benchmark": "dromeus-axl-baseline-v1",
                "kind": kind,
                "token": token,
                "sender_public_key": sender,
            }
        ),
    )


async def _receive_probe(
    transport: _MeasuredTransport,
    *,
    kind: str,
    token: str,
    sender: str,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        inbound = await transport.recv(
            timeout_seconds=max(0.001, deadline - time.monotonic())
        )
        if inbound is None:
            continue
        value = cast(
            object,
            msgpack.unpackb(inbound.payload, raw=False),  # pyright: ignore[reportUnknownMemberType]
        )
        if not isinstance(value, dict):
            continue
        record = cast(dict[object, object], value)
        if (
            record.get("benchmark") == "dromeus-axl-baseline-v1"
            and record.get("kind") == kind
            and record.get("token") == token
            and inbound.sender_public_key == sender
        ):
            return
    raise TimeoutError(f"AXL {kind} probe timed out")


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _validate_loopback_bridges(bridge_urls: Sequence[str]) -> None:
    for bridge_url in bridge_urls:
        parsed = urlparse(bridge_url)
        if parsed.scheme != "http" or parsed.hostname not in {
            "127.0.0.1",
            "::1",
            "localhost",
        }:
            raise ValueError("AXL bridge URLs must use HTTP on loopback")


def _validate_axl_build_record(path: Path, binary_sha256: str) -> None:
    value = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise ValueError("AXL build record must be a JSON object")
    record = cast(dict[object, object], value)
    if set(record) != {"source_commit", "binary_sha256"}:
        raise ValueError("AXL build record fields do not match")
    if record["source_commit"] != PINNED_AXL_COMMIT:
        raise ValueError("AXL build record does not use the pinned commit")
    if record["binary_sha256"] != binary_sha256:
        raise ValueError("AXL binary hash does not match its build record")


def preflight_artifacts(
    artifacts: Sequence[ArtifactCase],
    manifest: SealedManifest,
    axl_max_message_bytes: int,
) -> tuple[dict[str, str], int, int]:
    if axl_max_message_bytes <= 0:
        raise ValueError("AXL max message bytes must be positive")
    artifact_hashes: dict[str, str] = {}
    largest_payload = 0
    largest_message = 0
    sender = max(
        (participant.public_key for participant in manifest.participants),
        key=len,
    )
    transfer_id = "00000000-0000-4000-8000-000000000000"
    for artifact in artifacts:
        if not artifact.path.is_file():
            raise ValueError(f"baseline artifact is missing: {artifact.name}")
        tensors = load_file(str(artifact.path))
        expected = {tensor.name: tensor for tensor in artifact.tensor_schema.tensors}
        if set(tensors) != set(expected):
            raise ValueError(f"baseline artifact schema mismatch: {artifact.name}")
        for name, tensor in tensors.items():
            contract = expected[name]
            if str(tensor.dtype) != contract.dtype or tensor.shape != contract.shape:
                raise ValueError(f"baseline artifact schema mismatch: {artifact.name}")
        data = artifact.path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        artifact_hashes[artifact.name] = digest
        chunk = Chunk(
            transfer_id=transfer_id,
            chunk_index=0,
            chunk_count=1,
            chunk_sha256=digest,
            data=data,
        )
        chunk_payload = cast(
            bytes,
            msgpack.packb(  # pyright: ignore[reportUnknownMemberType]
                chunk.model_dump(mode="python")
            ),
        )
        envelope = encode_envelope(
            create_envelope(
                message_type=MessageType.CHUNK,
                message_id=f"{transfer_id}-chunk-0",
                run_id=manifest.run_id,
                manifest_hash=canonical_hash(manifest),
                sender_public_key=sender,
                algorithm_id=manifest.algorithm_id,
                correlation_id=transfer_id,
                payload=chunk_payload,
            )
        )
        largest_payload = max(largest_payload, len(chunk_payload))
        largest_message = max(largest_message, len(envelope))
    if largest_payload > manifest.transport.max_payload_bytes:
        raise ValueError(
            "baseline manifest max_payload_bytes must permit the 12 MiB case"
        )
    if largest_message > axl_max_message_bytes:
        raise ValueError("AXL max_message_size must permit the 12 MiB case")
    return artifact_hashes, largest_payload, largest_message


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(MIB):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-url", action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--axl-binary", type=Path, required=True)
    parser.add_argument("--axl-build-record", type=Path, required=True)
    parser.add_argument("--axl-max-message-bytes", type=int, required=True)
    parser.add_argument("--samples-per-pair", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = SealedManifest.model_validate_json(
        args.manifest.read_text(encoding="utf-8")
    )
    asyncio.run(
        run_axl_baseline(
            bridge_urls=args.bridge_url,
            manifest=manifest,
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            axl_binary_path=args.axl_binary,
            axl_build_record_path=args.axl_build_record,
            axl_max_message_bytes=args.axl_max_message_bytes,
            samples_per_pair=args.samples_per_pair,
        )
    )


if __name__ == "__main__":
    main()
