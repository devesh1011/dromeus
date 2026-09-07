from __future__ import annotations

from pathlib import Path

import pytest
from support.sample_manifest import manifest_data, write_checkpoint

from benchmarks.cifar10.axl_baseline import (
    ArtifactCase,
    TransferSample,
    preflight_artifacts,
    summarize_transfer_samples,
)
from dromeus.manifests.models import SealedManifest


def test_transfer_summary_covers_every_directed_pair_and_payload() -> None:
    peers = ("a", "b", "c", "d")
    samples = tuple(
        TransferSample(
            source=source,
            destination=destination,
            artifact=f"payload-{payload_bytes}",
            payload_bytes=payload_bytes,
            elapsed_seconds=elapsed_seconds,
            retry_count=retry_count,
            checksum_failure_count=0,
            dromeus_wire_bytes=payload_bytes + 128,
        )
        for source in peers
        for destination in peers
        if source != destination
        for payload_bytes in (1, 4, 8, 12)
        for elapsed_seconds, retry_count in ((1.0, 0), (2.0, 1))
    )

    summaries = summarize_transfer_samples(
        samples,
        participant_keys=peers,
        expected_payload_bytes=(1, 4, 8, 12),
    )

    assert len(summaries) == 48
    first = summaries[0]
    assert first.sample_count == 2
    assert first.p50_seconds == pytest.approx(1.0)
    assert first.p95_seconds == pytest.approx(2.0)
    assert first.p99_seconds == pytest.approx(2.0)
    assert first.retry_rate == pytest.approx(0.5)
    assert first.checksum_failure_rate == 0.0
    assert first.mean_goodput_mib_per_second == pytest.approx(
        ((1 / 1.0) + (1 / 2.0)) / 2 / (1024 * 1024)
    )
    assert first.mean_dromeus_overhead_bytes == 128.0


def test_transfer_summary_rejects_missing_pair_measurement() -> None:
    with pytest.raises(ValueError, match="missing transfer samples"):
        summarize_transfer_samples(
            (),
            participant_keys=("a", "b", "c", "d"),
            expected_payload_bytes=(1,),
        )


def test_artifact_preflight_enforces_exact_encoded_boundaries(
    tmp_path: Path,
) -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    checkpoint = tmp_path / "checkpoint.safetensors"
    write_checkpoint(checkpoint)
    artifact = ArtifactCase(
        name="checkpoint",
        path=checkpoint,
        tensor_schema=manifest.tensor_schema,
    )
    _, payload_bytes, message_bytes = preflight_artifacts(
        (artifact,),
        manifest.model_copy(
            update={
                "transport": manifest.transport.model_copy(
                    update={"max_payload_bytes": 16 * 1024 * 1024}
                )
            }
        ),
        16 * 1024 * 1024,
    )
    exact_manifest = manifest.model_copy(
        update={
            "transport": manifest.transport.model_copy(
                update={"max_payload_bytes": payload_bytes}
            )
        }
    )

    preflight_artifacts((artifact,), exact_manifest, message_bytes)
    with pytest.raises(ValueError, match="manifest max_payload_bytes"):
        preflight_artifacts(
            (artifact,),
            exact_manifest.model_copy(
                update={
                    "transport": exact_manifest.transport.model_copy(
                        update={"max_payload_bytes": payload_bytes - 1}
                    )
                }
            ),
            message_bytes,
        )
    with pytest.raises(ValueError, match="AXL max_message_size"):
        preflight_artifacts((artifact,), exact_manifest, message_bytes - 1)
