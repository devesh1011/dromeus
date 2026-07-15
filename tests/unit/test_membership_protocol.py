from __future__ import annotations

import pytest
from support.sample_manifest import manifest_data

from dromeus.manifests.models import SealedManifest
from dromeus.membership.protocol import ReadyValidationError, validate_ready


def test_validate_ready_accepts_matching_environment() -> None:
    manifest = SealedManifest.model_validate(manifest_data())

    validate_ready(
        manifest=manifest,
        local_public_key="peer-0",
        environment=manifest.environment,
        dataset=manifest.dataset,
        checkpoint_hash=manifest.initial_checkpoint_hash,
    )


def test_environment_mismatch_prevents_ready() -> None:
    manifest = SealedManifest.model_validate(manifest_data())
    environment = manifest.environment.model_copy(update={"axl_version": "2.0.0"})

    with pytest.raises(
        ReadyValidationError, match="environment fingerprint does not match manifest"
    ):
        validate_ready(
            manifest=manifest,
            local_public_key="peer-0",
            environment=environment,
            dataset=manifest.dataset,
            checkpoint_hash=manifest.initial_checkpoint_hash,
        )
