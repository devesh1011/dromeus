"""Durable reservations survive run rollback and compose mechanism histories."""

from pathlib import Path

import pytest

from dromeus.manifests.models import PrivacyReservation
from dromeus.persistence.privacy_ledger import PrivacyLedger


def test_reservations_are_not_refunded_or_duplicated(tmp_path: Path) -> None:
    ledger = PrivacyLedger.create(tmp_path / "ledger.json", lineage="synthetic-fixed")
    first = PrivacyReservation(
        run_id="run-1",
        policy_hash="a" * 64,
        sample_rate=0.25,
        noise_multiplier=1,
        logical_steps=4,
    )
    ledger.reserve(
        first,
        epsilon_limit=2,
        forecast=lambda history: sum(x.logical_steps for x in history) / 4,
    )
    ledger.check(first)
    reopened = PrivacyLedger(tmp_path / "ledger.json", lineage="synthetic-fixed")
    with pytest.raises(ValueError, match="already reserved"):
        reopened.reserve(first, epsilon_limit=None, forecast=lambda _: 0)
    second = first.model_copy(update={"run_id": "run-2", "logical_steps": 5})
    with pytest.raises(ValueError, match="budget"):
        reopened.reserve(
            second,
            epsilon_limit=2,
            forecast=lambda history: sum(x.logical_steps for x in history) / 4,
        )
    assert reopened.reservations() == (first,)
    (tmp_path / "ledger.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        reopened.check(first)


def test_failed_persistence_does_not_authorize_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    ledger = PrivacyLedger.create(tmp_path / "ledger.json", lineage="synthetic")
    reservation = PrivacyReservation(
        run_id="fail",
        policy_hash="b" * 64,
        sample_rate=0.5,
        noise_multiplier=1,
        logical_steps=2,
    )

    def fail(*args: object) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        ledger.reserve(reservation, epsilon_limit=None, forecast=lambda _: 1)
    with pytest.raises(ValueError, match="reservation"):
        ledger.check(reservation)
    assert ledger.reservations() == ()


def test_corrupt_or_wrong_lineage_ledger_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    PrivacyLedger.create(path, lineage="first")
    with pytest.raises(ValueError, match="lineage"):
        PrivacyLedger(path, lineage="second").reservations()
    path.write_text('{"version":1}')
    with pytest.raises(ValueError, match="corrupt"):
        PrivacyLedger(path, lineage="first").reservations()
