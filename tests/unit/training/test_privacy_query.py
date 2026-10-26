"""Query copies preserve the pinned mechanism bound and durable reservations."""

from pathlib import Path
from typing import Literal

import pytest

from dromeus.manifests.canonical import canonical_hash
from dromeus.manifests.models import PrivacyPolicy, PrivacyReservation
from dromeus.persistence.privacy_ledger import PrivacyLedger
from dromeus.training.privacy import opacus_module, privacy_epsilon, reserve_horizon

pytest.importorskip("opacus")

History = tuple[tuple[float, float, int], ...]


def policy(**changes: object) -> PrivacyPolicy:
    return PrivacyPolicy.model_validate(
        {
            "max_grad_norm": 1,
            "noise_multiplier": 1,
            "delta": 1e-5,
            "max_logical_steps": 100,
            "randomness_profile": "reproducible_public_benchmark",
            **changes,
        }
    )


def direct_epsilon(history: History, setting: PrivacyPolicy, kind: str) -> float:
    accountants = opacus_module("opacus.accountants")
    engine = (
        accountants.PRVAccountant() if kind == "prv" else accountants.RDPAccountant()
    )
    engine.history = list(history)
    if kind == "prv":
        return float(
            engine.get_epsilon(
                delta=setting.delta,
                eps_error=setting.prv_epsilon_error,
                delta_error=setting.delta * setting.prv_delta_error_ratio,
            )
        )
    return float(engine.get_epsilon(delta=setting.delta))


@pytest.mark.parametrize("kind", ["prv", "rdp"])
@pytest.mark.parametrize(
    "history",
    [
        ((1.0, 0.1, 2), (2.0, 0.1, 3), (1.0, 0.1, 4)),
        ((1.0, 0.1, 2), (1.0, 0.2, 3), (1.0, 0.1, 4)),
        ((1.0, 0.1, 2), (2.0, 0.2, 3), (1.0, 0.1, 4), (2.0, 0.2, 1)),
    ],
)
def test_query_matches_direct_raw_opacus_without_mutating_input(
    history: History, kind: Literal["prv", "rdp"]
) -> None:
    original = tuple(tuple(row) for row in history)
    setting = policy()
    expected = direct_epsilon(history, setting, kind)
    actual = privacy_epsilon(history, setting, accountant=kind)
    tolerance = 1e-6 if kind == "prv" else 1e-9
    assert actual == pytest.approx(expected, abs=tolerance, rel=1e-9)
    assert history == original


@pytest.mark.parametrize("kind", ["prv", "rdp"])
def test_empty_and_singleton_query_keep_their_meaning(kind: str) -> None:
    setting = policy(prv_epsilon_error=0.005, prv_delta_error_ratio=0.002)
    assert privacy_epsilon((), setting, accountant=kind) == 0
    history = ((1.0, 0.1, 3),)
    assert privacy_epsilon(history, setting, accountant=kind) == pytest.approx(
        direct_epsilon(history, setting, kind), abs=1e-9, rel=1e-9
    )


def test_query_does_not_merge_nearby_but_different_mechanisms() -> None:
    setting = policy()
    history = ((1.0, 0.1, 20), (1.001, 0.1001, 20), (1.0, 0.1, 10))
    expected = direct_epsilon(history, setting, "rdp")
    wrongly_rounded = direct_epsilon(((1.0, 0.1, 50),), setting, "rdp")
    assert abs(expected - wrongly_rounded) > 1e-6
    assert privacy_epsilon(history, setting, accountant="rdp") == pytest.approx(
        expected, abs=1e-9, rel=1e-9
    )


def test_reservation_query_preserves_raw_entries_and_rejects_over_budget(
    tmp_path: Path,
) -> None:
    setting = policy()
    ledger = PrivacyLedger.create(tmp_path / "ledger.json", lineage="public-query-test")
    first = PrivacyReservation(
        run_id="first",
        policy_hash=canonical_hash(setting),
        sample_rate=0.1,
        noise_multiplier=1,
        logical_steps=2,
    )
    second = first.model_copy(update={"run_id": "second", "logical_steps": 3})
    reserve_horizon(ledger, first, setting)
    reserve_horizon(ledger, second, setting)
    assert ledger.reservations() == (first, second)
    before = ledger.path.read_bytes()
    history = ((1.0, 0.1, 2), (1.0, 0.1, 3))
    assert privacy_epsilon(history, setting) == pytest.approx(
        direct_epsilon(history, setting, "prv"), abs=1e-6, rel=1e-9
    )
    assert ledger.path.read_bytes() == before
    with pytest.raises(ValueError, match="budget"):
        reserve_horizon(
            ledger,
            first.model_copy(update={"run_id": "over-budget"}),
            policy(epsilon_limit=0.0001),
        )
    assert ledger.reservations() == (first, second)
    assert ledger.path.read_bytes() == before
