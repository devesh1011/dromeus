"""Lazy optional Opacus boundary and mechanism accounting."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable
from typing import Any, Protocol

from dromeus.manifests.models import PrivacyPolicy, PrivacyReservation


class ReservationStore(Protocol):
    def reserve(
        self,
        reservation: PrivacyReservation,
        *,
        epsilon_limit: float | None,
        forecast: Callable[[tuple[PrivacyReservation, ...]], float],
    ) -> None: ...

    def check(self, reservation: PrivacyReservation) -> None: ...


def opacus_module(name: str = "opacus") -> Any:
    """Keep untyped upstream API confined to the optional implementation."""
    try:
        module = importlib.import_module(name)
        if importlib.import_module("opacus").__version__ != "1.6.0":
            raise ValueError("private training requires Opacus 1.6.0")
        return module
    except ImportError:
        raise ValueError(
            "install dromeus[privacy] and secure RNG when required"
        ) from None


def privacy_epsilon(
    history: tuple[tuple[float, float, int], ...],
    policy: PrivacyPolicy,
    *,
    accountant: str = "prv",
) -> float:
    if not history:
        return 0.0
    accountants = opacus_module("opacus.accountants")
    engine = (
        accountants.PRVAccountant()
        if accountant == "prv"
        else accountants.RDPAccountant()
    )
    engine.history = list(history)
    kwargs = (
        {
            "eps_error": policy.prv_epsilon_error,
            "delta_error": policy.delta * policy.prv_delta_error_ratio,
        }
        if accountant == "prv"
        else {}
    )
    value = float(engine.get_epsilon(delta=policy.delta, **kwargs))
    if not math.isfinite(value) or value < 0:
        raise ValueError("privacy accounting failed")
    return value


def reserve_horizon(
    store: ReservationStore,
    reservation: PrivacyReservation,
    policy: PrivacyPolicy,
) -> None:
    store.reserve(
        reservation,
        epsilon_limit=policy.epsilon_limit,
        forecast=lambda records: privacy_epsilon(
            tuple(
                (r.noise_multiplier, r.sample_rate, r.logical_steps) for r in records
            ),
            policy,
        ),
    )
