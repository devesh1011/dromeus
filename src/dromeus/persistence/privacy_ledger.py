"""Local, non-rollbackable full-horizon privacy reservations.

The trusted operator must preserve this file and compose external dataset uses.
An explicitly created new lineage is not evidence that data has never been used.
"""

from __future__ import annotations

import fcntl
import hashlib
import math
import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal
from uuid import uuid4

from dromeus.manifests.canonical import canonical_json
from dromeus.manifests.models import DomainModel, PrivacyReservation, Sha256


class _LedgerState(DomainModel):
    version: Literal[1] = 1
    lineage_hash: Sha256
    reservations: tuple[PrivacyReservation, ...] = ()


class PrivacyLedger:
    """Atomic, locked storage independent of run/model checkpoints."""

    def __init__(self, path: Path, *, lineage: str) -> None:
        if not lineage.strip():
            raise ValueError("local dataset lineage is required")
        self.path = path
        self._lineage_hash = hashlib.sha256(lineage.encode()).hexdigest()

    @classmethod
    def create(cls, path: Path, *, lineage: str) -> PrivacyLedger:
        """Explicit bootstrap for a new controlled lineage; never overwrite."""
        ledger = cls(path, lineage=lineage)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with ledger._locked():
            if path.exists() or path.is_symlink():
                raise ValueError("privacy ledger already exists")
            ledger._write(_LedgerState(lineage_hash=ledger._lineage_hash))
        return ledger

    @contextmanager
    def _locked(self) -> Generator[None]:
        lock_path = self.path.with_name(self.path.name + ".lock")
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _read(self) -> _LedgerState:
        if not self.path.is_file() or self.path.is_symlink():
            raise ValueError("privacy ledger missing or unsafe")
        try:
            state = _LedgerState.model_validate_json(self.path.read_bytes())
        except Exception:
            raise ValueError("privacy ledger corrupt") from None
        if state.lineage_hash != self._lineage_hash:
            raise ValueError("privacy ledger lineage mismatch")
        ids = [r.run_id for r in state.reservations]
        if len(ids) != len(set(ids)):
            raise ValueError("privacy ledger has duplicate reservations")
        return state

    def _write(self, state: _LedgerState) -> None:
        temp = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            descriptor = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(canonical_json(state))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temp.unlink(missing_ok=True)

    def reservations(self) -> tuple[PrivacyReservation, ...]:
        with self._locked():
            return self._read().reservations

    def reserve(
        self,
        reservation: PrivacyReservation,
        *,
        epsilon_limit: float | None,
        forecast: Callable[[tuple[PrivacyReservation, ...]], float],
    ) -> None:
        """Forecast the composed history, persist, then permit output."""
        with self._locked():
            state = self._read()
            if any(r.run_id == reservation.run_id for r in state.reservations):
                raise ValueError("run already reserved; use a new run identity")
            history = (*state.reservations, reservation)
            expenditure = forecast(history)
            if not math.isfinite(expenditure) or expenditure < 0:
                raise ValueError("invalid privacy budget forecast")
            if epsilon_limit is not None and expenditure > epsilon_limit:
                raise ValueError("composed privacy budget exceeds limit")
            self._write(state.model_copy(update={"reservations": history}))

    def check(self, reservation: PrivacyReservation) -> None:
        if reservation not in self.reservations():
            raise ValueError("privacy reservation missing or changed")
