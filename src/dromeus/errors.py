"""Errors raised by Dromeus.

Catch `DromeusError` for any library failure. Input errors are also `ValueError`
and run failures are also `RuntimeError`, following Python convention.
"""

from __future__ import annotations


class DromeusError(Exception):
    """Base class for every error Dromeus raises on purpose."""


class ConfigError(DromeusError, ValueError):
    """A node configuration or local setting is invalid."""


class TaskError(DromeusError, ValueError):
    """A training task, its model, batches, loss or optimizer is not admissible."""


class DraftError(DromeusError, ValueError):
    """A draft is malformed or does not match the local task or environment."""


class PrivacyError(DromeusError, ValueError):
    """A privacy policy, ledger, budget or private execution check failed."""


class FormationError(DromeusError, RuntimeError):
    """The group could not form: invitation, membership or readiness failed."""


class RunError(DromeusError, RuntimeError):
    """The run failed after formation: peers, transport, rounds or persistence."""


def one_line(message: str) -> str:
    """Collapse a multi-line error message, such as pydantic's, to one line."""
    lines = (line.strip() for line in message.splitlines())
    return "; ".join(
        line
        for line in lines
        if line and not line.startswith("For further information visit")
    )


__all__ = [
    "ConfigError",
    "DraftError",
    "DromeusError",
    "FormationError",
    "PrivacyError",
    "RunError",
    "TaskError",
    "one_line",
]
