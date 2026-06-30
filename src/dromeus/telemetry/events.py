"""Structured event output shared by Dromeus runtime modules."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from datetime import UTC, datetime


def emit_event(
    event: str,
    *,
    run_id: str | None = None,
    message_id: str | None = None,
    transfer_id: str | None = None,
    peer_id: str | None = None,
    round_id: int | None = None,
    **fields: object,
) -> None:
    """Write one JSON event, retaining correlation IDs when supplied."""
    record: dict[str, object] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": event,
    }
    identifiers: Mapping[str, str | int | None] = {
        "run_id": run_id,
        "message_id": message_id,
        "transfer_id": transfer_id,
        "peer_id": peer_id,
        "round_id": round_id,
    }
    record.update(
        {key: value for key, value in identifiers.items() if value is not None}
    )
    record.update(fields)
    print(json.dumps(record, separators=(",", ":"), sort_keys=True), file=sys.stdout)
