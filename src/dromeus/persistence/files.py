"""Crash-safe local file replacement shared by run state, ledgers and node files."""

from __future__ import annotations

import os
import uuid
from pathlib import Path


def atomic_write_bytes(path: Path, data: bytes, *, mode: int = 0o666) -> None:
    """Durably replace ``path`` with ``data`` or leave the old file in place.

    Writes a uniquely named sibling temp file (created exclusively with ``mode``,
    subject to the umask), fsyncs it, renames it over ``path`` and fsyncs the
    directory. Raises ``OSError`` on failure after removing the temp file.
    """
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def fsync_directory(path: Path) -> None:
    """Persist directory entries; skipped where directories cannot be opened."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
