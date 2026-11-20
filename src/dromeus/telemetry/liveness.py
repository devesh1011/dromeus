"""Whether a node process is running in a run root.

A running node holds an exclusive lock on `run_root/node.lock`. The operating
system releases the lock when the process ends, however it ends, so a reader of
the node's log can tell a live node from one that stopped without logging a
result. The same lock keeps two nodes from sharing one run root.
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from dromeus.errors import ConfigError


def node_lock_path(run_root: Path) -> Path:
    return run_root / "node.lock"


@contextmanager
def hold_node_lock(run_root: Path) -> Generator[None, None, None]:
    """Mark a node as running in `run_root` until the block exits."""
    run_root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(node_lock_path(run_root), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ConfigError(
                f"another node is already running in run_root {run_root}"
            ) from None
        yield
    finally:
        os.close(descriptor)  # closing the descriptor releases the lock


def node_is_running(run_root: Path) -> bool:
    """Whether a node currently holds the lock on `run_root`."""
    try:
        descriptor = os.open(node_lock_path(run_root), os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(descriptor)
    return False
