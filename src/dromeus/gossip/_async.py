"""Cancellation-safe offloading shared by gossip orchestration and adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TypeVar

_T = TypeVar("_T")


async def run_blocking(callback: Callable[..., _T], *args: object) -> _T:
    task = asyncio.create_task(asyncio.to_thread(callback, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise
