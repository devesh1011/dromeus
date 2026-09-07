"""Cancellation-safe offloading shared by gossip orchestration and adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Callable


async def run_blocking[T](callback: Callable[..., T], *args: object) -> T:
    task = asyncio.create_task(asyncio.to_thread(callback, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise
