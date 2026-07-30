"""Mojo implementations of uvloop's batchable event-loop primitives."""

from __future__ import annotations

import asyncio
import sys
import threading
import warnings
from typing import Any, Coroutine, TypeVar

from .loop import Loop
from .primitives import (
    MAX_SLEEP,
    TimerBatch,
    coalesce_events,
    compact_ready,
    due_indices,
    order_timers,
    quantize_delays,
)

__version__ = "0.1.0"
__all__ = ("new_event_loop", "run", "install", "EventLoopPolicy")
T = TypeVar("T")


def new_event_loop() -> Loop:
    """Return a new event loop."""
    return Loop()


def run(
    main: Coroutine[Any, Any, T],
    *,
    loop_factory=new_event_loop,
    debug: bool | None = None,
    **run_kwargs: Any,
) -> T:
    """Run a coroutine using a mojo-uvloop Loop."""

    async def wrapper() -> T:
        loop = asyncio.get_running_loop()
        if not isinstance(loop, Loop):
            main.close()
            raise TypeError("mojo_uvloop.run() uses a non-mojo-uvloop event loop")
        return await main

    return asyncio.run(
        wrapper(),
        loop_factory=loop_factory,
        debug=debug,
        **run_kwargs,
    )


class EventLoopPolicy(asyncio.AbstractEventLoopPolicy):
    """Deprecated event loop policy matching uvloop's compatibility API."""

    class _Local(threading.local):
        loop: asyncio.AbstractEventLoop | None = None

    def __init__(self) -> None:
        self._local = self._Local()

    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        if self._local.loop is None:
            raise RuntimeError(
                f"There is no current event loop in thread {threading.current_thread().name!r}."
            )
        return self._local.loop

    def set_event_loop(self, loop: asyncio.AbstractEventLoop | None) -> None:
        if loop is not None and not isinstance(loop, asyncio.AbstractEventLoop):
            raise TypeError(
                "loop must be an instance of AbstractEventLoop or None, "
                f"not {type(loop).__name__!r}"
            )
        self._local.loop = loop

    def new_event_loop(self) -> Loop:
        return new_event_loop()


def install() -> None:
    """Install the deprecated event loop policy compatibility API."""
    if sys.version_info[:2] >= (3, 12):
        warnings.warn(
            "mojo_uvloop.install() is deprecated in favor of mojo_uvloop.run()",
            DeprecationWarning,
            stacklevel=1,
        )
    asyncio.set_event_loop_policy(EventLoopPolicy())
