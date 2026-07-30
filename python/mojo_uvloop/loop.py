"""uvloop-shaped event loop API backed by the stdlib reactor and Mojo batches."""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from .primitives import MAX_SLEEP, order_timers, quantize_delays


def _quantize_delay(delay: float) -> float:
    if delay < 0:
        delay = 0.0
    elif delay == float("inf") or delay > MAX_SLEEP:
        delay = float(MAX_SLEEP)
    return round(delay * 1000) / 1000


class Loop(asyncio.SelectorEventLoop):
    """Selector event loop with uvloop-compatible timer semantics."""

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__module__.split('.')[0]}.Loop "
            f"running={self.is_running()} closed={self.is_closed()} "
            f"debug={self.get_debug()}>"
        )

    def call_later(
        self,
        delay: float,
        callback: Callable[..., Any],
        *args: Any,
        context: contextvars.Context | None = None,
    ) -> asyncio.Handle:
        quantized = _quantize_delay(delay)
        if quantized == 0:
            return self.call_soon(callback, *args, context=context)
        return asyncio.BaseEventLoop.call_at(
            self, self.time() + quantized, callback, *args, context=context
        )

    def call_at(
        self,
        when: float,
        callback: Callable[..., Any],
        *args: Any,
        context: contextvars.Context | None = None,
    ) -> asyncio.Handle:
        quantized = _quantize_delay(when - self.time())
        if quantized == 0:
            return self.call_soon(callback, *args, context=context)
        return asyncio.BaseEventLoop.call_at(
            self, self.time() + quantized, callback, *args, context=context
        )

    def call_soon_many(
        self,
        callbacks: Iterable[Callable[..., Any]],
        args: Sequence[Sequence[Any]] | None = None,
        *,
        context: contextvars.Context | None = None,
    ) -> list[asyncio.Handle]:
        callback_list = list(callbacks)
        arg_list = _normalize_args(callback_list, args)
        return [
            self.call_soon(callback, *callback_args, context=context)
            for callback, callback_args in zip(callback_list, arg_list, strict=True)
        ]

    def call_later_many(
        self,
        delays: Iterable[float],
        callbacks: Iterable[Callable[..., Any]],
        args: Sequence[Sequence[Any]] | None = None,
        *,
        context: contextvars.Context | None = None,
    ) -> list[asyncio.Handle]:
        delay_ms = quantize_delays(delays)
        callback_list = list(callbacks)
        if delay_ms.size != len(callback_list):
            raise ValueError("delays and callbacks must have equal length")
        arg_list = _normalize_args(callback_list, args)
        order = order_timers(delay_ms.astype("float64", copy=False))
        handles: list[asyncio.Handle | None] = [None] * len(callback_list)
        base = self.time()
        for raw_index in order:
            index = int(raw_index)
            delay = float(delay_ms[index]) / 1000
            if delay == 0:
                handle = self.call_soon(
                    callback_list[index], *arg_list[index], context=context
                )
            else:
                handle = asyncio.BaseEventLoop.call_at(
                    self,
                    base + delay,
                    callback_list[index],
                    *arg_list[index],
                    context=context,
                )
            handles[index] = handle
        return [handle for handle in handles if handle is not None]

    def call_at_many(
        self,
        when: Iterable[float],
        callbacks: Iterable[Callable[..., Any]],
        args: Sequence[Sequence[Any]] | None = None,
        *,
        context: contextvars.Context | None = None,
    ) -> list[asyncio.Handle]:
        now = self.time()
        return self.call_later_many(
            (deadline - now for deadline in when), callbacks, args, context=context
        )


def _normalize_args(
    callbacks: Sequence[Callable[..., Any]], args: Sequence[Sequence[Any]] | None
) -> list[Sequence[Any]]:
    if args is None:
        return [()] * len(callbacks)
    if len(args) != len(callbacks):
        raise ValueError("args and callbacks must have equal length")
    return list(args)
