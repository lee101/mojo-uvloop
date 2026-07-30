"""Batch event-loop primitives implemented by the Mojo shared library."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from numbers import Integral
from typing import Generic, TypeVar

import numpy as np

from ._lib import addr, ensure_parallel_runtime, lib

MAX_SLEEP = 3600 * 24 * 365 * 100
_COMPACT_PARALLEL_THRESHOLD = 1_000_000
_COMPACT_TASKS = 8
T = TypeVar("T")


def _f64(values: Iterable[float]) -> np.ndarray:
    if isinstance(values, np.ndarray):
        result = np.ascontiguousarray(values, dtype=np.float64)
    else:
        result = np.fromiter(values, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError("values must be one-dimensional")
    return result


def _i64(values: Iterable[int], name: str) -> np.ndarray:
    source = values if isinstance(values, np.ndarray) else list(values)
    array = np.asarray(source)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        return np.empty(0, dtype=np.int64)
    if array.dtype.kind not in "iu":
        if array.dtype.kind != "O" or not all(
            isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
            for value in array
        ):
            raise TypeError(f"{name} must contain integers")
    try:
        result = np.ascontiguousarray(array, dtype=np.int64)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} values must fit in int64") from error
    if array.dtype.kind == "u" and array.size and np.any(array > np.iinfo(np.int64).max):
        raise ValueError(f"{name} values must fit in int64")
    return result


def _flags(values: Iterable[bool]) -> np.ndarray:
    source = values if isinstance(values, np.ndarray) else list(values)
    array = np.asarray(source)
    if array.ndim != 1:
        raise ValueError("cancelled must be one-dimensional")
    if array.dtype != np.bool_ and (
        array.dtype.kind != "O"
        or not all(isinstance(value, (bool, np.bool_)) for value in array)
    ):
        raise TypeError("cancelled must contain booleans")
    return np.ascontiguousarray(array, dtype=np.uint8)


def quantize_delays(delays: Iterable[float]) -> np.ndarray:
    """Apply uvloop's clamp and nearest-millisecond delay conversion in bulk."""
    values = _f64(delays)
    if np.isnan(values).any():
        raise ValueError("cannot convert float NaN to integer")
    result = np.empty(values.size, dtype=np.uint64)
    if values.size:
        lib().muv_quantize_delays(addr(values), values.size, addr(result), MAX_SLEEP)
    return result


def order_timers(deadlines: Iterable[float]) -> np.ndarray:
    """Return a stable ascending permutation for finite timer deadlines."""
    values = _f64(deadlines)
    if not np.isfinite(values).all():
        raise ValueError("timer deadlines must be finite")
    indices = np.empty(values.size, dtype=np.int64)
    work = np.empty(values.size, dtype=np.int64)
    if values.size:
        lib().muv_order_timers(addr(values), values.size, addr(indices), addr(work))
    return indices


def compact_ready(cancelled: Iterable[bool]) -> np.ndarray:
    """Return live ready-queue positions in FIFO order."""
    flags = _flags(cancelled)
    indices = np.empty(flags.size, dtype=np.int64)
    count = 0
    if flags.size:
        counts_addr = 0
        if flags.size >= _COMPACT_PARALLEL_THRESHOLD:
            ensure_parallel_runtime()
            counts = np.empty(_COMPACT_TASKS, dtype=np.int64)
            counts_addr = addr(counts)
        count = lib().muv_compact_ready(
            addr(flags), flags.size, addr(indices), counts_addr
        )
        if count < 0 or count > flags.size:
            raise RuntimeError("Mojo returned an invalid compacted length")
    return indices[:count]


def due_indices(
    sorted_deadlines: Iterable[float],
    cancelled: Iterable[bool],
    now: float,
    *,
    start: int = 0,
) -> tuple[np.ndarray, int]:
    """Extract live due positions from an ascending deadline array."""
    deadlines = _f64(sorted_deadlines)
    flags = _flags(cancelled)
    if deadlines.size != flags.size:
        raise ValueError("deadlines and cancelled must be equal-length 1D arrays")
    if not np.isfinite(deadlines).all():
        raise ValueError("timer deadlines must be finite")
    if np.any(deadlines[1:] < deadlines[:-1]):
        raise ValueError("deadlines must be sorted")
    if not np.isfinite(now):
        raise ValueError("now must be finite")
    if not isinstance(start, Integral):
        raise TypeError("start must be an integer")
    if start < 0 or start > deadlines.size:
        raise ValueError("start must be between zero and the input length")
    indices = np.empty(deadlines.size, dtype=np.int64)
    meta = np.zeros(2, dtype=np.int64)
    if deadlines.size:
        lib().muv_due_indices(
            addr(deadlines),
            addr(flags),
            deadlines.size,
            float(now),
            int(start),
            addr(indices),
            addr(meta),
        )
    cursor = int(meta[0])
    count = int(meta[1])
    if cursor < start or cursor > deadlines.size or count < 0 or count > cursor - start:
        raise RuntimeError("Mojo returned invalid due-timer metadata")
    return indices[:count], cursor


def coalesce_events(
    fds: Iterable[int], masks: Iterable[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Merge repeated fd readiness masks, preserving first-seen fd order."""
    fd_values = _i64(fds, "fds")
    mask_values = _i64(masks, "masks")
    if fd_values.size != mask_values.size:
        raise ValueError("fds and masks must be equal-length 1D arrays")
    if np.any(fd_values < 0):
        raise ValueError("file descriptors must be non-negative")
    n = fd_values.size
    result_fds = np.empty(n, dtype=np.int64)
    result_masks = np.empty(n, dtype=np.int64)
    if not n:
        return result_fds, result_masks
    capacity = 1
    while capacity < n * 2:
        capacity <<= 1
    hash_keys = np.empty(capacity, dtype=np.int64)
    hash_positions = np.empty(capacity, dtype=np.int64)
    count = lib().muv_coalesce_events(
        addr(fd_values),
        addr(mask_values),
        n,
        addr(result_fds),
        addr(result_masks),
        addr(hash_keys),
        addr(hash_positions),
        capacity,
    )
    if count < 0 or count > n:
        raise RuntimeError("Mojo returned an invalid coalesced length")
    return result_fds[:count], result_masks[:count]


class TimerBatch(Generic[T]):
    """A sorted, cancellable batch of deadlines with O(k) due extraction."""

    def __init__(self, deadlines: Sequence[float], payloads: Sequence[T] | None = None):
        values = _f64(deadlines)
        if values.ndim != 1:
            raise ValueError("deadlines must be one-dimensional")
        if payloads is None:
            payloads = list(range(values.size))  # type: ignore[assignment]
        if len(payloads) != values.size:
            raise ValueError("deadlines and payloads must have equal length")
        order = order_timers(values)
        self.deadlines = values[order]
        self.payloads = [payloads[int(i)] for i in order]
        self._cancelled = np.zeros(values.size, dtype=np.bool_)
        self._inverse = np.empty(values.size, dtype=np.int64)
        self._inverse[order] = np.arange(values.size, dtype=np.int64)
        self._cursor = 0

    def cancel(self, index: int) -> None:
        """Cancel an item by its original input position."""
        if index < 0 or index >= self._inverse.size:
            raise IndexError(index)
        self._cancelled[self._inverse[index]] = 1

    def pop_due(self, now: float) -> list[T]:
        positions, self._cursor = due_indices(
            self.deadlines, self._cancelled, now, start=self._cursor
        )
        return [self.payloads[int(i)] for i in positions]

    def __len__(self) -> int:
        if self._cursor >= self.deadlines.size:
            return 0
        return int(np.count_nonzero(self._cancelled[self._cursor :] == 0))
