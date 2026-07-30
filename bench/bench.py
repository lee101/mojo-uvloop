"""Benchmarks against uvloop and source-equivalent Python/NumPy references."""

from __future__ import annotations

import os
import platform
import sys
import time
from collections import OrderedDict

import numpy as np
import uvloop

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojo_uvloop as muv  # noqa: E402


def best_time(function, repetitions=3):
    best = float("inf")
    result = None
    for _ in range(repetitions):
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - start)
    return best, result


def numpy_quantize(values):
    clipped = np.minimum(np.maximum(values, 0.0), float(muv.MAX_SLEEP))
    return np.rint(clipped * 1000).astype(np.uint64)


def python_coalesce(fds, masks):
    result = OrderedDict()
    for fd, mask in zip(fds, masks, strict=True):
        key = int(fd)
        result[key] = result.get(key, 0) | int(mask)
    return result


def callback_dispatch(factory, n):
    loop = factory()
    try:
        for _ in range(n):
            loop.call_soon(lambda: None)
        loop.call_soon(loop.stop)
        loop.run_forever()
    finally:
        loop.close()


def scalar_timer_schedule(factory, delays):
    loop = factory()
    try:
        handles = [loop.call_later(float(delay), lambda: None) for delay in delays]
        return len(handles)
    finally:
        loop.close()


def batch_timer_schedule(delays):
    loop = muv.new_event_loop()
    callback = lambda: None
    try:
        return len(loop.call_later_many(delays, [callback] * delays.size))
    finally:
        loop.close()


def fmt(seconds):
    if seconds < 0.001:
        return f"{seconds * 1e6:.1f} us"
    return f"{seconds * 1e3:.2f} ms"


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    rng = np.random.default_rng(0)
    rows = []

    delays = rng.random(5_000_000) * 1000
    mojo_s, got = best_time(lambda: muv.quantize_delays(delays))
    ref_s, expected = best_time(lambda: numpy_quantize(delays))
    assert np.array_equal(got, expected)
    rows.append(("quantize 5M delays", mojo_s, ref_s, "NumPy uvloop formula"))

    deadlines = rng.integers(0, 100_000, size=1_000_000).astype(np.float64)
    mojo_s, got = best_time(lambda: muv.order_timers(deadlines))
    ref_s, expected = best_time(lambda: np.argsort(deadlines, kind="stable"))
    assert np.array_equal(got, expected)
    rows.append(("order 1M timers", mojo_s, ref_s, "NumPy stable argsort"))

    cancelled = rng.random(5_000_000) < 0.3
    mojo_s, got = best_time(lambda: muv.compact_ready(cancelled))
    ref_s, expected = best_time(lambda: np.flatnonzero(~cancelled))
    assert np.array_equal(got, expected)
    rows.append(("compact 5M ready", mojo_s, ref_s, "NumPy flatnonzero"))

    fds = rng.integers(0, 65_536, size=1_000_000, dtype=np.int64)
    masks = rng.choice(np.array([1, 2, 4], dtype=np.int64), size=fds.size)
    mojo_s, got = best_time(lambda: muv.coalesce_events(fds, masks))
    ref_s, expected = best_time(lambda: python_coalesce(fds, masks))
    assert got[0].tolist() == list(expected)
    assert got[1].tolist() == list(expected.values())
    rows.append(("coalesce 1M poll events", mojo_s, ref_s, "Python ordered dict"))

    callback_n = 200_000
    mojo_s, _ = best_time(
        lambda: callback_dispatch(muv.new_event_loop, callback_n), repetitions=2
    )
    ref_s, _ = best_time(
        lambda: callback_dispatch(uvloop.new_event_loop, callback_n), repetitions=2
    )
    rows.append(("schedule + dispatch 200k callbacks", mojo_s, ref_s, "upstream uvloop"))

    timer_delays = rng.random(100_000) * 60 + 1
    mojo_s, count = best_time(lambda: batch_timer_schedule(timer_delays), repetitions=2)
    ref_s, ref_count = best_time(
        lambda: scalar_timer_schedule(uvloop.new_event_loop, timer_delays),
        repetitions=2,
    )
    assert count == ref_count == timer_delays.size
    rows.append(("schedule 100k timers", mojo_s, ref_s, "upstream scalar API"))

    print(
        f"Machine: {cpu_name()}; {platform.system()} {platform.release()}; "
        f"Python {platform.python_version()}; uvloop {uvloop.__version__}"
    )
    print()
    print("| benchmark | mojo-uvloop | reference | ratio | reference |")
    print("| --- | ---: | ---: | ---: | --- |")
    for name, mojo_s, ref_s, reference in rows:
        print(
            f"| {name} | {fmt(mojo_s)} | {fmt(ref_s)} | "
            f"{ref_s / mojo_s:.2f}x | {reference} |"
        )


if __name__ == "__main__":
    main()
