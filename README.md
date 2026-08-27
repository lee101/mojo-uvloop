# mojo-uvloop

`mojo-uvloop` ports the batchable event-loop primitives around
[uvloop](https://github.com/MagicStack/uvloop) to Mojo. It provides an
independent, uvloop-shaped Python API for Linux: upstream uvloop is used for
tests and benchmarks, not as the runtime backend.

This is deliberately not a rewrite of libuv. CPython's selector event loop
provides the reactor and standard transports; Mojo handles operations that
benefit from crossing the FFI boundary in batches.

## Covered subset

- `new_event_loop()`, `run()`, deprecated `install()`, and `EventLoopPolicy`
- `Loop.call_soon`, `call_soon_threadsafe`, `call_later`, and `call_at`
- uvloop-compatible timer clamping and half-even millisecond quantization
- tasks, futures, executors, selector-based TCP servers and clients
- bulk `call_soon_many`, `call_later_many`, and `call_at_many` extensions
- Mojo primitives for timer ordering, due-timer extraction, cancelled-ready
  compaction, and repeated poll-event coalescing
- `TimerBatch`, a stable cancellable deadline batch

The public scalar behavior above is parity-tested against the upstream uvloop
version installed by the locked Pixi environment.

Not covered are uvloop's libuv reactor and its specialized TCP/UDP/Unix
transports, DNS cache, SSL state machine, subprocess implementation, filesystem
events, or signal-handler optimizations. Methods supplied by
`asyncio.SelectorEventLoop` may still work outside the covered subset, but they
are not claimed as ports. This project therefore is not a performance drop-in
for a whole uvloop installation: callback dispatch is slower than upstream,
as the benchmark table shows.

## Install and build

```bash
pixi install
pixi run build
pixi run test
```

The build creates `dist/libmojo-uvloop.so`. The Python package is available
inside Pixi through the configured `PYTHONPATH`.

## Usage

The normal entry point has the same shape as `uvloop.run`:

```python
import asyncio
import mojo_uvloop as uvloop


async def main():
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    loop.call_later(0.005, future.set_result, 42)
    return await future


print(uvloop.run(main()))
```

Bulk scheduling amortizes conversion and orders insertions before they enter
the selector loop:

```python
import mojo_uvloop

loop = mojo_uvloop.new_event_loop()
seen = []
loop.call_later_many(
    [0.003, 0.001, 0.002],
    [seen.append] * 3,
    args=[("c",), ("a",), ("b",)],
)
loop.call_later(0.010, loop.stop)
loop.run_forever()
loop.close()
assert seen == ["a", "b", "c"]
```

The kernels are also directly usable:

```python
from mojo_uvloop import coalesce_events, quantize_delays

assert quantize_delays([0.0005, 0.0015]).tolist() == [0, 2]
fds, masks = coalesce_events([7, 4, 7], [1, 2, 4])
assert (fds.tolist(), masks.tolist()) == ([7, 4], [5, 2])
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136-generic, Python 3.13.14, and uvloop 0.22.1. The ratio is
reference time divided by mojo-uvloop time, so values above 1 mean
mojo-uvloop was faster.

| benchmark | mojo-uvloop | reference | ratio | reference |
| --- | ---: | ---: | ---: | --- |
| quantize 5M delays | 40.98 ms | 377.96 ms | 9.22x | NumPy uvloop formula |
| order 1M timers | 78.66 ms | 143.06 ms | 1.82x | NumPy stable argsort |
| compact 5M ready | 34.56 ms | 15.41 ms | 0.45x | NumPy flatnonzero |
| coalesce 1M poll events | 54.95 ms | 497.57 ms | 9.06x | Python ordered dict |
| schedule + dispatch 200k callbacks | 432.06 ms | 266.04 ms | 0.62x | upstream uvloop |
| schedule 100k timers | 248.16 ms | 316.81 ms | 1.28x | upstream scalar API |

Batch delay quantization and poll-event coalescing remain the largest wins.
The four-pass stable timer radix sort now beats NumPy's stable argsort. Ready
compaction lost to NumPy on this run, while the complete bulk timer path beat
upstream's scalar API. Scheduling and dispatching callbacks remains slower
than upstream uvloop; it is Python object execution rather than a batchable
Mojo kernel. Those results are included to keep the performance boundary
explicit.

No GPU path is provided. Sorting, compaction, hashing, and Python object
scheduling perform roughly zero to one simple operation per 8--16 bytes moved,
well below the two-flops-per-byte threshold where transfer and launch costs
could be justified.

## How it works

`src/capi.mojo` is one compilation unit with a small C ABI. NumPy owns every
input, result, hash table, and ordering scratch buffer. Contiguous `float64`,
`int64`, `uint64`, and one-byte boolean buffers cross `ctypes` as integer
addresses and are reconstructed as `UnsafePointer[..., AnyOrigin[mut=True]]`
inside Mojo. The kernels allocate no memory and retain no pointers after a
call.

Timer ordering uses a four-pass stable 16-bit radix sort for large batches and
a bottom-up merge sort for small batches, so equal deadlines retain
registration order. Index initialization and copy-back use native-width SIMD
with scalar tails. Quantized `uint64` timer buffers are ordered in place across
the FFI boundary without a temporary `float64` copy. Ready compaction uses a
single-pass SIMD emitter. Due extraction advances a cursor while filtering
cancelled entries. Poll coalescing uses caller-owned open-addressed hash arrays
and preserves the first occurrence of each file descriptor while OR-ing
readiness masks.

The Python `Loop` subclasses `asyncio.SelectorEventLoop`. Scalar timers mirror
uvloop's negative-delay handling, 100-year cap, and nearest-millisecond
rounding. Bulk calls quantize and order in Mojo, then create ordinary asyncio
handles, preserving cancellation, contexts, tasks, and transport integration.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The benchmark task holds a machine-wide file lock. Run it through Pixi rather
than invoking `bench/bench.py` directly.

MIT licensed.
