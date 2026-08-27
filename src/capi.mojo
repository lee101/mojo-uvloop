"""C ABI for the event-loop primitive kernels."""

from std.collections import Array
from std.math import round
from std.sys.info import simd_width_of as simdwidthof

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime UPtr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime COMPACT_PARALLEL_THRESHOLD = 1_000_000
comptime COMPACT_TASKS = 8


@export("KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice")
def muv_parallel_runtime_compat() abi("C") -> Int:
    """Keep the pre-1.1 Python runtime bootstrap ABI available.

    Parallel algorithms moved out of the Mojo standard library in 1.1, so the
    compaction kernel no longer needs an async runtime.  Existing Python
    bindings still probe this symbol and only require a non-null result.
    """
    return 1


@export("muv_quantize_delays")
def muv_quantize_delays(
    delays_addr: Int, n: Int, result_addr: Int, max_sleep: Float64
) abi("C"):
    var delays = FPtr(unsafe_from_address=delays_addr)
    var result = UPtr(unsafe_from_address=result_addr)
    for i in range(n):
        var delay = delays[i]
        if delay < 0.0:
            delay = 0.0
        elif delay > max_sleep:
            delay = max_sleep
        result[i] = UInt64(round(delay * 1000.0))


def timer_key(bits: UInt64) -> UInt64:
    comptime SIGN = UInt64(0x8000000000000000)
    if (bits & ~SIGN) == 0:
        return SIGN
    if (bits & SIGN) != 0:
        return ~bits
    return bits ^ SIGN


def merge_order(
    deadlines: FPtr, n: Int, indices: IPtr, work: IPtr
):
    for i in range(n):
        indices[i] = Int64(i)

    var width = 1
    while width < n:
        var begin = 0
        while begin < n:
            var middle = min(begin + width, n)
            var end = min(begin + 2 * width, n)
            var left = begin
            var right = middle
            var dst = begin
            while left < middle and right < end:
                var li = Int(indices[left])
                var ri = Int(indices[right])
                if deadlines[li] <= deadlines[ri]:
                    work[dst] = indices[left]
                    left += 1
                else:
                    work[dst] = indices[right]
                    right += 1
                dst += 1
            while left < middle:
                work[dst] = indices[left]
                left += 1
                dst += 1
            while right < end:
                work[dst] = indices[right]
                right += 1
                dst += 1
            begin += 2 * width
        for i in range(n):
            indices[i] = work[i]
        width *= 2


def radix_order(
    deadlines: FPtr, n: Int, indices: IPtr, work: IPtr
):
    var words = deadlines.bitcast[UInt64]()
    comptime RADIX_SIZE = 256
    var counts = Array[Int, RADIX_SIZE](fill=0)
    for i in range(n):
        indices[i] = Int64(i)
    var source = indices
    var destination = work
    for radix_pass in range(8):
        for bucket in range(RADIX_SIZE):
            counts[bucket] = 0
        var shift = radix_pass * 8
        for i in range(n):
            var index = Int(source[i])
            var bucket = Int(
                (timer_key(words[index]) >> UInt64(shift)) & UInt64(255)
            )
            counts[bucket] += 1
        var offset = 0
        for bucket in range(RADIX_SIZE):
            var size = counts[bucket]
            counts[bucket] = offset
            offset += size
        for i in range(n):
            var index = Int(source[i])
            var bucket = Int(
                (timer_key(words[index]) >> UInt64(shift)) & UInt64(255)
            )
            destination[counts[bucket]] = source[i]
            counts[bucket] += 1
        var temporary = source
        source = destination
        destination = temporary


@export("muv_order_timers")
def muv_order_timers(
    deadlines_addr: Int, n: Int, indices_addr: Int, work_addr: Int
) abi("C"):
    var deadlines = FPtr(unsafe_from_address=deadlines_addr)
    var indices = IPtr(unsafe_from_address=indices_addr)
    var work = IPtr(unsafe_from_address=work_addr)
    if n < 2048:
        merge_order(deadlines, n, indices, work)
    else:
        radix_order(deadlines, n, indices, work)


def count_live(cancelled: BPtr, start: Int, end: Int) -> Int:
    comptime W = simdwidthof[DType.float64]()
    var count = 0
    var vector_count = SIMD[DType.int64, W](0)
    var i = start
    while i + W <= end:
        vector_count += (
            cancelled.load[width=W](i).eq(0).cast[DType.int64]()
        )
        i += W
    count = Int(vector_count.reduce_add())
    while i < end:
        count += Int(cancelled[i] == 0)
        i += 1
    return count


def emit_live(
    cancelled: BPtr,
    indices: IPtr,
    start: Int,
    end: Int,
    destination: Int,
) -> Int:
    comptime W = simdwidthof[DType.float64]()
    var count = destination
    var i = start
    while i + W <= end:
        var live = cancelled.load[width=W](i).eq(0)
        comptime for lane in range(W):
            if live[lane]:
                indices[count] = Int64(i + lane)
                count += 1
        i += W
    while i < end:
        if cancelled[i] == 0:
            indices[count] = Int64(i)
            count += 1
        i += 1
    return count


@export("muv_compact_ready")
def muv_compact_ready(
    cancelled_addr: Int, n: Int, indices_addr: Int, counts_addr: Int
) abi("C") -> Int:
    var cancelled = BPtr(unsafe_from_address=cancelled_addr)
    var indices = IPtr(unsafe_from_address=indices_addr)
    if n < COMPACT_PARALLEL_THRESHOLD:
        return emit_live(cancelled, indices, 0, n, 0)

    var counts = IPtr(unsafe_from_address=counts_addr)
    var chunk = (n + COMPACT_TASKS - 1) // COMPACT_TASKS

    for task in range(COMPACT_TASKS):
        var start = task * chunk
        var end = min(start + chunk, n)
        counts[task] = Int64(count_live(cancelled, start, end))

    var total = 0
    for task in range(COMPACT_TASKS):
        var size = Int(counts[task])
        counts[task] = Int64(total)
        total += size

    for task in range(COMPACT_TASKS):
        var start = task * chunk
        var end = min(start + chunk, n)
        _ = emit_live(cancelled, indices, start, end, Int(counts[task]))
    return total


@export("muv_due_indices")
def muv_due_indices(
    deadlines_addr: Int,
    cancelled_addr: Int,
    n: Int,
    now: Float64,
    start: Int,
    indices_addr: Int,
    meta_addr: Int,
) abi("C"):
    var deadlines = FPtr(unsafe_from_address=deadlines_addr)
    var cancelled = BPtr(unsafe_from_address=cancelled_addr)
    var indices = IPtr(unsafe_from_address=indices_addr)
    var meta = IPtr(unsafe_from_address=meta_addr)
    var cursor = max(start, 0)
    var count = 0
    while cursor < n and deadlines[cursor] <= now:
        if cancelled[cursor] == 0:
            indices[count] = Int64(cursor)
            count += 1
        cursor += 1
    meta[0] = Int64(cursor)
    meta[1] = Int64(count)


@export("muv_coalesce_events")
def muv_coalesce_events(
    fds_addr: Int,
    masks_addr: Int,
    n: Int,
    result_fds_addr: Int,
    result_masks_addr: Int,
    hash_keys_addr: Int,
    hash_positions_addr: Int,
    capacity: Int,
) abi("C") -> Int:
    var fds = IPtr(unsafe_from_address=fds_addr)
    var masks = IPtr(unsafe_from_address=masks_addr)
    var result_fds = IPtr(unsafe_from_address=result_fds_addr)
    var result_masks = IPtr(unsafe_from_address=result_masks_addr)
    var hash_keys = IPtr(unsafe_from_address=hash_keys_addr)
    var hash_positions = IPtr(unsafe_from_address=hash_positions_addr)

    for i in range(capacity):
        hash_keys[i] = -1

    var count = 0
    for i in range(n):
        var fd = fds[i]
        var slot = Int(
            (UInt64(fd) * UInt64(11400714819323198485))
            & UInt64(capacity - 1)
        )
        while hash_keys[slot] != -1 and hash_keys[slot] != fd:
            slot = (slot + 1) & (capacity - 1)
        if hash_keys[slot] == -1:
            hash_keys[slot] = fd
            hash_positions[slot] = Int64(count)
            result_fds[count] = fd
            result_masks[count] = masks[i]
            count += 1
        else:
            var position = Int(hash_positions[slot])
            result_masks[position] = result_masks[position] | masks[i]
    return count
