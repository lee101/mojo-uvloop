"""C ABI for the event-loop primitive kernels."""

from std.collections import Array
from std.math import iota, round
from std.sys.info import simd_width_of as simdwidthof

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime UPtr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]


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


def order_key[float_keys: Bool](bits: UInt64) -> UInt64:
    comptime if float_keys:
        return timer_key(bits)
    return bits


def initialize_indices(indices: IPtr, n: Int):
    comptime W = simdwidthof[DType.float64]()
    var i = 0
    while i + W <= n:
        indices.store(i, iota[DType.int64, W](Int64(i)))
        i += W
    while i < n:
        indices[i] = Int64(i)
        i += 1


def copy_indices(source: IPtr, destination: IPtr, n: Int):
    comptime W = simdwidthof[DType.float64]()
    var i = 0
    while i + W <= n:
        destination.store(i, source.load[width=W](i))
        i += W
    while i < n:
        destination[i] = source[i]
        i += 1


def merge_order[float_keys: Bool](
    words: UPtr, n: Int, indices: IPtr, work: IPtr
):
    initialize_indices(indices, n)

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
                if order_key[float_keys](words[li]) <= order_key[float_keys](
                    words[ri]
                ):
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


def radix_order[float_keys: Bool](
    words: UPtr, n: Int, indices: IPtr, work: IPtr
):
    comptime RADIX_BITS = 16
    comptime RADIX_SIZE = 1 << RADIX_BITS
    comptime RADIX_MASK = UInt64(RADIX_SIZE - 1)
    comptime RADIX_PASSES = (64 + RADIX_BITS - 1) // RADIX_BITS
    var counts = Array[Int, RADIX_SIZE](fill=0)
    initialize_indices(indices, n)
    var source = indices
    var destination = work
    for radix_pass in range(RADIX_PASSES):
        for bucket in range(RADIX_SIZE):
            counts[bucket] = 0
        var shift = radix_pass * RADIX_BITS
        for i in range(n):
            var index = Int(source[i])
            var bucket = Int(
                (order_key[float_keys](words[index]) >> UInt64(shift))
                & RADIX_MASK
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
                (order_key[float_keys](words[index]) >> UInt64(shift))
                & RADIX_MASK
            )
            destination[counts[bucket]] = source[i]
            counts[bucket] += 1
        var temporary = source
        source = destination
        destination = temporary
    comptime if RADIX_PASSES % 2 != 0:
        copy_indices(source, indices, n)


@export("muv_order_timers")
def muv_order_timers(
    deadlines_addr: Int, n: Int, indices_addr: Int, work_addr: Int
) abi("C"):
    var deadlines = FPtr(unsafe_from_address=deadlines_addr)
    var words = deadlines.bitcast[UInt64]()
    var indices = IPtr(unsafe_from_address=indices_addr)
    var work = IPtr(unsafe_from_address=work_addr)
    if n < 2048:
        merge_order[True](words, n, indices, work)
    else:
        radix_order[True](words, n, indices, work)


@export("muv_order_uint64")
def muv_order_uint64(
    values_addr: Int, n: Int, indices_addr: Int, work_addr: Int
) abi("C"):
    var values = UPtr(unsafe_from_address=values_addr)
    var indices = IPtr(unsafe_from_address=indices_addr)
    var work = IPtr(unsafe_from_address=work_addr)
    if n < 2048:
        merge_order[False](values, n, indices, work)
    else:
        radix_order[False](values, n, indices, work)


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
    cancelled_addr: Int, n: Int, indices_addr: Int, _counts_addr: Int
) abi("C") -> Int:
    var cancelled = BPtr(unsafe_from_address=cancelled_addr)
    var indices = IPtr(unsafe_from_address=indices_addr)
    return emit_live(cancelled, indices, 0, n, 0)


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
