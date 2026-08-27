from collections import OrderedDict

import numpy as np
import pytest

import mojo_uvloop as muv
from mojo_uvloop.primitives import _order_uint64


def test_quantize_delays_matches_uvloop_formula():
    values = np.array(
        [-np.inf, -1.0, 0.0, 0.0004, 0.0005, 0.0015, 2.3456, np.inf, 1e20]
    )
    expected = []
    for value in values:
        if value < 0:
            value = 0.0
        elif value == np.inf or value > muv.MAX_SLEEP:
            value = float(muv.MAX_SLEEP)
        expected.append(round(value * 1000))
    assert np.array_equal(muv.quantize_delays(values), expected)


def test_quantize_rejects_nan_like_uvloop():
    with pytest.raises(ValueError, match="NaN"):
        muv.quantize_delays([0.1, np.nan])
    with pytest.raises(ValueError, match="one-dimensional"):
        muv.quantize_delays(np.zeros((2, 2)))


def test_order_timers_matches_stable_numpy():
    rng = np.random.default_rng(0)
    deadlines = rng.integers(0, 100, size=20_003).astype(float)
    got = muv.order_timers(deadlines)
    expected = np.argsort(deadlines, kind="stable")
    assert np.array_equal(got, expected)
    assert np.array_equal(np.sort(got), np.arange(deadlines.size))


def test_order_timers_edge_cases():
    assert muv.order_timers([]).tolist() == []
    assert muv.order_timers([4.0]).tolist() == [0]
    with pytest.raises(ValueError, match="finite"):
        muv.order_timers([1.0, np.inf])


def test_order_timers_radix_path_is_stable_for_signed_values_and_zero():
    pattern = np.array(
        [3.0, -0.0, -2.0, 0.0, -2.0, 3.0, -1e100, 1e100],
        dtype=np.float64,
    )
    deadlines = np.tile(pattern, 300)
    assert deadlines.size >= 2048
    got = muv.order_timers(deadlines)
    expected = np.argsort(deadlines, kind="stable")
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("size", [2047, 2048, 2051])
def test_order_timers_threshold_and_simd_tail(size):
    deadlines = (np.arange(size, dtype=np.float64) * 37) % 101
    got = muv.order_timers(deadlines)
    expected = np.argsort(deadlines, kind="stable")
    assert np.array_equal(got, expected)


def test_uint64_ordering_radix_path_is_stable_with_simd_tail():
    values = (np.arange(2051, dtype=np.uint64) * 37) % 101
    got = _order_uint64(values)
    expected = np.argsort(values, kind="stable")
    assert np.array_equal(got, expected)


def test_compact_ready_matches_fifo_reference():
    rng = np.random.default_rng(1)
    cancelled = rng.random(100_007) < 0.35
    assert np.array_equal(muv.compact_ready(cancelled), np.flatnonzero(~cancelled))


@pytest.mark.parametrize("size", [999_999, 1_000_000, 1_000_007])
def test_compact_ready_large_inputs_and_simd_tail(size):
    cancelled = np.zeros(size, dtype=np.bool_)
    cancelled[::7] = True
    cancelled[-1] = False
    assert np.array_equal(muv.compact_ready(cancelled), np.flatnonzero(~cancelled))


def test_due_indices_matches_reference_and_advances_cursor():
    deadlines = np.array([1.0, 1.0, 2.0, 3.0, 5.0])
    cancelled = np.array([False, True, False, False, False])
    first, cursor = muv.due_indices(deadlines, cancelled, 2.0)
    second, cursor = muv.due_indices(deadlines, cancelled, 4.0, start=cursor)
    third, cursor = muv.due_indices(deadlines, cancelled, 9.0, start=cursor)
    assert first.tolist() == [0, 2]
    assert second.tolist() == [3]
    assert third.tolist() == [4]
    assert cursor == len(deadlines)


def test_due_indices_validates_sorted_equal_inputs():
    with pytest.raises(ValueError, match="equal-length"):
        muv.due_indices([1.0], [False, True], 1.0)
    with pytest.raises(ValueError, match="sorted"):
        muv.due_indices([2.0, 1.0], [False, False], 2.0)
    with pytest.raises(ValueError, match="now must be finite"):
        muv.due_indices([1.0], [False], np.nan)
    with pytest.raises(ValueError, match="start"):
        muv.due_indices([1.0], [False], 1.0, start=2)
    with pytest.raises(TypeError, match="booleans"):
        muv.due_indices([1.0], [0], 1.0)


def test_coalesce_events_matches_ordered_dict_reference():
    rng = np.random.default_rng(2)
    fds = rng.integers(0, 4096, size=100_000, dtype=np.int64)
    masks = rng.choice(np.array([1, 2, 4], dtype=np.int64), size=fds.size)
    expected = OrderedDict()
    for fd, mask in zip(fds, masks, strict=True):
        expected[int(fd)] = expected.get(int(fd), 0) | int(mask)
    got_fds, got_masks = muv.coalesce_events(fds, masks)
    assert got_fds.tolist() == list(expected)
    assert got_masks.tolist() == list(expected.values())


def test_coalesce_events_empty_and_validation():
    fds, masks = muv.coalesce_events([], [])
    assert fds.size == masks.size == 0
    with pytest.raises(ValueError, match="equal-length"):
        muv.coalesce_events([1], [1, 2])
    with pytest.raises(ValueError, match="non-negative"):
        muv.coalesce_events([-1], [1])
    with pytest.raises(TypeError, match="integers"):
        muv.coalesce_events([1.5], [1])
    with pytest.raises(ValueError, match="fit in int64"):
        muv.coalesce_events([2**63], [1])
    with pytest.raises(ValueError, match="one-dimensional"):
        muv.coalesce_events(np.array([[1]]), [1])


def test_timer_batch_orders_cancels_and_pops_once():
    batch = muv.TimerBatch([3.0, 1.0, 2.0, 1.0], list("abcd"))
    batch.cancel(3)
    assert len(batch) == 3
    assert batch.pop_due(1.0) == ["b"]
    assert batch.pop_due(2.0) == ["c"]
    assert batch.pop_due(10.0) == ["a"]
    assert batch.pop_due(10.0) == []
    assert len(batch) == 0
