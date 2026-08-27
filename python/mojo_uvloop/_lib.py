"""ctypes bindings for the Mojo event-loop primitives."""

from __future__ import annotations

import ctypes
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_UVLOOP_LIB", os.path.join(ROOT, "dist", "libmojo-uvloop.so"))

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "muv_quantize_delays": ([I, I, I, F], None),
    "muv_order_timers": ([I, I, I, I], None),
    "muv_order_uint64": ([I, I, I, I], None),
    "muv_compact_ready": ([I, I, I, I], I),
    "muv_due_indices": ([I, I, I, F, I, I, I], None),
    "muv_coalesce_events": ([I, I, I, I, I, I, I, I], I),
}

_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIB):
            raise RuntimeError("Mojo library is not built; run `pixi run build`")
        _library = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    if array.size == 0:
        raise ValueError("empty buffers must not cross the FFI boundary")
    address = int(array.ctypes.data)
    if address == 0:
        raise RuntimeError("NumPy returned a null buffer address")
    return address
