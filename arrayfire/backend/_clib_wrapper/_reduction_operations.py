from __future__ import annotations

import ctypes
from collections.abc import Callable
from typing import TYPE_CHECKING

from arrayfire.backend._backend import _backend

from ._error_handler import safe_call

if TYPE_CHECKING:
    from ._base import AFArrayType


def count_all(x: AFArrayType) -> int | float | complex:
    # TODO reconsider original arith.count
    return _reduce_all(x, _backend.clib.af_count_all)


def _reduce_all(arr: AFArrayType, c_func: Callable) -> int | float | complex:
    real = ctypes.c_double(0)
    imag = ctypes.c_double(0)
    safe_call(c_func(ctypes.pointer(real), ctypes.pointer(imag), arr))
    return real.value if imag.value == 0 else real.value + imag.value * 1j


def all_true(arr: AFArrayType, axis: int, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__reduce__func__all__true.htm#ga068708be5177a0aa3788af140bb5ebd6
    """
    out = ctypes.c_void_p(0)
    safe_call(_backend.clib.af_all_true(ctypes.pointer(out), arr, axis))
    return out


def all_true_all(arr: AFArrayType, /) -> complex:
    """
    source: https://arrayfire.org/docs/group__reduce__func__all__true.htm#ga068708be5177a0aa3788af140bb5ebd6
    """
    real = ctypes.c_double(0)
    imag = ctypes.c_double(0)
    safe_call(_backend.clib.af_all_true(ctypes.pointer(real), ctypes.pointer(imag), arr))
    return real.value  # NOTE imag is always set to 0 in C library


def any_true(arr: AFArrayType, axis: int, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__reduce__func__any__true.htm#ga7c275cda2cfc8eb0bd20ea86472ca0d5
    """
    out = ctypes.c_void_p(0)
    safe_call(_backend.clib.af_all_true(ctypes.pointer(out), arr, axis))
    return out


def any_true_all(arr: AFArrayType, /) -> complex:
    """
    source: https://arrayfire.org/docs/group__reduce__func__any__true.htm#ga47d991276bb5bf8cdba8340e8751e536
    """
    real = ctypes.c_double(0)
    imag = ctypes.c_double(0)
    safe_call(_backend.clib.af_all_true(ctypes.pointer(real), ctypes.pointer(imag), arr))
    return real.value  # NOTE imag is always set to 0 in C library


def sum(arr: AFArrayType, axis: int, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__reduce__func__sum.htm#gacd4917c2e916870ebdf54afc2f61d533
    """
    out = ctypes.c_void_p(0)
    safe_call(_backend.clib.af_sum(ctypes.pointer(out), arr, axis))
    return out


def sum_all(arr: AFArrayType, /) -> complex:
    """
    source: https://arrayfire.org/docs/group__reduce__func__sum.htm#gabc009d04df0faf29ba1e381c7badde58
    """
    real = ctypes.c_double(0)
    imag = ctypes.c_double(0)
    safe_call(_backend.clib.af_sum_all(ctypes.pointer(real), ctypes.pointer(imag), arr))
    return real.value  # NOTE imag is always set to 0 in C library


def sum_nan(arr: AFArrayType, axis: int, nan_value: float, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__reduce__func__sum.htm#ga52461231e2d9995f689b7f23eea0e798
    """
    out = ctypes.c_void_p(0)
    safe_call(_backend.clib.af_sum_nan(ctypes.pointer(out), arr, axis, ctypes.c_double(nan_value)))
    return out


def sum_nan_all(arr: AFArrayType, nan_value: float, /) -> complex:
    """
    source: https://arrayfire.org/docs/group__reduce__func__sum.htm#gabc009d04df0faf29ba1e381c7badde58
    """
    real = ctypes.c_double(0)
    imag = ctypes.c_double(0)
    safe_call(_backend.clib.af_sum_all(ctypes.pointer(real), ctypes.pointer(imag), arr, ctypes.c_double(nan_value)))
    return real.value  # NOTE imag is always set to 0 in C library
