from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Tuple, Union

from arrayfire.backend._backend import _backend
from arrayfire.dtypes import CShape, Dtype, implicit_dtype, int64, uint64, is_complex_dtype, complex64, complex128

from ._error_handler import safe_call

if TYPE_CHECKING:
    from ._base import AFArrayType


def _constant_complex(number: Union[int, float, complex], shape: Tuple[int, ...], dtype: Dtype, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__data__func__constant.htm#ga5a083b1f3cd8a72a41f151de3bdea1a2
    """
    out = ctypes.c_void_p(0)
    c_shape = CShape(*shape)

    safe_call(
        _backend.clib.af_constant_complex(
            ctypes.pointer(out),
            ctypes.c_double(number.real),
            ctypes.c_double(number.imag),
            4,
            ctypes.pointer(c_shape.c_array),
            dtype.c_api_value,
        )
    )
    return out


def _constant_long(number: Union[int, float], shape: Tuple[int, ...], dtype: Dtype, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__data__func__constant.htm#ga10f1c9fad1ce9e9fefd885d5a1d1fd49
    """
    out = ctypes.c_void_p(0)
    c_shape = CShape(*shape)

    safe_call(
        _backend.clib.af_constant_long(
            ctypes.pointer(out), ctypes.c_longlong(int(number.real)), 4, ctypes.pointer(c_shape.c_array)
        )
    )
    return out


def _constant_ulong(number: Union[int, float], shape: Tuple[int, ...], dtype: Dtype, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__data__func__constant.htm#ga67af670cc9314589f8134019f5e68809
    """
    # return _backend.clib.af_constant_ulong(arr, val, ndims, dims)
    out = ctypes.c_void_p(0)
    c_shape = CShape(*shape)

    safe_call(
        _backend.clib.af_constant_ulong(
            ctypes.pointer(out), ctypes.c_ulonglong(int(number.real)), 4, ctypes.pointer(c_shape.c_array)
        )
    )
    return out


def _constant(number: Union[int, float], shape: Tuple[int, ...], dtype: Dtype, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__data__func__constant.htm#gafc51b6a98765dd24cd4139f3bde00670
    """
    out = ctypes.c_void_p(0)
    c_shape = CShape(*shape)

    safe_call(
        _backend.clib.af_constant(
            ctypes.pointer(out), ctypes.c_double(number), 4, ctypes.pointer(c_shape.c_array), dtype.c_api_value
        )
    )
    return out


def create_constant_array(number: Union[int, float, complex], shape: Tuple[int, ...], dtype: Dtype, /) -> AFArrayType:
    dtype = implicit_dtype(number, dtype)

    if isinstance(number, complex):
        return _constant_complex(number, shape, dtype if is_complex_dtype(dtype) else complex64)

    if dtype == int64:
        return _constant_long(number, shape, dtype)

    if dtype == uint64:
        return _constant_ulong(number, shape, dtype)

    return _constant(number, shape, dtype)
