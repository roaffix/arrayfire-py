from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Callable, Union

from arrayfire.backend._backend import _backend

from ._error_handler import safe_call

if TYPE_CHECKING:
    from ._base import AFArrayType


def count_all(x: AFArrayType) -> Union[int, float, complex]:
    # TODO reconsider original arith.count
    return _reduce_all(x, _backend.clib.af_count_all)


def _reduce_all(arr: AFArrayType, c_func: Callable) -> Union[int, float, complex]:
    real = ctypes.c_double(0)
    imag = ctypes.c_double(0)
    safe_call(c_func(ctypes.pointer(real), ctypes.pointer(imag), arr))
    return real.value if imag.value == 0 else real.value + imag.value * 1j
