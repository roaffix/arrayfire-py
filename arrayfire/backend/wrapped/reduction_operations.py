import ctypes
from typing import Callable, Union

from ..backend import backend_api, safe_call

AFArray = ctypes.c_void_p


def count_all(x: AFArray) -> Union[int, float, complex]:
    # TODO reconsider original arith.count
    return _reduce_all(x, backend_api.af_count_all)


def _reduce_all(arr: AFArray, c_func: Callable) -> Union[int, float,  complex]:
    real = ctypes.c_double(0)
    imag = ctypes.c_double(0)
    safe_call(c_func(ctypes.pointer(real), ctypes.pointer(imag), arr))
    return real.value if imag.value == 0 else real.value + imag.value * 1j
