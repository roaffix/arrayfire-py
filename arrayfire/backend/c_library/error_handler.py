import ctypes
from enum import Enum

from arrayfire.backend.api import backend_api
from arrayfire.dtypes.helpers import c_dim_t, to_str


class _ErrorCodes(Enum):
    none = 0


def safe_call(c_err: int) -> None:
    if c_err == _ErrorCodes.none.value:
        return

    err_str = ctypes.c_char_p(0)
    backend_api.af_get_last_error(ctypes.pointer(err_str), ctypes.pointer(c_dim_t(0)))
    raise RuntimeError(to_str(err_str))
