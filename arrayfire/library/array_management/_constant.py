import arrayfire_wrapper.lib as wrapper
from arrayfire_wrapper import AFArray

from arrayfire.dtypes import Dtype, complex32, complex64, implicit_dtype, int64, uint64


def _is_complex_dtype(dtype: Dtype) -> bool:
    return dtype in {complex32, complex64}


def create_constant_array(number: int | float | complex, shape: tuple[int, ...], dtype: Dtype, /) -> AFArray:
    if not dtype:
        dtype = implicit_dtype(number, dtype)

    if isinstance(number, complex):
        return wrapper.constant_complex(number, shape, dtype if _is_complex_dtype(dtype) else complex32)

    if dtype == int64:
        return wrapper.constant_long(number, shape, dtype)

    if dtype == uint64:
        return wrapper.constant_ulong(number, shape, dtype)

    return wrapper.constant(number, shape, dtype)
