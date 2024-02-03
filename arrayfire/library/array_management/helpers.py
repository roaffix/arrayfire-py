from typing import cast as typing_cast

import arrayfire_wrapper.lib as wrapper

from arrayfire import Array
from arrayfire.array_object import afarray_as_array
from arrayfire.dtypes import Dtype


@afarray_as_array
def cast(array: Array, dtype: Dtype, /) -> Array:
    """
    Cast an array to a specified type.

    Parameters
    ----------
    array : Array
        Multi-dimensional arrayfire array to be cast.
    dtype : Dtype
        The target data type to which the array will be cast. Must be one of the following:
        - Dtype.int8 for signed 8-bit integer
        - Dtype.int16 for signed 16-bit integer
        - Dtype.int32 for signed 32-bit integer
        - Dtype.int64 for signed 64-bit integer
        - Dtype.uint8 for unsigned 8-bit integer
        - Dtype.uint16 for unsigned 16-bit integer
        - Dtype.uint32 for unsigned 32-bit integer
        - Dtype.uint64 for unsigned 64-bit integer
        - Dtype.float16 for 16-bit floating-point
        - Dtype.float32 for 32-bit floating-point
        - Dtype.float64 for 64-bit floating-point
        - Dtype.complex64 for 64-bit complex number
        - Dtype.complex128 for 128-bit complex number
        - Dtype.bool for boolean

    Returns
    -------
    Array
        An array containing the values from `array` after conversion to the specified `dtype`.
    """
    return typing_cast(Array, wrapper.cast(array.arr, dtype))


@afarray_as_array
def isinf(array: Array, /) -> Array:
    return typing_cast(Array, wrapper.isinf(array.arr))


@afarray_as_array
def isnan(array: Array, /) -> Array:
    return typing_cast(Array, wrapper.isnan(array.arr))


@afarray_as_array
def iszero(array: Array, /) -> Array:
    return typing_cast(Array, wrapper.iszero(array.arr))
