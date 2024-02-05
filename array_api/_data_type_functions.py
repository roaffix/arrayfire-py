from dataclasses import dataclass

import arrayfire as af

from ._array_object import Array
from ._dtypes import all_dtypes, float16, float32, float64, int8, int16, int32, int64, promote_types


def astype(x: Array, dtype: af.Dtype, /, *, copy: bool = True) -> Array:
    return NotImplemented


def broadcast_arrays(*arrays: Array) -> list[Array]:
    return NotImplemented


def broadcast_to(x: Array, /, shape: tuple[int, ...]) -> Array:
    return NotImplemented


def can_cast(from_: af.Dtype | Array, to: af.Dtype, /) -> bool:
    return NotImplemented


@dataclass
class finfo_object:
    bits: int
    eps: float
    max: float
    min: float
    smallest_normal: float
    dtype: af.Dtype


@dataclass
class iinfo_object:
    bits: int
    max: int
    min: int
    dtype: af.Dtype


def finfo(type: af.Dtype | Array, /) -> finfo_object:
    if isinstance(type, af.Dtype):
        dtype = type
    elif isinstance(type, Array):
        dtype = Array.dtype  # type: ignore[assignment]
    else:
        raise ValueError("Unsupported dtype.")

    if dtype == float32:
        return finfo_object(32, 1.19209290e-7, 3.4028234e38, -3.4028234e38, 1.1754943e-38, float32)
    if dtype == float64:
        return finfo_object(
            64,
            2.2204460492503131e-16,
            1.7976931348623157e308,
            -1.7976931348623157e308,
            2.2250738585072014e-308,
            float64,
        )
    if dtype == float16:
        return finfo_object(16, 0.00097656, 65504, -65504, 0.00006103515625, float16)

    raise ValueError("Unsupported dtype.")


def iinfo(type: af.Dtype | Array, /) -> iinfo_object:
    if isinstance(type, af.Dtype):
        dtype = type
    elif isinstance(type, Array):
        dtype = Array.dtype  # type: ignore[assignment]
    else:
        raise ValueError("Unsupported dtype.")

    if dtype == int32:
        return iinfo_object(32, 2147483648, -2147483647, int32)
    if dtype == int16:
        return iinfo_object(16, 32767, -32768, int16)
    if dtype == int8:
        return iinfo_object(8, 127, -128, int8)
    if dtype == int64:
        return iinfo_object(64, 9223372036854775807, -9223372036854775808, int64)

    raise ValueError("Unsupported dtype.")


def isdtype(dtype: af.Dtype, kind: af.Dtype | str | tuple[af.Dtype | str, ...]) -> bool:
    return NotImplemented


def result_type(*arrays_and_dtypes: Array | af.Dtype) -> af.Dtype:
    """
    Array API compatible wrapper for :py:func:`np.result_type <numpy.result_type>`.

    See its docstring for more information.
    """
    # Note: we use a custom implementation that gives only the type promotions
    # required by the spec rather than using np.result_type. NumPy implements
    # too many extra type promotions like int64 + uint64 -> float64, and does
    # value-based casting on scalar arrays.
    A = []
    for a in arrays_and_dtypes:
        if isinstance(a, Array):
            a = a.dtype
        elif isinstance(a, af.Array) or a not in all_dtypes:
            raise TypeError("result_type() inputs must be array_api arrays or dtypes")
        A.append(a)

    if len(A) == 0:
        raise ValueError("at least one array or dtype is required")
    elif len(A) == 1:
        return A[0]
    else:
        t = A[0]
        for t2 in A[1:]:
            t = promote_types(t, t2)
        return t
