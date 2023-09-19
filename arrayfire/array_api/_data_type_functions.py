from dataclasses import dataclass
from typing import List, Tuple, Union

from arrayfire import Array as AFArray
from arrayfire.array_api._array_object import Array
from arrayfire.array_api._dtypes import all_dtypes, promote_types
from arrayfire.dtypes import Dtype


def astype(x: Array, dtype: Dtype, /, *, copy: bool = True) -> Array:
    return NotImplemented


def broadcast_arrays(*arrays: Array) -> List[Array]:
    return NotImplemented


def broadcast_to(x: Array, /, shape: Tuple[int, ...]) -> Array:
    return NotImplemented


def can_cast(from_: Union[Dtype, Array], to: Dtype, /) -> bool:
    return NotImplemented


@dataclass
class finfo_object:
    bits: int
    eps: float
    max: float
    min: float
    smallest_normal: float
    dtype: Dtype


# TODO fix and add complex support
# float32 : finfo(32, 1.19209290 * 10^-7, 3.4028234 * 10^38, -3.4028234 * 10^38, 1.1754943 *10^−38, float32)
# float64 : finfo(64, 2.2204460492503131*10^-16, 1.7976931348623157 · 10^308, -1.7976931348623157 · 10^308, 2.2250738585072014 * 10^−308 , float64)
# float16 : finfo(16, 0.00097656, 65504, -65504, 0.00006103515625, float16)


# TODO separate API supported dtypes, add aliases
# Common
# int16 = Dtype("int16", "h", ctypes.c_short, "short int", 10) == s16
# int32 = Dtype("int32", "i", ctypes.c_int, "int", 5) == s32
# int64 = Dtype("int64", "l", ctypes.c_longlong, "long int", 8) == s64
# uint8 = Dtype("uint8", "B", ctypes.c_ubyte, "unsigned_char", 7) == u8
# uint16 = Dtype("uint16", "H", ctypes.c_ushort, "unsigned short int", 11) == u16
# uint32 = Dtype("uint32", "I", ctypes.c_uint, "unsigned int", 6) == u32
# uint64 = Dtype("uint64", "L", ctypes.c_ulonglong, "unsigned long int", 9) == u64
# float16 = Dtype("float16", "e", ctypes.c_uint16, "half", 12) == f16
# float32 = Dtype("float32", "f", ctypes.c_float, "float", 0) == f32
# float64 = Dtype("float64", "d", ctypes.c_double, "double", 2) == f64
# bool = Dtype("bool", "b", ctypes.c_char, "bool", 4) == b8

# AF API
# complex32 = Dtype("complex32", "F", ctypes.c_float * 2, "float complex", 1) == c32
# complex64 = Dtype("complex64", "D", ctypes.c_double * 2, "double complex", 3) == c64

# Array API
# int8 = Dtype("int8", "b8", ctypes.c_char, "int8", 4)  # HACK int8 - not supported in AF -> same as b8
# complex64 = Dtype("complex64", "F", ctypes.c_float * 2, "float complex", 1)  # type: ignore[arg-type]
# complex128 = Dtype("complex128", "D", ctypes.c_double * 2, "double complex", 3)  # type: ignore[arg-type]


@dataclass
class iinfo_object:
    bits: int
    max: int
    min: int
    dtype: Dtype


def finfo(type: Union[Dtype, Array], /) -> finfo_object:
    return NotImplemented


# TODO fix bug
# def iinfo(type: Union[Dtype, Array], /) -> iinfo_object:
#     # Reference: https://en.cppreference.com/w/cpp/language/types

#     if isinstance(type, Dtype):
#         type_ = type
#     elif isinstance(type, Array):
#         type_ = Array.dtype
#     else:
#         raise ValueError("Wrong type.")

#     match type_:
#         case int32:
#             return iinfo_object(32, 2147483648, -2147483647, int32)
#         case int16:
#             return iinfo_object(16,32767, -32768, int16)
#         case int8:
#             return iinfo_object(8, 127, -128, int8)
#         case int64:
#             return iinfo_object(64, 9223372036854775807, -9223372036854775808, int64)


def isdtype(dtype: Dtype, kind: Union[Dtype, str, Tuple[Union[Dtype, str], ...]]) -> bool:
    return NotImplemented


def result_type(*arrays_and_dtypes: Union[Array, Dtype]) -> Dtype:
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
        elif isinstance(a, AFArray) or a not in all_dtypes:
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
