from __future__ import annotations

from typing import cast

from arrayfire import Array
from arrayfire.array_object import afarray_as_array
from arrayfire.backend import _clib_wrapper as wrapper
from arrayfire.dtypes import is_complex_dtype


@afarray_as_array
def add(x1: Array, x2: Array, /) -> Array:
    return cast(Array, wrapper.add(x1.arr, x2.arr))


@afarray_as_array
def sub(x1: Array, x2: Array, /) -> Array:
    return wrapper.sub(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def mul(x1: Array, x2: Array, /) -> Array:
    return wrapper.mul(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def div(x1: Array, x2: Array, /) -> Array:
    return wrapper.div(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def mod(x1: int | float | Array, x2: int | float | Array, /) -> Array:
    """
    Calculate the modulus of two arrays or a scalar and an array.

    Parameters
    ----------
    x1 : int |float |Array
        The first array or scalar operand.
    x2 : int |float |Array
        The second array or scalar operand.

    Returns
    -------
    result : Array
        The array containing the modulus values after performing the operation.

    Raises
    ------
    ValueError
        If both operands are scalars or if the arrays' shapes do not match.
    """

    _check_operands_fit_requirements(x1, x2)

    x1_ = x1.arr if isinstance(x1, Array) else x1
    x2_ = x2.arr if isinstance(x2, Array) else x2

    result = wrapper.mod(x1_, x2_)
    return cast(Array, result)


@afarray_as_array
def pow(x1: int | float | Array, x2: int | float | Array, /) -> Array:
    _check_operands_fit_requirements(x1, x2)

    return wrapper.pow(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def bitnot(x: Array, /) -> Array:
    return wrapper.bitnot(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def bitand(x1: Array, x2: Array, /) -> Array:
    return wrapper.bitand(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def bitor(x1: Array, x2: Array, /) -> Array:
    return wrapper.bitor(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def bitxor(x1: Array, x2: Array, /) -> Array:
    return wrapper.bitxor(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def bitshiftl(x1: Array, x2: Array, /) -> Array:
    return wrapper.bitshiftl(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def bitshiftr(x1: Array, x2: Array, /) -> Array:
    return wrapper.bitshiftr(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def lt(x1: Array, x2: Array, /) -> Array:
    return wrapper.lt(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def le(x1: Array, x2: Array, /) -> Array:
    return wrapper.le(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def gt(x1: Array, x2: Array, /) -> Array:
    return wrapper.gt(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def ge(x1: Array, x2: Array, /) -> Array:
    return wrapper.ge(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def eq(x1: Array, x2: Array, /) -> Array:
    return wrapper.eq(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def neq(x1: Array, x2: Array, /) -> Array:
    return wrapper.neq(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


# @afarray_as_array
# def clamp(x: Array, /, lo: float, hi: float) -> Array:
#     return NotImplemented


# #     """
# #     source: https://arrayfire.org/docs/group__arith__func__clamp.htm#gac4e785c5c877c7905e56f44ef0cb5e61
# #     """
# #     # TODO: check if lo and hi are of type float. Can be ArrayFire array as well
# #     out = ctypes.c_void_p(0)
# #     safe_call(_backend.clib.af_clamp(ctypes.pointer(out), arr, lo, hi))
# #     return out


@afarray_as_array
def minof(x1: int | float | Array, x2: int | float | Array, /) -> Array:
    _check_operands_fit_requirements(x1, x2)

    return wrapper.minof(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def maxof(x1: int | float | Array, x2: int | float | Array, /) -> Array:
    _check_operands_fit_requirements(x1, x2)

    return wrapper.maxof(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def rem(x1: int | float | Array, x2: int | float | Array, /) -> Array:
    _check_operands_fit_requirements(x1, x2)

    return wrapper.rem(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def abs(x: Array, /) -> Array:
    return wrapper.abs(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def arg(x: Array, /) -> Array:
    return wrapper.arg(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def sign(x: Array, /) -> Array:
    return wrapper.sign(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def round(x: Array, /) -> Array:
    return wrapper.round(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def trunc(x: Array, /) -> Array:
    return wrapper.trunc(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def floor(x: Array, /) -> Array:
    return wrapper.floor(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def ceil(x: Array, /) -> Array:
    return wrapper.ceil(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def hypot(x1: int | float | Array, x2: int | float | Array, /) -> Array:
    _check_operands_fit_requirements(x1, x2)

    return wrapper.hypot(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def sin(x: Array, /) -> Array:
    _check_array_values_not_complex(x)

    return wrapper.sin(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def cos(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.cos(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def tan(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.tan(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def asin(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.asin(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def acos(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.acos(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def atan(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.atan(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def atan2(x1: int | float | Array, x2: int | float | Array, /) -> Array:
    _check_operands_fit_requirements(x1, x2)
    return wrapper.atan2(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def cplx(x1: int | float | Array, x2: int | float | Array | None, /) -> Array:
    if x2 is None:
        return wrapper.cplx1(x1)  # type: ignore[arg-type, return-value]
    else:
        return wrapper.cplx2(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def real(x: Array, /) -> Array:
    return wrapper.real(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def imag(x: Array, /) -> Array:
    return wrapper.imag(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def conjg(x: Array, /) -> Array:
    return wrapper.conjg(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def sinh(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.sinh(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def cosh(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.cosh(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def tanh(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.tanh(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def asinh(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.asinh(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def acosh(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.acosh(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def atanh(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.atanh(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def root(x1: int | float | Array, x2: int | float | Array, /) -> Array:
    _check_operands_fit_requirements(x1, x2)
    return wrapper.root(x1.arr, x2.arr)


@afarray_as_array
def pow2(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.pow2(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def sigmoid(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.sigmoid(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def exp(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.exp(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def expm1(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.expm1(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def erf(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.erf(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def erfc(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.erfc(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def log(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.log(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def log1p(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.log1p(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def log10(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.log10(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def log2(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.log2(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def sqrt(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.sqrt(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def rsqrt(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.rsqrt(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def cbrt(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.cbrt(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def factorial(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.factorial(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def tgamma(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.tgamma(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def lgamma(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.lgamma(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def iszero(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.iszero(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def isinf(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.isinf(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def isnan(x: Array, /) -> Array:
    _check_array_values_not_complex(x)
    return wrapper.isnan(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def land(x1: Array, x2: Array, /) -> Array:
    return wrapper.land(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def lor(x1: Array, x2: Array, /) -> Array:
    return wrapper.lor(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def lnot(x: Array, /) -> Array:
    return wrapper.lnot(x.arr)  # type: ignore[arg-type, return-value]


def _check_operands_fit_requirements(x1: int | float | Array, x2: int | float | Array) -> None:
    if isinstance(x1, Array) and isinstance(x2, Array):
        if x1.shape != x2.shape:
            raise ValueError("Array shapes must match.")

    if not isinstance(x1, Array) and not isinstance(x2, Array):
        raise ValueError("At least one operand must be an Array.")


def _check_array_values_not_complex(x: Array) -> None:
    if is_complex_dtype(x.dtype):
        raise TypeError("Values of an Array should not be the complex numbers.")
    pass
