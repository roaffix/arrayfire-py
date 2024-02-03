__all__ = [
    "add",
    "bitshiftl",
    "bitshiftr",
    "div",
    "mul",
    "sub",
    "conjg",
    "cplx",
    "imag",
    "real",
    "cbrt",
    "erf",
    "erfc",
    "exp",
    "expm1",
    "factorial",
    "lgamma",
    "log",
    "log1p",
    "log2",
    "log10",
    "pow",
    "pow2",
    "root",
    "rsqrt",
    "sqrt",
    "tgamma",
    "acosh",
    "asinh",
    "atanh",
    "cosh",
    "sinh",
    "tanh",
    "logical_and",
    "bitand",
    "bitnot",
    "bitor",
    "bitxor",
    "eq",
    "ge",
    "gt",
    "le",
    "lt",
    "neq",
    "logical_not",
    "logical_or",
    "abs",
    "arg",
    "ceil",
    "clamp",
    "floor",
    "hypot",
    "maxof",
    "minof",
    "mod",
    "neg",
    "rem",
    "round",
    "sign",
    "trunc",
    "acos",
    "asin",
    "atan",
    "atan2",
    "cos",
    "sin",
    "tan",
]

from typing import cast

import arrayfire_wrapper.lib as wrapper

from arrayfire import Array
from arrayfire.array_object import afarray_as_array


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

    x1_ = x1.arr if isinstance(x1, Array) else x1
    x2_ = x2.arr if isinstance(x2, Array) else x2

    result = wrapper.mod(x1_, x2_)
    return cast(Array, result)


@afarray_as_array
def pow(x1: int | float | Array, x2: int | float | Array, /) -> Array:
    x1_ = x1.arr if isinstance(x1, Array) else x1
    x2_ = x2.arr if isinstance(x2, Array) else x2

    return cast(Array, wrapper.pow(x1_, x2_))


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


@afarray_as_array
def clamp(x: Array, /, lo: float, hi: float) -> Array:
    return NotImplemented


@afarray_as_array
def minof(x1: int | float | Array, x2: int | float | Array, /) -> Array:
    return cast(Array, wrapper.minof(x1.arr, x2.arr))


@afarray_as_array
def maxof(x1: int | float | Array, x2: int | float | Array, /) -> Array:
    return wrapper.maxof(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def rem(x1: int | float | Array, x2: int | float | Array, /) -> Array:
    return wrapper.rem(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def abs(x: Array, /) -> Array:
    return wrapper.abs_(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def arg(x: Array, /) -> Array:
    return wrapper.arg(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def sign(x: Array, /) -> Array:
    return wrapper.sign(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def round(x: Array, /) -> Array:
    return wrapper.round_(x.arr)  # type: ignore[arg-type, return-value]


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
    return wrapper.hypot(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def sin(x: Array, /) -> Array:
    return wrapper.sin(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def cos(x: Array, /) -> Array:
    return wrapper.cos(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def tan(x: Array, /) -> Array:
    return wrapper.tan(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def asin(x: Array, /) -> Array:
    return wrapper.asin(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def acos(x: Array, /) -> Array:
    return wrapper.acos(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def atan(x: Array, /) -> Array:
    return wrapper.atan(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def atan2(x1: int | float | Array, x2: int | float | Array, /) -> Array:
    return wrapper.atan2(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def cplx(x1: int | float | Array, x2: int | float | Array | None, /) -> Array:
    if x2 is None:
        return cast(Array, wrapper.cplx(x1.arr))
    else:
        return cast(Array, wrapper.cplx2(x1.arr, x2.arr))


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
    return wrapper.sinh(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def cosh(x: Array, /) -> Array:
    return wrapper.cosh(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def tanh(x: Array, /) -> Array:
    return wrapper.tanh(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def asinh(x: Array, /) -> Array:
    return wrapper.asinh(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def acosh(x: Array, /) -> Array:
    return wrapper.acosh(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def atanh(x: Array, /) -> Array:
    return wrapper.atanh(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def root(x1: int | float | Array, x2: int | float | Array, /) -> Array:
    return wrapper.root(x1.arr, x2.arr)


@afarray_as_array
def pow2(x: Array, /) -> Array:
    return wrapper.pow2(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def sigmoid(x: Array, /) -> Array:
    return wrapper.sigmoid(x.arr)


@afarray_as_array
def exp(x: Array, /) -> Array:
    return wrapper.exp(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def expm1(x: Array, /) -> Array:
    return wrapper.expm1(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def erf(x: Array, /) -> Array:
    return wrapper.erf(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def erfc(x: Array, /) -> Array:
    return wrapper.erfc(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def log(x: Array, /) -> Array:
    return wrapper.log(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def log1p(x: Array, /) -> Array:
    return wrapper.log1p(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def log10(x: Array, /) -> Array:
    return wrapper.log10(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def log2(x: Array, /) -> Array:
    return wrapper.log2(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def sqrt(x: Array, /) -> Array:
    return wrapper.sqrt(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def rsqrt(x: Array, /) -> Array:
    return wrapper.rsqrt(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def cbrt(x: Array, /) -> Array:
    return wrapper.cbrt(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def factorial(x: Array, /) -> Array:
    return wrapper.factorial(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def tgamma(x: Array, /) -> Array:
    return wrapper.tgamma(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def lgamma(x: Array, /) -> Array:
    return wrapper.lgamma(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def iszero(x: Array, /) -> Array:
    return wrapper.iszero(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def isinf(x: Array, /) -> Array:
    return wrapper.isinf(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def isnan(x: Array, /) -> Array:
    return wrapper.isnan(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def logical_and(x1: Array, x2: Array, /) -> Array:
    return wrapper.and_(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def logical_or(x1: Array, x2: Array, /) -> Array:
    return wrapper.or_(x1.arr, x2.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def logical_not(x: Array, /) -> Array:
    return wrapper.not_(x.arr)  # type: ignore[arg-type, return-value]


@afarray_as_array
def neg(x: Array, /) -> Array:
    return cast(Array, wrapper.neg(x.arr))
