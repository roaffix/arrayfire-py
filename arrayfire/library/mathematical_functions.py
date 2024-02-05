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
    "neg",
]

from typing import cast

import arrayfire_wrapper.lib as wrapper

from arrayfire.array_object import Array, afarray_as_array, process_c_function


def add(x1: Array | int | float, x2: Array | int | float, /) -> Array:
    return process_c_function(x1, x2, wrapper.add)


def sub(x1: Array | int | float, x2: Array | int | float, /) -> Array:
    return process_c_function(x1, x2, wrapper.sub)


def mul(x1: Array | int | float, x2: Array | int | float, /) -> Array:
    return process_c_function(x1, x2, wrapper.mul)


def div(x1: Array | int | float, x2: Array | int | float, /) -> Array:
    return process_c_function(x1, x2, wrapper.div)


def mod(x1: int | float | Array, x2: int | float | Array, /) -> Array:
    """
    Calculate the modulus of two arrays or a scalar and an array.

    Parameters
    ----------
    x1 : int | float | Array
        The first array or scalar operand.
    x2 : int | float | Array
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
    return process_c_function(x1, x2, wrapper.mod)


def pow(x1: int | float | Array, x2: int | float | Array, /) -> Array:
    return process_c_function(x1, x2, wrapper.pow)


@afarray_as_array
def bitnot(x: Array, /) -> Array:
    return cast(Array, wrapper.bitnot(x.arr))


def bitand(x1: Array | int | float, x2: Array | int | float, /) -> Array:
    return process_c_function(x1, x2, wrapper.bitand)


def bitor(x1: Array | int | float, x2: Array | int | float, /) -> Array:
    return process_c_function(x1, x2, wrapper.bitor)


def bitxor(x1: Array | int | float, x2: Array | int | float, /) -> Array:
    return process_c_function(x1, x2, wrapper.bitxor)


def bitshiftl(x1: Array | int | float, x2: Array | int | float, /) -> Array:
    return process_c_function(x1, x2, wrapper.bitshiftl)


def bitshiftr(x1: Array | int | float, x2: Array | int | float, /) -> Array:
    return process_c_function(x1, x2, wrapper.bitshiftr)


def lt(x1: Array | int | float, x2: Array | int | float, /) -> Array:
    return process_c_function(x1, x2, wrapper.lt)


def le(x1: Array | int | float, x2: Array | int | float, /) -> Array:
    return process_c_function(x1, x2, wrapper.le)


def gt(x1: Array | int | float, x2: Array | int | float, /) -> Array:
    return process_c_function(x1, x2, wrapper.gt)


def ge(x1: Array | int | float, x2: Array | int | float, /) -> Array:
    return process_c_function(x1, x2, wrapper.ge)


def eq(x1: Array | int | float, x2: Array | int | float, /) -> Array:
    return process_c_function(x1, x2, wrapper.eq)


def neq(x1: Array | int | float, x2: Array | int | float, /) -> Array:
    return process_c_function(x1, x2, wrapper.neq)


@afarray_as_array
def clamp(x: Array, /, lo: float, hi: float) -> Array:
    # TODO
    return NotImplemented


def minof(x1: int | float | Array, x2: int | float | Array, /) -> Array:
    return process_c_function(x1, x2, wrapper.minof)


def maxof(x1: int | float | Array, x2: int | float | Array, /) -> Array:
    return process_c_function(x1, x2, wrapper.maxof)


def rem(x1: int | float | Array, x2: int | float | Array, /) -> Array:
    return process_c_function(x1, x2, wrapper.rem)


@afarray_as_array
def abs(x: Array, /) -> Array:
    return cast(Array, wrapper.abs_(x.arr))


@afarray_as_array
def arg(x: Array, /) -> Array:
    return cast(Array, wrapper.arg(x.arr))


@afarray_as_array
def sign(x: Array, /) -> Array:
    return cast(Array, wrapper.sign(x.arr))


@afarray_as_array
def round(x: Array, /) -> Array:
    return cast(Array, wrapper.round_(x.arr))


@afarray_as_array
def trunc(x: Array, /) -> Array:
    return cast(Array, wrapper.trunc(x.arr))


@afarray_as_array
def floor(x: Array, /) -> Array:
    return cast(Array, wrapper.floor(x.arr))


@afarray_as_array
def ceil(x: Array, /) -> Array:
    return cast(Array, wrapper.ceil(x.arr))


@afarray_as_array
def hypot(x1: int | float | Array, x2: int | float | Array, /) -> Array:
    return process_c_function(x1, x2, wrapper.hypot)


@afarray_as_array
def sin(x: Array, /) -> Array:
    return cast(Array, wrapper.sin(x.arr))


@afarray_as_array
def cos(x: Array, /) -> Array:
    return cast(Array, wrapper.cos(x.arr))


@afarray_as_array
def tan(x: Array, /) -> Array:
    return cast(Array, wrapper.tan(x.arr))


@afarray_as_array
def asin(x: Array, /) -> Array:
    return cast(Array, wrapper.asin(x.arr))


@afarray_as_array
def acos(x: Array, /) -> Array:
    return cast(Array, wrapper.acos(x.arr))


@afarray_as_array
def atan(x: Array, /) -> Array:
    return cast(Array, wrapper.atan(x.arr))


def atan2(x1: int | float | Array, x2: int | float | Array, /) -> Array:
    return process_c_function(x1, x2, wrapper.atan2)


@afarray_as_array
def cplx(x1: int | float | Array, /, x2: int | float | Array | None = None) -> Array:
    if x2 is None:
        if not isinstance(x1, Array):
            raise TypeError("x1 can not be int or tuple when x2 is None.")
        return cast(Array, wrapper.cplx(x1.arr))
    else:
        return process_c_function(x1, x2, wrapper.cplx2)


@afarray_as_array
def real(x: Array, /) -> Array:
    return cast(Array, wrapper.real(x.arr))


@afarray_as_array
def imag(x: Array, /) -> Array:
    return cast(Array, wrapper.imag(x.arr))


@afarray_as_array
def conjg(x: Array, /) -> Array:
    return cast(Array, wrapper.conjg(x.arr))


@afarray_as_array
def sinh(x: Array, /) -> Array:
    return cast(Array, wrapper.sinh(x.arr))


@afarray_as_array
def cosh(x: Array, /) -> Array:
    return cast(Array, wrapper.cosh(x.arr))


@afarray_as_array
def tanh(x: Array, /) -> Array:
    return cast(Array, wrapper.tanh(x.arr))


@afarray_as_array
def asinh(x: Array, /) -> Array:
    return cast(Array, wrapper.asinh(x.arr))


@afarray_as_array
def acosh(x: Array, /) -> Array:
    return cast(Array, wrapper.acosh(x.arr))


@afarray_as_array
def atanh(x: Array, /) -> Array:
    return cast(Array, wrapper.atanh(x.arr))


def root(x1: int | float | Array, x2: int | float | Array, /) -> Array:
    return process_c_function(x1, x2, wrapper.root)


@afarray_as_array
def pow2(x: Array, /) -> Array:
    return cast(Array, wrapper.pow2(x.arr))


@afarray_as_array
def sigmoid(x: Array, /) -> Array:
    return cast(Array, wrapper.sigmoid(x.arr))


@afarray_as_array
def exp(x: Array, /) -> Array:
    return cast(Array, wrapper.exp(x.arr))


@afarray_as_array
def expm1(x: Array, /) -> Array:
    return cast(Array, wrapper.expm1(x.arr))


@afarray_as_array
def erf(x: Array, /) -> Array:
    return cast(Array, wrapper.erf(x.arr))


@afarray_as_array
def erfc(x: Array, /) -> Array:
    return cast(Array, wrapper.erfc(x.arr))


@afarray_as_array
def log(x: Array, /) -> Array:
    return cast(Array, wrapper.log(x.arr))


@afarray_as_array
def log1p(x: Array, /) -> Array:
    return cast(Array, wrapper.log1p(x.arr))


@afarray_as_array
def log10(x: Array, /) -> Array:
    return cast(Array, wrapper.log10(x.arr))


@afarray_as_array
def log2(x: Array, /) -> Array:
    return cast(Array, wrapper.log2(x.arr))


@afarray_as_array
def sqrt(x: Array, /) -> Array:
    return cast(Array, wrapper.sqrt(x.arr))


@afarray_as_array
def rsqrt(x: Array, /) -> Array:
    return cast(Array, wrapper.rsqrt(x.arr))


@afarray_as_array
def cbrt(x: Array, /) -> Array:
    return cast(Array, wrapper.cbrt(x.arr))


@afarray_as_array
def factorial(x: Array, /) -> Array:
    return cast(Array, wrapper.factorial(x.arr))


@afarray_as_array
def tgamma(x: Array, /) -> Array:
    return cast(Array, wrapper.tgamma(x.arr))


@afarray_as_array
def lgamma(x: Array, /) -> Array:
    return cast(Array, wrapper.lgamma(x.arr))


@afarray_as_array
def iszero(x: Array, /) -> Array:
    return cast(Array, wrapper.iszero(x.arr))


@afarray_as_array
def isinf(x: Array, /) -> Array:
    return cast(Array, wrapper.isinf(x.arr))


@afarray_as_array
def isnan(x: Array, /) -> Array:
    return cast(Array, wrapper.isnan(x.arr))


def logical_and(x1: Array | int | float, x2: Array | int | float, /) -> Array:
    return process_c_function(x1, x2, wrapper.and_)


def logical_or(x1: Array | int | float, x2: Array | int | float, /) -> Array:
    return process_c_function(x1, x2, wrapper.or_)


@afarray_as_array
def logical_not(x: Array, /) -> Array:
    return cast(Array, wrapper.not_(x.arr))


@afarray_as_array
def neg(x: Array, /) -> Array:
    return cast(Array, wrapper.neg(x.arr))
