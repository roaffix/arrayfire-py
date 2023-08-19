from __future__ import annotations

from arrayfire.library import operators

from ._array_object import Array


def abs(x: Array, /) -> Array:
    return Array._new(operators.abs(x._array.arr))


def acos(x: Array, /) -> Array:
    return Array._new(operators.acos(x._array.arr))


def acosh(x: Array, /) -> Array:
    return Array._new(operators.acosh(x._array.arr))


def add(x1: Array, x2: Array, /) -> Array:
    return Array._new(operators.add(x1._array.arr, x2._array.arr))


def asin(x: Array, /) -> Array:
    return Array._new(operators.asin(x._array.arr))


def asinh(x: Array, /) -> Array:
    return Array._new(operators.asinh(x._array.arr))


def atan(x: Array, /) -> Array:
    return Array._new(operators.atan(x._array.arr))


def atan2(x1: Array, x2: Array, /) -> Array:
    return Array._new(operators.atan2(x1._array.arr, x2._array.arr))


def atanh(x: Array, /) -> Array:
    return Array._new(operators.atanh(x._array.arr))


def bitwise_and(x1: Array, x2: Array, /) -> Array:
    return Array._new(operators.bitand(x1._array.arr, x2._array.arr))


def bitwise_left_shift(x1: Array, x2: Array, /) -> Array:
    return Array._new(operators.bitshiftl(x1._array.arr, x2._array.arr))


def bitwise_invert(x: Array, /) -> Array:
    return Array._new(operators.bitnot(x._array.arr))


def bitwise_or(x1: Array, x2: Array, /) -> Array:
    return Array._new(operators.bitor(x1._array.arr, x2._array.arr))


def bitwise_right_shift(x1: Array, x2: Array, /) -> Array:
    return Array._new(operators.bitshiftr(x1._array.arr, x2._array.arr))


def bitwise_xor(x1: Array, x2: Array, /) -> Array:
    return Array._new(operators.bitxor(x1._array.arr, x2._array.arr))


def ceil(x: Array, /) -> Array:
    return Array._new(operators.ceil(x._array.arr))


def conj(x: Array, /) -> Array:
    return Array._new(operators.conjg(x._array.arr))


def cos(x: Array, /) -> Array:
    return Array._new(operators.cos(x._array.arr))


def cosh(x: Array, /) -> Array:
    return Array._new(operators.cosh(x._array.arr))


def divide(x1: Array, x2: Array, /) -> Array:
    return Array._new(operators.div(x1._array.arr, x2._array.arr))


def equal(x1: Array, x2: Array, /) -> Array:
    return Array._new(operators.eq(x1._array.arr, x2._array.arr))


def exp(x: Array, /) -> Array:
    return Array._new(operators.exp(x._array.arr))


def expm1(x: Array, /) -> Array:
    return Array._new(operators.expm1(x._array.arr))


def floor(x: Array, /) -> Array:
    return Array._new(operators.floor(x._array.arr))


def floor_divide(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


def greater(x1: Array, x2: Array, /) -> Array:
    return Array._new(operators.gt(x1._array.arr, x2._array.arr))


def greater_equal(x1: Array, x2: Array, /) -> Array:
    return Array._new(operators.ge(x1._array.arr, x2._array.arr))


def imag(x: Array, /) -> Array:
    return Array._new(operators.imag(x._array.arr))


def isfinite(x: Array, /) -> Array:
    return NotImplemented


def isinf(x: Array, /) -> Array:
    return Array._new(operators.isinf(x._array.arr))


def isnan(x: Array, /) -> Array:
    return Array._new(operators.isnan(x._array.arr))


def less(x1: Array, x2: Array, /) -> Array:
    return Array._new(operators.lt(x1._array.arr, x2._array.arr))


def less_equal(x1: Array, x2: Array, /) -> Array:
    return Array._new(operators.le(x1._array.arr, x2._array.arr))


def log(x: Array, /) -> Array:
    return Array._new(operators.log(x._array.arr))


def log1p(x: Array, /) -> Array:
    return Array._new(operators.log1p(x._array.arr))


def log2(x: Array, /) -> Array:
    return Array._new(operators.log2(x._array.arr))


def log10(x: Array, /) -> Array:
    return Array._new(operators.log10(x._array.arr))


def logaddexp(x1: Array, x2: Array) -> Array:
    return NotImplemented


def logical_and(x1: Array, x2: Array, /) -> Array:
    return Array._new(operators.land(x1._array.arr, x2._array.arr))


def logical_not(x: Array, /) -> Array:
    return Array._new(operators.lnot(x._array.arr))


def logical_or(x1: Array, x2: Array, /) -> Array:
    return Array._new(operators.lor(x1._array.arr, x2._array.arr))


def logical_xor(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


def multiply(x1: Array, x2: Array, /) -> Array:
    return Array._new(operators.mul(x1._array.arr, x2._array.arr))


def negative(x: Array, /) -> Array:
    return NotImplemented


def not_equal(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


def positive(x: Array, /) -> Array:
    return NotImplemented


def pow(x1: Array, x2: Array, /) -> Array:
    return Array._new(operators.pow(x1._array.arr, x2._array.arr))


def real(x: Array, /) -> Array:
    return Array._new(operators.real(x._array.arr))


def remainder(x1: Array, x2: Array, /) -> Array:
    return Array._new(operators.rem(x1._array.arr, x2._array.arr))


def round(x: Array, /) -> Array:
    return Array._new(operators.round(x._array.arr))


def sign(x: Array, /) -> Array:
    return Array._new(operators.sign(x._array.arr))


def sin(x: Array, /) -> Array:
    return Array._new(operators.sin(x._array.arr))


def sinh(x: Array, /) -> Array:
    return Array._new(operators.sinh(x._array.arr))


def square(x: Array, /) -> Array:
    return NotImplemented


def sqrt(x: Array, /) -> Array:
    return Array._new(operators.sqrt(x._array.arr))


def subtract(x1: Array, x2: Array, /) -> Array:
    return Array._new(operators.sub(x1._array.arr, x2._array.arr))


def tan(x: Array, /) -> Array:
    return Array._new(operators.tan(x._array.arr))


def tanh(x: Array, /) -> Array:
    return Array._new(operators.tanh(x._array.arr))


def trunc(x: Array, /) -> Array:
    return Array._new(operators.trunc(x._array.arr))
