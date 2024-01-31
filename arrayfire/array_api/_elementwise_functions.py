from __future__ import annotations

from arrayfire.library import mathematical_functions

from ._array_object import Array


def abs(x: Array, /) -> Array:
    return Array._new(mathematical_functions.abs(x._array.arr))


def acos(x: Array, /) -> Array:
    return Array._new(mathematical_functions.acos(x._array.arr))


def acosh(x: Array, /) -> Array:
    return Array._new(mathematical_functions.acosh(x._array.arr))


def add(x1: Array, x2: Array, /) -> Array:
    return Array._new(mathematical_functions.add(x1._array.arr, x2._array.arr))


def asin(x: Array, /) -> Array:
    return Array._new(mathematical_functions.asin(x._array.arr))


def asinh(x: Array, /) -> Array:
    return Array._new(mathematical_functions.asinh(x._array.arr))


def atan(x: Array, /) -> Array:
    return Array._new(mathematical_functions.atan(x._array.arr))


def atan2(x1: Array, x2: Array, /) -> Array:
    return Array._new(mathematical_functions.atan2(x1._array.arr, x2._array.arr))


def atanh(x: Array, /) -> Array:
    return Array._new(mathematical_functions.atanh(x._array.arr))


def bitwise_and(x1: Array, x2: Array, /) -> Array:
    return Array._new(mathematical_functions.bitand(x1._array.arr, x2._array.arr))


def bitwise_left_shift(x1: Array, x2: Array, /) -> Array:
    return Array._new(mathematical_functions.bitshiftl(x1._array.arr, x2._array.arr))


def bitwise_invert(x: Array, /) -> Array:
    return Array._new(mathematical_functions.bitnot(x._array.arr))


def bitwise_or(x1: Array, x2: Array, /) -> Array:
    return Array._new(mathematical_functions.bitor(x1._array.arr, x2._array.arr))


def bitwise_right_shift(x1: Array, x2: Array, /) -> Array:
    return Array._new(mathematical_functions.bitshiftr(x1._array.arr, x2._array.arr))


def bitwise_xor(x1: Array, x2: Array, /) -> Array:
    return Array._new(mathematical_functions.bitxor(x1._array.arr, x2._array.arr))


def ceil(x: Array, /) -> Array:
    return Array._new(mathematical_functions.ceil(x._array.arr))


def conj(x: Array, /) -> Array:
    return Array._new(mathematical_functions.conjg(x._array.arr))


def cos(x: Array, /) -> Array:
    return Array._new(mathematical_functions.cos(x._array.arr))


def cosh(x: Array, /) -> Array:
    return Array._new(mathematical_functions.cosh(x._array.arr))


def divide(x1: Array, x2: Array, /) -> Array:
    return Array._new(mathematical_functions.div(x1._array.arr, x2._array.arr))


def equal(x1: Array, x2: Array, /) -> Array:
    return Array._new(mathematical_functions.eq(x1._array.arr, x2._array.arr))


def exp(x: Array, /) -> Array:
    return Array._new(mathematical_functions.exp(x._array.arr))


def expm1(x: Array, /) -> Array:
    return Array._new(mathematical_functions.expm1(x._array.arr))


def floor(x: Array, /) -> Array:
    return Array._new(mathematical_functions.floor(x._array.arr))


def floor_divide(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


def greater(x1: Array, x2: Array, /) -> Array:
    return Array._new(mathematical_functions.gt(x1._array.arr, x2._array.arr))


def greater_equal(x1: Array, x2: Array, /) -> Array:
    return Array._new(mathematical_functions.ge(x1._array.arr, x2._array.arr))


def imag(x: Array, /) -> Array:
    return Array._new(mathematical_functions.imag(x._array.arr))


def isfinite(x: Array, /) -> Array:
    return NotImplemented


def isinf(x: Array, /) -> Array:
    return Array._new(mathematical_functions.isinf(x._array.arr))


def isnan(x: Array, /) -> Array:
    return Array._new(mathematical_functions.isnan(x._array.arr))


def less(x1: Array, x2: Array, /) -> Array:
    return Array._new(mathematical_functions.lt(x1._array.arr, x2._array.arr))


def less_equal(x1: Array, x2: Array, /) -> Array:
    return Array._new(mathematical_functions.le(x1._array.arr, x2._array.arr))


def log(x: Array, /) -> Array:
    return Array._new(mathematical_functions.log(x._array.arr))


def log1p(x: Array, /) -> Array:
    return Array._new(mathematical_functions.log1p(x._array.arr))


def log2(x: Array, /) -> Array:
    return Array._new(mathematical_functions.log2(x._array.arr))


def log10(x: Array, /) -> Array:
    return Array._new(mathematical_functions.log10(x._array.arr))


def logaddexp(x1: Array, x2: Array) -> Array:
    return NotImplemented


def logical_and(x1: Array, x2: Array, /) -> Array:
    return Array._new(mathematical_functions.land(x1._array.arr, x2._array.arr))


def logical_not(x: Array, /) -> Array:
    return Array._new(mathematical_functions.lnot(x._array.arr))


def logical_or(x1: Array, x2: Array, /) -> Array:
    return Array._new(mathematical_functions.lor(x1._array.arr, x2._array.arr))


def logical_xor(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


def multiply(x1: Array, x2: Array, /) -> Array:
    return Array._new(mathematical_functions.mul(x1._array.arr, x2._array.arr))


def negative(x: Array, /) -> Array:
    return NotImplemented


def not_equal(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


def positive(x: Array, /) -> Array:
    return NotImplemented


def pow(x1: Array, x2: Array, /) -> Array:
    return Array._new(mathematical_functions.pow(x1._array.arr, x2._array.arr))


def real(x: Array, /) -> Array:
    return Array._new(mathematical_functions.real(x._array.arr))


def remainder(x1: Array, x2: Array, /) -> Array:
    return Array._new(mathematical_functions.rem(x1._array.arr, x2._array.arr))


def round(x: Array, /) -> Array:
    return Array._new(mathematical_functions.round(x._array.arr))


def sign(x: Array, /) -> Array:
    return Array._new(mathematical_functions.sign(x._array.arr))


def sin(x: Array, /) -> Array:
    return Array._new(mathematical_functions.sin(x._array.arr))


def sinh(x: Array, /) -> Array:
    return Array._new(mathematical_functions.sinh(x._array.arr))


def square(x: Array, /) -> Array:
    return NotImplemented


def sqrt(x: Array, /) -> Array:
    return Array._new(mathematical_functions.sqrt(x._array.arr))


def subtract(x1: Array, x2: Array, /) -> Array:
    return Array._new(mathematical_functions.sub(x1._array.arr, x2._array.arr))


def tan(x: Array, /) -> Array:
    return Array._new(mathematical_functions.tan(x._array.arr))


def tanh(x: Array, /) -> Array:
    return Array._new(mathematical_functions.tanh(x._array.arr))


def trunc(x: Array, /) -> Array:
    return Array._new(mathematical_functions.trunc(x._array.arr))
