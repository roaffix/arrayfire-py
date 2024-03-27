from __future__ import annotations

import arrayfire as af

from ._array_object import Array


def abs(x: Array, /) -> Array:
    return Array._new(af.abs(x._array))


def acos(x: Array, /) -> Array:
    return Array._new(af.acos(x._array))


def acosh(x: Array, /) -> Array:
    return Array._new(af.acosh(x._array))


def add(x1: Array, x2: Array, /) -> Array:
    return Array._new(af.add(x1._array, x2._array))


def asin(x: Array, /) -> Array:
    return Array._new(af.asin(x._array))


def asinh(x: Array, /) -> Array:
    return Array._new(af.asinh(x._array))


def atan(x: Array, /) -> Array:
    return Array._new(af.atan(x._array))


def atan2(x1: Array, x2: Array, /) -> Array:
    return Array._new(af.atan2(x1._array, x2._array))


def atanh(x: Array, /) -> Array:
    return Array._new(af.atanh(x._array))


def bitwise_and(x1: Array, x2: Array, /) -> Array:
    return Array._new(af.bitand(x1._array, x2._array))


def bitwise_left_shift(x1: Array, x2: Array, /) -> Array:
    return Array._new(af.bitshiftl(x1._array, x2._array))


def bitwise_invert(x: Array, /) -> Array:
    return Array._new(af.bitnot(x._array))


def bitwise_or(x1: Array, x2: Array, /) -> Array:
    return Array._new(af.bitor(x1._array, x2._array))


def bitwise_right_shift(x1: Array, x2: Array, /) -> Array:
    return Array._new(af.bitshiftr(x1._array, x2._array))


def bitwise_xor(x1: Array, x2: Array, /) -> Array:
    return Array._new(af.bitxor(x1._array, x2._array))


def ceil(x: Array, /) -> Array:
    return Array._new(af.ceil(x._array))


def conj(x: Array, /) -> Array:
    return Array._new(af.conjg(x._array))


def cos(x: Array, /) -> Array:
    return Array._new(af.cos(x._array))


def cosh(x: Array, /) -> Array:
    return Array._new(af.cosh(x._array))


def divide(x1: Array, x2: Array, /) -> Array:
    return Array._new(af.div(x1._array, x2._array))


def equal(x1: Array, x2: Array, /) -> Array:
    return Array._new(af.eq(x1._array, x2._array))


def exp(x: Array, /) -> Array:
    return Array._new(af.exp(x._array))


def expm1(x: Array, /) -> Array:
    return Array._new(af.expm1(x._array))


def floor(x: Array, /) -> Array:
    return Array._new(af.floor(x._array))


def floor_divide(x1: Array, x2: Array, /) -> Array:
    # TODO
    return NotImplemented


def greater(x1: Array, x2: Array, /) -> Array:
    return Array._new(af.gt(x1._array, x2._array))


def greater_equal(x1: Array, x2: Array, /) -> Array:
    return Array._new(af.ge(x1._array, x2._array))


def imag(x: Array, /) -> Array:
    return Array._new(af.imag(x._array))


def isfinite(x: Array, /) -> Array:
    # TODO
    return NotImplemented


def isinf(x: Array, /) -> Array:
    return Array._new(af.isinf(x._array))


def isnan(x: Array, /) -> Array:
    return Array._new(af.isnan(x._array))


def less(x1: Array, x2: Array, /) -> Array:
    return Array._new(af.lt(x1._array, x2._array))


def less_equal(x1: Array, x2: Array, /) -> Array:
    return Array._new(af.le(x1._array, x2._array))


def log(x: Array, /) -> Array:
    return Array._new(af.log(x._array))


def log1p(x: Array, /) -> Array:
    return Array._new(af.log1p(x._array))


def log2(x: Array, /) -> Array:
    return Array._new(af.log2(x._array))


def log10(x: Array, /) -> Array:
    return Array._new(af.log10(x._array))


def logaddexp(x1: Array, x2: Array) -> Array:
    # TODO
    return NotImplemented


def logical_and(x1: Array, x2: Array, /) -> Array:
    return Array._new(af.logical_and(x1._array, x2._array))


def logical_not(x: Array, /) -> Array:
    return Array._new(af.logical_not(x._array))


def logical_or(x1: Array, x2: Array, /) -> Array:
    return Array._new(af.logical_or(x1._array, x2._array))


def logical_xor(x1: Array, x2: Array, /) -> Array:
    # TODO
    return NotImplemented


def multiply(x1: Array, x2: Array, /) -> Array:
    return Array._new(af.mul(x1._array, x2._array))


def negative(x: Array, /) -> Array:
    return Array._new(af.lt(x._array, 1))


def not_equal(x1: Array, x2: Array, /) -> Array:
    return Array._new(af.neq(x1._array, x2._array))


def positive(x: Array, /) -> Array:
    return Array._new(af.gt(x._array, 1))


def pow(x1: Array, x2: Array, /) -> Array:
    return Array._new(af.pow(x1._array, x2._array))


def real(x: Array, /) -> Array:
    return Array._new(af.real(x._array))


def remainder(x1: Array, x2: Array, /) -> Array:
    return Array._new(af.rem(x1._array, x2._array))


def round(x: Array, /) -> Array:
    return Array._new(af.round(x._array))


def sign(x: Array, /) -> Array:
    return Array._new(af.sign(x._array))


def sin(x: Array, /) -> Array:
    return Array._new(af.sin(x._array))


def sinh(x: Array, /) -> Array:
    return Array._new(af.sinh(x._array))


def square(x: Array, /) -> Array:
    return Array._new(af.pow(x._array, 2))


def sqrt(x: Array, /) -> Array:
    return Array._new(af.sqrt(x._array))


def subtract(x1: Array, x2: Array, /) -> Array:
    return Array._new(af.sub(x1._array, x2._array))


def tan(x: Array, /) -> Array:
    return Array._new(af.tan(x._array))


def tanh(x: Array, /) -> Array:
    return Array._new(af.tanh(x._array))


def trunc(x: Array, /) -> Array:
    return Array._new(af.trunc(x._array))
