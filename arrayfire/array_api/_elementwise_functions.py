from __future__ import annotations

from ._array_object import Array

from arrayfire.library import operators


def abs(x: Array, /) -> Array:
    return NotImplemented


def acos(x: Array, /) -> Array:
    return NotImplemented


def acosh(x: Array, /) -> Array:
    return NotImplemented


def add(x1: Array, x2: Array, /) -> Array:
    return Array._new(operators.add(x1._array, x2._array))


def asin(x: Array, /) -> Array:
    return NotImplemented


def asinh(x: Array, /) -> Array:
    return NotImplemented


def atan(x: Array, /) -> Array:
    return NotImplemented


def atan2(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


def atanh(x: Array, /) -> Array:
    return NotImplemented


def bitwise_and(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


def bitwise_left_shift(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


def bitwise_invert(x: Array, /) -> Array:
    return NotImplemented


def bitwise_or(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


def bitwise_right_shift(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


def bitwise_xor(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


def ceil(x: Array, /) -> Array:
    return NotImplemented


def conj(x: Array, /) -> Array:
    return NotImplemented


def cos(x: Array, /) -> Array:
    return NotImplemented


def cosh(x: Array, /) -> Array:
    return NotImplemented


def divide(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


def equal(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


def exp(x: Array, /) -> Array:
    return NotImplemented


def expm1(x: Array, /) -> Array:
    return NotImplemented


def floor(x: Array, /) -> Array:
    return NotImplemented


def floor_divide(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


def greater(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


def greater_equal(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


def imag(x: Array, /) -> Array:
    return NotImplemented


def isfinite(x: Array, /) -> Array:
    return NotImplemented


def isinf(x: Array, /) -> Array:
    return NotImplemented


def isnan(x: Array, /) -> Array:
    return NotImplemented


def less(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


def less_equal(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


def log(x: Array, /) -> Array:
    return NotImplemented


def log1p(x: Array, /) -> Array:
    return NotImplemented


def log2(x: Array, /) -> Array:
    return NotImplemented


def log10(x: Array, /) -> Array:
    return NotImplemented


def logaddexp(x1: Array, x2: Array) -> Array:
    return NotImplemented


def logical_and(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


def logical_not(x: Array, /) -> Array:
    return NotImplemented


def logical_or(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


def logical_xor(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


def multiply(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


def negative(x: Array, /) -> Array:
    return NotImplemented


def not_equal(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


def positive(x: Array, /) -> Array:
    return NotImplemented


# Note: the function name is different here
def pow(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


def real(x: Array, /) -> Array:
    return NotImplemented


def remainder(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


def round(x: Array, /) -> Array:
    return NotImplemented


def sign(x: Array, /) -> Array:
    return NotImplemented


def sin(x: Array, /) -> Array:
    return NotImplemented


def sinh(x: Array, /) -> Array:
    return NotImplemented


def square(x: Array, /) -> Array:
    return NotImplemented


def sqrt(x: Array, /) -> Array:
    return NotImplemented


def subtract(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


def tan(x: Array, /) -> Array:
    return NotImplemented


def tanh(x: Array, /) -> Array:
    return NotImplemented


def trunc(x: Array, /) -> Array:
    return NotImplemented
