from typing import Callable

from arrayfire import Array
from arrayfire.backend import _clib_wrapper as wrapper


class return_copy:
    # TODO merge with process_c_function in array_object
    def __init__(self, func: Callable) -> None:
        self.func = func

    def __call__(self, x1: Array, x2: Array) -> Array:
        out = Array()
        out.arr = self.func(x1.arr, x2.arr)
        return out


@return_copy
def abs(x: Array, /) -> Array:
    return wrapper.abs(x)  # type: ignore[arg-type, return-value]


@return_copy
def acos(x: Array, /) -> Array:
    return wrapper.acos(x)  # type: ignore[arg-type, return-value]


@return_copy
def acosh(x: Array, /) -> Array:
    return wrapper.acosh(x)  # type: ignore[arg-type, return-value]


@return_copy
def add(x1: Array, x2: Array, /) -> Array:
    return wrapper.add(x1, x2)  # type: ignore[arg-type, return-value]


@return_copy
def asin(x: Array, /) -> Array:
    return wrapper.asin(x)  # type: ignore[arg-type, return-value]


@return_copy
def asinh(x: Array, /) -> Array:
    return wrapper.asinh(x)  # type: ignore[arg-type, return-value]


@return_copy
def atan(x: Array, /) -> Array:
    return wrapper.atan(x)  # type: ignore[arg-type, return-value]


@return_copy
def atan2(x1: Array, x2: Array, /) -> Array:
    return wrapper.atan2(x1, x2)  # type: ignore[arg-type, return-value]


@return_copy
def atanh(x: Array, /) -> Array:
    return wrapper.atanh(x)  # type: ignore[arg-type, return-value]


@return_copy
def bitwise_and(x1: Array, x2: Array, /) -> Array:
    return wrapper.bitand(x1, x2)  # type: ignore[arg-type, return-value]


@return_copy
def bitwise_left_shift(x1: Array, x2: Array, /) -> Array:
    return wrapper.bitshiftl(x1, x2)  # type: ignore[arg-type, return-value]


@return_copy
def bitwise_invert(x: Array, /) -> Array:
    return wrapper.bitnot(x)  # type: ignore[arg-type, return-value]


@return_copy
def bitwise_or(x1: Array, x2: Array, /) -> Array:
    return wrapper.bitor(x1, x2)  # type: ignore[arg-type, return-value]


@return_copy
def bitwise_right_shift(x1: Array, x2: Array, /) -> Array:
    return wrapper.bitshiftr(x1, x2)  # type: ignore[arg-type, return-value]


@return_copy
def bitwise_xor(x1: Array, x2: Array, /) -> Array:
    return wrapper.bitxor(x1, x2)  # type: ignore[arg-type, return-value]


@return_copy
def ceil(x: Array, /) -> Array:
    return wrapper.ceil(x)  # type: ignore[arg-type, return-value]


@return_copy
def conj(x: Array, /) -> Array:
    return wrapper.conjg(x)  # type: ignore[arg-type, return-value]


@return_copy
def cos(x: Array, /) -> Array:
    return wrapper.cos(x)  # type: ignore[arg-type, return-value]


@return_copy
def cosh(x: Array, /) -> Array:
    return wrapper.cosh(x)  # type: ignore[arg-type, return-value]


@return_copy
def divide(x1: Array, x2: Array, /) -> Array:
    return wrapper.div(x1, x2)  # type: ignore[arg-type, return-value]


@return_copy
def equal(x1: Array, x2: Array, /) -> Array:
    return wrapper.eq(x1, x2)  # type: ignore[arg-type, return-value]


@return_copy
def exp(x: Array, /) -> Array:
    return wrapper.exp(x)  # type: ignore[arg-type, return-value]


@return_copy
def expm1(x: Array, /) -> Array:
    return wrapper.expm1(x)  # type: ignore[arg-type, return-value]


@return_copy
def floor(x: Array, /) -> Array:
    return wrapper.floor(x)  # type: ignore[arg-type, return-value]


@return_copy
def floor_divide(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


@return_copy
def greater(x1: Array, x2: Array, /) -> Array:
    return wrapper.gt(x1, x2)  # type: ignore[arg-type, return-value]


@return_copy
def greater_equal(x1: Array, x2: Array, /) -> Array:
    return wrapper.ge(x1, x2)  # type: ignore[arg-type, return-value]


@return_copy
def imag(x: Array, /) -> Array:
    return wrapper.imag(x)  # type: ignore[arg-type, return-value]


@return_copy
def isfinite(x: Array, /) -> Array:
    return NotImplemented


@return_copy
def isinf(x: Array, /) -> Array:
    return wrapper.isinf(x)  # type: ignore[arg-type, return-value]


@return_copy
def isnan(x: Array, /) -> Array:
    return wrapper.isnan(x)  # type: ignore[arg-type, return-value]


@return_copy
def less(x1: Array, x2: Array, /) -> Array:
    return wrapper.le(x1, x2)  # type: ignore[arg-type, return-value]


@return_copy
def less_equal(x1: Array, x2: Array, /) -> Array:
    return wrapper.lt(x1, x2)  # type: ignore[arg-type, return-value]


@return_copy
def log(x: Array, /) -> Array:
    return wrapper.log(x)  # type: ignore[arg-type, return-value]


@return_copy
def log1p(x: Array, /) -> Array:
    return wrapper.log1p(x)  # type: ignore[arg-type, return-value]


@return_copy
def log2(x: Array, /) -> Array:
    return wrapper.log2(x)  # type: ignore[arg-type, return-value]


@return_copy
def log10(x: Array, /) -> Array:
    return wrapper.log10(x)  # type: ignore[arg-type, return-value]


@return_copy
def logaddexp(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


@return_copy
def logical_and(x1: Array, x2: Array, /) -> Array:
    return wrapper.land(x1, x2)  # type: ignore[arg-type, return-value]


@return_copy
def logical_not(x: Array, /) -> Array:
    return wrapper.lnot(x)  # type: ignore[arg-type, return-value]


@return_copy
def logical_or(x1: Array, x2: Array, /) -> Array:
    return wrapper.lor(x1, x2)  # type: ignore[arg-type, return-value]


@return_copy
def logical_xor(x1: Array, x2: Array, /) -> Array:
    return NotImplemented


@return_copy
def multiply(x1: Array, x2: Array, /) -> Array:
    return wrapper.mul(x1, x2)  # type: ignore[arg-type, return-value]


@return_copy
def negative(x: Array, /) -> Array:
    return wrapper.sub(0, x)  # type: ignore[arg-type, return-value]


@return_copy
def not_equal(x1: Array, x2: Array, /) -> Array:
    return wrapper.neq(x1, x2)  # type: ignore[arg-type, return-value]


@return_copy
def positive(x: Array, /) -> Array:
    return x  # type: ignore[arg-type, return-value]


@return_copy
def pow(x1: Array, x2: Array, /) -> Array:
    return wrapper.pow(x1, x2)  # type: ignore[arg-type, return-value]


@return_copy
def real(x: Array, /) -> Array:
    return wrapper.real(x)  # type: ignore[arg-type, return-value]


@return_copy
def remainder(x1: Array, x2: Array, /) -> Array:
    return wrapper.rem(x1, x2)  # type: ignore[arg-type, return-value]


@return_copy
def round(x: Array, /) -> Array:
    return wrapper.round(x)  # type: ignore[arg-type, return-value]


@return_copy
def sign(x: Array, /) -> Array:
    return wrapper.sign(x)  # type: ignore[arg-type, return-value]


@return_copy
def sin(x: Array, /) -> Array:
    return wrapper.sin(x)  # type: ignore[arg-type, return-value]


@return_copy
def sinh(x: Array, /) -> Array:
    return wrapper.sinh(x)  # type: ignore[arg-type, return-value]


@return_copy
def square(x: Array, /) -> Array:
    return wrapper.pow(x, 2)  # type: ignore[arg-type, return-value]


@return_copy
def sqrt(x: Array, /) -> Array:
    return wrapper.sqrt(x)  # type: ignore[arg-type, return-value]


@return_copy
def subtract(x1: Array, x2: Array, /) -> Array:
    return wrapper.sub(x1, x2)  # type: ignore[arg-type, return-value]


@return_copy
def tan(x: Array, /) -> Array:
    return wrapper.tan(x)  # type: ignore[arg-type, return-value]


@return_copy
def tanh(x: Array, /) -> Array:
    return wrapper.tanh(x)  # type: ignore[arg-type, return-value]


@return_copy
def trunc(x: Array, /) -> Array:
    return wrapper.trunc(x)  # type: ignore[arg-type, return-value]
