from __future__ import annotations

from typing import Union

from arrayfire import Array, return_copy
from arrayfire.backend import _clib_wrapper as wrapper


@return_copy
def add(x1: Array, x2: Array, /) -> Array:
    return wrapper.add(x1, x2)  # type: ignore[arg-type, return-value]


@return_copy
def sub(x1: Array, x2: Array, /) -> Array:
    return wrapper.sub(x1, x2)  # type: ignore[arg-type, return-value]


@return_copy
def mul(x1: Array, x2: Array, /) -> Array:
    return wrapper.mul(x1, x2)  # type: ignore[arg-type, return-value]


@return_copy
def div(x1: Array, x2: Array, /) -> Array:
    return wrapper.div(x1, x2)  # type: ignore[arg-type, return-value]


@return_copy
def mod(x1: Union[int, float, Array], x2: Union[int, float, Array], /) -> Array:
    """
    Calculate the modulus of two arrays or a scalar and an array.

    Parameters
    ----------
    x1 : Union[int, float, Array]
        The first array or scalar operand.
    x2 : Union[int, float, Array]
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

    if isinstance(x1, Array) and isinstance(x2, Array):
        if x1.shape != x2.shape:
            raise ValueError("Array shapes must match.")
    elif not isinstance(x1, Array) and not isinstance(x2, Array):
        raise ValueError("At least one operand must be an Array.")

    return wrapper.mod(x1, x2)  # type: ignore[arg-type, return-value]


@return_copy
def pow(x1: Array, x2: Array, /) -> Array:
    """
    source: https://arrayfire.org/docs/group__arith__func__pow.htm#ga0f28be1a9c8b176a78c4a47f483e7fc6
    """
    return wrapper.pow(x1, x2)  # type: ignore[arg-type, return-value]


@return_copy
def bitnot(x: Array, /) -> Array:
    """
    source: https://arrayfire.org/docs/group__arith__func__bitnot.htm#gaf97e8a38aab59ed2d3a742515467d01e
    """
    return wrapper.bitnot(x1, x2)  # type: ignore[arg-type, return-value]


@return_copy
def bitand(x1: Array, x2: Array, /) -> Array:
    """
    source: https://arrayfire.org/docs/group__arith__func__bitand.htm#ga45c0779ade4703708596df11cca98800
    """
    return wrapper.bitand(x1, x2)  # type: ignore[arg-type, return-value]


@return_copy
def bitor(x1: Array, x2: Array, /) -> Array:
    """
    source: https://arrayfire.org/docs/group__arith__func__bitor.htm#ga84c99f77d1d83fd53f949b4d67b5b210
    """
    return wrapper.bitor(x1, x2)  # type: ignore[arg-type, return-value]


# @return_copy
# def bitxor(x1: Array, x2: Array, /) -> Array:
#     """
#     source: https://arrayfire.org/docs/group__arith__func__bitxor.htm#ga8188620da6b432998e55fdd1fad22100
#     """
#     return _binary_op(_backend.clib.af_bitxor, x1, x2)  # type: ignore[arg-type, return-value]


# @return_copy
# def bitshiftl(x1: Array, x2: Array, /) -> Array:
#     """
#     source: https://arrayfire.org/docs/group__arith__func__shiftl.htm#ga3139645aafe6f045a5cab454e9c13137
#     """
#     return _binary_op(_backend.clib.af_butshiftl, x1, x2)  # type: ignore[arg-type, return-value]


# @return_copy
# def bitshiftr(x1: Array, x2: Array, /) -> Array:
#     """
#     source: https://arrayfire.org/docs/group__arith__func__shiftr.htm#ga4c06b9977ecf96cdfc83b5dfd1ac4895
#     """
#     return _binary_op(_backend.clib.af_bitshiftr, x1, x2)  # type: ignore[arg-type, return-value]


# @return_copy
# def lt(x1: Array, x2: Array, /) -> Array:
#     """
#     source: https://arrayfire.org/docs/arith_8h.htm#ae7aa04bf23b32bb11c4bab8bdd637103
#     """
#     return _binary_op(_backend.clib.af_lt, x1, x2)  # type: ignore[arg-type, return-value]


# @return_copy
# def le(x1: Array, x2: Array, /) -> Array:
#     """
#     source: https://arrayfire.org/docs/group__arith__func__le.htm#gad5535ce64dbed46d0773fd494e84e922
#     """
#     return _binary_op(_backend.clib.af_le, x1, x2)  # type: ignore[arg-type, return-value]


# @return_copy
# def gt(x1: Array, x2: Array, /) -> Array:
#     """
#     source: https://arrayfire.org/docs/group__arith__func__gt.htm#ga4e65603259515de8939899a163ebaf9e
#     """
#     return _binary_op(_backend.clib.af_gt, x1, x2)  # type: ignore[arg-type, return-value]


# @return_copy
# def ge(x1: Array, x2: Array, /) -> Array:
#     """
#     source: https://arrayfire.org/docs/group__arith__func__ge.htm#ga4513f212e0b0a22dcf4653e89c85e3d9
#     """
#     return _binary_op(_backend.clib.af_ge, x1, x2)  # type: ignore[arg-type, return-value]


# @return_copy
# def eq(x1: Array, x2: Array, /) -> Array:
#     """
#     source: https://arrayfire.org/docs/group__arith__func__eq.htm#ga76d2da7716831616bb81effa9e163693
#     """
#     return _binary_op(_backend.clib.af_eq, x1, x2)  # type: ignore[arg-type, return-value]


# @return_copy
# def neq(x1: Array, x2: Array, /) -> Array:
#     """
#     source: https://arrayfire.org/docs/group__arith__func__neq.htm#gae4ee8bd06a410f259f1493fb811ce441
#     """
#     return _binary_op(_backend.clib.af_neq, x1, x2)  # type: ignore[arg-type, return-value]


# @return_copy
# def clamp(x: Array, /, lo: float, hi: float) -> Array:
#     return NotImplemented


# #     """
# #     source: https://arrayfire.org/docs/group__arith__func__clamp.htm#gac4e785c5c877c7905e56f44ef0cb5e61
# #     """
# #     # TODO: check if lo and hi are of type float. Can be ArrayFire array as well
# #     out = ctypes.c_void_p(0)
# #     safe_call(_backend.clib.af_clamp(ctypes.pointer(out), arr, lo, hi))
# #     return out


# @return_copy
# def minof(x1: Array, x2: Array, /) -> Array:
#     """
#     source: https://arrayfire.org/docs/group__arith__func__min.htm#ga2b842c2d86df978ff68699aeaafca794
#     """
#     return _binary_op(_backend.clib.af_minof, x1, x2)  # type: ignore[arg-type, return-value]


# @return_copy
# def maxof(x1: Array, x2: Array, /) -> Array:
#     """
#     source: https://arrayfire.org/docs/group__arith__func__max.htm#ga0cd47e70cf82b48730a97c59f494b421
#     """
#     return _binary_op(_backend.clib.af_maxof, x1, x2)  # type: ignore[arg-type, return-value]


# @return_copy
# def rem(x1: Array, x2: Array, /) -> Array:
#     """
#     source: https://arrayfire.org/docs/group__arith__func__clamp.htm#gac4e785c5c877c7905e56f44ef0cb5e61
#     """
#     return _binary_op(_backend.clib.af_rem, x1, x2)  # type: ignore[arg-type, return-value]


# @return_copy
# def abs(x: Array, /) -> Array:
#     """
#     source: https://arrayfire.org/docs/group__arith__func__abs.htm#ga7e8b3c848e6cda3d1f3b0c8b2b4c3f8f
#     """
#     return _unary_op(_backend.clib.af_abs, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def arg(x: Array, /) -> Array:
#     """
#     source: https://arrayfire.org/docs/group__arith__func__arg.htm#gad04de0f7948688378dcd3628628a7424
#     """
#     return _unary_op(_backend.clib.af_arg, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def sign(x: Array, /) -> Array:
#     """
#     source: https://arrayfire.org/docs/group__arith__func__sign.htm#ga2d55dfb9b25e0a1316b70f01d5b44b35
#     """
#     return _unary_op(_backend.clib.af_sign, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def round(x: Array, /) -> Array:
#     """
#     source: https://arrayfire.org/docs/group__arith__func__sign.htm#ga2d55dfb9b25e0a1316b70f01d5b44b35
#     """
#     return _unary_op(_backend.clib.af_round, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def trunc(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_trunc, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def floor(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_floor, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def ceil(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_ceil, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def hypot(x1: Array, x2: Array, /) -> Array:
#     """
#     source:
#     """
#     return _binary_op(_backend.clib.af_hypot, x1, x2)  # type: ignore[arg-type, return-value]


# @return_copy
# def sin(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_sin, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def cos(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_cos, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def tan(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_tan, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def asin(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_asin, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def acos(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_acos, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def atan(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_atan, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def atan2(x1: Array, x2: Array, /) -> Array:
#     """
#     source:
#     """
#     return _binary_op(_backend.clib.af_atan2, x1, x2)  # type: ignore[arg-type, return-value]


# @return_copy
# def cplx(x1: Array, x2: Optional[Array], /) -> Array:
#     """
#     source:
#     """
#     if x2 is None:
#         return _unary_op(_backend.clib.af_cplx, x1)  # type: ignore[arg-type, return-value]
#     else:
#         return _binary_op(_backend.clib.af_cplx2, x1, x2)  # type: ignore[arg-type, return-value]


# @return_copy
# def real(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_real, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def imag(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_imag, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def conjg(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_conjg, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def sinh(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_sinh, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def cosh(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_cosh, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def tanh(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_tanh, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def asinh(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_asinh, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def acosh(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_acosh, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def atanh(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_atanh, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def root(x1: Array, x2: Array, /) -> Array:
#     """
#     source:
#     """
#     return _binary_op(_backend.clib.af_root, x1, x2)  # type: ignore[arg-type, return-value]


# @return_copy
# def pow2(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_pow2, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def sigmoid(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_sigmoid, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def exp(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_exp, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def expm1(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_expm1, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def erf(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_erf, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def erfc(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_erfc, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def log(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_log, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def log1p(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_log1p, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def log10(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_log10, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def log2(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_log2, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def sqrt(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_sqrt, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def rsqrt(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_rsqrt, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def cbrt(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_cbrt, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def factorial(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_factorial, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def tgamma(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_tgamma, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def lgamma(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_lgamma, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def iszero(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_iszero, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def isinf(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_isinf, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def isnan(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_isnan, x)  # type: ignore[arg-type, return-value]


# @return_copy
# def land(x1: Array, x2: Array, /) -> Array:
#     """
#     source:
#     """
#     return _binary_op(_backend.clib.af_and, x1, x2)  # type: ignore[arg-type, return-value]


# @return_copy
# def lor(x1: Array, x2: Array, /) -> Array:
#     """
#     source:
#     """
#     return _binary_op(_backend.clib.af_or, x1, x2)  # type: ignore[arg-type, return-value]


# @return_copy
# def lnot(x: Array, /) -> Array:
#     """
#     source:
#     """
#     return _unary_op(_backend.clib.af_not, x)  # type: ignore[arg-type, return-value]
