from __future__ import annotations

import ctypes
from collections.abc import Callable
from typing import TYPE_CHECKING

from arrayfire.backend._backend import _backend
from arrayfire.library.broadcast import bcast_var

from ._error_handler import safe_call

if TYPE_CHECKING:
    from ._base import AFArrayType

# Arithmetic Operators


def add(lhs: AFArrayType, rhs: AFArrayType, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__arith__func__add.htm#ga1dfbee755fedd680f4476803ddfe06a7
    """
    return _binary_op(_backend.clib.af_add, lhs, rhs)


def sub(lhs: AFArrayType, rhs: AFArrayType, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__arith__func__sub.htm#ga80ff99a2e186c23614ea9f36ffc6f0a4
    """
    return _binary_op(_backend.clib.af_sub, lhs, rhs)


def mul(lhs: AFArrayType, rhs: AFArrayType, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__arith__func__mul.htm#ga5f7588b2809ff7551d38b6a0bd583a02
    """
    return _binary_op(_backend.clib.af_mul, lhs, rhs)


def div(lhs: AFArrayType, rhs: AFArrayType, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__arith__func__div.htm#ga21f3f97755702692ec8976934e75fde6
    """
    return _binary_op(_backend.clib.af_div, lhs, rhs)


def mod(lhs: AFArrayType, rhs: AFArrayType, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__arith__func__mod.htm#ga01924d1b59d8886e46fabd2dc9b27e0f
    """
    return _binary_op(_backend.clib.af_mod, lhs, rhs)


def pow(lhs: AFArrayType, rhs: AFArrayType, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__arith__func__pow.htm#ga0f28be1a9c8b176a78c4a47f483e7fc6
    """
    return _binary_op(_backend.clib.af_pow, lhs, rhs)


# Bitwise Operators


def bitnot(arr: AFArrayType, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__arith__func__bitnot.htm#gaf97e8a38aab59ed2d3a742515467d01e
    """
    return _unary_op(_backend.clib.af_bitnot, arr)


def bitand(lhs: AFArrayType, rhs: AFArrayType, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__arith__func__bitand.htm#ga45c0779ade4703708596df11cca98800
    """
    return _binary_op(_backend.clib.af_bitand, lhs, rhs)


def bitor(lhs: AFArrayType, rhs: AFArrayType, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__arith__func__bitor.htm#ga84c99f77d1d83fd53f949b4d67b5b210
    """
    return _binary_op(_backend.clib.af_bitor, lhs, rhs)


def bitxor(lhs: AFArrayType, rhs: AFArrayType, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__arith__func__bitxor.htm#ga8188620da6b432998e55fdd1fad22100
    """
    return _binary_op(_backend.clib.af_bitxor, lhs, rhs)


def bitshiftl(lhs: AFArrayType, rhs: AFArrayType, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__arith__func__shiftl.htm#ga3139645aafe6f045a5cab454e9c13137
    """
    return _binary_op(_backend.clib.af_bitshiftl, lhs, rhs)


def bitshiftr(lhs: AFArrayType, rhs: AFArrayType, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__arith__func__shiftr.htm#ga4c06b9977ecf96cdfc83b5dfd1ac4895
    """
    return _binary_op(_backend.clib.af_bitshiftr, lhs, rhs)


def lt(lhs: AFArrayType, rhs: AFArrayType, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/arith_8h.htm#ae7aa04bf23b32bb11c4bab8bdd637103
    """
    return _binary_op(_backend.clib.af_lt, lhs, rhs)


def le(lhs: AFArrayType, rhs: AFArrayType, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__arith__func__le.htm#gad5535ce64dbed46d0773fd494e84e922
    """
    return _binary_op(_backend.clib.af_le, lhs, rhs)


def gt(lhs: AFArrayType, rhs: AFArrayType, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__arith__func__gt.htm#ga4e65603259515de8939899a163ebaf9e
    """
    return _binary_op(_backend.clib.af_gt, lhs, rhs)


def ge(lhs: AFArrayType, rhs: AFArrayType, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__arith__func__ge.htm#ga4513f212e0b0a22dcf4653e89c85e3d9
    """
    return _binary_op(_backend.clib.af_ge, lhs, rhs)


def eq(lhs: AFArrayType, rhs: AFArrayType, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__arith__func__eq.htm#ga76d2da7716831616bb81effa9e163693
    """
    return _binary_op(_backend.clib.af_eq, lhs, rhs)


def neq(lhs: AFArrayType, rhs: AFArrayType, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__arith__func__neq.htm#gae4ee8bd06a410f259f1493fb811ce441
    """
    return _binary_op(_backend.clib.af_neq, lhs, rhs)


# Numeric Functions


def clamp(arr: AFArrayType, /, lo: float, hi: float) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__arith__func__clamp.htm#gac4e785c5c877c7905e56f44ef0cb5e61
    """
    # TODO: check if lo and hi are of type float. Can be ArrayFire array as well
    out = ctypes.c_void_p(0)
    safe_call(_backend.clib.af_clamp(ctypes.pointer(out), arr, lo, hi))
    return out


def minof(lhs: AFArrayType, rhs: AFArrayType, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__arith__func__min.htm#ga2b842c2d86df978ff68699aeaafca794
    """
    return _binary_op(_backend.clib.af_minof, lhs, rhs)


def maxof(lhs: AFArrayType, rhs: AFArrayType, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__arith__func__max.htm#ga0cd47e70cf82b48730a97c59f494b421
    """
    return _binary_op(_backend.clib.af_maxof, lhs, rhs)


def rem(lhs: AFArrayType, rhs: AFArrayType, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__arith__func__clamp.htm#gac4e785c5c877c7905e56f44ef0cb5e61
    """
    return _binary_op(_backend.clib.af_rem, lhs, rhs)


def abs(arr: AFArrayType, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__arith__func__abs.htm#ga7e8b3c848e6cda3d1f3b0c8b2b4c3f8f
    """
    return _unary_op(_backend.clib.af_abs, arr)


def arg(arr: AFArrayType, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__arith__func__arg.htm#gad04de0f7948688378dcd3628628a7424
    """
    return _unary_op(_backend.clib.af_arg, arr)


def sign(arr: AFArrayType, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__arith__func__sign.htm#ga2d55dfb9b25e0a1316b70f01d5b44b35
    """
    return _unary_op(_backend.clib.af_sign, arr)


def round(arr: AFArrayType, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__arith__func__sign.htm#ga2d55dfb9b25e0a1316b70f01d5b44b35
    """
    return _unary_op(_backend.clib.af_round, arr)


def trunc(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_trunc, arr)


def floor(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_floor, arr)


def ceil(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_ceil, arr)


def hypot(lhs: AFArrayType, rhs: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _binary_op(_backend.clib.af_hypot, lhs, rhs)


def sin(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_sin, arr)


def cos(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_cos, arr)


def tan(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_tan, arr)


def asin(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_asin, arr)


def acos(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_acos, arr)


def atan(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_atan, arr)


def atan2(lhs: AFArrayType, rhs: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _binary_op(_backend.clib.af_atan2, lhs, rhs)


def cplx1(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_cplx, arr)


def cplx2(lhs: AFArrayType, rhs: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _binary_op(_backend.clib.af_cplx2, lhs, rhs)


def real(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_real, arr)


def imag(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_imag, arr)


def conjg(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_conjg, arr)


def sinh(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_sinh, arr)


def cosh(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_cosh, arr)


def tanh(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_tanh, arr)


def asinh(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_asinh, arr)


def acosh(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_acosh, arr)


def atanh(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_atanh, arr)


def root(lhs: AFArrayType, rhs: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _binary_op(_backend.clib.af_root, lhs, rhs)


def pow2(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_pow2, arr)


def sigmoid(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_sigmoid, arr)


def exp(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_exp, arr)


def expm1(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_expm1, arr)


def erf(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_erf, arr)


def erfc(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_erfc, arr)


def log(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_log, arr)


def log1p(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_log1p, arr)


def log10(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_log10, arr)


def log2(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_log2, arr)


def sqrt(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_sqrt, arr)


def rsqrt(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_rsqrt, arr)


def cbrt(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_cbrt, arr)


def factorial(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_factorial, arr)


def tgamma(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_tgamma, arr)


def lgamma(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_lgamma, arr)


def iszero(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_iszero, arr)


def isinf(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_isinf, arr)


def isnan(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_isnan, arr)


def land(lhs: AFArrayType, rhs: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _binary_op(_backend.clib.af_and, lhs, rhs)


def lor(lhs: AFArrayType, rhs: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _binary_op(_backend.clib.af_or, lhs, rhs)


def lnot(arr: AFArrayType, /) -> AFArrayType:
    """
    source:
    """
    return _unary_op(_backend.clib.af_not, arr)


def _binary_op(c_func: Callable, lhs: AFArrayType, rhs: AFArrayType, /) -> AFArrayType:
    out = ctypes.c_void_p(0)
    safe_call(c_func(ctypes.pointer(out), lhs, rhs, bcast_var.get()))
    return out


def _unary_op(c_func: Callable, arr: AFArrayType, /) -> AFArrayType:
    out = ctypes.c_void_p(0)
    safe_call(c_func(ctypes.pointer(out), arr))
    return out
