__all__ = [
    "dot",
    "gemm",
    "matmul",
    "is_lapack_available",
    "cholesky",
    "lu",
    "qr",
    "svd",
    "det",
    "inverse",
    "norm",
    "pinverse",
    "rank",
    "solve",
]

from typing import cast

import arrayfire_wrapper.lib as wrapper
from arrayfire_wrapper.lib import is_lapack_available

from arrayfire import Array
from arrayfire.array_object import afarray_as_array
from arrayfire.library.constants import MatProp, Norm


def dot(
    lhs: Array,
    rhs: Array,
    /,
    lhs_opts: MatProp = MatProp.NONE,
    rhs_opts: MatProp = MatProp.NONE,
    *,
    return_scalar: bool = False,
) -> int | float | complex | Array:
    if return_scalar:
        return wrapper.dot_all(lhs.arr, rhs.arr, lhs_opts, rhs_opts)

    return Array.from_afarray(wrapper.dot(lhs.arr, rhs.arr, lhs_opts, rhs_opts))


@afarray_as_array
def gemm(
    lhs: Array,
    rhs: Array,
    /,
    lhs_opts: MatProp = MatProp.NONE,
    rhs_opts: MatProp = MatProp.NONE,
    alpha: int | float = 1.0,
    beta: int | float = 0.0,
) -> Array:
    return cast(Array, wrapper.gemm(lhs.arr, rhs.arr, lhs_opts, rhs_opts, alpha, beta))


@afarray_as_array
def matmul(lhs: Array, rhs: Array, /, lhs_opts: MatProp = MatProp.NONE, rhs_opts: MatProp = MatProp.NONE) -> Array:
    return cast(Array, wrapper.matmul(lhs.arr, rhs.arr, lhs_opts, rhs_opts))


def cholesky(array: Array, /, is_upper: bool = True, *, inplace: bool = False) -> int | tuple[Array, int]:
    if inplace:
        return wrapper.cholesky_inplace(array.arr, is_upper)

    matrix, info = wrapper.cholesky(array.arr, is_upper)
    return Array.from_afarray(matrix), info


def lu(array: Array, /, *, inplace: bool = False, is_lapack_pivot: bool = True) -> Array | tuple[Array, ...]:
    if inplace:
        return Array.from_afarray(wrapper.lu_inplace(array.arr, is_lapack_pivot))

    lower, upper, pivot = wrapper.lu(array.arr)
    return Array.from_afarray(lower), Array.from_afarray(upper), Array.from_afarray(pivot)


def qr(array: Array, /, *, inplace: bool = False) -> Array | tuple[Array, ...]:
    if inplace:
        return Array.from_afarray(wrapper.qr_inplace(array.arr))

    q, r, tau = wrapper.qr(array.arr)
    return Array.from_afarray(q), Array.from_afarray(r), Array.from_afarray(tau)


def svd(array: Array, /, *, inplace: bool = False) -> tuple[Array, ...]:
    if inplace:
        u, s, vt, arr = wrapper.svd_inplace(array.arr)
        return Array.from_afarray(u), Array.from_afarray(s), Array.from_afarray(vt), Array.from_afarray(arr)

    u, s, vt = wrapper.svd(array.arr)
    return Array.from_afarray(u), Array.from_afarray(s), Array.from_afarray(vt)


def det(array: Array, /) -> int | float | complex:
    return wrapper.det(array.arr)


@afarray_as_array
def inverse(array: Array, /, options: MatProp = MatProp.NONE) -> Array:
    return cast(Array, wrapper.inverse(array.arr, options))


def norm(array: Array, /, *, norm_type: Norm = Norm.EUCLID, p: float = 1.0, q: float = 1.0) -> float:
    return wrapper.norm(array.arr, norm_type, p, q)


@afarray_as_array
def pinverse(array: Array, /, *, tol: float = 1e-6, options: MatProp = MatProp.NONE) -> Array:
    return cast(Array, wrapper.pinverse(array.arr, tol, options))


def rank(array: Array, /, *, tol: float = 1e-5) -> int:
    return wrapper.rank(array.arr, tol)


@afarray_as_array
def solve(a: Array, b: Array, /, *, options: MatProp = MatProp.NONE, pivot: None | Array = None) -> Array:
    if pivot:
        return cast(Array, wrapper.solve_lu(a.arr, b.arr, pivot.arr, options))

    return cast(Array, wrapper.solve(a.arr, b.arr, options))


# TODO
# Add Sparse functions? #good_first_issue
