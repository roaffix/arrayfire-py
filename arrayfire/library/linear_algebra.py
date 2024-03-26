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
    *,
    return_scalar: bool = False,
) -> int | float | complex | Array:
    """
    Calculates the dot product of two input arrays, with options to modify the operation
    on the input arrays and the possibility to return the result as a scalar.

    Parameters
    ----------
    lhs : Array
        A 1-dimensional, int of float Array instance, representing an array.

    rhs : Array
        A 1-dimensional, int of float Array instance, representing another array.

    return_scalar : bool, optional
        When set to True, the input arrays are flattened, and the output is a scalar value.
        Default is False.

    Returns
    -------
    out : int | float | complex | Array
        The result of the dot product. Returns an Array unless `return_scalar` is True,
        in which case a scalar value (int, float, or complex) is returned based on the
        data type of the inputs.

    Note
    -----
    - The data types of `lhs` and `rhs` should be the same.
    - Batch operations are not supported.
    - Modification options for `lhs` and `rhs` are currently disabled as function supports only `MatProp.NONE`.
    """
    # TODO
    # Add support of lhs_opts and rhs_opts and return them as key arguments.
    lhs_opts: MatProp = MatProp.NONE
    rhs_opts: MatProp = MatProp.NONE

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
    """
    Performs BLAS general matrix multiplication (GEMM) on two Array instances.

    The operation is defined as: C = alpha * op(lhs) * op(rhs) + beta * C, where op(X) is
    one of no operation, transpose, or Hermitian transpose, determined by lhs_opts and rhs_opts.

    Parameters
    ----------
    lhs : Array
        A 2-dimensional, real or complex array representing the left-hand side matrix.

    rhs : Array
        A 2-dimensional, real or complex array representing the right-hand side matrix.

    lhs_opts : MatProp, optional
        Operation to perform on `lhs` before multiplication. Default is MatProp.NONE. Options include:
         - MatProp.NONE: No operation.
         - MatProp.TRANS: Transpose.
         - MatProp.CTRANS: Hermitian transpose.

    rhs_opts : MatProp, optional
        Operation to perform on `rhs` before multiplication. Default is MatProp.NONE. Options include:
         - MatProp.NONE: No operation.
         - MatProp.TRANS: Transpose.
         - MatProp.CTRANS: Hermitian transpose.

    alpha : int | float, optional
        Scalar multiplier for the product of `lhs` and `rhs`. Default is 1.0.

    beta : int | float, optional
        Scalar multiplier for the existing matrix C in the accumulation. Default is 0.0.

    Returns
    -------
    Array
        The result of the matrix multiplication operation.

    Note
    -----
    - The data types of `lhs` and `rhs` must be compatible.
    - Batch operations are not supported in this version.
    """
    return cast(Array, wrapper.gemm(lhs.arr, rhs.arr, lhs_opts, rhs_opts, alpha, beta))


@afarray_as_array
def matmul(lhs: Array, rhs: Array, /, lhs_opts: MatProp = MatProp.NONE, rhs_opts: MatProp = MatProp.NONE) -> Array:
    """
    Performs generalized matrix multiplication between two arrays with optional
    transposition or hermitian transposition operations on the input matrices.

    Parameters
    ----------
    lhs : af.Array
        A 2-dimensional, real or complex ArrayFire array representing the left-hand side matrix.

    rhs : af.Array
        A 2-dimensional, real or complex ArrayFire array representing the right-hand side matrix.

    lhs_opts : af.MATPROP, optional
        Operation to perform on the `lhs` matrix before multiplication. Defaults to af.MATPROP.NONE.
        Options include:
        - af.MATPROP.NONE: No operation.
        - af.MATPROP.TRANS: Transpose `lhs`.
        - af.MATPROP.CTRANS: Hermitian transpose (conjugate transpose) `lhs`.

    rhs_opts : af.MATPROP, optional
        Operation to perform on the `rhs` matrix before multiplication. Defaults to af.MATPROP.NONE.
        Options include:
        - af.MATPROP.NONE: No operation.
        - af.MATPROP.TRANS: Transpose `rhs`.
        - af.MATPROP.CTRANS: Hermitian transpose (conjugate transpose) `rhs`.

    Returns
    -------
    out : af.Array
        The result of the matrix multiplication. The output is a 2-dimensional ArrayFire array.

    Notes
    -----
    - The data types of `lhs` and `rhs` must be the same.
    - Batch operations (multiplying multiple pairs of matrices at once) are not supported in this implementation.

    Examples
    --------
    Basic matrix multiplication:

        A = af.randu(5, 4, dtype=af.Dtype.f32)
        B = af.randu(4, 6, dtype=af.Dtype.f32)
        C = matmul(A, B)

    Matrix multiplication with the left-hand side transposed:

        C = matmul(A, B, lhs_opts=af.MATPROP.TRANS)

    Matrix multiplication with both matrices transposed:

        C = matmul(A, B, lhs_opts=af.MATPROP.TRANS, rhs_opts=af.MATPROP.TRANS)
    """
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
def inverse(array: Array, /, *, options: MatProp = MatProp.NONE) -> Array:
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


# TODO #good_first_issue
# Add Sparse functions
