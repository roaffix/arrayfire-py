from typing import cast

from arrayfire_wrapper import lib as wrapper

from arrayfire import Array
from arrayfire.array_object import afarray_as_array
from arrayfire.library.constants import BinaryOperator


@afarray_as_array
def accum(array: Array, /, axis: int = 0) -> Array:
    """
    Calculate the cumulative sum of elements along a specified dimension.

    Parameters
    ----------
    array : af.Array
        Multi-dimensional ArrayFire array.

    axis : int, optional, default: 0
        Dimension along which the cumulative sum is required.

    Returns
    -------
    af.Array
        An ArrayFire array of the same size as `array` containing the cumulative sum along the specified dimension.

    Note
    ----
    If `axis` is not specified, the cumulative sum is calculated along the first dimension (default: 0).
    """
    return cast(Array, wrapper.accum(array.arr, axis))


@afarray_as_array
def scan(
    array: Array,
    /,
    key: None | Array = None,
    axis: int = 0,
    op: BinaryOperator = BinaryOperator.ADD,
    inclusive_scan: bool = True,
) -> Array:
    """
    Perform a generalized scan of an array, optionally with a key.

    Parameters
    ----------
    array : af.Array
        Multi-dimensional ArrayFire array.

    key : af.Array, optional, default: None
        Key array for generalized scan. If None, a standard scan is performed.

    axis : int, optional, default: 0
        Dimension along which the scan is performed.

    op : af.BINARYOP, optional, default: af.BINARYOP.ADD
        Binary operation that the scan algorithm uses. Can be one of:
        - af.BINARYOP.ADD
        - af.BINARYOP.MUL
        - af.BINARYOP.MIN
        - af.BINARYOP.MAX

    inclusive_scan : bool, optional, default: True
        Specifies if the scan is inclusive.

    Returns
    -------
    out : af.Array
        Array containing the result of the generalized scan.
    """
    if key:
        return cast(Array, wrapper.scan_by_key(key.arr, array.arr, axis, op, inclusive_scan))

    return cast(Array, wrapper.scan(array.arr, axis, op, inclusive_scan))


@afarray_as_array
def where(array: Array, /) -> Array:
    """
    Find the indices of non-zero elements.

    Parameters
    ----------
    array : af.Array
        Multi-dimensional ArrayFire array.

    Returns
    -------
    af.Array
        Linear indices for non-zero elements.
    """
    return cast(Array, wrapper.where(array.arr))
