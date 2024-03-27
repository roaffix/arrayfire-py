__all__ = [
    "accum",
    "scan",
    "where",
    "all_true",
    "any_true",
    "sum",
    "product",
    "count",
    "imax",
    "max",
    "imin",
    "min",
    "diff1",
    "diff2",
    "gradient",
    "set_intersect",
    "set_union",
    "set_unique",
    "sort",
]

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
    keys: None | Array = None,
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

    keys : af.Array, optional, default: None
        Keys array for generalized scan. If None, a standard scan is performed.

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
    if keys:
        return cast(Array, wrapper.scan_by_key(keys.arr, array.arr, axis, op, inclusive_scan))

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


def all_true(array: Array, axis: int | None = None) -> bool | Array:
    """
    Check if all the elements along a specified dimension are true.

    Parameters
    ----------
    array : Array
        Multi-dimensional ArrayFire array.

    axis : int, optional, default: None
        Dimension along which the product is required.

    Returns
    -------
    bool | Array
        An ArrayFire array containing True if all elements in `array` along the specified dimension are True.
        If `axis` is `None`, the output is True if `array` does not have any zeros, else False.

    Note
    ----
    If `axis` is `None`, output is True if the array does not have any zeros, else False.
    """
    if axis is None:
        return bool(wrapper.all_true_all(array.arr))

    return Array.from_afarray(wrapper.all_true(array.arr, axis))


def any_true(array: Array, axis: int | None = None) -> bool | Array:
    """
    Check if any of the elements along a specified dimension are true.

    Parameters
    ----------
    array : Array
        Multi-dimensional ArrayFire array.

    axis : int, optional, default: None
        Dimension along which the product is required.

    Returns
    -------
    bool | Array
        An ArrayFire array containing True if any of the elements in `array` along the specified dimension are True.
        If `axis` is `None`, the output is True if `array` does not have any zeros, else False.

    Note
    ----
    If `axis` is `None`, output is True if the array does not have any zeros, else False.
    """
    if axis is None:
        return bool(wrapper.any_true_all(array.arr))

    return Array.from_afarray(wrapper.any_true(array.arr, axis))


def sum(array: Array, /, *, axis: int | None = None, nan_value: float | None = None) -> int | float | complex | Array:
    # FIXME documentation issues
    """
    Calculate the sum of elements along a specified dimension or the entire array.

    Parameters
    ----------
    array : Array
        The multi-dimensional array to calculate the sum of.

    axis : int or None, optional, default: None
        The dimension along which the sum is calculated.
        If None, the sum of all elements in the entire array is calculated.

    nan_value : float or None, optional, default: None
        The value to replace NaN (Not-a-Number) values in the array before summing. If None, NaN values are ignored.

    Returns
    -------
    Array or bool or scalar
        - If `axis` is None and `nan_value` is None, returns a boolean indicating if the sum contains NaN or Inf.
        - If `axis` is None and `nan_value` is not None, returns a boolean indicating if the sum contains NaN or
          Inf after replacing NaN values.
        - If `axis` is not None, returns an Array containing the sum along the specified dimension.
        - If `axis` is not None and `nan_value` is not None, returns an Array containing the sum along the specified
          dimension after replacing NaN values.
    """

    if axis is None:
        if nan_value is None:
            return wrapper.sum_all(array.arr)

        return wrapper.sum_nan_all(array.arr, nan_value)

    if nan_value is not None:
        return Array.from_afarray(wrapper.sum_nan(array.arr, axis, nan_value))

    return Array.from_afarray(wrapper.sum(array.arr, axis))  # type: ignore[call-arg]


def product(
    array: Array, /, *, axis: int | None = None, nan_value: float | None = None
) -> int | float | complex | Array:
    # FIXME documentation issues
    """
    Calculate the product of all the elements along a specified dimension.

    Parameters
    ----------
    array : Array
        The multi-dimensional array to calculate the product of.

    axis : int or None, optional, default: None
        The dimension along which the product is calculated.
        If None, the product of all elements in the entire array is returned.

    nan_value : float or None, optional, default: None
        The value to replace NaN (Not-a-Number) values in the array before computing the product.
        If None, NaN values are ignored.

    Returns
    -------
    Array or scalar number
        The product of all elements in `array` along dimension `axis`.
        If `axis` is `None`, the product of the entire array is returned.
    """
    if axis is None:
        if nan_value is None:
            return wrapper.product_all(array.arr)

        return wrapper.product_nan_all(array.arr, nan_value)

    if nan_value is None:
        return Array.from_afarray(wrapper.product(array.arr, axis))

    return Array.from_afarray(wrapper.product_nan(array.arr, axis, nan_value))  # type: ignore[call-arg]


def count(
    array: Array, /, *, axis: int | None = None, keys: Array | None = None
) -> int | float | complex | Array | tuple[Array, Array]:
    if keys:
        axis_ = -1 if axis is None else axis
        key, value = wrapper.count_by_key(keys.arr, array.arr, axis_)
        return Array.from_afarray(key), Array.from_afarray(value)

    if axis is None:
        return wrapper.count_all(array.arr)

    return Array.from_afarray(wrapper.count(array.arr, axis))


def imax(array: Array, /, *, axis: int | None = None) -> tuple[int | float | complex, int] | tuple[Array, Array]:
    if axis is None:
        return wrapper.imax_all(array.arr)

    maximum, location = wrapper.imax(array.arr, axis)
    return Array.from_afarray(maximum), Array.from_afarray(location)


def max(
    array: Array, /, *, axis: int | None = None, keys: Array | None = None, ragged_len: Array | None = None
) -> int | float | complex | Array | tuple[Array, Array]:
    if keys and ragged_len:
        raise RuntimeError("To process ragged max function, the keys value should be None and vice versa.")

    if keys:
        axis_ = -1 if axis is None else axis
        key, value = wrapper.max_by_key(keys.arr, array.arr, axis_)
        return Array.from_afarray(key), Array.from_afarray(value)

    if ragged_len:
        axis_ = -1 if axis is None else axis
        values, indices = wrapper.max_ragged(array.arr, ragged_len.arr, axis_)
        return Array.from_afarray(values), Array.from_afarray(indices)

    if axis is None:
        return wrapper.max_all(array.arr)

    return Array.from_afarray(wrapper.max(array.arr, axis))


def imin(array: Array, /, *, axis: int | None = None) -> tuple[int | float | complex, int] | tuple[Array, Array]:
    if axis is None:
        return wrapper.imin_all(array.arr)

    minimum, location = wrapper.imin(array.arr, axis)
    return Array.from_afarray(minimum), Array.from_afarray(location)


def min(array: Array, /, *, axis: int | None = None) -> int | float | complex | Array:
    if axis is None:
        return wrapper.min_all(array.arr)

    return Array.from_afarray(wrapper.min(array.arr, axis))


@afarray_as_array
def diff1(array: Array, /, axis: int = 0) -> Array:
    return cast(Array, wrapper.diff1(array.arr, axis))


@afarray_as_array
def diff2(array: Array, /, axis: int = 0) -> Array:
    return cast(Array, wrapper.diff2(array.arr, axis))


def gradient(array: Array, /) -> tuple[Array, Array]:
    dx, dy = wrapper.gradient(array.arr)
    return Array.from_afarray(dx), Array.from_afarray(dy)


@afarray_as_array
def set_intersect(x: Array, y: Array, /, *, is_unique: bool = False) -> Array:
    return cast(Array, wrapper.set_intersect(x.arr, y.arr, is_unique))


@afarray_as_array
def set_union(x: Array, y: Array, /, *, is_unique: bool = False) -> Array:
    return cast(Array, wrapper.set_union(x.arr, y.arr, is_unique))


@afarray_as_array
def set_unique(array: Array, /, *, is_sorted: bool = False) -> Array:
    return cast(Array, wrapper.set_unique(array.arr, is_sorted))


def sort(
    array: Array,
    /,
    axis: int = 0,
    is_ascending: bool = True,
    *,
    keys: Array | None = None,
    is_index_array: bool = False,
) -> Array | tuple[Array, Array]:
    if keys and is_index_array:
        raise RuntimeError("Could not process sorting by keys when `is_index_array` is True. Select only one option.")

    if keys:
        key, value = wrapper.sort_by_key(keys.arr, array.arr, axis, is_ascending)
        return Array.from_afarray(key), Array.from_afarray(value)

    if is_index_array:
        values, indices = wrapper.sort_index(array.arr, axis, is_ascending)
        return Array.from_afarray(values), Array.from_afarray(indices)

    return Array.from_afarray(wrapper.sort(array.arr, axis, is_ascending))
