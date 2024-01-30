from collections.abc import Callable
from typing import Any, cast

from arrayfire_wrapper import lib as wrapper

from arrayfire import Array
from arrayfire.array_object import afarray_as_array


@afarray_as_array
def _reduce_to_array(func: Callable, array: Array, axis: int, /, **kwargs: Any) -> Array:
    result = func(array.arr, axis, **kwargs)
    return cast(Array, result)


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

    return _reduce_to_array(wrapper.all_true, array, axis)


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

    return _reduce_to_array(wrapper.any_true, array, axis)


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

    if nan_value is None:
        return _reduce_to_array(wrapper.sum, array, axis)

    return _reduce_to_array(wrapper.sum_nan, array, axis, nan_value=nan_value)


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
        return _reduce_to_array(wrapper.product, array, axis)

    return _reduce_to_array(wrapper.product_nan, array, axis, nan_value=nan_value)
