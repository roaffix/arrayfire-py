from collections.abc import Callable
from typing import Any, cast

from arrayfire import Array
from arrayfire.array_object import afarray_as_array
from arrayfire.backend import _clib_wrapper as wrapper


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


def sum(array: Array, /, *, axis: int | None = None, nan_value: float | None = None) -> bool | Array:
    if axis is None:
        if nan_value is None:
            return bool(wrapper.sum_all(array.arr))

        return bool(wrapper.sum_nan_all(array.arr, nan_value))

    if nan_value is None:
        return _reduce_to_array(wrapper.sum, array, axis)

    return _reduce_to_array(wrapper.sum_nan, array, axis, nan_value=nan_value)
