from __future__ import annotations

import arrayfire as af

from ._array_object import Array


def argsort(x: Array, /, *, axis: int = -1, descending: bool = False, stable: bool = True) -> Array:
    """
    Returns the indices that would sort an array along a specified axis.

    Parameters
    ----------
    x : Array
        Input array. Should be real-valued as complex numbers have unspecified ordering in this context.
    axis : int, optional
        Axis along which to sort the array. If -1, the array is sorted along the last axis. Default is -1.
    descending : bool, optional
        Sort order. If True, sorts in descending order. If False, sorts in ascending order. Default is False.
    stable : bool, optional
        Sort stability. If True, maintains the relative order of elements that compare as equal. If False,
        the order of such elements is implementation-dependent. Default is True.

    Returns
    -------
    out : Array
        An array of indices that sort the array `x` along the specified axis. Has the same shape as `x`.

    Notes
    -----
    - The function currently does not support complex number data types due to unspecified ordering rules.
    - While the `stable` parameter is accepted to match API requirements, actual stability depends on ArrayFire's
      implementation.
    """
    if axis == -1:
        axis = x.ndim - 1

    _, indices = af.sort(x._array, axis=axis, is_ascending=not descending, is_index_array=True)
    return Array._new(indices)


def sort(x: Array, /, *, axis: int = -1, descending: bool = False, stable: bool = True) -> Array:
    """
    Returns a sorted copy of an input array along a specified axis, with options for order and stability.

    Parameters
    ----------
    x : Array
        Input array. Should have a real-valued data type as the ordering of complex numbers is unspecified.
    axis : int, optional
        Axis along which to sort. If set to -1, the function sorts along the last axis. Default is -1.
    descending : bool, optional
        If True, the array is sorted in descending order. If False, the array is sorted in ascending order.
        Default is False.
    stable : bool, optional
        If True, the sort is stable, meaning that the relative order of elements with equal values is preserved.
        If False, the sort may not be stable, and the order of elements with equal values is implementation-dependent.
        Default is True.

    Returns
    -------
    out : Array
        A sorted array with the same shape and data type as the input array.

    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.randu(5)  # Create a random 1D array of size 5
    >>> sorted_a = sort(a)
    >>> print(sorted_a)  # Displays the sorted array

    Notes
    -----
    - The function does not support complex number data types due to unspecified ordering rules for such values.
    - The `stable` flag may be limited by the capabilities of the underlying ArrayFire library.
    """
    if axis == -1:
        axis = x.ndim - 1

    return Array._new(af.sort(x._array, axis=axis, is_ascending=not descending))
