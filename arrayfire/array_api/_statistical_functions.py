from __future__ import annotations

import arrayfire as af

from ._array_object import Array


def max(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """
    Calculates the maximum value of the input array along specified axes, optionally keeping the reduced dimensions.

    Parameters
    ----------
    x : Array
        Input array. Should have a real-valued data type.
    axis : int | Tuple[int, ...] | None, optional
        Axis or axes along which to compute the maximum. If None, the maximum is computed over the entire array.
        If an axis or tuple of axes is specified, the maximum is computed over those axes.
    keepdims : bool, optional
        If True, keeps the reduced dimensions as singleton dimensions in the output. If False, reduces the dimensions.

    Returns
    -------
    out : Array
        If the maximum value is computed over the entire array, a zero-dimensional array containing the maximum value;
        otherwise, an array containing the maximum values. The returned array has the same data type as x.

    Raises
    ------
    ValueError
        If the specified axis is out of bounds or if no elements are present to compute the maximum.

    Notes
    -----
    - The function does not support complex number data types due to unspecified ordering rules.
    - NaN values in floating-point arrays propagate. If a NaN is present in the reduction, the result is NaN.
    """
    if axis is None:
        result = af.max(x._array)  # stands for max_all in wrapper

        if keepdims:
            result = af.constant(result, (1, 1))  # type: ignore[arg-type]

        return Array._new(result)

    if isinstance(axis, tuple):
        # TODO
        # if keepdims:
        #     new_dims = (1 if i in axis else s for i, s in enumerate(x.shape))
        #     return af.moddims(x, *new_dims)
        return NotImplemented

    return Array._new(af.max(x._array, axis=axis))


def mean(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    return NotImplemented


def min(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    return NotImplemented


def prod(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: af.Dtype | None = None,
    keepdims: bool = False,
) -> Array:
    return NotImplemented


def std(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: int | float = 0.0,
    keepdims: bool = False,
) -> Array:
    return NotImplemented


def sum(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: af.Dtype | None = None,
    keepdims: bool = False,
) -> Array:
    return NotImplemented


def var(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: int | float = 0.0,
    keepdims: bool = False,
) -> Array:
    return NotImplemented
