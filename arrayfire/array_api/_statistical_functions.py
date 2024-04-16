from __future__ import annotations

from typing import Any, Callable

import arrayfire as af

from ._array_object import Array
from ._data_type_functions import astype


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
    Array
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
    return _compute_statistic(x, af.max, axis=axis, keepdims=keepdims)


def mean(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """
    Calculates the arithmetic mean of the input array along specified axes.

    Parameters
    ----------
    x : Array
        Input array. Should be real-valued and floating-point.
    axis : int | Tuple[int, ...] | None, optional
        Axis or axes along which the arithmetic means are to be computed. If None, the mean is computed over
        the entire array. If a tuple of integers, means are computed over multiple axes. Default is None.
    keepdims : bool, optional
        If True, keeps the reduced dimensions as singleton dimensions in the result, making the result
        compatible with the input array. If False, reduced dimensions are not included in the result. Default is False.

    Returns
    -------
    Array
        If the mean is computed over the entire array, a zero-dimensional array containing the arithmetic mean;
        otherwise, an array containing the arithmetic means. The returned array has the same data type as x.

    Raises
    ------
    ValueError
        If specified axes are out of range or the data type of x is not floating-point.

    Notes
    -----
    - NaN values in the array propagate; if any element is NaN, the corresponding mean is NaN.
    - Only supports real-valued floating-point data types for accurate computations.
    """
    return _compute_statistic(x, af.mean, axis=axis, keepdims=keepdims)


def min(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """
    Calculates the minimum value of the input array along specified axes, optionally keeping the reduced dimensions.

    Parameters
    ----------
    x : Array
        Input array. Should have a real-valued data type.
    axis : int | Tuple[int, ...] | None, optional
        Axis or axes along which the minimum values are to be computed. If None, the minimum is computed over
        the entire array. If a tuple of integers, minimums are computed over multiple axes. Default is None.
    keepdims : bool, optional
        If True, keeps the reduced dimensions as singleton dimensions in the result, making the result
        compatible with the input array. If False, reduced dimensions are not included in the result. Default is False.

    Returns
    -------
    Array
        If the minimum value is computed over the entire array, a zero-dimensional array containing the minimum value;
        otherwise, an array containing the minimum values. The returned array has the same data type as x.

    Raises
    ------
    ValueError
        If specified axes are out of range.

    Notes
    -----
    - NaN values in the array propagate; if any element is NaN, the corresponding minimum is NaN.
    - Only supports real-valued floating-point data types for accurate computations.
    """
    return _compute_statistic(x, af.min, axis=axis, keepdims=keepdims)


def prod(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: af.Dtype | None = None,
    keepdims: bool = False,
) -> Array:
    """
    Calculates the product of elements in the input array along specified axes, with an option to specify the data
    type.

    Parameters
    ----------
    x : Array
        Input array. Should have a numeric data type.
    axis : int | Tuple[int, ...] | None, optional
        Axis or axes along which products are to be computed. If None, the product is computed over the entire array.
        If a tuple of integers, products are computed over multiple axes. Default is None.
    dtype : dtype | None, optional
        Data type of the returned array. If None, the data type is determined based on the data type of x, with
        adjustments to prevent overflow. If specified, the input array x is cast to this data type before computing
        the product.
    keepdims : bool, optional
        If True, the reduced dimensions are kept as singleton dimensions in the result, making the result compatible
        with the input array. If False, the reduced dimensions are not included in the result. Default is False.

    Returns
    -------
    Array
        If the product is computed over the entire array, a zero-dimensional array containing the product;
        otherwise, an array containing the products. The returned array has the data type as specified or determined.

    Notes
    -----
    - If N is 0, the product is 1 (empty product).
    - NaN values in floating-point arrays will result in NaN for the entire product if encountered.
    - Proper handling of data type to prevent overflow is crucial, especially for integer types.
    """
    if dtype is not None:
        x = astype(x, dtype)

    return _compute_statistic(x, af.product, axis=axis, keepdims=keepdims)


def std(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: int | float = 0.0,
    keepdims: bool = False,
) -> Array:
    """
    Calculates the standard deviation of the input array along specified axes, with an option for degrees of freedom
    adjustment.

    Parameters
    ----------
    x : Array
        Input array. Should have a real-valued floating-point data type.
    axis : int | Tuple[int, ...] | None, optional
        Axis or axes along which standard deviations are to be computed. If None, the standard deviation is computed
        over the entire array. If a tuple of integers, standard deviations are computed over multiple axes.
        Default is None.
    correction : int | float, optional
        Degrees of freedom adjustment. Setting this to 0 computes the population standard deviation. Setting this to 1
        (Bessel's correction) computes the sample standard deviation. Default is 0.
    keepdims : bool, optional
        If True, keeps the reduced dimensions as singleton dimensions in the result, making the result compatible
        with the input array. If False, reduced dimensions are not included in the result. Default is False.

    Returns
    -------
    out : Array
        If the standard deviation is computed over the entire array, a zero-dimensional array containing the standard
        deviation; otherwise, an array containing the standard deviations. The returned array has the same data type
        as x.

    Notes
    -----
    - If N - correction is less than or equal to 0, the standard deviation is NaN.
    - NaN values in the array propagate; if any element is NaN, the corresponding standard deviation is NaN.
    """
    if correction == 0:
        bias = af.VarianceBias.POPULATION
    elif correction == 1:
        bias = af.VarianceBias.SAMPLE
    else:
        raise ValueError("Correction can only be set as 0 or 1. Other values are unsupported with arrayfire.")

    return _compute_statistic(x, af.stdev, axis=axis, keepdims=keepdims, bias=bias)


def sum(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: af.Dtype | None = None,
    keepdims: bool = False,
) -> Array:
    """
    Calculates the sum of the input array along specified axes, with an option to specify the data type.

    Parameters
    ----------
    x : Array
        Input array. Should have a numeric data type.
    axis : int | Tuple[int, ...] | None, optional
        Axis or axes along which sums are to be computed. If None, the sum is computed over the entire array.
        If a tuple of integers, sums are computed over multiple axes. Default is None.
    dtype : dtype | None, optional
        Data type of the returned array. If None, the returned array matches the data type of x. If specified,
        the input array x is cast to this data type before computing the sum.
    keepdims : bool, optional
        If True, keeps the reduced dimensions as singleton dimensions in the result, making the result compatible
        with the input array. If False, reduced dimensions are not included in the result. Default is False.

    Returns
    -------
    out : Array
        If the sum is computed over the entire array, a zero-dimensional array containing the sum;
        otherwise, an array containing the sums. The returned array has the data type as specified or determined.

    Notes
    -----
    - If N is 0, the sum is 0.
    - NaN values in floating-point arrays will propagate.
    - Careful consideration of data type can help prevent overflow in integer arrays or loss of precision in
      floating-point arrays.
    """
    if dtype is not None:
        x = astype(x, dtype)

    return _compute_statistic(x, af.sum, axis=axis, keepdims=keepdims)


def var(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: int | float = 0.0,
    keepdims: bool = False,
) -> Array:
    """
    Calculates the variance of the input array along specified axes, with an option for degrees of freedom adjustment.

    Parameters
    ----------
    x : Array
        Input array. Should be real-valued and floating-point.
    axis : int | Tuple[int, ...] | None, optional
        Axis or axes along which variances are to be computed. If None, the variance is computed over
        the entire array. If a tuple of integers, variances are computed over multiple axes. Default is None.
    correction : int | float, optional
        Degrees of freedom adjustment. Setting this to 0 computes the population variance. Setting this to 1
        (Bessel's correction) computes the sample variance. Default is 0.
    keepdims : bool, optional
        If True, keeps the reduced dimensions as singleton dimensions in the result, making the result compatible
        with the input array. If False, reduced dimensions are not included in the result. Default is False.

    Returns
    -------
    out : Array
        If the variance is computed over the entire array, a zero-dimensional array containing the variance;
        otherwise, an array containing the variances. The returned array has the same data type as x.

    Notes
    -----
    - If N - correction is less than or equal to 0, the variance is NaN.
    - NaN values in the array propagate; if any element is NaN, the corresponding variance is NaN.
    """
    if correction == 0:
        bias = af.VarianceBias.POPULATION
    elif correction == 1:
        bias = af.VarianceBias.SAMPLE
    else:
        raise ValueError("Correction can only be set as 0 or 1. Other values are unsupported with arrayfire.")

    return _compute_statistic(x, af.var, axis=axis, keepdims=keepdims, bias=bias)


def _compute_statistic(
    x: Array, operation: Callable, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False, **kwargs: Any
) -> Array:
    if axis is None:
        result = operation(x._array, **kwargs)

        if keepdims:
            result = af.constant(result, (1, 1), dtype=x.dtype)

        return Array._new(result)

    if isinstance(axis, int):
        axis = (axis,)

    result = x._array
    for ax in axis:
        result = operation(result, axis=ax, **kwargs)

    if keepdims:
        new_shape = tuple(1 if i in axis else s for i, s in enumerate(x.shape))
        result = af.moddims(result, new_shape)

    return Array._new(result)
