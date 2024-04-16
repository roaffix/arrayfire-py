from __future__ import annotations

import arrayfire as af

from ._array_object import Array
from ._data_type_functions import astype
from ._dtypes import bool as array_api_bool
from ._statistical_functions import _compute_statistic


def all(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """
    Tests whether all input array elements evaluate to True along a specified axis.

    Parameters
    ----------
    x : Array
        Input array.
    axis : int | Tuple[int, ...] | None, optional
        Axis or axes along which to perform a logical AND reduction. If None, a logical AND reduction is
        performed over the entire array. If a tuple of integers, logical AND reductions are performed over
        multiple axes. Default is None.
    keepdims : bool, optional
        If True, keeps the reduced dimensions as singleton dimensions in the result, making the result
        compatible with the input array. If False, reduced dimensions are not included in the result. Default is False.

    Returns
    -------
    out : Array
        If a logical AND reduction was performed over the entire array, the returned array is a zero-dimensional
        array containing the test result; otherwise, the returned array is a non-zero-dimensional array containing
        the test results. The returned array has a data type of bool.

    Notes
    -----
    - Positive infinity, negative infinity, and NaN must evaluate to True.
    - If x has a complex floating-point data type, elements having a non-zero component must evaluate to True.
    - If x is an empty array or the size of the axis along which to evaluate elements is zero, the test result must
      be True.
    """
    if axis is None:
        axis = x.shape

    result = _compute_statistic(x, af.all_true, axis=axis, keepdims=keepdims)
    return astype(result, array_api_bool)


def any(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """
    Tests whether any input array elements evaluate to True along a specified axis.

    Parameters
    ----------
    x : Array
        Input array. Should have a numeric or boolean data type.
    axis : int | Tuple[int, ...] | None, optional
        Axis or axes along which to perform a logical OR reduction. If None, a logical OR reduction is
        performed over the entire array. If a tuple of integers, logical OR reductions are performed over
        multiple axes. A valid axis must be an integer within the interval [-N, N), where N is the rank
        (number of dimensions) of x. Default is None.
    keepdims : bool, optional
        If True, the reduced axes (dimensions) are included in the result as singleton dimensions, making the
        result compatible with the input array. If False, the reduced axes are not included in the result.
        Default is False.

    Returns
    -------
    out : Array
        If a logical OR reduction was performed over the entire array, the returned array is a zero-dimensional
        array containing the test result; otherwise, the returned array is a non-zero-dimensional array containing
        the test results. The returned array has a data type of bool.

    Notes
    -----
    - Positive infinity, negative infinity, and NaN are considered as True.
    - If x has a complex floating-point data type, elements having a non-zero component must evaluate to True.
    - If x is an empty array or the size of the axis along which to evaluate elements is zero, the test result must
      be False.
    """
    result = _compute_statistic(x, af.any_true, axis=axis, keepdims=keepdims)
    return astype(result, array_api_bool)
