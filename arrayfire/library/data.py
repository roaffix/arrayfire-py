from typing import Callable, Tuple, Union, cast

from typing_extensions import ParamSpec

from arrayfire import Array
from arrayfire.backend import _clib_wrapper as wrapper
from arrayfire.dtypes import Dtype, float32

_pyrange = range

P = ParamSpec("P")


def _afarray_as_array(func: Callable[P, Array]) -> Callable[P, Array]:
    """
    Decorator that converts a function returning an array to return an ArrayFire Array.

    Parameters
    ----------
    func : Callable[P, Array]
        The original function that returns an array.

    Returns
    -------
    Callable[P, Array]
        A decorated function that returns an ArrayFire Array.
    """

    def decorated(*args: P.args, **kwargs: P.kwargs) -> Array:
        out = Array()
        result = func(*args, **kwargs)
        out.arr = result  # type: ignore[assignment]
        return out

    return decorated


@_afarray_as_array
def constant(scalar: Union[int, float, complex], shape: Tuple[int, ...] = (1,), dtype: Dtype = float32) -> Array:
    """
    Create a multi-dimensional array filled with a constant value.

    Parameters
    ----------
    scalar : Union[int, float, complex]
        The value to fill each element of the constant array with.

    shape : Tuple[int, ...], optional, default: (1,)
        The shape of the constant array.

    dtype : Dtype, optional, default: float32
        Data type of the array.

    Returns
    -------
    Array
        A multi-dimensional ArrayFire array filled with the specified value.

    Notes
    -----
    The shape parameter determines the dimensions of the resulting array:
    - If shape is (x1,), the output is a 1D array of size (x1,).
    - If shape is (x1, x2), the output is a 2D array of size (x1, x2).
    - If shape is (x1, x2, x3), the output is a 3D array of size (x1, x2, x3).
    - If shape is (x1, x2, x3, x4), the output is a 4D array of size (x1, x2, x3, x4).
    """
    result = wrapper.create_constant_array(scalar, shape, dtype)
    return cast(Array, result)  # HACK actually it return AFArrayType, but decorator makes it an ArrayFire Array.


@_afarray_as_array
def range(shape: Tuple[int, ...], axis: int = 0, dtype: Dtype = float32) -> Array:
    """
    Create a multi-dimensional array using the length of a dimension as a range.

    Parameters
    ----------
    shape : Tuple[int, ...]
        The shape of the resulting array. Each element represents the length
        of a corresponding dimension.

    axis : int, optional, default: 0
        The dimension along which the range is calculated.

    dtype : Dtype, optional, default: float32
        Data type of the array.

    Returns
    -------
    Array
        A multi-dimensional ArrayFire array whose elements along `axis` fall
        between [0, self.ndims[axis]-1].

    Raises
    ------
    ValueError
        If axis value is greater than the number of axes in resulting Array.

    Notes
    -----
    The `shape` parameter determines the dimensions of the resulting array:
    - If shape is (x1,), the output is a 1D array of size (x1,).
    - If shape is (x1, x2), the output is a 2D array of size (x1, x2).
    - If shape is (x1, x2, x3), the output is a 3D array of size (x1, x2, x3).
    - If shape is (x1, x2, x3, x4), the output is a 4D array of size (x1, x2, x3, x4).

    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.range((3, 2))  # axis is not specified, range is along the first dimension.
    >>> af.display(a)  # The data ranges from [0 - 2] (3 elements along the first dimension)
    [3 2 1 1]
        0.0000     0.0000
        1.0000     1.0000
        2.0000     2.0000

    >>> a = af.range((3, 2), axis=1)  # axis is 1, range is along the second dimension.
    >>> af.display(a)  # The data ranges from [0 - 1] (2 elements along the second dimension)
    [3 2 1 1]
        0.0000     1.0000
        0.0000     1.0000
        0.0000     1.0000
    """
    if axis > len(shape):
        raise ValueError(
            f"Can not calculate along {axis} dimension. The resulting Array is set to has {len(shape)} dimensions."
        )

    result = wrapper.af_range(shape, axis, dtype)
    return cast(Array, result)  # HACK actually it return AFArrayType, but decorator makes it an ArrayFire Array.
