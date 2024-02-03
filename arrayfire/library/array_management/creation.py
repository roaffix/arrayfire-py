from typing import cast

import arrayfire_wrapper.lib as wrapper

from arrayfire.array_object import Array, afarray_as_array
from arrayfire.dtypes import Dtype, float32
from arrayfire.library.constants import Pad

from ._constant import create_constant_array


@afarray_as_array
def constant(scalar: int | float | complex, shape: tuple[int, ...] = (1,), dtype: Dtype = float32) -> Array:
    """
    Create a multi-dimensional array filled with a constant value.

    Parameters
    ----------
    scalar : int | float | complex
        The value to fill each element of the constant array with.

    shape : tuple[int, ...], optional, default: (1,)
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
    return cast(Array, create_constant_array(scalar, shape, dtype))


@afarray_as_array
def diag(array: Array, /, *, diag_index: int = 0, extract: bool = True) -> Array:
    if extract:
        return cast(Array, wrapper.diag_extract(array.arr, diag_index))

    return cast(Array, wrapper.diag_create(array.arr, diag_index))


@afarray_as_array
def identity(shape: tuple[int, ...], dtype: Dtype = float32) -> Array:
    """
    Create an identity matrix or batch of identity matrices.

    Parameters
    ----------
    shape : tuple[int, ...]
        The shape of the resulting identity array or batch of arrays.
        Must have at least 2 values.

    dtype : Dtype, optional, default: float32
        Data type of the array.

    Returns
    -------
    Array
        A multi-dimensional ArrayFire array where the first two dimensions
        form an identity matrix or batch of matrices.

    Notes
    -----
    The `shape` parameter determines the dimensions of the resulting array:
    - If shape is (x1, x2), the output is a 2D array of size (x1, x2).
    - If shape is (x1, x2, x3), the output is a 3D array of size (x1, x2, x3).
    - If shape is (x1, x2, x3, x4), the output is a 4D array of size (x1, x2, x3, x4).

    Raises
    ------
    ValueError
        If shape is not a tuple or has less than two values.

    Examples
    --------
    >>> import arrayfire as af
    >>> identity_matrix = af.identity((3, 3))  # Create a 3x3 identity matrix
    >>> af.display(identity_matrix)
    [3 3 1 1]
        1.0000     0.0000     0.0000
        0.0000     1.0000     0.0000
        0.0000     0.0000     1.0000

    >>> identity_batch = af.identity((2, 2, 3))  # Create a batch of 3 identity 2x2 matrices
    >>> af.display(identity_batch)
    [2 2 3 1]
        1.0000     0.0000     1.0000     0.0000     1.0000     0.0000
        0.0000     1.0000     0.0000     1.0000     0.0000     1.0000
    """
    return cast(Array, wrapper.identity(shape, dtype))


@afarray_as_array
def iota(shape: tuple[int, ...], /, *, tile_shape: tuple[int, ...] = (), dtype: Dtype = float32) -> Array:
    # if tile_shape:
    #     min_length = min(len(shape), len(tile_shape))
    #     tile_shape = tuple(tile_shape[:min_length] + shape[min_length:])
    # else:
    #     tile_shape = (1,) * len(shape)

    return cast(Array, wrapper.iota(shape, tile_shape, dtype))


@afarray_as_array
def lower(array: Array, /, *, is_unit_diag: bool = False) -> Array:
    return cast(Array, wrapper.lower(array.arr, is_unit_diag))


@afarray_as_array
def upper(array: Array, /, *, is_unit_diag: bool = False) -> Array:
    return cast(Array, wrapper.upper(array.arr, is_unit_diag))


@afarray_as_array
def pad(
    array: Array, start_shape: tuple[int, ...], end_shape: tuple[int, ...], /, *, fill_type: Pad = Pad.ZERO
) -> Array:
    return cast(Array, wrapper.pad(array.arr, start_shape, end_shape, fill_type))


@afarray_as_array
def range(shape: tuple[int, ...], /, *, axis: int = 0, dtype: Dtype = float32) -> Array:
    """
    Create a multi-dimensional array using the length of a dimension as a range.

    Parameters
    ----------
    shape : tuple[int, ...]
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

    return cast(Array, wrapper.range(shape, axis, dtype))
