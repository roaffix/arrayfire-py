__all__ = [
    "constant",
    "diag",
    "identity",
    "iota",
    "lower",
    "upper",
    "pad",
    "range",
    "isinf",
    "isnan",
    "iszero",
    "set_manual_eval_flag",
    "eval",
    "copy_array",
    "flat",
    "flip",
    "join",
    "moddims",
    "reorder",
    "replace",
    "select",
    "shift",
    "tile",
    "transpose",
]

import warnings
from typing import cast

import arrayfire_wrapper.lib as wrapper
from arrayfire_wrapper.lib import set_manual_eval_flag

from arrayfire.array_object import Array, afarray_as_array
from arrayfire.dtypes import Dtype, float32
from arrayfire.library.constants import Pad

# Array creation


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
    return cast(Array, wrapper.create_constant_array(scalar, shape, dtype))


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


# Helper functions


@afarray_as_array
def isinf(array: Array, /) -> Array:
    return cast(Array, wrapper.isinf(array.arr))


@afarray_as_array
def isnan(array: Array, /) -> Array:
    return cast(Array, wrapper.isnan(array.arr))


@afarray_as_array
def iszero(array: Array, /) -> Array:
    return cast(Array, wrapper.iszero(array.arr))


# Functions to manage array


@afarray_as_array
def copy_array(array: Array, /) -> Array:
    return cast(Array, wrapper.copy_array(array.arr))


def eval(*arrays: Array) -> None:
    if len(arrays) == 1:
        wrapper.eval(arrays[0].arr)
        return

    arrs = [array.arr for array in arrays]
    wrapper.eval_multiple(len(arrays), *arrs)


# Move and reorder


@afarray_as_array
def flat(array: Array, /) -> Array:
    """
    Flatten the input multi-dimensional array into a 1D array.

    Parameters
    ----------
    array : Array
        The input multi-dimensional array to be flattened.

    Returns
    -------
    Array
        A 1D array containing all the elements from the input array.

    Examples
    --------
    >>> import arrayfire as af
    >>> arr = af.randu(3, 2)  # Create a 3x2 random array
    >>> flattened = af.flat(arr)  # Flatten the array
    >>> af.display(flattened)
    [6 1 1 1]
        0.8364
        0.5604
        0.6352
        0.0062
        0.7052
        0.1676
    """
    return cast(Array, wrapper.flat(array.arr))


@afarray_as_array
def flip(array: Array, /, *, axis: int = 0) -> Array:
    return cast(Array, wrapper.flip(array.arr, axis))


@afarray_as_array
def join(axis: int, /, *arrays: Array) -> Array:
    if len(arrays) < 2:
        raise ValueError("Shape should be at least 2 dimensional.")
    if len(arrays) == 2:
        return cast(Array, wrapper.join(axis, arrays[0].arr, arrays[1].arr))
    if len(arrays) > 10:
        warnings.warn("API is limited to process max of 10 arrays, thus only first 10 units will be processed.")

    afarrays = [array.arr for array in arrays]
    return cast(Array, wrapper.join_many(axis, len(arrays), *afarrays))


@afarray_as_array
def moddims(array: Array, shape: tuple[int, ...], /) -> Array:
    """
    Modify the shape of the array without changing the data layout.

    Parameters
    ----------
    array : af.Array
        Multi-dimensional array to be reshaped.

    shape : tuple of int
        The desired shape of the output array. It should be a tuple of integers
        representing the dimensions of the output array. The product of these
        dimensions must match the total number of elements in the input array.

    Returns
    -------
    out : af.Array
        - An array containing the same data as `array` with the specified shape.
        - The total number of elements in `array` must match the product of the
          dimensions specified in the `shape` tuple.

    Raises
    ------
    ValueError
        If the total number of elements in the input array does not match the
        product of the dimensions specified in the `shape` tuple.

    Notes
    -----
    This function modifies the shape of the input array without changing the
    data layout. The resulting array will have the same data, but with a
    different shape as specified by the `shape` parameter.

    Examples
    --------
    >>> a = af.randu(2, 3, 4)  # Create a random 3D array
    >>> b = moddims(a, (6, 2))  # Reshape to a 2D array with 6 rows and 2 columns
    """

    return cast(Array, wrapper.moddims(array.arr, shape))


@afarray_as_array
def reorder(array: Array, /, *, shape: tuple[int, ...] = (1, 0, 2, 3)) -> Array:
    return cast(Array, wrapper.reorder(array.arr, *shape))


def replace(lhs: Array, rhs: Array | int | float, conditional: Array, /) -> None:
    if isinstance(rhs, Array):
        wrapper.replace(lhs.arr, conditional.arr, rhs.arr)
        return

    wrapper.replace_scalar(lhs.arr, conditional.arr, rhs)


def select(lhs: Array | int | float, rhs: Array | int | float, conditional: Array, /) -> None:
    if isinstance(lhs, Array) and isinstance(rhs, Array):
        wrapper.select(lhs.arr, conditional.arr, rhs.arr)
        return

    if isinstance(lhs, Array) and not isinstance(rhs, Array):
        wrapper.select_scalar_r(lhs.arr, conditional.arr, rhs)
        return

    if not isinstance(lhs, Array) and isinstance(rhs, Array):
        wrapper.select_scalar_l(lhs, conditional.arr, rhs.arr)
        return

    raise TypeError("At least one array (lhr or rhs) must be of type af.Array.")


@afarray_as_array
def shift(array: Array, shape: tuple[int, ...], /) -> Array:
    if len(shape) > 4:
        raise ValueError("Max 4-dimensional arrays are supported.")

    return cast(Array, wrapper.shift(array.arr, *shape))


@afarray_as_array
def tile(array: Array, /, shape: tuple[int, ...]) -> Array:
    if len(shape) > 4:
        raise ValueError("Max 4-dimensional arrays are supported.")

    return cast(Array, wrapper.tile(array.arr, *shape))


@afarray_as_array
def transpose(array: Array, /, *, conjugate: bool = False, inplace: bool = False) -> Array:
    if inplace:
        wrapper.transpose_inplace(array.arr, conjugate)
        return array

    return cast(Array, wrapper.transpose(array.arr, conjugate))
