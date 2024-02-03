import warnings
from typing import cast

import arrayfire_wrapper.lib as wrapper

from arrayfire import Array
from arrayfire.array_object import afarray_as_array

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
def tile(array: Array, shape: tuple[int, ...], /) -> Array:
    if len(shape) > 4:
        raise ValueError("Max 4-dimensional arrays are supported.")

    return cast(Array, wrapper.tile(array.arr, *shape))


@afarray_as_array
def transpose(array: Array, /, *, conjugate: bool = False, inplace: bool = False) -> Array:
    if inplace:
        wrapper.transpose_inplace(array.arr, conjugate)
        return array

    return cast(Array, wrapper.transpose(array.arr, conjugate))
