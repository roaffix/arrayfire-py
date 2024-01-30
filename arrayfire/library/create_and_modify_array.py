from typing import cast

import arrayfire_wrapper.lib as wrapper

from arrayfire import Array
from arrayfire.array_object import afarray_as_array


@afarray_as_array
def moddims(array: Array, shape: tuple[int, ...]) -> Array:
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
