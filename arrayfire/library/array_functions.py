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
    "lookup",
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
    """
    Extract a diagonal from or create a diagonal matrix based on an input array.

    This method operates on an ArrayFire array, allowing for the extraction of a specified diagonal
    from a 2-dimensional array or the creation of a diagonal matrix from a 1-dimensional array.

    Parameters
    ----------
    array : Array
        The input ArrayFire array. For diagonal extraction, this should be a 2-dimensional array.
        For diagonal matrix creation, this should be a 1-dimensional array.

    diag_index : int, optional, keyword-only, default: 0
        The index of the diagonal that the operation pertains to.
        - diag_index == 0 signifies the main diagonal.
        - diag_index > 0 signifies a super diagonal.
        - diag_index < 0 signifies a sub diagonal.

    extract : bool, optional, keyword-only, default: True
        Determines the operation to perform:
        - If True, the method extracts the specified diagonal from a 2-dimensional array.
        - If False, the method creates a diagonal matrix with the input array populating the specified diagonal.

    Returns
    -------
    Array
        - If `extract` is True, the returned Array contains the `diag_index`'th diagonal elements from the input array.
        - If `extract` is False, the returned Array is a diagonal matrix with the input array elements placed along
        the `diag_index`'th diagonal.

    Notes
    -----
    The `diag_index` parameter allows for flexible selection of diagonals, enabling operations not just on the main
    diagonal but also on any super or sub diagonals relative to the main.
    """
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
    >>> af.identity((3, 3))  # Create a 3x3 identity matrix
    [3 3 1 1]
        1.0000     0.0000     0.0000
        0.0000     1.0000     0.0000
        0.0000     0.0000     1.0000

    >>> af.identity((2, 2, 3))  # Create a batch of 3 identity 2x2 matrices
    [2 2 3 1]
        1.0000     0.0000     1.0000     0.0000     1.0000     0.0000
        0.0000     1.0000     0.0000     1.0000     0.0000     1.0000
    """
    if not isinstance(shape, tuple) and len(shape) < 2:
        raise ValueError("Argument shape must be a tuple with at least 2 values.")

    return cast(Array, wrapper.identity(shape, dtype))


@afarray_as_array
def iota(shape: int | tuple[int, ...], /, *, tile_shape: tuple[int, ...] = (), dtype: Dtype = float32) -> Array:
    """
    Generate a multi-dimensional ArrayFire array with values populated based on their linear index within the array,
    optionally tiling the result to create larger arrays.

    This function creates an array where each element's value represents its linear index within the array, starting
    from 0. It supports optional tiling, which repeats the array across specified dimensions to create a larger array.

    Parameters
    ----------
    shape : tuple[int, ...]
        The shape of the array to be generated. This parameter defines the dimensions of the array.
        For example, `shape=5` creates a 1-dimensional array of length 5, `shape=(5, 4)` creates a 2D array
        of size 5x4, and so on.

    tile_shape : tuple[int, ...], optional, keyword-only, default: ()
        The shape used for tiling the generated array. Each element in the tuple represents the number of times
        the array is repeated along that dimension. By default, no tiling is applied.
        For example, `tile_shape=(2, 3)` will tile the generated array 2 times along the first dimension and
        3 times along the second dimension.

    dtype : Dtype, optional, keyword-only, default: float32
        The data type of the array elements. This determines the type of the values in the generated array.

    Returns
    -------
    Array
        A multi-dimensional ArrayFire array with elements populated based on their linear index, optionally tiled
        according to `tile_shape`.

    Raises
    ------
    ValueError
        If `shape` is not int or tuple with less than one value.

    Examples
    --------
    >>> import arrayfire as af
    >>> af.iota((3, 3))  # Generate a 3x3 array without tiling
    [3 3 1 1]
        0.0000     3.0000     6.0000
        1.0000     4.0000     7.0000
        2.0000     5.0000     8.0000

    >>> af.iota((3, 3), tile_shape=(1, 2))  # Generate and tile the array along the second dimension
    [3 6 1 1]
        0.0000     3.0000     6.0000     0.0000     3.0000     6.0000
        1.0000     4.0000     7.0000     1.0000     4.0000     7.0000
        2.0000     5.0000     8.0000     2.0000     5.0000     8.0000
    """
    if isinstance(shape, int):
        shape = (shape,)

    if not isinstance(shape, tuple) or not shape:
        raise ValueError("Argument shape must be a tuple with at least 1 value.")

    return cast(Array, wrapper.iota(shape, tile_shape, dtype))


@afarray_as_array
def lower(array: Array, /, *, is_unit_diag: bool = False) -> Array:
    """
    Extract the lower triangular part of a given multi-dimensional ArrayFire array.

    This function returns the lower triangular matrix from the input array. If the `is_unit_diag` flag
    is set to True, the diagonal is considered to be all ones, and therefore not explicitly stored in the
    output.

    Parameters
    ----------
    array : Array
        The input ArrayFire array from which to extract the lower triangular part. This array must be
        at least 2-dimensional.

    is_unit_diag : bool, optional, keyword-only, default: False
        A flag that specifies whether the diagonal elements of the lower triangular matrix are to be considered as 1.
        If True, the diagonal elements are assumed to be 1, and thus not explicitly included in the output array.
        If False, the diagonal elements are taken as they appear in the input array.

    Returns
    -------
    Array
        An ArrayFire array containing the lower triangular part of the input array. If `is_unit_diag` is True,
        the diagonal elements are considered to be 1 but are not explicitly included in the array.

    Notes
    -----
    - The function does not alter the elements above the main diagonal; it simply does not include them in the output.
    - This function can be useful for mathematical operations that require lower triangular matrices, such as certain
    types of matrix factorizations.

    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.randu((3, 3))  # Generate a random 3x3 array
    >>> a
    [3 3 1 1]
        0.6010     0.2126     0.2864
        0.0278     0.0655     0.3410
        0.9806     0.5497     0.7509

    >>> af.lower(a)  # Extract lower triangular part without unit diagonal
    [3 3 1 1]
        0.6010     0.0000     0.0000
        0.0278     0.0655     0.0000
        0.9806     0.5497     0.7509

    >>> af.lower(a, is_unit_diag=True)  # Extract lower triangular part with unit diagonal
    [3 3 1 1]
        1.0000     0.0000     0.0000
        0.0278     1.0000     0.0000
        0.9806     0.5497     1.0000
    """
    return cast(Array, wrapper.lower(array.arr, is_unit_diag))


@afarray_as_array
def upper(array: Array, /, *, is_unit_diag: bool = False) -> Array:
    """
    Extract the upper triangular part of a given multi-dimensional ArrayFire array.

    This function returns the upper triangular matrix from the input array. If the `is_unit_diag` flag
    is set to True, the diagonal elements are considered to be all ones, and therefore not explicitly stored
    in the output.

    Parameters
    ----------
    array : Array
        The input ArrayFire array from which to extract the upper triangular part. This array must be
        at least 2-dimensional.

    is_unit_diag : bool, optional, keyword-only, default: False
        A flag that specifies whether the diagonal elements of the upper triangular matrix are to be considered as 1.
        If True, the diagonal elements are assumed to be 1, and thus not explicitly included in the output array.
        If False, the diagonal elements are taken as they appear in the input array.

    Returns
    -------
    Array
        An ArrayFire array containing the upper triangular part of the input array. If `is_unit_diag` is True,
        the diagonal elements are considered to be 1 but are not explicitly included in the array.

    Notes
    -----
    - The function does not alter the elements below the main diagonal; it simply does not include them in the output.
    - This function can be useful for mathematical operations that require upper triangular matrices, such as certain
    types of matrix factorizations.

    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.randu((3, 3))  # Generate a random 3x3 array
    >>> a
    [3 3 1 1]
        0.8962     0.6105     0.7896
        0.3712     0.5232     0.8966
        0.6755     0.5567     0.0536

    >>> af.upper(a)  # Extract upper triangular part without unit diagonal
    [3 3 1 1]
        0.8962     0.6105     0.7896
        0.0000     0.5232     0.8966
        0.0000     0.0000     0.0536

    >>> af.upper(a, is_unit_diag=True)  # Extract upper triangular part with unit diagonal
    [3 3 1 1]
        1.0000     0.6105     0.7896
        0.0000     1.0000     0.8966
        0.0000     0.0000     1.0000
    """
    return cast(Array, wrapper.upper(array.arr, is_unit_diag))


@afarray_as_array
def pad(
    array: Array, start_shape: tuple[int, ...], end_shape: tuple[int, ...], /, *, fill_type: Pad = Pad.ZERO
) -> Array:
    """
    Pads an ArrayFire array with specified sizes of padding around its edges and fills the padding
    with a specified value.

    This function allows for the padding of an array on all sides with a variety of filling options
    for the new elements added by the padding process. The amount of padding to add at the start and
    end of each dimension is specified by `start_shape` and `end_shape`, respectively.

    Parameters
    ----------
    array : Array
        The input multi-dimensional ArrayFire array to be padded.

    start_shape : tuple[int, ...]
        The amount of padding to add at the beginning of each dimension of the array. Each value in the
        tuple corresponds to a dimension in the array.

    end_shape : tuple[int, ...]
        The amount of padding to add at the end of each dimension of the array. Each value in the tuple
        corresponds to a dimension in the array.

    fill_type : Pad, optional, keyword-only, default: Pad.ZERO
        The type of value to fill the padded areas with. The default is `Pad.ZERO`, which fills the padded
        area with zeros. Other options may include constant values or methods of padding such as edge value
        replication, depending on the library's implementation.

    Returns
    -------
    Array
        The padded ArrayFire array. The shape of the output array will be larger than the input array by
        the amounts specified in `start_shape` and `end_shape`.

    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.randu((3, 3))
    >>> a
    [3 3 1 1]
        0.4107     0.1794     0.3775
        0.8224     0.4198     0.3027
        0.9518     0.0081     0.6456

    >>> af.pad(a, (1, 1), (1, 1))
    [5 5 1 1]
        0.0000     0.0000     0.0000     0.0000     0.0000
        0.0000     0.4107     0.1794     0.3775     0.0000
        0.0000     0.8224     0.4198     0.3027     0.0000
        0.0000     0.9518     0.0081     0.6456     0.0000
        0.0000     0.0000     0.0000     0.0000     0.0000
    """
    return cast(Array, wrapper.pad(array.arr, start_shape, end_shape, fill_type))


@afarray_as_array
def range(shape: int | tuple[int, ...], /, *, axis: int = 0, dtype: Dtype = float32) -> Array:
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

    ValueError
        If `shape` is not int or tuple with less than one value.

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
    >>> a  # The data ranges from [0 - 2] (3 elements along the first dimension)
    [3 2 1 1]
        0.0000     0.0000
        1.0000     1.0000
        2.0000     2.0000

    >>> a = af.range((3, 2), axis=1)  # axis is 1, range is along the second dimension.
    >>> a  # The data ranges from [0 - 1] (2 elements along the second dimension)
    [3 2 1 1]
        0.0000     1.0000
        0.0000     1.0000
        0.0000     1.0000
    """
    if isinstance(shape, int):
        shape = (shape,)

    if not isinstance(shape, tuple) or not shape:
        raise ValueError("Argument shape must be a tuple with at least 1 value.")

    if axis > len(shape):
        raise ValueError(
            f"Can not calculate along {axis} dimension. The resulting Array is set to has {len(shape)} dimensions."
        )

    return cast(Array, wrapper.range(shape, axis, dtype))


# Helper functions


@afarray_as_array
def isinf(array: Array, /) -> Array:
    """
    Check if each element of the input ArrayFire array is infinity.

    This function iterates over each element of the input array to determine if it is infinity. The result is an array
    where each element is a boolean indicating whether the corresponding element in the input array is infinity.

    Parameters
    ----------
    array : Array
        The input multi-dimensional ArrayFire array.

    Returns
    -------
    Array
        An ArrayFire array of booleans, where each element indicates whether the corresponding element in the input
        array is infinity.

    Notes
    -----
    - The input array must not be of a complex data type.
    """
    return cast(Array, wrapper.isinf(array.arr))


@afarray_as_array
def isnan(array: Array, /) -> Array:
    """
    Check if each element of the input ArrayFire array is NaN (Not a Number).

    This function iterates over each element of the input array to determine if it is NaN. The result is an array
    where each element is a boolean indicating whether the corresponding element in the input array is NaN.

    Parameters
    ----------
    array : Array
        The input multi-dimensional ArrayFire array.

    Returns
    -------
    Array
        An ArrayFire array of booleans, where each element indicates whether the corresponding element in the input
        array is NaN.

    Notes
    -----
    - The input array must not be of a complex data type.
    """
    return cast(Array, wrapper.isnan(array.arr))


@afarray_as_array
def iszero(array: Array, /) -> Array:
    """
    Check if each element of the input ArrayFire array is zero.

    This function iterates over each element of the input array to determine if it is zero. The result is an array
    where each element is a boolean indicating whether the corresponding element in the input array is zero.

    Parameters
    ----------
    array : Array
        The input multi-dimensional ArrayFire array.

    Returns
    -------
    Array
        An ArrayFire array of booleans, where each element indicates whether the corresponding element in the input
        array is zero.

    Notes
    -----
    - The input array must not be of a complex data type.
    """
    return cast(Array, wrapper.iszero(array.arr))


# Functions to manage array


@afarray_as_array
def copy_array(array: Array, /) -> Array:
    """
    Performs a deep copy of the given ArrayFire array.

    This function creates a new ArrayFire array that is an identical copy of the input array.
    It ensures that the new array is a separate instance with its own memory, independent of the original array.

    Parameters
    ----------
    array : Array
        The input ArrayFire array to be copied.

    Returns
    -------
    Array
        A new ArrayFire array that is an identical copy of the input array.
        This copy is independent of the original, allowing for modifications without affecting the original array.

    Example
    -------
    >>> import arrayfire as af
    >>> original_array = af.randu((3, 3))  # Create a random 3x3 array
    >>> copied_array = af.copy_array(original_array)  # Make a deep copy of the original array
    >>> original_array == copied_array
    [3 3 1 1]
         1          1          1
         1          1          1
         1          1          1

    >>> original_array.arr == copied_array.arr
    False
    """
    return cast(Array, wrapper.copy_array(array.arr))


def eval(*arrays: Array) -> None:
    """
    Forces the evaluation of one or more ArrayFire arrays.

    In ArrayFire, operations are typically executed lazily, meaning that computations are not actually performed
    until the results are needed. This function forces the evaluation of its input arrays, potentially optimizing
    the execution by combining multiple operations into a single kernel launch.

    Parameters
    ----------
    *arrays : Array
        Variable number of ArrayFire arrays to be evaluated. All input arrays should be of the same size.

    Note
    ----
    It's important to ensure that all input arrays are of the same size to avoid runtime errors. This function
    facilitates performing multiple array operations in a single step, improving performance by reducing the
    number of kernel launches.

    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.constant(1, (3, 3))
    >>> b = af.constant(2, (3, 3))
    >>> c = a + b
    >>> d = a - b
    >>> af.eval(c, d)  # Forces evaluation, optimizing the execution
    >>> c
    [3 3 1 1]
        3.0000     3.0000     3.0000
        3.0000     3.0000     3.0000
        3.0000     3.0000     3.0000

    >>> d
    [3 3 1 1]
        -1.0000    -1.0000    -1.0000
        -1.0000    -1.0000    -1.0000
        -1.0000    -1.0000    -1.0000

    In this example, `eval` is used to force the evaluation of `c` and `d`. Instead of executing two separate
    operations (addition and subtraction), ArrayFire optimizes the process into a single kernel execution, thus
    enhancing performance.
    """
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
    >>> af.flat(arr)  # Flatten the array
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
    """
    Flip an ArrayFire array along a specified dimension.

    This function reverses the order of the elements of the input array along the specified axis (dimension).
    Flipping an array along its vertical axis (0) will invert the rows, whereas flipping along the horizontal
    axis (1) will invert the columns, and so on for higher dimensions.

    Parameters
    ----------
    array : Array
        The input multi-dimensional ArrayFire array to be flipped.

    axis : int, optional, keyword-only, default: 0
        The dimension along which to flip the array. For a 2D array, 0 flips it vertically, and 1 flips it
        horizontally.

    Returns
    -------
    Array
        The ArrayFire array resulting from flipping the input array along the specified axis.

    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.randu((3, 3))  # Generate a random 3x3 array
    >>> a
    [3 3 1 1]
        0.7269     0.3569     0.3341
        0.7104     0.1437     0.0899
        0.5201     0.4563     0.5363

    >>> af.flip(a, axis=0)  # Flip vertically
    [3 3 1 1]
        0.5201     0.4563     0.5363
        0.7104     0.1437     0.0899
        0.7269     0.3569     0.3341

    >>> af.flip(a, axis=1)  # Flip horizontally
    [3 3 1 1]
        0.3341     0.3569     0.7269
        0.0899     0.1437     0.7104
        0.5363     0.4563     0.5201
    """
    return cast(Array, wrapper.flip(array.arr, axis))


@afarray_as_array
def join(axis: int, /, *arrays: Array) -> Array:
    """
    Join two or more ArrayFire arrays along a specified dimension.

    This function concatenates the given arrays along the specified axis (dimension). The arrays must have compatible
    shapes in all dimensions except for the dimension along which they are being joined.

    Parameters
    ----------
    axis : int
        The dimension along which to join the arrays. For example, for 2D arrays, 0 would join the arrays vertically
        (adding rows), and 1 would join them horizontally (adding columns).

    *arrays : Array
        A variable number of ArrayFire arrays to be joined. At least two arrays must be provided. The function is
        capable of joining up to 10 arrays due to API limitations.

    Returns
    -------
    Array
        An ArrayFire array resulting from concatenating the input arrays along the specified axis.

    Raises
    ------
    ValueError
        If fewer than two arrays are provided as input.

    Warning
        If more than 10 arrays are provided, only the first 10 will be processed due to API limitations.

    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.randu((2, 3))
    >>> b = af.randu((2, 3))
    >>> c = af.join(0, a, b)  # Join vertically
    >>> d = af.join(1, a, b)  # Join horizontally
    >>> с
    [4 3 1 1]
        0.9508     0.2591     0.7928
        0.5367     0.8359     0.8719
        0.3266     0.6009     0.2442
        0.6275     0.0495     0.6591

    >>> d
    [2 6 1 1]
        0.9508     0.2591     0.7928     0.3266     0.6009     0.2442
        0.5367     0.8359     0.8719     0.6275     0.0495     0.6591
    """
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
    """
    # TODO add examples to doc
    return cast(Array, wrapper.moddims(array.arr, shape))


@afarray_as_array
def reorder(array: Array, /, *, shape: tuple[int, ...] = (1, 0, 2, 3)) -> Array:
    """
    Reorders the dimensions of the given ArrayFire array according to the specified order.

    This function changes the order of the dimensions of the input array, which can be useful for data rearrangement
    or alignment before further processing. The new dimension order is specified by the `shape` parameter.

    Parameters
    ----------
    array : Array
        The input multi-dimensional ArrayFire array whose dimensions are to be reordered.

    shape : tuple[int, ...], optional, keyword-only, default: (1, 0, 2, 3)
        The new order of the dimensions. The default value swaps the first two dimensions, similar to performing
        a transpose operation on a 2D array. Each element in the tuple represents the index of the dimension in the
        input array that should be moved to this position.

    Returns
    -------
    Array
        An ArrayFire array with its dimensions reordered according to the specified `shape`.

    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.randu((5, 5, 3))  # Generate a 5x5x3 random array
    >>> a
    [5 5 3 1]
        0.4107     0.0081     0.6600     0.1046     0.8395
        0.8224     0.3775     0.0764     0.8827     0.1933
        0.9518     0.3027     0.0901     0.1647     0.7270
        0.1794     0.6456     0.5933     0.8060     0.0322
        0.4198     0.5591     0.1098     0.5938     0.0012

        0.8703     0.9250     0.4387     0.6530     0.4224
        0.5259     0.3063     0.3784     0.5476     0.5293
        0.1443     0.9313     0.4002     0.8577     0.0212
        0.3253     0.8684     0.4390     0.8370     0.1103
        0.5081     0.6592     0.4718     0.0618     0.4420

        0.8355     0.6767     0.1033     0.9426     0.9276
        0.4878     0.6742     0.2119     0.4817     0.8662
        0.2055     0.4523     0.5955     0.9097     0.3578
        0.1794     0.1236     0.3745     0.6821     0.6263
        0.5606     0.7924     0.9165     0.6056     0.9747

    >>> b = af.reorder(a, shape=(2, 0, 1))  # Reorder dimensions: move the third dimension to the first
    >>> b
    [3 5 5 1]
        0.4107     0.8224     0.9518     0.1794     0.4198
        0.8703     0.5259     0.1443     0.3253     0.5081
        0.8355     0.4878     0.2055     0.1794     0.5606

        0.0081     0.3775     0.3027     0.6456     0.5591
        0.9250     0.3063     0.9313     0.8684     0.6592
        0.6767     0.6742     0.4523     0.1236     0.7924

        0.6600     0.0764     0.0901     0.5933     0.1098
        0.4387     0.3784     0.4002     0.4390     0.4718
        0.1033     0.2119     0.5955     0.3745     0.9165

        0.1046     0.8827     0.1647     0.8060     0.5938
        0.6530     0.5476     0.8577     0.8370     0.0618
        0.9426     0.4817     0.9097     0.6821     0.6056

        0.8395     0.1933     0.7270     0.0322     0.0012
        0.4224     0.5293     0.0212     0.1103     0.4420
        0.9276     0.8662     0.3578     0.6263     0.9747

    Note
    ----
    - The `shape` tuple must contain all integers from 0 up to the number of dimensions in `array` - 1, without
    repetition.
    - Reordering dimensions can be particularly useful in preparation for operations that require a specific dimension
    order.
    """
    return cast(Array, wrapper.reorder(array.arr, *shape))


def replace(lhs: Array, rhs: Array | int | float, conditional: Array, /) -> None:
    """
    Conditionally replaces elements of one ArrayFire array with elements from another array or a scalar value.

    This function iterates over each element of the `lhs` array and replaces it with the corresponding element from
    `rhs` if the corresponding element in the `conditional` array is True. If `rhs` is a scalar, `lhs` is updated with
    this scalar value where the condition is True.

    Parameters
    ----------
    lhs : Array
        The left-hand side ArrayFire array whose elements may be replaced based on the condition.

    rhs : Array | int | float
        The right-hand side value(s) used for replacement. This can be an ArrayFire array, integer, or floating-point
        scalar. If `rhs` is an array, it must be the same size as `lhs`.

    conditional : Array
        An ArrayFire array of boolean values indicating where replacement should occur. Must be the same size as `lhs`.

    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.randu((3,3))  # Generate a random 3x3 array
    >>> a
    [3 3 1 1]
        0.4107     0.1794     0.3775
        0.8224     0.4198     0.3027
        0.9518     0.0081     0.6456

    >>> cond = (a >= 0.25) & (a <= 0.75)  # Generate a condition array
    >>> cond
    [3 3 1 1]
        1          0          1
        0          1          1
        0          0          1

    >>> af.replace(a, 0.3333, cond)  # Replace where condition is True with 0.3333
    >>> a
    [3 3 1 1]
        0.3333     0.1794     0.3333
        0.8224     0.3333     0.3333
        0.9518     0.0081     0.3333

    Note
    ----
    - The `lhs`, `rhs` (if an array), and `conditional` arrays must be of the same size.
    """
    if isinstance(rhs, Array):
        wrapper.replace(lhs.arr, conditional.arr, rhs.arr)
        return

    wrapper.replace_scalar(lhs.arr, conditional.arr, rhs)


def select(lhs: Array | int | float, rhs: Array | int | float, conditional: Array, /) -> Array:
    """
    Conditionally selects elements from one of two sources (ArrayFire arrays or scalars) based on a condition array.

    This function iterates over each element of the `conditional` array. For elements where `conditional` is True,
    it selects the corresponding element from `lhs`; otherwise, it selects from `rhs`. The `lhs` and `rhs` can be
    either ArrayFire arrays or scalar values, but at least one of them must be an ArrayFire array.

    Parameters
    ----------
    lhs : Array | int | float
        The left-hand side source for selection. Can be an ArrayFire array or a scalar value. Elements from `lhs`
        are selected where `conditional` is True.

    rhs : Array | int | float
        The right-hand side source for selection. Can be an ArrayFire array or a scalar value. Elements from `rhs`
        are selected where `conditional` is False.

    conditional : Array
        An ArrayFire array of boolean values that serve as the condition for selection.

    Raises
    ------
    TypeError
        If neither `lhs` nor `rhs` is an ArrayFire array.

    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.randu((3,3))  # Generate a random 3x3 array
    >>> b = af.randu((3,3))  # Generate another random 3x3 array
    >>> cond = a > b  # Generate a boolean condition array

    >>> a
    [3 3 1 1]
        0.4107     0.1794     0.3775
        0.8224     0.4198     0.3027
        0.9518     0.0081     0.6456

    >>> b
    [3 3 1 1]
        0.7269     0.3569     0.3341
        0.7104     0.1437     0.0899
        0.5201     0.4563     0.5363

    >>> af.select(cond, a, b)  # Conditionally select between `a` and `b`
    [3 3 1 1]
        0.7269     0.3569     0.3775
        0.8224     0.4198     0.3027
        0.9518     0.4563     0.6456

    Note
    ----
    - The `conditional` array must be of the same size as both `lhs` and `rhs` if they are arrays.
    - At least one of `lhs` or `rhs` must be an ArrayFire array.
    """
    if isinstance(lhs, Array) and isinstance(rhs, Array):
        return cast(Array, wrapper.select(lhs.arr, conditional.arr, rhs.arr))

    if isinstance(lhs, Array) and not isinstance(rhs, Array):
        return cast(Array, wrapper.select_scalar_r(lhs.arr, conditional.arr, rhs))

    if not isinstance(lhs, Array) and isinstance(rhs, Array):
        return cast(Array, wrapper.select_scalar_l(lhs, conditional.arr, rhs.arr))

    raise TypeError("At least one array (lhr or rhs) must be of type af.Array.")


@afarray_as_array
def shift(array: Array, shape: tuple[int, ...], /) -> Array:
    """
    Shifts the input ArrayFire array along each dimension by specified amounts.

    This function cyclically shifts the elements of the input array along each dimension. The amount of shift for each
    dimension is specified in the `shape` tuple. A positive shift moves elements towards higher indices, while a
    negative shift moves them towards lower indices.

    Parameters
    ----------
    array : Array
        The input multi-dimensional ArrayFire array to be shifted.

    shape : tuple[int, ...]
        A tuple specifying the amount of shift along each dimension. Can contain up to four values, corresponding to
        the shift along the first, second, third, and fourth dimensions, respectively. Unspecified dimensions are
        assumed to have a shift of 0.

    Raises
    ------
    ValueError
        If the `shape` tuple contains more than four elements, as only up to 4-dimensional arrays are supported.

    Returns
    -------
    Array
        An ArrayFire array of the same shape as `array`, shifted by the specified amounts along each dimension.

    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.randu((3, 3))  # Generate a random 3x3 array
    >>> a
    [3 3 1 1]
        0.7269     0.3569     0.3341
        0.7104     0.1437     0.0899
        0.5201     0.4563     0.5363

    >>> b = af.shift(a, (2,))  # Shift along the first dimension by 2
    >>> b
    [3 3 1 1]
        0.7104     0.1437     0.0899
        0.5201     0.4563     0.5363
        0.7269     0.3569     0.3341

    >>> c = af.shift(a, (1, -1))  # Shift along the first dimension by 1 and the second by -1
    >>> c
    [3 3 1 1]
        0.4563     0.5363     0.5201
        0.3569     0.3341     0.7269
        0.1437     0.0899     0.7104

    Note
    ----
    - Shifts are performed cyclically, meaning that elements shifted "off" one end of the array reappear at the other.
    """
    if len(shape) > 4:
        raise ValueError("Max 4-dimensional arrays are supported.")

    return cast(Array, wrapper.shift(array.arr, *shape))


@afarray_as_array
def tile(array: Array, /, shape: tuple[int, ...]) -> Array:
    """
    Repeats an ArrayFire array along specified dimensions to create a tiled array.

    This function creates a larger array by repeating the input array a specified number of times along each dimension.
    The amount of repetition for each dimension is specified in the `shape` tuple. This can be used to duplicate data
    along one or more axes.

    Parameters
    ----------
    array : Array
        The input multi-dimensional ArrayFire array to be tiled.

    shape : tuple[int, ...]
        A tuple specifying the number of times the input array should be repeated along each dimension. Can contain up
        to four values, corresponding to the repetition factor along the first, second, third, and fourth dimensions,
        respectively. Dimensions not specified will not be tiled (i.e., treated as if they have a repetition factor
        of 1).

    Raises
    ------
    ValueError
        If the `shape` tuple contains more than four elements, as only up to 4-dimensional arrays are supported.

    Returns
    -------
    Array
        An ArrayFire array resulting from tiling the input array according to the specified `shape`.

    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.randu((2, 3))  # Generate a 2x3 random array
    >>> a
    [2 3 1 1]
        0.9508     0.2591     0.7928
        0.5367     0.8359     0.8719

    >>> af.tile(a, (2,))  # Tile along the first dimension by 2
    [4 3 1 1]
        0.9508     0.2591     0.7928
        0.5367     0.8359     0.8719
        0.9508     0.2591     0.7928
        0.5367     0.8359     0.8719

    >>> af.tile(a, (1, 2))  # Tile along the second dimension by 2
    [2 6 1 1]
        0.9508     0.2591     0.7928     0.9508     0.2591     0.7928
        0.5367     0.8359     0.8719     0.5367     0.8359     0.8719

    >>> af.tile(a, (2, 2))  # Tile both dimensions by 2
    [4 6 1 1]
        0.9508     0.2591     0.7928     0.9508     0.2591     0.7928
        0.5367     0.8359     0.8719     0.5367     0.8359     0.8719
        0.9508     0.2591     0.7928     0.9508     0.2591     0.7928
        0.5367     0.8359     0.8719     0.5367     0.8359     0.8719

    Note
    ----
    - The repetition factor of 1 means the dimension is not tiled.
    """
    if len(shape) > 4:
        raise ValueError("Max 4-dimensional arrays are supported.")

    return cast(Array, wrapper.tile(array.arr, *shape))


@afarray_as_array
def transpose(array: Array, /, *, conjugate: bool = False, inplace: bool = False) -> Array:
    """
    Perform the transpose (and optionally, the complex conjugate transpose) of an ArrayFire array.
    The operation can be performed in-place for square matrices or square matrix batches.

    Parameters
    ----------
    array : Array
        The input multi-dimensional ArrayFire array to be transposed. If `inplace` is True, `array` must be a square
        matrix or a batch of square matrices.

    conjugate : bool, optional, keyword-only, default: False
        If True, performs a complex conjugate transpose. This is only relevant for complex data types and is ignored
        for other data types.

    inplace : bool, optional, keyword-only, default: False
        If True, performs the transpose operation in-place, modifying the input `array`. The input `array` must be a
        square matrix or a batch of square matrices.

    Returns
    -------
    Array
        If `inplace` is False, returns a new ArrayFire array containing the transpose of `array`. If `inplace` is True,
        returns the modified `array` containing its own transpose. For in-place operations, the input `array` is
        directly modified.

    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.randu(3, 3)  # Generate a random 3x3 array
    >>> a
    [3 3 1 1]
        0.7269     0.3569     0.3341
        0.7104     0.1437     0.0899
        0.5201     0.4563     0.5363

    >>> af.transpose(a)  # Transpose the array
    [3 3 1 1]
        0.7269     0.7104     0.5201
        0.3569     0.1437     0.4563
        0.3341     0.0899     0.5363

    Note
    ----
    - The `inplace` operation requires the input array to be a square matrix or a batch of square matrices.
    Attempting an in-place transpose on non-square matrices will result in an error.
    - For complex matrices, setting `conjugate` to True applies the complex conjugate in addition to the transpose
    operation.
    """
    if inplace:
        wrapper.transpose_inplace(array.arr, conjugate)
        return array

    return cast(Array, wrapper.transpose(array.arr, conjugate))


@afarray_as_array
def lookup(array: Array, indices: Array, /, *, axis: int = 0) -> Array:
    """
    Performs a lookup operation on the input ArrayFire array based on the specified indices along a given dimension.

    This function gathers elements from the input array `array` at positions specified by the `indices` array.
    The operation is performed along the dimension specified by `axis`.

    Parameters
    ----------
    array : Array
        The input multi-dimensional ArrayFire array from which elements are to be gathered.

    indices : Array
        An ArrayFire array containing the indices of elements to gather. The values in `indices` should be of integer
        type.

    axis : int, optional, keyword-only, default: 0
        The dimension along which the lookup is performed.

    Returns
    -------
    Array
        An ArrayFire array containing the elements of `array` at the locations specified by `indices`. The shape of
        the output array is determined by the shape of `indices` and the dimension specified by `axis`.

    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.Array([1, 0, 3, 4, 5, 6], shape=(2, 3))  # Create a 2x3 array
    >>> a
    [2 3 1 1]
        1.0000     3.0000     5.0000
        0.0000     4.0000     6.0000

    >>> idx = af.Array([0, 2])  # Indices for lookup
    >>> af.lookup(a, idx, axis=1)  # Lookup along the second dimension
    [2 2 1 1]
        1.0000     5.0000
        0.0000     6.0000

    >>> idx = af.Array([0])  # Indices for lookup
    >>> af.lookup(arr, idx, axis=0)  # Lookup along the first dimension
    [1 3 1 1]
        1.0000     3.0000     5.0000

    Note
    ----
    - The `indices` array must contain integer values indicating the positions of elements to gather from `array`.
    - The dimension specified by `axis` must not exceed the number of dimensions in `array`.
    """
    return cast(Array, wrapper.lookup(array.arr, indices.arr, axis))
