from __future__ import annotations

from functools import reduce

import arrayfire as af

from ._array_object import Array


def concat(arrays: tuple[Array, ...] | list[Array], /, *, axis: int | None = 0) -> Array:
    """
    Concatenates a sequence of arrays along a specified axis. If `axis` is None, all arrays are flattened before
    concatenation. Negative `axis` values are interpreted as counting from the last dimension backwards.

    Parameters
    ----------
    arrays : tuple of Array or list of Array
        A tuple or list of ArrayFire arrays to be concatenated. All arrays must have compatible shapes except along
        the concatenation axis.
    axis : int | None, optional
        The axis along which the arrays will be joined. If None, arrays are flattened before concatenation.
        If negative, the axis is determined from the last dimension. The default is 0.

    Returns
    -------
    Array
        An ArrayFire array resulting from the concatenation of the input arrays along the specified axis.

    Raises
    ------
    ValueError
        If the `arrays` argument is empty.
    TypeError
        If any element in `arrays` is not an Array.

    Notes
    -----
    - Concatenation is performed on the specified `axis`. If `axis` is 0, it concatenates along the rows. If `axis`
      is 1, it concatenates along the columns, etc.
    - It is essential that all arrays have the same shape in every dimension except for the dimension corresponding
      to `axis`.
    - When `axis` is None, all input arrays are flattened into 1-D arrays before concatenation.
    """
    if not arrays:
        raise ValueError("At least one array requires to concatenate.")

    for array in arrays:
        if not isinstance(array, Array):
            raise TypeError("All elements must be Array arrays.")

    if axis is None:
        afarrays = [af.flat(array._array) for array in arrays]
        axis = 0
    elif axis < 0:
        axes = arrays[0].ndim
        axis += axes
        afarrays = [array._array for array in arrays]
    else:
        afarrays = [array._array for array in arrays]

    return Array._new(af.join(axis, *afarrays))


def expand_dims(x: Array, /, *, axis: int) -> Array:
    """
    Expands the shape of an array by inserting a new axis (dimension) of size one at the position specified by axis.

    Parameters
    ----------
    x : Array
        Input ArrayFire array.
    axis : int, optional
        Axis position (zero-based). If x has rank N, a valid axis must reside on the closed-interval [-N-1, N].
        If provided a negative axis, the axis position at which to insert a singleton dimension is computed
        as N + axis + 1.
        For example, if provided -1, the resolved axis position will be N, appending a dimension at the end.
        If provided -N-1, the resolved axis position will be 0, prepending a dimension. An IndexError is raised
        if provided an invalid axis position.

    Returns
    -------
    out : Array
        An expanded output array having the same data type as x with a new axis of size one inserted.

    Raises
    ------
    IndexError
        If the specified axis is out of the valid range for the input array's dimensions.
    """
    N = x.ndim

    if axis < -N - 1 or axis > N:
        raise IndexError(f"axis {axis} is out of bounds for array of dimension {N}.")

    if axis < 0:
        axis += N + 1

    new_shape = [1 if i == axis else x.shape[i] for i in range(max(N, axis) + 1)]
    if len(new_shape) < N + 1:
        new_shape += [x.shape[i] for i in range(len(new_shape), N)]

    return Array._new(af.moddims(x._array, tuple(new_shape)))


def flip(x: Array, /, *, axis: int | tuple[int, ...] | None = None) -> Array:
    """
    Reverses the order of elements in an array along the given axis or axes. The shape of the array is preserved.

    Parameters
    ----------
    x : Array
        Input ArrayFire array.
    axis : int | Tuple[int, ...] | None, optional
        Axis or axes along which to flip the elements. If axis is None, the function flips the elements along all axes.
        If axis is a negative number, it counts from the last dimension. If a tuple of axes is provided, it flips only
        the specified axes. Default is None.

    Returns
    -------
    out : Array
        An output array having the same data type and shape as x, but with elements reversed along the specified axes.

    Notes
    -----
    - The array's shape is maintained, and only the order of elements is reversed.
    - Negative axis values are interpreted as counting from the last dimension towards the first.
    """
    if axis is None:
        # TODO
        return NotImplemented

    if isinstance(axis, int):
        if axis < 0:
            axis += x.ndim
        return Array._new(af.flip(x._array, axis=axis))

    if isinstance(axis, tuple):
        # TODO
        return NotImplemented

    raise TypeError("Axis must be an integer, a tuple of integers, or None")


def permute_dims(x: Array, /, axes: tuple[int, ...]) -> Array:
    """
    Permutes the axes (dimensions) of an array according to a specified tuple.

    Parameters
    ----------
    x : Array
        Input ArrayFire array.
    axes : Tuple[int, ...]
        Tuple containing a permutation of indices (0, 1, ..., N-1) where N is the number of dimensions of x.
        Each element in the tuple specifies the new position of the dimension at that index.

    Returns
    -------
    out : Array
        An array with the same data type as x, with its dimensions permuted according to the `axes` tuple.

    Notes
    -----
    - The function requires that the `axes` tuple be a complete permutation of the array dimensions indices,
      meaning all indices must be present, with no repeats and no omissions.
    - Misconfiguration in the axes tuple, such as duplicate indices or indices out of bounds, will result
      in a runtime error.
    """
    if len(axes) != x.shape:
        raise ValueError("Length of axes tuple must match the number of axes in the array.")

    if sorted(axes) != list(range(x.ndim)):
        raise ValueError("Axes tuple must be a permutation of [0, ..., N-1] where N is the number of dimensions.")

    return Array._new(af.reorder(x._array, shape=axes))


def reshape(x: Array, /, shape: tuple[int, ...], *, copy: bool | None = None) -> Array:
    """
    Reshapes an array to a specified shape without changing the underlying data layout, with an option to copy the
    data.

    Parameters
    ----------
    x : Array
        Input ArrayFire array to be reshaped.
    shape : Tuple[int, ...]
        A new shape for the array. One shape dimension can be -1, in which case that dimension is inferred from the
        remaining dimensions and the total number of elements in the array.
    copy : bool | None, optional
        Specifies whether to forcibly copy the array:
        - If True, the function always copies the array.
        - If False, the function will not copy the array and will raise a ValueError if a copy is necessary for
          reshaping.
        - If None (default), the function reuses the existing memory buffer if possible, and copies otherwise.

    Returns
    -------
    out : Array
        An output array with the specified shape, having the same data type and elements as x.

    Raises
    ------
    ValueError
        If `copy` is False and a copy is necessary to achieve the requested shape, or if the specified shape is not
        compatible.

    Notes
    -----
    - Reshaping is done without altering the underlying data order when possible.
    - The product of the new dimensions must exactly match the total number of elements in the input array.
    """
    if -1 in shape:
        product_of_non_negative_dimensions = 1
        negative_count = 0
        for s in shape:
            if s != -1:
                product_of_non_negative_dimensions *= s
            else:
                negative_count += 1

        if negative_count > 1:
            raise ValueError("Only one dimension can be -1")

        inferred_dimension = x.size // product_of_non_negative_dimensions
        shape = tuple(inferred_dimension if s == -1 else s for s in shape)

    if reduce(lambda x, y: x * y, shape) != x.size:
        raise ValueError("Total elements mismatch between input array and new shape")

    if copy is True:
        # Explicitly copy the array if requested
        new_array = af.copy_array(x._array)
        return Array._new(af.moddims(new_array, shape))
    elif copy is False and not x._array.is_linear:
        raise ValueError("Reshape cannot be done without copying, but 'copy' is set to False")
    else:
        # Default case, reshape without copying if possible
        return Array._new(af.moddims(x._array, shape))


def roll(
    x: Array,
    /,
    shift: int | tuple[int, ...],
    *,
    axis: int | tuple[int, ...] | None = None,
) -> Array:
    """
    Rolls the elements of an array along specified axes. Elements that roll beyond the last position are reintroduced
    at the first.

    Parameters
    ----------
    x : Array
        Input ArrayFire array to be rolled.
    shift : int | Tuple[int, ...]
        The number of places by which elements are to be shifted. If `shift` is an integer and `axis` is a tuple,
        the same shift is applied to all specified axes. If both `shift` and `axis` are tuples, they must have
        the same length, and each axis will be shifted by the corresponding element in `shift`.
    axis : int | Tuple[int, ...] | None, optional
        The axis or axes along which elements are to be shifted. If None, the array is flattened, shifted, and then
        restored to its original shape.

    Returns
    -------
    out : Array
        An output array with the same data type as x, whose elements have been shifted as specified.

    Notes
    -----
    - Positive shifts move elements toward higher indices, while negative shifts move them toward lower indices.
    - The function wraps around the edges of the array, reintroducing elements that are shifted out of bounds.
    """
    if isinstance(shift, int):
        shift = (shift,)

    if axis is None:
        flat_x = af.flat(x._array)
        rolled_x = af.shift(flat_x, shift)
        return Array._new(af.moddims(rolled_x, x.shape))

    if isinstance(axis, int):
        axis = (axis,)

    # If axis and shift are tuples, validate their lengths
    if isinstance(shift, tuple) and isinstance(axis, tuple) and len(shift) != len(axis):
        raise ValueError("If both 'shift' and 'axis' are tuples, they must have the same length.")

    result = x._array
    for ax, sh in zip(axis, shift):
        result = af.shift(result, (sh if ax == 0 else 0, sh if ax == 1 else 0, sh if ax == 2 else 0))

    return Array._new(result)


def squeeze(x: Array, /, axis: int | tuple[int, ...]) -> Array:
    """
    Removes singleton dimensions from an array along specified axes.

    Parameters
    ----------
    x : Array
        Input ArrayFire array.
    axis : int | Tuple[int, ...]
        Axis or axes to remove. These must be singleton dimensions in the array (i.e., dimensions with size 1).
        If a specified axis has a size greater than one, a ValueError is raised.

    Returns
    -------
    out : Array
        An output array with singleton dimensions removed along the specified axes, having the same data type and
        elements as x.

    Raises
    ------
    ValueError
        If any specified axis is not a singleton dimension.

    Notes
    -----
    - If no axis is provided, all singleton dimensions are removed.
    - If the dimensions along the specified axis are not of size 1, a ValueError is raised.
    """
    if isinstance(axis, int):
        axis = (axis,)

    new_dims = []
    for i in range(len(x.shape)):
        if i in axis:
            if x.shape[i] != 1:
                raise ValueError(f"Axis {i} is not a singleton dimension and cannot be squeezed.")
        else:
            new_dims.append(x.shape[i])

    return Array._new(af.moddims(x._array, tuple(new_dims)))


def stack(arrays: tuple[Array, ...] | list[Array], /, *, axis: int = 0) -> Array:
    """
    Joins a sequence of arrays along a new axis.

    Parameters
    ----------
    arrays : Union[Tuple[Array, ...], List[Array]]
        Input arrays to join. Each array must have the same shape.
    axis : int, optional
        Axis along which the arrays will be joined. This axis refers to the new axis in the output array.
        The value of `axis` can range from -N-1 to N (exclusive of N), where N is the rank (number of dimensions)
        of each input array. Default is 0.

    Returns
    -------
    out : Array
        An output array having rank N+1, where N is the rank (number of dimensions) of the input arrays.
        The output array will have the same data type as the input arrays if they are of the same type.

    Raises
    ------
    ValueError
        If not all input arrays have the same shape or if the axis is out of the allowed range.

    Notes
    -----
    - The new axis is inserted before the dimension specified by `axis`.
    - Each array must have exactly the same shape.
    """
    if not arrays:
        raise ValueError("No arrays provided for stacking.")

    if not all(arr.shape == arrays[0].shape for arr in arrays):
        raise ValueError("All input arrays must have the same shape.")

    afarrays = [
        af.moddims(arr._array, tuple([1 if i == axis else arr.shape[i] for i in range(arr.ndim)] + [1]))
        for arr in arrays
    ]

    result = afarrays[0]
    for arr in afarrays[1:]:
        result = af.join(axis, result, arr)

    return Array._new(result)
