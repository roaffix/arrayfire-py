import arrayfire as af

from ._array_object import Array


def take(x: Array, indices: Array, /, *, axis: int | None = None) -> Array:
    """
    Returns elements of an array along a specified axis using a set of indices.

    This function extracts elements from the input array `x` at positions specified by the `indices` array. This
    operation is similar to fancy indexing in NumPy, but it's limited to using one-dimensional arrays for indices.
    The function allows selection along a specified axis. If no axis is specified and the input array is flat, it
    behaves as if operating along the first axis.

    Parameters
    ----------
    x : Array
        The input array from which to take elements.
    indices : Array
        A one-dimensional array of integer indices specifying which elements to extract.
    axis : int | None, optional
        The axis over which to select values. If the axis is negative, the selection is made from the last dimension.
        For one-dimensional `x`, `axis` is optional; for multi-dimensional `x`, `axis` is required.

    Returns
    -------
    Array
        An array that contains the selected elements. This output array will have the same rank as `x`, but the size of
        the dimension along the specified `axis` will correspond to the number of elements in `indices`.

    Notes
    -----
    - The function mimics part of the behavior of advanced indexing in NumPy but is constrained by the current
      specification that avoids __setitem__ and mutation of the array through indexing.
    - If `axis` is None and `x` is multi-dimensional, an exception will be raised since an axis must be specified.

    Raises
    ------
    ValueError
        If `axis` is None and `x` is multi-dimensional, or if `indices` is not a one-dimensional array.

    """
    if axis is None:
        flat_array = af.flat(x._array)
        return Array._new(af.lookup(flat_array, indices._array))

    if axis != 0:
        shape = (x._array.size,)
        afarray = af.moddims(x._array, shape)

    return Array._new(af.lookup(afarray, indices._array, axis=axis))
