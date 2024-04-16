__all__ = [
    "accum",
    "scan",
    "where",
    "all_true",
    "any_true",
    "sum",
    "product",
    "count",
    "imax",
    "max",
    "imin",
    "min",
    "diff1",
    "diff2",
    "gradient",
    "set_intersect",
    "set_union",
    "set_unique",
    "sort",
]

from typing import cast

from arrayfire_wrapper import lib as wrapper

from arrayfire import Array
from arrayfire.array_object import afarray_as_array
from arrayfire.library.constants import BinaryOperator


@afarray_as_array
def accum(array: Array, /, axis: int = 0) -> Array:
    """
    Calculate the cumulative sum of elements along a specified dimension.

    Parameters
    ----------
    array : af.Array
        Multi-dimensional ArrayFire array.

    axis : int, optional, default: 0
        Dimension along which the cumulative sum is required.

    Returns
    -------
    af.Array
        An ArrayFire array of the same size as `array` containing the cumulative sum along the specified dimension.

    Note
    ----
    If `axis` is not specified, the cumulative sum is calculated along the first dimension (default: 0).
    """
    return cast(Array, wrapper.accum(array.arr, axis))


@afarray_as_array
def scan(
    array: Array,
    /,
    keys: None | Array = None,
    axis: int = 0,
    op: BinaryOperator = BinaryOperator.ADD,
    inclusive_scan: bool = True,
) -> Array:
    """
    Perform a generalized scan of an array, optionally with a key.

    Parameters
    ----------
    array : af.Array
        Multi-dimensional ArrayFire array.

    keys : af.Array, optional, default: None
        Keys array for generalized scan. If None, a standard scan is performed.

    axis : int, optional, default: 0
        Dimension along which the scan is performed.

    op : af.BINARYOP, optional, default: af.BINARYOP.ADD
        Binary operation that the scan algorithm uses. Can be one of:
        - af.BINARYOP.ADD
        - af.BINARYOP.MUL
        - af.BINARYOP.MIN
        - af.BINARYOP.MAX

    inclusive_scan : bool, optional, default: True
        Specifies if the scan is inclusive.

    Returns
    -------
    out : af.Array
        Array containing the result of the generalized scan.
    """
    if keys:
        return cast(Array, wrapper.scan_by_key(keys.arr, array.arr, axis, op, inclusive_scan))

    return cast(Array, wrapper.scan(array.arr, axis, op, inclusive_scan))


@afarray_as_array
def where(array: Array, /) -> Array:
    """
    Find the indices of non-zero elements.

    Parameters
    ----------
    array : af.Array
        Multi-dimensional ArrayFire array.

    Returns
    -------
    af.Array
        Linear indices for non-zero elements.
    """
    return cast(Array, wrapper.where(array.arr))


def all_true(array: Array, axis: int | None = None) -> bool | Array:
    """
    Check if all the elements along a specified dimension are true.

    Parameters
    ----------
    array : Array
        Multi-dimensional ArrayFire array.

    axis : int, optional, default: None
        Dimension along which the product is required.

    Returns
    -------
    bool | Array
        An ArrayFire array containing True if all elements in `array` along the specified dimension are True.
        If `axis` is `None`, the output is True if `array` does not have any zeros, else False.

    Note
    ----
    If `axis` is `None`, output is True if the array does not have any zeros, else False.
    """
    if axis is None:
        return bool(wrapper.all_true_all(array.arr))

    return Array.from_afarray(wrapper.all_true(array.arr, axis))


def any_true(array: Array, axis: int | None = None) -> bool | Array:
    """
    Check if any of the elements along a specified dimension are true.

    Parameters
    ----------
    array : Array
        Multi-dimensional ArrayFire array.

    axis : int, optional, default: None
        Dimension along which the product is required.

    Returns
    -------
    bool | Array
        An ArrayFire array containing True if any of the elements in `array` along the specified dimension are True.
        If `axis` is `None`, the output is True if `array` does not have any zeros, else False.

    Note
    ----
    If `axis` is `None`, output is True if the array does not have any zeros, else False.
    """
    if axis is None:
        return bool(wrapper.any_true_all(array.arr))

    return Array.from_afarray(wrapper.any_true(array.arr, axis))


def sum(array: Array, /, *, axis: int | None = None, nan_value: float | None = None) -> int | float | complex | Array:
    # FIXME documentation issues
    """
    Calculate the sum of elements along a specified dimension or the entire array.

    Parameters
    ----------
    array : Array
        The multi-dimensional array to calculate the sum of.

    axis : int or None, optional, default: None
        The dimension along which the sum is calculated.
        If None, the sum of all elements in the entire array is calculated.

    nan_value : float or None, optional, default: None
        The value to replace NaN (Not-a-Number) values in the array before summing. If None, NaN values are ignored.

    Returns
    -------
    Array or bool or scalar
        - If `axis` is None and `nan_value` is None, returns a boolean indicating if the sum contains NaN or Inf.
        - If `axis` is None and `nan_value` is not None, returns a boolean indicating if the sum contains NaN or
          Inf after replacing NaN values.
        - If `axis` is not None, returns an Array containing the sum along the specified dimension.
        - If `axis` is not None and `nan_value` is not None, returns an Array containing the sum along the specified
          dimension after replacing NaN values.
    """

    if axis is None:
        if nan_value is None:
            return wrapper.sum_all(array.arr)

        return wrapper.sum_nan_all(array.arr, nan_value)

    if nan_value is None:
        return Array.from_afarray(wrapper.sum(array.arr, axis))

    return Array.from_afarray(wrapper.sum_nan(array.arr, axis, nan_value))  # type: ignore[call-arg]


def product(
    array: Array, /, *, axis: int | None = None, nan_value: float | None = None
) -> int | float | complex | Array:
    # FIXME documentation issues
    """
    Calculate the product of all the elements along a specified dimension.

    Parameters
    ----------
    array : Array
        The multi-dimensional array to calculate the product of.

    axis : int or None, optional, default: None
        The dimension along which the product is calculated.
        If None, the product of all elements in the entire array is returned.

    nan_value : float or None, optional, default: None
        The value to replace NaN (Not-a-Number) values in the array before computing the product.
        If None, NaN values are ignored.

    Returns
    -------
    Array or scalar number
        The product of all elements in `array` along dimension `axis`.
        If `axis` is `None`, the product of the entire array is returned.
    """
    if axis is None:
        if nan_value is None:
            return wrapper.product_all(array.arr)

        return wrapper.product_nan_all(array.arr, nan_value)

    if nan_value is None:
        return Array.from_afarray(wrapper.product(array.arr, axis))

    return Array.from_afarray(wrapper.product_nan(array.arr, axis, nan_value))  # type: ignore[call-arg]


def count(
    array: Array, /, *, axis: int | None = None, keys: Array | None = None
) -> int | float | complex | Array | tuple[Array, Array]:
    """
    Count the number of non-zero elements in an ArrayFire array along a specified dimension or across the entire array.
    Optionally, perform counting based on unique keys.

    Parameters
    ----------
    array : Array
        The input multi-dimensional ArrayFire array whose non-zero elements are to be counted.

    axis : int, optional, keyword-only
        The dimension along which the non-zero elements are counted. If None, the total number of non-zero elements
        across the entire array is returned.

    keys : Array, optional, keyword-only
        An optional one-dimensional ArrayFire array with keys for counting non-zero elements according to unique keys.
        If provided, `axis` determines the dimension along which elements are counted per key. If `axis` is None, it
        defaults to counting across all dimensions for each key.

    Returns
    -------
    int | float | complex | Array | tuple[Array, Array]
        - If `keys` is None and `axis` is None, returns a scalar (int, float, or complex) representing the total count
          of non-zero elements in `array`.
        - If `keys` is None and `axis` is specified, returns an ArrayFire array representing the count of non-zero
          elements along the specified `axis`.
        - If `keys` is provided, returns a tuple containing two ArrayFire arrays: the unique keys and their
          corresponding counts. The counts are performed along the specified `axis` (or across all dimensions if
          `axis` is None).

    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.randu((3, 3))
    >>> a
    [3 3 1 1]
        0.6010     0.2126     0.2864
        0.0278     0.0655     0.3410
        0.9806     0.5497     0.7509

    >>> b = a > 0.5
    >>> b
    [3 3 1 1]
        1          0          0
        0          0          0
        1          1          1

    >>> af.count(b)
    4.0

    >>> af.count(b, axis=0)
    [1 3 1 1]
        2          1          1
    """
    if keys:
        axis_ = -1 if axis is None else axis
        key, value = wrapper.count_by_key(keys.arr, array.arr, axis_)
        return Array.from_afarray(key), Array.from_afarray(value)

    if axis is None:
        return wrapper.count_all(array.arr)

    return Array.from_afarray(wrapper.count(array.arr, axis))


def imax(array: Array, /, *, axis: int | None = None) -> tuple[int | float | complex, int] | tuple[Array, Array]:
    """
    Find the maximum value and its location within an ArrayFire array along a specified dimension, or globally if no
    dimension is specified.

    Parameters
    ----------
    array : Array
        The input multi-dimensional ArrayFire array.

    axis : int, optional, keyword-only
        The dimension along which to find the maximum value. If None (default), the global maximum value and its
        location in the array are returned.

    Returns
    -------
    tuple[int | float | complex, int] | tuple[Array, Array]
        - If `axis` is None, returns a tuple containing the global maximum value (int, float, or complex) and its
          linear index in the array.
        - If `axis` is specified, returns two ArrayFire arrays in a tuple: the first array contains the maximum values
          along the specified dimension, and the second array contains the locations of these maximum values along the
          same dimension.

    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.randu((3, 3))
    >>> a
    [3 3 1 1]
        0.6010     0.2126     0.2864
        0.0278     0.0655     0.3410
        0.9806     0.5497     0.7509

    >>> af.imax(a)
    (0.9805505871772766, 2)

    >>> af.imax(a, axis=1)
    (
    [3 1 1 1]
        0.6010
        0.3410
        0.9806
    ,

    [3 1 1 1]
            0
            2
            0
    )

    Note
    ----
    - When `axis` is None, the global maximum is found, and the index is returned as a linear index relative to the
      array's storage.
    - The maximum values and their locations are returned as separate arrays when an axis is specified.
    """
    if axis is None:
        return wrapper.imax_all(array.arr)

    maximum, location = wrapper.imax(array.arr, axis)
    return Array.from_afarray(maximum), Array.from_afarray(location)


def max(
    array: Array, /, *, axis: int | None = None, keys: Array | None = None, ragged_len: Array | None = None
) -> int | float | complex | Array | tuple[Array, Array]:
    """
    Find the maximum value(s) in an ArrayFire array along a specified dimension, optionally based on unique keys or
    with ragged array dimensions.

    Parameters
    ----------
    array : Array
        The input multi-dimensional ArrayFire array to find the maximum values in.

    axis : int, optional, keyword-only
        The dimension along which to find the maximum values. If None, finds the global maximum.

    keys : Array, optional, keyword-only
        A one-dimensional ArrayFire array containing keys for grouped maximum calculations. Cannot be used
        simultaneously with `ragged_len`.

    ragged_len : Array, optional, keyword-only
        A one-dimensional ArrayFire array containing lengths for ragged maximum calculations. Cannot be used
        simultaneously with `keys`.

    Returns
    -------
    int | float | complex | Array | tuple[Array, Array]
        - If neither `keys` nor `ragged_len` is provided, returns the maximum value across the entire array or along
          the specified `axis`.
        - If `keys` is provided, returns a tuple containing two ArrayFire arrays: the unique keys and their
          corresponding maximum values along the specified `axis`.
        - If `ragged_len` is provided, returns a tuple containing two ArrayFire arrays: the maximum values and their
          indices within each ragged segment along the specified `axis`.

    Raises
    ------
    RuntimeError
        If both `keys` and `ragged_len` are provided, as they cannot be used simultaneously.

    Note
    ----
    - `axis` is ignored when finding the global maximum.
    - `keys` and `ragged_len` cannot be used together.
    - The `ragged_len` array determines the lengths of each segment along `axis` for ragged maximum calculations.
    """
    if keys and ragged_len:
        raise RuntimeError("To process ragged max function, the keys value should be None and vice versa.")

    if keys:
        axis_ = -1 if axis is None else axis
        key, value = wrapper.max_by_key(keys.arr, array.arr, axis_)
        return Array.from_afarray(key), Array.from_afarray(value)

    if ragged_len:
        axis_ = -1 if axis is None else axis
        values, indices = wrapper.max_ragged(array.arr, ragged_len.arr, axis_)
        return Array.from_afarray(values), Array.from_afarray(indices)

    if axis is None:
        return wrapper.max_all(array.arr)

    return Array.from_afarray(wrapper.max(array.arr, axis))


def imin(array: Array, /, *, axis: int | None = None) -> tuple[int | float | complex, int] | tuple[Array, Array]:
    """
    Find the value and location of the minimum value in an ArrayFire array along a specified dimension, or globally
    across the entire array.

    Parameters
    ----------
    array : Array
        The input multi-dimensional ArrayFire array whose minimum value and location are to be determined.

    axis : int, optional, keyword-only
        The dimension along which the minimum value and its location are to be found. If None, the global minimum value
        and its location across the entire array are returned.

    Returns
    -------
    tuple[int | float | complex, int] | tuple[Array, Array]
        If `axis` is None, returns a tuple containing a scalar (the global minimum value of `array`) and an integer
        (the linear index where the global minimum occurs).
        If `axis` is specified, returns a tuple of two ArrayFire arrays: the first array contains the minimum values
        along the specified `axis`, and the second array contains the indices of these minimum values along the same
        `axis`.

    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.randu((3, 3))
    >>> a
    [3 3 1 1]
        0.6010     0.2126     0.2864
        0.0278     0.0655     0.3410
        0.9806     0.5497     0.7509

    >>> af.imin(a)
    (0.027758777141571045, 1)

    >>> af.imin(a, axis=0)
    (
    [1 3 1 1]
        0.0278     0.0655     0.2864 ,

    [1 3 1 1]
            1          1          0 )
    """
    if axis is None:
        return wrapper.imin_all(array.arr)

    minimum, location = wrapper.imin(array.arr, axis)
    return Array.from_afarray(minimum), Array.from_afarray(location)


def min(array: Array, /, *, axis: int | None = None) -> int | float | complex | Array:
    """
    Finds the minimum value in an ArrayFire array, optionally along a specified axis.

    Parameters
    ----------
    array : Array
        The input multi-dimensional ArrayFire array whose minimum value is sought.

    axis : int, optional, keyword-only
        The dimension along which to find the minimum value. If None (the default),
        the minimum value of the entire array is returned.

    Returns
    -------
    int | float | complex | Array
        The minimum value found in the array. If `axis` is specified, an ArrayFire array
        containing the minimum values along that axis is returned. If `axis` is None, a
        single scalar value (int, float, or complex) representing the minimum value of
        the entire array is returned.

    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.randu((3, 3))  # Generate a 3x3 array of random numbers
    >>> a
    [3 3 1 1]
        0.6010     0.2126     0.2864
        0.0278     0.0655     0.3410
        0.9806     0.5497     0.7509

    >>> af.min(a)  # Find the minimum value in the entire array
    0.027758777141571045

    >>> af.min(a, axis=0)  # Find the minimum values along the first axis (column-wise minimum)
    [1 3 1 1]
        0.0278     0.0655     0.2864

    Note
    ----
    - If the array contains NaN values, the operation will return NaN because NaNs propagate through operations as per
      IEEE standards.
    """
    if axis is None:
        return wrapper.min_all(array.arr)

    return Array.from_afarray(wrapper.min(array.arr, axis))


@afarray_as_array
def diff1(array: Array, /, axis: int = 0) -> Array:
    """
    Computes the first-order differences of an ArrayFire array along a specified dimension.

    The first-order difference is calculated as `array[i+1] - array[i]` along the specified axis.

    Parameters
    ----------
    array : Array
        The input ArrayFire array to compute differences on.

    axis : int, optional, default: 0
        The dimension along which the first-order differences are calculated. For a 2D array,
        `axis=0` computes the difference between consecutive rows, while `axis=1` computes the
        difference between consecutive columns.

    Returns
    -------
    Array
        An ArrayFire array of first-order differences. The size of this array along the specified
        axis is one less than the input array, as differences are computed between consecutive
        elements.

    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.Array([1, 2, 4, 7, 11])
    >>> af.diff1(a)
    [4 1 1 1]
        1.0000
        2.0000
        3.0000
        4.0000

    Note
    ----
    - The differences for a 2D array along `axis=0` would be row-wise differences,
    and along `axis=1` would be column-wise differences.
    """
    return cast(Array, wrapper.diff1(array.arr, axis))


@afarray_as_array
def diff2(array: Array, /, axis: int = 0) -> Array:
    """
    Computes the second-order differences of an ArrayFire array along a specified dimension.

    The second-order difference is calculated as `array[i+2] - 2*array[i+1] + array[i]` along the specified axis,
    which is analogous to the second derivative in continuous functions.

    Parameters
    ----------
    array : Array
        The input ArrayFire array to compute second-order differences on.

    axis : int, optional, default: 0
        The dimension along which the second-order differences are calculated. For a 2D array,
        `axis=0` computes the difference between consecutive rows (down each column), while
        `axis=1` computes the difference between consecutive columns (across each row).

    Returns
    -------
    Array
        An ArrayFire array of second-order differences. The size of this array along the specified
        axis is two less than the input array, as the operation effectively reduces the dimension
        size by two.

    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.Array([1, 2, 4, 7, 11])
    >>> af.diff2(a)
    [3 1 1 1]
        1.0000
        1.0000
        1.0000

    Note
    ----
    The operation requires that the array has at least three elements along the specified axis to compute
    the second-order differences. For arrays with fewer than three elements along the axis, the result will
    be an empty array.
    """
    return cast(Array, wrapper.diff2(array.arr, axis))


def gradient(array: Array, /) -> tuple[Array, Array]:
    """
    Computes the horizontal and vertical gradients of a 2D ArrayFire array or a batch of 2D arrays.

    The gradient is a vector that points in the direction of the greatest rate of increase of the function,
    and its magnitude is the slope of the graph in that direction. For images, this operation can highlight
    edges and changes in intensity.

    Parameters
    ----------
    array : Array
        The input ArrayFire array, which can be a 2D array representing a single image, or a multi-dimensional
        array representing a batch of images. For batch processing, the gradient is computed for each image
        in the batch independently.

    Returns
    -------
    tuple[Array, Array]
        A tuple containing two ArrayFire arrays:
        - The first array (`dx`) contains the horizontal gradients of the input array.
        - The second array (`dy`) contains the vertical gradients of the input array.

    Examples
    --------
    >>> import arrayfire as af
    >>> image = af.randu((3, 3))  # Generate a random 3x3 "image"
    >>> image
    [3 3 1 1]
        0.4105     0.3543     0.3636
        0.1583     0.6450     0.4165
        0.3712     0.9675     0.5814

    >>> dx, dy = af.gradient(image)
    >>> dx  # Display the horizontal gradients
    [3 3 1 1]
        -0.2522    0.2907     0.0528
        -0.0196    0.3066     0.1089
        0.2129     0.3225     0.1650

    >>> dy  # Display the vertical gradients
    [3 3 1 1]
        -0.0562    -0.0234    0.0093
        0.4867     0.1291    -0.2286
        0.5962     0.1051    -0.3860

    Note
    ----
    - The gradient operation is particularly useful in the context of image processing for identifying
      edges and textural patterns within images.
    - For higher-dimensional arrays representing batches of images, the gradient operation is applied
      independently to each image in the batch.
    """
    dx, dy = wrapper.gradient(array.arr)
    return Array.from_afarray(dx), Array.from_afarray(dy)


@afarray_as_array
def set_intersect(x: Array, y: Array, /, *, is_unique: bool = False) -> Array:
    """
    Calculates the intersection of two ArrayFire arrays, returning elements common to both arrays.

    Parameters
    ----------
    x : Array
        The first input 1D ArrayFire array.
    y : Array
        The second input 1D ArrayFire array.
    is_unique : bool, optional, keyword-only, default: False
        Specifies whether both input arrays contain unique elements. If set to True,
        the function assumes that the arrays do not have repeated elements, which
        can optimize the intersection operation.

    Returns
    -------
    Array
        An ArrayFire array containing the intersection of `x` and `y`. The returned array
        includes elements that are common to both input arrays. If `is_unique` is True,
        the function assumes no duplicates within each input array, potentially
        enhancing performance.

    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.Array([1, 2, 3, 4, 5])
    >>> b = af.Array([4, 5, 6, 7, 8])
    >>> af.set_intersect(a, b)
    [2 1 1 1]
        4.0000
        5.0000

    Note
    ----
    - Both `x` and `y` must be 1D arrays.
    - The `is_unique` parameter can be used to optimize the intersection calculation when both input arrays are known
      to contain unique elements.
    """
    return cast(Array, wrapper.set_intersect(x.arr, y.arr, is_unique))


@afarray_as_array
def set_union(x: Array, y: Array, /, *, is_unique: bool = False) -> Array:
    """
    Computes the union of two 1D ArrayFire arrays, effectively combining the elements from both arrays and removing
    duplicates.

    Parameters
    ----------
    x, y : Array
        The input 1D ArrayFire arrays whose union is to be computed. These arrays can contain any numerical type.

    is_unique : bool, optional, keyword-only, default: False
        A flag that indicates whether both input arrays are guaranteed to contain unique elements. Setting this to True
        can optimize the computation but should only be used if each element in both arrays is indeed unique.

    Returns
    -------
    Array
        An ArrayFire array containing the unique elements from both `x` and `y`.

    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.Array([1, 2, 3, 4, 5])
    >>> b = af.Array([4, 5, 6, 7, 8])
    >>> af.set_union(a, b)
    [8 1 1 1]
        1.0000
        2.0000
        3.0000
        4.0000
        5.0000
        6.0000
        7.0000
        8.0000

    Note
    ----
    The operation is performed on 1D arrays. For inputs that are not 1D, consider reshaping or flattening them before
    performing the operation to ensure correct results. The `is_unique` flag should be used with caution; incorrect
    usage (i.e., setting it to True when arrays are not composed of unique elements) may lead to unexpected results.
    """
    return cast(Array, wrapper.set_union(x.arr, y.arr, is_unique))


@afarray_as_array
def set_unique(array: Array, /, *, is_sorted: bool = False) -> Array:
    """
    Extracts unique elements from a 1D ArrayFire array.

    This function returns a new array containing only the unique elements of the input array. It can operate more
    efficiently if the input array is known to be sorted.

    Parameters
    ----------
    array : Array
        The input 1D ArrayFire array from which unique elements are to be extracted.

    is_sorted : bool, optional, keyword-only, default: False
        Indicates whether the input array is already sorted. If True, the function can skip the sorting step,
        potentially improving performance. However, setting this to True for an unsorted array will lead to incorrect
        results.

    Returns
    -------
    Array
        An ArrayFire array containing the unique elements extracted from the input array.

    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.Array([1, 2, 2, 3, 4, 4, 5, 5, 5])
    >>> af.set_unique(a)
    [5 1 1 1]
        1.0000
        2.0000
        3.0000
        4.0000
        5.0000

    >>> sorted_a = af.sort(a)  # Assuming 'a' is not sorted
    >>> af.set_unique(sorted_a, is_sorted=True)
    [5 1 1 1]
        1.0000
        2.0000
        3.0000
        4.0000
        5.0000

    Note
    ----
    The input array must be 1D. If you have a multi-dimensional array, consider reshaping or flattening it before
    using this function. Ensure the `is_sorted` flag accurately reflects the state of the input array to avoid
    incorrect results.
    """
    return cast(Array, wrapper.set_unique(array.arr, is_sorted))


def sort(
    array: Array,
    /,
    axis: int = 0,
    is_ascending: bool = True,
    *,
    keys: Array | None = None,
    is_index_array: bool = False,
) -> Array | tuple[Array, Array]:
    """
    Sorts the elements of an ArrayFire array along a specified dimension. Optionally, sorting can be performed based
    on keys, or sorted indices can be returned.

    Parameters
    ----------
    array : Array
        The input multi-dimensional ArrayFire array to be sorted.

    axis : int, default: 0
        The dimension along which the sorting is to be performed.

    is_ascending : bool, default: True
        Determines the direction of the sort. If True, the sorting is done in ascending order; otherwise, in descending
        order.

    keys : Array, optional
        An optional ArrayFire array containing keys based on which the sorting should be performed. If provided, the
        elements in `array` are sorted according to the order determined by these keys.

    is_index_array : bool, default: False
        If True, the function returns a tuple of arrays - the sorted array and an array of indices that maps the sorted
        array back to the original array.

    Returns
    -------
    Array | tuple[Array, Array]
        If neither `keys` nor `is_index_array` is provided, returns the sorted array.
        If `keys` is provided, returns a tuple (sorted_keys, sorted_values) where `sorted_keys` is the keys sorted and
        `sorted_values` are the elements of `array` sorted according to `sorted_keys`.
        If `is_index_array` is true, returns a tuple (sorted_array, indices) where `sorted_array` is the sorted array
        and `indices` maps the sorted array back to the original array.

    Raises
    ------
    RuntimeError
        If both `keys` and `is_index_array` are provided.

    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.randu(5)  # Create a random 1D array
    >>> a
    [5 1 1 1]
        0.6010
        0.0278
        0.9806
        0.2126
        0.0655

    >>> af.sort(a)  # Sort the array in ascending order
    [5 1 1 1]
        0.0278
        0.0655
        0.2126
        0.6010
        0.9806

    >>> keys = af.Array([3, 2, 1, 5, 4])
    >>> values = af.Array([10, 20, 30, 40, 50])
    >>> sorted_keys, sorted_values = af.sort(values, keys=keys)
    >>> sorted_keys
    [5 1 1 1]
        1.0000
        2.0000
        3.0000
        4.0000
        5.0000

    >>> sorted_values
    [5 1 1 1]
        30.0000
        20.0000
        10.0000
        50.0000
        40.0000

    >>> sorted_array, indices = af.sort(a, is_index_array=True)
    >>> sorted_array
    [5 1 1 1]
        0.0278
        0.0655
        0.2126
        0.6010
        0.9806

    >>> indices
    [5 1 1 1]
        1
        4
        3
        0
        2

    Note
    ----
    - The sorting based on keys (`sort_by_key`) or returning sorted indices (`sort_index`) cannot be performed
      simultaneously. Select only one option per function call.
    """
    if keys and is_index_array:
        raise RuntimeError("Could not process sorting by keys when `is_index_array` is True. Select only one option.")

    if keys:
        key, value = wrapper.sort_by_key(keys.arr, array.arr, axis, is_ascending)
        return Array.from_afarray(key), Array.from_afarray(value)

    if is_index_array:
        values, indices = wrapper.sort_index(array.arr, axis, is_ascending)
        return Array.from_afarray(values), Array.from_afarray(indices)

    return Array.from_afarray(wrapper.sort(array.arr, axis, is_ascending))
