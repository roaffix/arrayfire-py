from __future__ import annotations

import arrayfire as af

from ._array_object import Array
from ._constants import Device, NestedSequence, SupportsBufferProtocol
from ._dtypes import all_dtypes, float32, int32, float64


def _check_valid_dtype(dtype: af.Dtype | None) -> None:
    # Note: Only spelling dtypes as the dtype objects is supported.

    # We use this instead of "dtype in _all_dtypes" because the dtype objects
    # define equality with the sorts of things we want to disallow.
    if dtype not in (None,) + all_dtypes:
        raise ValueError("dtype must be one of the supported dtypes")


def asarray(
    obj: Array | bool | int | float | complex | NestedSequence | SupportsBufferProtocol,
    /,
    *,
    dtype: af.Dtype | None = None,
    device: Device | None = None,
    copy: bool | None = None,
) -> Array:
    _check_valid_dtype(dtype)

    # if device not in supported_devices:
    #     raise ValueError(f"Unsupported device {device!r}")

    if dtype is None and isinstance(obj, int) and (obj > 2**64 or obj < -(2**63)):
        raise OverflowError("Integer out of bounds for array dtypes")

    if device == Device.CPU or device is None:
        to_device = False
    elif device == Device.GPU:
        to_device = True
    else:
        raise ValueError(f"Unsupported device {device!r}")

    # if isinstance(obj, int | float):
    #     afarray = af.Array([obj], dtype=dtype, shape=(1,), to_device=to_device)
    #     return Array._new(afarray)

    afarray = af.Array(obj, dtype=dtype, to_device=to_device)
    return Array._new(afarray)


def arange(
    start: int | float,
    /,
    stop: int | float | None = None,
    step: int | float = 1,
    *,
    dtype: af.Dtype | None = None,
    device: Device | None = None,
) -> Array:
    """
    Converts the input to an array.

    This function creates an array from an object, or wraps an existing array. If `obj` is already an array of the
    desired data type and `copy` is False, no copy will be performed, and `obj` itself is returned. Otherwise, a new
    array is created, possibly copying the data from `obj`, depending on the `copy` parameter.

    Parameters
    ----------
    obj : Array | bool | int | float | complex | NestedSequence | SupportsBufferProtocol
        The input object to convert to an array. This can be a scalar, nested sequence (like lists of lists), or any
        object exposing the buffer interface, and of course, an existing array.
    dtype : af.Dtype | None, optional
        The desired data type for the array. If `None`, the data type is inferred from the input object. Explicitly
        specifying the data type allows the creation of an array with the intended data type.
    device : Device | None, optional
        The device on which to create the array. If `None`, the array is created on the default device. This parameter
        can be used to specify the computational device (CPU, GPU, etc.) for the array's storage.
    copy : bool | None, optional
        If True, a new array is always created. If False, a new array is only created if necessary (i.e., if `obj` is
        not already an array of the specified data type). If `None`, the behavior defaults to True (i.e., always copy).

    Returns
    -------
    Array
        An array representation of `obj`. If `obj` is already an array, it may be returned directly based on the `copy`
        parameter and the data type match.

    Examples
    --------
    >>> asarray([1, 2, 3])
    Array([1, 2, 3])

    >>> asarray([1, 2, 3], dtype=float32)
    Array([1.0, 2.0, 3.0])

    >>> a = array([1, 2, 3], device='cpu')
    >>> asarray(a, device='gpu')
    # A new array on GPU, containing [1, 2, 3]

    >>> asarray(1.0, dtype=int32)
    Array(1)

    Notes
    -----
    - The `asarray` function is a convenient way to create arrays or convert other objects to arrays with specified
      data type and storage device criteria. It is particularly useful for ensuring that numerical data is in the form
      of an array for computational tasks.
    - The `copy` parameter offers control over whether data is duplicated when creating the array, which can be
      important for managing memory use and computational efficiency.
    """
    return NotImplemented


def empty(
    shape: int | tuple[int, ...],
    *,
    dtype: af.Dtype | None = None,
    device: Device | None = None,
) -> Array:
    """
    Returns an uninitialized array of a specified shape. The contents of the array are uninitialized,
    meaning they will contain arbitrary data. This function is often used for efficiency in scenarios
    where the entire array will be populated with data immediately after its creation, eliminating the
    need for initial zeroing of memory.

    Parameters
    ----------
    shape : int or tuple of ints
        The desired shape of the output array. An integer creates a one-dimensional array, while a tuple
        creates an n-dimensional array with dimensions specified by the tuple.
    dtype : af.Dtype | None, optional
        The desired data type for the array. If not specified, the array's data type will be the default
        floating-point data type determined by the array library. Specifying a data type is useful when
        creating arrays intended for specific types of data.
    device : Device | None, optional
        The device on which to create the array. If not specified, the array will be created on the default
        device. This parameter can be used to specify the computational device (e.g., CPU or GPU) for the
        array's storage.

    Returns
    -------
    Array
        An array with the specified shape and data type, containing uninitialized data.

    Examples
    --------
    >>> empty(3)
    Array([..., ..., ...])  # Uninitialized data

    >>> empty((2, 2), dtype=float64)
    Array([[..., ...],
           [..., ...]])  # Uninitialized data of float64 type

    >>> empty((3, 4), device='gpu')
    Array([[..., ..., ..., ...],
           [..., ..., ..., ...],
           [..., ..., ..., ...]])  # Uninitialized data on GPU

    Notes
    -----
    - The contents of the returned array are uninitialized and accessing them without setting their values first
      can result in undefined behavior.
    - This function provides an efficient way to allocate memory for large arrays when it is known that all
      elements will be explicitly assigned before any computation is performed on the array.

    The use of `dtype` and `device` allows for control over the type and location of the allocated memory,
    optimizing performance and compatibility with specific hardware or computational tasks.
    """
    _check_valid_dtype(dtype)

    if isinstance(shape, int):
        shape = (shape,)

    if dtype is None:
        dtype = float32

    array = af.constant(0, shape, dtype)
    # TODO
    # device -> get_device -> set_device
    return Array._new(array)


def empty_like(x: Array, /, *, dtype: af.Dtype | None = None, device: Device | None = None) -> Array:
    """
    Returns an uninitialized array with the same shape and, optionally, the same data type as the input array `x`.
    The contents of the new array are uninitialized, meaning they may contain arbitrary data (including potentially
    sensitive data left over from other processes). This function is typically used for efficiency in scenarios where
    the user intends to populate the array with data immediately after its creation.

    Parameters
    ----------
    x : Array
        The input array from which to derive the shape of the output array. The data type of the output array is also
        inferred from `x` unless `dtype` is explicitly specified.
    dtype : af.Dtype | None, optional
        The desired data type for the new array. If `None` (the default), the data type of the input array `x` is used.
        This parameter allows the user to specify a different data type for the new array.
    device : Device | None, optional
        The device on which to create the new array. If `None`, the new array is created on the same device as the
        input array `x`. This allows for the creation of arrays on specific devices (e.g., on a GPU).

    Returns
    -------
    Array
        An array having the same shape as `x`, with uninitialized data. The data type of the array is determined by the
        `dtype` parameter if specified, otherwise by the data type of `x`.

    Examples
    --------
    >>> x = array([1, 2, 3])
    >>> y = empty_like(x)
    # y is an uninitialized array with the same shape and data type as x

    >>> x = array([[1.0, 2.0], [3.0, 4.0]])
    >>> y = empty_like(x, dtype=int32)
    # y is an uninitialized array with the same shape as x but with an int32 data type

    >>> x = array([1, 2, 3], device='cpu')
    >>> y = empty_like(x, device='gpu')
    # y is an uninitialized array with the same shape as x, created on a GPU

    Notes
    -----
    - The contents of the returned array are uninitialized. Accessing the data without first initializing it
      may result in unpredictable behavior.
    - The `dtype` and `device` parameters offer flexibility in the creation of the new array, allowing the user
      to specify the data type and the computational device for the array.
    """
    _check_valid_dtype(dtype)

    if dtype is None:
        dtype = x.dtype

    if device is None:
        device = x.device

    return empty(x.shape, dtype=dtype, device=device)


def eye(
    n_rows: int,
    n_cols: int | None = None,
    /,
    *,
    k: int = 0,
    dtype: af.Dtype | None = None,
    device: Device | None = None,
) -> Array:
    """
    Returns a two-dimensional array with ones on the kth diagonal and zeros elsewhere.

    This function is useful for creating identity matrices or variations thereof.

    Parameters
    ----------
    n_rows : int
        The number of rows in the output array.
    n_cols : int, optional
        The number of columns in the output array. If None (the default), the output array will be square,
        with the number of columns equal to `n_rows`.
    k : int, optional, default: 0
        The index of the diagonal to be filled with ones. `k=0` refers to the main diagonal. A positive `k`
        refers to an upper diagonal, and a negative `k` to a lower diagonal.
    dtype : af.Dtype | None, optional
        The desired data type of the output array. If None, the default floating-point data type is used.
        Specifying a data type allows for the creation of identity matrices with elements of that type.
    device : Device | None, optional
        The device on which to create the array. If None, the array is created on the default device. This
        can be useful for ensuring the array is created in the appropriate memory space (e.g., on a GPU).

    Returns
    -------
    Array
        A 2D array with ones on the specified diagonal and zeros elsewhere. The shape of the array is
        (n_rows, n_cols), with `n_cols` defaulting to `n_rows` if not specified.

    Examples
    --------
    >>> eye(3)
    Array([[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]])

    >>> eye(3, 4, k=1)
    Array([[0, 1, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1]])

    >>> eye(4, 3, k=-1, dtype=int32)
    Array([[0, 0, 0],
           [1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]])

    >>> eye(3, device='gpu')
    Array([[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]])

    Notes
    -----
    - The `dtype` and `device` parameters allow for detailed control over the type and location of the
      resulting array, which can be important for performance and compatibility with other arrays or
      operations in a program.
    """
    _check_valid_dtype(dtype)

    if n_cols is None:
        n_cols = n_rows

    if dtype is None:
        dtype = float32

    # TODO
    # device

    if n_rows <= abs(k):
        return Array._new(af.constant(0, (n_rows, n_cols), dtype))

    # Create an identity matrix as the base
    array = af.identity((n_rows, n_cols), dtype=dtype)

    if k == 0:
        # No shift needed, directly return the identity matrix
        return Array._new(array)

    # Prepare a zeros array for padding
    zeros_padding_vertical = af.constant(0, (abs(k), n_cols), dtype=dtype)
    zeros_padding_horizontal = af.constant(0, (n_rows, abs(k)), dtype=dtype)

    if k > 0:
        # Shift the diagonal upwards by removing the last k columns and padding with zeros on the left
        shifted_array = af.join(1, zeros_padding_horizontal, array[:, :-k])
    else:
        # Shift the diagonal downwards by removing the last k rows and padding with zeros on top
        shifted_array = af.join(0, zeros_padding_vertical, array[: -abs(k), :])

    return Array._new(shifted_array)


def full(
    shape: int | tuple[int, ...],
    fill_value: int | float,
    *,
    dtype: af.Dtype | None = None,
    device: Device | None = None,
) -> Array:
    """
    Returns a new array of a specified shape, filled with `fill_value`.

    Parameters
    ----------
    shape : int or tuple of ints
        The shape of the new array. If an integer is provided, the result will be a 1-D array of that length.
        A tuple can be used to specify the shape of a multi-dimensional array.
    fill_value : int or float
        The value used to fill the array. Can be an integer or floating-point number.
    dtype : af.Dtype | None, optional
        The desired data type for the new array. If not specified, the data type is inferred from the `fill_value`.
        If `fill_value` is an integer, the default integer data type is used. For a floating-point number, the
        default floating-point data type is chosen. The behavior is unspecified if `fill_value`'s precision exceeds
        the capabilities of the chosen data type.
    device : Device | None, optional
        The device on which to create the array. If not specified, the array is created on the default device.
        This can be used to specify creation of the array on a particular device, such as a GPU.

    Returns
    -------
    Array
        An array of shape `shape`, where each element is `fill_value`.

    Notes
    -----
    - The behavior is unspecified if the `fill_value` exceeds the precision of the chosen data type.
    - The `dtype` argument allows for specifying the desired data type explicitly. If `dtype` is None,
      the data type is inferred from the `fill_value`.

    Examples
    --------
    >>> full(3, 5)
    Array([5, 5, 5])

    >>> full((2, 2), 0.5)
    Array([[0.5, 0.5],
           [0.5, 0.5]])

    >>> full((2, 3), 1, dtype=float64)
    Array([[1.0, 1.0, 1.0],
           [1.0, 1.0, 1.0]])

    >>> full(4, True, dtype=bool)
    Array([True, True, True, True])

    This function is useful for creating arrays with a constant value throughout. The `device` parameter
    allows for control over where the computation is performed, which can be particularly important for
    performance in computational environments with multiple devices.
    """
    _check_valid_dtype(dtype)

    if isinstance(shape, int):
        shape = (shape,)

    # Default dtype handling based on 'fill_value' type if 'dtype' not provided
    if dtype is None:
        dtype = float32 if isinstance(fill_value, float) else int32

    # TODO
    # Device

    return Array._new(af.constant(fill_value, shape, dtype=dtype))


def full_like(
    x: Array,
    /,
    fill_value: int | float,
    *,
    dtype: af.Dtype | None = None,
    device: Device | None = None,
) -> Array:
    """
    Returns a new array with the same shape as the input array `x`, filled with `fill_value`.

    Parameters
    ----------
    x : Array
        The input array whose shape and data type (if `dtype` is None) are used to create the new array.
    fill_value : int | float
        The scalar value to fill the new array with.
    dtype : af.Dtype | None, optional
        The desired data type for the new array. If `None`, the data type of `x` is used. This allows
        overriding the data type of the input array while keeping its shape.
    device : Device | None, optional
        The device on which to place the created array. If `None`, the array is created on the same device as `x`.

    Returns
    -------
    Array
        A new array having the same shape as `x`, where every element is set to `fill_value`.

    Notes
    -----
    - If `fill_value` is of a different data type than `x` and `dtype` is `None`, the data type of the new array
      will be inferred in a way that can represent `fill_value`.
    - If `dtype` is specified, it determines the data type of the new array, irrespective of the data type of `x`.

    Examples
    --------
    >>> x = array([[1, 2], [3, 4]])
    >>> full_like(x, 5)
    Array([[5, 5],
           [5, 5]])

    >>> full_like(x, 0.5, dtype=float32)
    Array([[0.5, 0.5],
           [0.5, 0.5]])

    >>> full_like(x, 7, device='gpu')
    Array([[7, 7],
           [7, 7]])

    In the above examples, the `full_like` function creates new arrays with the same shape as `x`,
    but filled with the specified `fill_value`, and optionally with the specified data type and device.
    """
    _check_valid_dtype(dtype)

    if dtype is None:
        dtype = x.dtype

    if device is None:
        device = x.device

    return full(x.shape, fill_value, dtype=dtype, device=device)


def linspace(
    start: int | float,
    stop: int | float,
    /,
    num: int,
    *,
    dtype: af.Dtype | None = None,
    device: Device | None = None,
    endpoint: bool = True,
) -> Array:
    """
    Returns `num` evenly spaced samples, calculated over the interval [`start`, `stop`].

    The endpoint of the interval can optionally be excluded.

    Parameters
    ----------
    start : int | float
        The starting value of the sequence.
    stop : int | float
        The end value of the sequence, unless `endpoint` is set to False. In that case, the sequence
        consists of all but the last of `num + 1` evenly spaced samples, so that `stop` is excluded.
        Note that the step size changes when `endpoint` is False.
    num : int
        Number of samples to generate. Must be non-negative.
    dtype : af.Dtype | None, optional, default: None
        The data type of the output array. If `None`, the data type is inferred from `start` and `stop`.
        The inferred data type will never be an integer data type because `linspace` always generates
        floating point values.
    device : Device | None, optional, default: None
        The device on which to place the created array. If `None`, the device is inferred from the current
        device context.
    endpoint : bool, optional, default: True
        If True, `stop` is the last sample. Otherwise, it is not included.

    Returns
    -------
    Array
        An array of `num` equally spaced samples in the closed interval [`start`, `stop`] or the
        half-open interval [`start`, `stop`) (depending on whether `endpoint` is True or False).

    See Also
    --------
    arange : Similar to `linspace`, but uses a step size (instead of the number of samples).

    Notes
    -----
    - The output array is always a floating point array, even if `start`, `stop`, and `dtype` are all integers.
    - If `num` is 1, the output array only contains the `start` value.

    Examples
    --------
    >>> linspace(2.0, 3.0, num=5)
    Array([2. , 2.25, 2.5 , 2.75, 3. ])

    >>> linspace(2.0, 3.0, num=5, endpoint=False)
    Array([2. , 2.2, 2.4, 2.6, 2.8])
    """
    # BUG
    # # Default dtype handling based on 'start' and 'stop' types if 'dtype' not provided
    # if dtype is None:
    #     dtype = float32

    # # TODO
    # # Device

    # # Generate the linearly spaced array
    # if endpoint:
    #     step = (stop - start) / (num - 1) if num > 1 else 0
    # else:
    #     step = (stop - start) / num

    # array = af.seq(start, stop, step, dtype=dtype)
    # result_array = array if array.size == num else af.moddims(array, (num,))

    # return Array._new(result_array)
    return NotImplemented


def meshgrid(*arrays: Array, indexing: str = "xy") -> list[Array]:
    """
    Returns coordinate matrices from coordinate vectors.

    Make N-D coordinate arrays for vectorized evaluations of N-D scalar/vector fields over N-D grids,
    given one-dimensional coordinate arrays x1, x2,..., xn.

    Parameters
    ----------
    *arrays : Array
        One-dimensional arrays representing the coordinates of a grid.
    indexing : {'xy', 'ij'}, optional, default: 'xy'
        Cartesian ('xy', default) or matrix ('ij') indexing of output.
        In Cartesian indexing, the first dimension corresponds to the x-coordinate, and the second to the y-coordinate.
        In matrix indexing, the first dimension corresponds to the row index, and the second to the column index.

    Returns
    -------
    list[Array]
        List of N arrays, where `N` is the number of provided one-dimensional input arrays.
        Each returned array must have the same shape. If `indexing` is 'xy', the last dimension
        of the arrays corresponds to changes in x1, and the second-to-last corresponds to changes in x2,
        and so forth. If `indexing` is 'ij', then the first dimension of the arrays corresponds to changes
        in x1, the second dimension to changes in x2, and so forth.

    See Also
    --------
    arange, linspace

    Notes
    -----
    This function supports both indexing conventions through the `indexing` keyword argument.
    Giving the string 'xy' returns the grid with Cartesian indexing, while 'ij' returns the grid
    with matrix indexing. The difference is that 'xy' indexing returns arrays where the last
    dimension represents points that vary along what would conventionally be considered the x-axis,
    and 'ij' indexing returns arrays where the first dimension represents points that vary along
    what would conventionally be considered the row index of a matrix or 2D array.

    Examples
    --------
    >>> x = array([1, 2, 3])
    >>> y = array([4, 5, 6, 7])
    >>> xv, yv = meshgrid(x, y)
    >>> xv
    Array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]])
    >>> yv
    Array([[4, 4, 4],
           [5, 5, 5],
           [6, 6, 6],
           [7, 7, 7]])

    With `indexing='ij'`, the shape is transposed.
    >>> xv, yv = meshgrid(x, y, indexing='ij')
    >>> xv
    Array([[1, 1, 1, 1],
           [2, 2, 2, 2],
           [3, 3, 3, 3]])
    >>> yv
    Array([[4, 5, 6, 7],
           [4, 5, 6, 7],
           [4, 5, 6, 7]])
    """
    # BUG
    # arrays_af = [arr._array for arr in arrays]  # Convert custom Array to ArrayFire array for processing
    # dims = [arr.size for arr in arrays_af]  # Get the number of elements in each array

    # result = []
    # for i, arr in enumerate(arrays_af):
    #     import ipdb; ipdb.set_trace()
    #     shape = [1] * len(arrays)
    #     shape[i] = dims[i]
    #     tiled_arr = af.tile(arr, tuple(shape))

    #     # Expand each array to have the correct shape
    #     for j, dim in enumerate(dims):
    #         if j != i:
    #             import ipdb; ipdb.set_trace()
    #             tiled_arr = af.moddims(tiled_arr, tuple(shape))
    #             shape[j] = dim
    #         else:
    #             shape[j] = 1

    #     if indexing == "xy" and len(arrays) > 1 and i == 0:
    #         # Swap the first two dimensions for the x array in 'xy' indexing
    #         tiled_arr = af.moddims(tiled_arr, (dims[1], dims[0], *dims[2:]))
    #     elif indexing == "xy" and len(arrays) > 1 and i == 1:
    #         # Ensure the y array is correctly shaped in 'xy' indexing
    #         tiled_arr = af.reorder(tiled_arr, shape=(1, 0, *range(2, len(dims))))

    #     result.append(Array._new(tiled_arr))

    # if indexing == "ij":
    #     # No need to modify the order for 'ij' indexing
    #     pass

    # return result
    return NotImplemented


def ones(
    shape: int | tuple[int, ...],
    *,
    dtype: af.Dtype | None = None,
    device: Device | None = None,
) -> Array:
    """
    Returns a new array of given shape and type, filled with ones.

    Parameters
    ----------
    shape : int or tuple of ints
        The shape of the new array, e.g., (2, 3) or 2. If the shape is an integer, the output will be
        a one-dimensional array of that length. If the shape is a tuple, the output will have the
        specified shape.
    dtype : af.Dtype | None, optional
        The desired data type for the array, e.g., float32. If dtype is None, the default data type
        (float64) is used. Note: The availability of data types can vary by implementation.
    device : Device | None, optional
        The device on which to place the created array. If None, the array is created on the default
        device. This can be used to create arrays on, for example, a GPU.

    Returns
    -------
    Array
        An array of shape `shape` and data type `dtype`, where all elements are ones.

    Notes
    -----
    - While the `dtype` parameter is optional, specifying it can be important for ensuring that
      the array has the desired type, especially for operations that are sensitive to the data type.
    - The `device` parameter allows control over where the array is stored, which can be important
      for computational efficiency, especially in environments with multiple devices like CPUs and GPUs.

    Examples
    --------
    >>> ones(5)
    Array([1, 1, 1, 1, 1])

    >>> ones((2, 3), dtype=float32)
    Array([[1.0, 1.0, 1.0],
           [1.0, 1.0, 1.0]])

    >>> ones((2, 2), device='gpu')
    Array([[1, 1],
           [1, 1]])

    The behavior of the `dtype` and `device` parameters may vary depending on the implementation
    and the available hardware.
    """
    _check_valid_dtype(dtype)

    if isinstance(shape, int):
        shape = (shape,)

    if dtype is None:
        dtype = float32

    return Array._new(af.constant(1, shape, dtype))


def ones_like(x: Array, /, *, dtype: af.Dtype | None = None, device: Device | None = None) -> Array:
    """
    Returns a new array with the same shape and type as a given array, filled with ones.

    Parameters
    ----------
    x : Array
        The shape and data type of `x` are used to create the new array. The contents of `x` are not used.
    dtype : af.Dtype | None, optional
        The desired data type for the new array. If None, the data type of `x` is used. This allows overriding
        the data type of the input array while keeping its shape.
    device : Device | None, optional
        The device on which to place the created array. If None, the array is created on the same device as `x`.
        This parameter allows the new array to be placed on a specified device, which can be different from the
        device of `x`.

    Returns
    -------
    Array
        An array of the same shape as `x`, with all elements set to one (1). The data type of the array is determined
        by the `dtype` parameter if specified, otherwise by the data type of `x`.

    Notes
    -----
    - The `ones_like` function is useful for creating arrays that are initialized to 1 and have the same shape
      and data type (or a specified data type) as another array.
    - Specifying the `dtype` and `device` parameters can be useful for creating an array with a specific data
      type or for placing the array on a specific computational device, which can be important for performance
      and memory usage considerations.

    Examples
    --------
    >>> x = array([[0, 1], [2, 3]])
    >>> ones_like(x)
    Array([[1, 1],
           [1, 1]])

    >>> ones_like(x, dtype=float32)
    Array([[1.0, 1.0],
           [1.0, 1.0]])

    >>> ones_like(x, device='gpu')
    Array([[1, 1],
           [1, 1]])

    The output reflects the shape and, optionally, the specified data type or device of the input array.
    """
    _check_valid_dtype(dtype)

    if dtype is None:
        dtype = x.dtype

    if device is None:
        device = x.device

    return ones(x.shape, dtype=dtype, device=device)


def tril(x: Array, /, *, k: int = 0) -> Array:
    """
    Returns the lower triangular part of the array `x`, with elements above the kth diagonal zeroed.

    The kth diagonal refers to the diagonal that runs from the top-left corner of the matrix to the bottom-right
    corner. For k = 0, the diagonal is the main diagonal. A positive value of k includes elements above the main
    diagonal, and a negative value excludes elements below the main diagonal.

    Parameters
    ----------
    x : Array
        The input array from which the lower triangular part is extracted. The array must be at least two-dimensional.
    k : int, optional, default: 0
        Diagonal above which to zero elements. `k=0` is the main diagonal, `k>0` is above the main diagonal, and `k<0`
        is below the main diagonal.

    Returns
    -------
    Array
        An array with the same shape and data type as `x`, where elements above the kth diagonal are zeroed, and the
        lower triangular part is retained.

    Examples
    --------
    >>> x = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> tril(x)
    Array([[1, 0, 0],
           [4, 5, 0],
           [7, 8, 9]])

    >>> tril(x, k=1)
    Array([[1, 2, 0],
           [4, 5, 6],
           [7, 8, 9]])

    >>> tril(x, k=-1)
    Array([[0, 0, 0],
           [4, 0, 0],
           [7, 8, 0]])

    Notes
    -----
    - The function is used to extract the lower triangular part of a matrix, which can be useful in various
      numerical and algebraic operations.
    - The behavior of `tril` on arrays with more than two dimensions is not specified; it primarily operates
      on two-dimensional matrices.

    The `k` parameter allows for flexibility in defining which part of the array is considered the lower
    triangular portion, offering a straightforward way to manipulate matrices for lower-triangular matrix operations.
    """
    dtype = x.dtype

    _check_valid_dtype(dtype)

    n_rows, n_cols = x.shape

    array = x._array
    row_indices = af.tile(af.iota((1, n_rows), tile_shape=(n_cols, 1)), (1, 1))
    col_indices = af.tile(af.iota((n_cols, 1), tile_shape=(1, n_rows)), (1, 1))

    mask = row_indices <= (col_indices + k)

    return Array._new(array * af.cast(mask, dtype))


def triu(x: Array, /, *, k: int = 0) -> Array:
    """
    Returns the upper triangular part of the array `x`, with elements below the kth diagonal zeroed.

    The kth diagonal refers to the diagonal that runs from the top-left corner of the matrix to the bottom-right
    corner. For k = 0, the diagonal is the main diagonal. A positive value of k includes elements above and on the main
    diagonal, and a negative value includes elements further below the main diagonal.

    Parameters
    ----------
    x : Array
        The input array from which the upper triangular part is extracted. The array must be at least two-dimensional.
    k : int, optional, default: 0
        Diagonal below which to zero elements. `k=0` zeroes elements below the main diagonal, `k>0` zeroes elements
        further above the main diagonal, and `k<0` zeroes elements starting from diagonals below the main diagonal.

    Returns
    -------
    Array
        An array with the same shape and data type as `x`, where elements below the kth diagonal are zeroed, and the
        upper triangular part is retained.

    Examples
    --------
    >>> x = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> triu(x)
    Array([[1, 2, 3],
           [0, 5, 6],
           [0, 0, 9]])

    >>> triu(x, k=1)
    Array([[0, 2, 3],
           [0, 0, 6],
           [0, 0, 0]])

    >>> triu(x, k=-1)
    Array([[1, 2, 3],
           [4, 5, 6],
           [0, 8, 9]])

    Notes
    -----
    - The function is used to extract the upper triangular part of a matrix, which can be useful in various
      numerical and algebraic operations, such as solving linear equations or matrix factorization.
    - The behavior of `triu` on arrays with more than two dimensions is not specified; it primarily operates
      on two-dimensional matrices.

    The `k` parameter allows adjusting the definition of the "upper triangular" part, enabling more flexible
    matrix manipulations based on the specific requirements of the operation or analysis being performed.
    """
    dtype = x.dtype

    _check_valid_dtype(dtype)

    n_rows, n_cols = x.shape

    array = x._array
    row_indices = af.tile(af.iota((1, n_rows), tile_shape=(n_cols, 1)), (1, 1))
    col_indices = af.tile(af.iota((n_cols, 1), tile_shape=(1, n_rows)), (1, 1))

    mask = col_indices <= (row_indices - k)

    return Array._new(array * af.cast(mask, dtype))


def zeros(
    shape: int | tuple[int, ...],
    *,
    dtype: af.Dtype | None = None,
    device: Device | None = None,
) -> Array:
    """
    Returns a new array of given shape and type, filled with zeros.

    This function is useful for creating arrays that serve as initial placeholders in various numerical
    computations, providing a base for algorithms that require an initial array filled with zeros.

    Parameters
    ----------
    shape : int or tuple of ints
        The shape of the new array. If an integer is provided, the result is a one-dimensional array of that length.
        A tuple specifies the dimensions of the array for more than one dimension.
    dtype : af.Dtype | None, optional
        The desired data type for the array. If not specified, the default data type (typically float64) is used.
        Specifying a data type can be important for the precise control of numerical computations.
    device : Device | None, optional
        The device on which to create the array. If not specified, the array is created on the default device.
        This parameter can be used to control where the computation is performed, such as on a CPU or GPU,
        which might be important for performance reasons or when working within specific computational environments.

    Returns
    -------
    Array
        An array with the specified shape and data type, where all elements are zeros.

    Examples
    --------
    >>> zeros(5)
    Array([0, 0, 0, 0, 0])

    >>> zeros((2, 3), dtype=int32)
    Array([[0, 0, 0],
           [0, 0, 0]])

    >>> zeros((2, 2), device='gpu')
    Array([[0, 0],
           [0, 0]])
    # Note: The actual device specification will depend on the array library's implementation and the available
    hardware.

    Notes
    -----
    - The `dtype` and `device` parameters offer the flexibility to create the zero-filled array with specific
      characteristics tailored to the needs of different computational tasks or hardware requirements.

    The use of zeros in computational algorithms is widespread, serving as initial states, placeholders, or
    default values in various numerical and data processing operations.
    """
    _check_valid_dtype(dtype)

    if isinstance(shape, int):
        shape = (shape,)

    if dtype is None:
        dtype = float32

    return Array._new(af.constant(0, shape, dtype))


def zeros_like(x: Array, /, *, dtype: af.Dtype | None = None, device: Device | None = None) -> Array:
    """
    Returns a new array with the same shape as the input array `x`, filled with zeros.

    This function is commonly used to create a new array with the same dimensions as an existing array
    but initialized to zeros, which is useful for algorithms that require a zero-filled array of the same
    shape as some input data.

    Parameters
    ----------
    x : Array
        The input array from which to derive the shape and, unless `dtype` is specified, the data type
        of the output array.
    dtype : af.Dtype | None, optional
        The desired data type for the new array. If `None`, the data type of `x` is used. This parameter
        allows the user to specify a data type different from that of `x` for the output array.
    device : Device | None, optional
        The device on which to create the new array. If `None`, the new array is created on the same device
        as `x`. This parameter can be used to specify where the array should be stored, particularly
        important in environments with multiple computational devices (e.g., CPUs and GPUs).

    Returns
    -------
    Array
        A new array having the same shape as `x`, with all elements set to zero. The data type of the
        array is determined by the `dtype` parameter if specified, otherwise by the data type of `x`.

    Examples
    --------
    >>> x = array([[1, 2, 3], [4, 5, 6]])
    >>> zeros_like(x)
    Array([[0, 0, 0],
           [0, 0, 0]])

    >>> zeros_like(x, dtype=float32)
    Array([[0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0]])

    >>> zeros_like(x, device='gpu')
    Array([[0, 0, 0],
           [0, 0, 0]])
    # Note: The actual device specification will depend on the array library's implementation and the available
    hardware.

    Notes
    -----
    - The function `zeros_like` simplifies the process of creating new arrays that are intended to be
      used as initial or default states in various numerical computations, matching the size and shape
      of existing data structures.
    - Specifying the `dtype` and `device` parameters provides flexibility, enabling the creation of the
      zero-filled array with specific attributes tailored to the needs of different computational tasks
      or hardware environments.
    """
    _check_valid_dtype(dtype)

    if dtype is None:
        dtype = x.dtype

    if device is None:
        device = x.device

    return zeros(x.shape, dtype=dtype, device=device)
