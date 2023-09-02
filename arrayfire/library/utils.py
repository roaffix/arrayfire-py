from typing import cast

from arrayfire import Array
from arrayfire._array_helpers import afarray_as_array
from arrayfire.backend import _clib_wrapper as wrapper


@afarray_as_array
def _all_true(array: Array, axis: int, /) -> Array:
    result = wrapper.all_true(array.arr, axis)
    return cast(Array, result)


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
        return wrapper.all_true_all(array.arr)

    return _all_true(array, axis)


# from time import time
# import math

# def timeit(af_func, *args):
#     """
#     Function to time arrayfire functions.

#     Parameters
#     ----------

#     af_func    : arrayfire function

#     *args      : arguments to `af_func`

#     Returns
#     --------

#     t   : Time in seconds
#     """

#     sample_trials = 3

#     sample_time = 1E20

#     for i in range(sample_trials):
#         start = time()
#         res = af_func(*args)
#         eval(res)
#         sync()
#         sample_time = min(sample_time, time() - start)

#     if (sample_time >= 0.5):
#         return sample_time

#     num_iters = max(math.ceil(1.0 / sample_time), 3.0)

#     start = time()
#     for i in range(int(num_iters)):
#         res = af_func(*args)
#         eval(res)
#     sync()
#     sample_time = (time() - start) / num_iters
#     return sample_time
