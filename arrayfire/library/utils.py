from arrayfire import Array


def all_true(array: Array, axis: int | None = None) -> Array:
    return NotImplemented


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
