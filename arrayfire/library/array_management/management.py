__all__ = ["set_manual_eval_flag", "eval", "copy_array"]

from typing import cast

import arrayfire_wrapper.lib as wrapper
from arrayfire_wrapper.lib import set_manual_eval_flag

from arrayfire import Array
from arrayfire.array_object import afarray_as_array


@afarray_as_array
def copy_array(array: Array, /) -> Array:
    return cast(Array, wrapper.copy_array(array.arr))


def eval(*arrays: Array) -> None:
    if len(arrays) == 1:
        wrapper.eval(arrays[0].arr)

    arrs = [array.arr for array in arrays]
    wrapper.eval_multiple(len(arrays), *arrs)
