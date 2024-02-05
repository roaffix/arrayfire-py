__all__ = ["convolve2_gradient_nn", "ConvGradient"]

from typing import cast

import arrayfire_wrapper.lib as wrapper

from arrayfire import Array
from arrayfire.array_object import afarray_as_array
from arrayfire.library.constants import ConvGradient


@afarray_as_array
def convolve2_gradient_nn(
    incoming_gradient: Array,
    original_signal: Array,
    original_filter: Array,
    convolved_output: Array,
    /,
    *,
    strides: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    grad_type: ConvGradient = ConvGradient.DEFAULT,
) -> Array:
    return cast(
        Array,
        wrapper.convolve2_gradient_nn(
            incoming_gradient.arr,
            original_signal.arr,
            original_filter.arr,
            convolved_output.arr,
            strides,
            padding,
            dilation,
            grad_type,
        ),
    )
