#!/usr/bin/env python

#######################################################
# Copyright (c) 2024, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

import sys
from array import array
from timeit import timeit

import arrayfire as af


def set_device_from_args() -> None:
    """Sets the ArrayFire device based on the command line argument."""
    if len(sys.argv) > 1:
        af.set_device(int(sys.argv[1]))
    af.info()


def create_arrays() -> tuple[af.Array, ...]:
    """Creates and returns initialized ArrayFire arrays for convolution."""
    h_dx = array("f", (1.0 / 12, -8.0 / 12, 0, 8.0 / 12, 1.0 / 12))
    h_spread = array("f", (1.0 / 5, 1.0 / 5, 1.0 / 5, 1.0 / 5, 1.0 / 5))

    img = af.randu((640, 480))
    dx = af.Array(h_dx, shape=(5, 1))
    spread = af.Array(h_spread, shape=(1, 5))

    return img, dx, spread


def perform_convolution(img: af.Array, dx: af.Array, spread: af.Array) -> tuple[af.Array, af.Array]:
    """Performs and returns the result of full and separable 2D convolution."""
    kernel = af.matmul(dx, spread)
    full_res = af.convolve2(img, kernel)
    sep_res = af.convolve2_separable(dx, spread, img)
    return full_res, sep_res


def af_assert(left: af.Array, right: af.Array, eps: float = 1e-6) -> None:
    """Asserts that two arrays are equal within a specified precision."""
    max_diff = af.max(af.abs(left - right))
    if isinstance(max_diff, complex):
        max_diff = max_diff.real
    if max_diff > eps:
        raise ValueError("Arrays not within dictated precision")


def time_convolution_operations(img: af.Array, dx: af.Array, spread: af.Array, kernel: af.Array) -> None:
    """Times and prints the convolution operations."""
    time_convolve2 = timeit(lambda: af.convolve2(img, kernel), number=1000)
    time_convolve2_sep = timeit(lambda: af.convolve2_separable(dx, spread, img), number=1000)

    print(f"Full 2D convolution time: {time_convolve2 * 1000:.5f} ms")
    print(f"Full separable 2D convolution time: {time_convolve2_sep * 1000:.5f} ms")


def main() -> None:
    try:
        set_device_from_args()
        img, dx, spread = create_arrays()
        full_res, sep_res = perform_convolution(img, dx, spread)
        af_assert(full_res, sep_res)
        kernel = af.matmul(dx, spread)  # Reconstruct kernel for timing
        time_convolution_operations(img, dx, spread, kernel)
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
