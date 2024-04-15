#!/usr/bin/python

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
from arrayfire.library.signal_processing import convolve2, convolve2_separable


def af_assert(left, right, eps=1e-6):
    if af.max(af.abs(left - right)) > eps:
        raise ValueError("Arrays not within dictated precision")
    return


if __name__ == "__main__":
    if len(sys.argv) > 1:
        af.set_device(int(sys.argv[1]))
    af.info()

    h_dx = array("f", (1.0 / 12, -8.0 / 12, 0, 8.0 / 12, 1.0 / 12))
    h_spread = array("f", (1.0 / 5, 1.0 / 5, 1.0 / 5, 1.0 / 5, 1.0 / 5))

    img = af.randu((640, 480))
    dx = af.Array(h_dx, shape=(5, 1))
    spread = af.Array(h_spread, shape=(1, 5))
    kernel = af.matmul(dx, spread)

    full_res = convolve2(img, kernel)
    sep_res = convolve2_separable(dx, spread, img)

    af_assert(full_res, sep_res)

    time_convolve2 = timeit(lambda: convolve2(img, kernel), number=1000)
    print(f"full 2D convolution time: {time_convolve2 * 1000:.5f} ms")

    time_convolve2_sep = timeit(lambda: convolve2_separable(dx, spread, img), number=1000)
    print(f"full separable 2D convolution time: {time_convolve2_sep * 1000:.5f} ms")
