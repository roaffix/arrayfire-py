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
from time import time
from typing import Any, Callable

import arrayfire as af

try:
    import numpy as np
except ImportError:
    raise ImportError("Please install arrayfire-python[benchmarks] or numpy directly to run this example.")


def calc_arrayfire(n: int) -> Callable:
    A = af.randu((n, n))
    af.sync(-1)

    def run(iters: int) -> None:
        for t in range(iters):
            B = af.fft2(A)  # noqa: F841

        af.sync(-1)

    return run


def calc_numpy(n: int) -> Callable:
    np.random.seed(1)
    A = np.random.rand(n, n).astype(np.float32)

    def run(iters: int) -> None:
        for t in range(iters):
            B = np.fft.fft2(A)  # noqa: F841

    return run


def bench(calc: Any, iters: int = 100, upto: int = 13) -> None:
    _, name = calc.__name__.split("_")
    print("Benchmark N x N 2D fft on %s" % name)

    for M in range(7, upto):
        N = 1 << M
        run = calc(N)
        start = time()
        run(iters)
        t = (time() - start) / iters
        gflops = (10.0 * N * N * M) / (t * 1e9)
        print("Time taken for %4d x %4d: %0.4f Gflops" % (N, N, gflops))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        af.set_device(int(sys.argv[1]))

    af.info()

    bench(calc_arrayfire)
    if np:
        bench(calc_numpy, upto=10)
