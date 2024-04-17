#!/usr/bin/env python

#######################################################
# Copyright (c) 2024, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

import math
import sys
from time import time

import arrayfire as af


def initialize_device() -> None:
    """Initialize the ArrayFire device based on command line arguments."""
    device_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    af.set_device(device_id)
    af.info()


def cumulative_normal_distribution(x: af.Array) -> af.Array:
    """Calculate the cumulative normal distribution using ArrayFire."""
    sqrt2 = math.sqrt(2.0)
    condition = x > 0
    lhs = condition * (0.5 + af.erf(x / sqrt2) / 2)
    rhs = (1 - condition) * (0.5 - af.erf((-x) / sqrt2) / 2)
    return lhs + rhs


def black_scholes(S: af.Array, X: af.Array, R: af.Array, V: af.Array, T: af.Array) -> tuple[af.Array, af.Array]:
    """Compute call and put options prices using the Black-Scholes formula."""
    d1 = (af.log(S / X) + (R + 0.5 * V**2) * T) / (V * af.sqrt(T))
    d2 = d1 - V * af.sqrt(T)

    cnd_d1 = cumulative_normal_distribution(d1)
    cnd_d2 = cumulative_normal_distribution(d2)

    C = S * cnd_d1 - X * af.exp(-R * T) * cnd_d2
    P = X * af.exp(-R * T) * (1 - cnd_d2) - S * (1 - cnd_d1)
    return C, P


def benchmark_black_scholes(num_elements: int, num_iter: int = 100) -> None:
    """Benchmark the Black-Scholes model over varying matrix sizes."""
    M = 4000
    for N in range(50, 501, 50):
        S, X, R, V, T = (af.randu((M, N)) for _ in range(5))

        print(f"Input data size: {M * N} elements")

        start = time()
        for _ in range(num_iter):
            C, P = black_scholes(S, X, R, V, T)
            af.eval(C, P)
        af.sync()

        sec = (time() - start) / num_iter
        print(f"Mean GPU Time: {1000.0 * sec:.6f} ms\n")


def main() -> None:
    initialize_device()

    # Run a small test to ensure that everything is set up correctly.
    M = 4000
    test_arrays = (af.randu((M, 1)) for _ in range(5))
    C, P = black_scholes(*test_arrays)
    af.eval(C, P)
    af.sync()

    # Benchmark Black-Scholes over varying sizes of input data.
    benchmark_black_scholes(M)


if __name__ == "__main__":
    main()
