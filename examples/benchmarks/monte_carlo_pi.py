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
from random import random
from time import time
from typing import Any, overload

try:
    import numpy as np
except ImportError:
    raise ImportError("Please install arrayfire-python[benchmarks] or numpy directly to run this example.")

import arrayfire as af

try:
    frange = xrange  # type: ignore[name-defined]
except NameError:
    frange = range  # Python3


@overload
def in_circle(x: af.Array, y: af.Array) -> af.Array:
    return (x * x + y * y) < 1


@overload
def in_circle(x: np.ndarray, y: np.ndarray) -> np.ndarray: ...


@overload
def in_circle(x: float, y: float) -> bool: ...


# Having the function outside is faster than the lambda inside
def in_circle(x: af.Array | np.ndarray | float, y: af.Array | np.ndarray | float) -> af.Array | np.ndarray | float:
    return (x * x + y * y) < 1  # type: ignore[operator]  # NOTE no override for np.ndarray


def calc_pi_device(samples: int) -> af.Array:
    x = af.randu((samples,))
    y = af.randu((samples,))
    res = in_circle(x, y)
    return 4 * af.sum(res) / samples  # type: ignore[return-value, operator]


def calc_pi_numpy(samples: int) -> af.Array:
    np.random.seed(1)
    x = np.random.rand(samples).astype(np.float32)
    y = np.random.rand(samples).astype(np.float32)
    res = in_circle(x, y)
    return 4.0 * np.sum(res) / samples  # type: ignore[no-any-return]


def calc_pi_host(samples: int) -> float:
    count = sum(1 for k in frange(samples) if in_circle(random(), random()))
    return 4 * float(count) / samples


def bench(calc_pi: Any, samples: int = 1000000, iters: int = 25) -> None:
    func_name = calc_pi.__name__[8:]
    print(
        "Monte carlo estimate of pi on %s with %d million samples: %f" % (func_name, samples / 1e6, calc_pi(samples))
    )

    start = time()
    for k in frange(iters):
        calc_pi(samples)
    end = time()

    print("Average time taken: %f ms" % (1000 * (end - start) / iters))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        af.set_device(int(sys.argv[1]))
    af.info()

    bench(calc_pi_device)
    if np:
        bench(calc_pi_numpy)
    bench(calc_pi_host)
