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
from random import random
from time import time

import numpy as np

import arrayfire as af

try:
    frange = xrange  # type: ignore[name-defined]
except NameError:
    frange = range  # Python3


# Having the function outside is faster than the lambda inside
def in_circle(x, y):
    return (x * x + y * y) < 1


def calc_pi_device(samples):
    x = af.randu((samples,))
    y = af.randu((samples,))
    return 4 * af.sum(in_circle(x, y)) / samples


def calc_pi_numpy(samples):
    np.random.seed(1)
    x = np.random.rand(samples).astype(np.float32)
    y = np.random.rand(samples).astype(np.float32)
    return 4.0 * np.sum(in_circle(x, y)) / samples


def calc_pi_host(samples):
    count = sum(1 for k in frange(samples) if in_circle(random(), random()))
    return 4 * float(count) / samples


def bench(calc_pi, samples=1000000, iters=25):
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
