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
from typing import cast

import arrayfire as af


def monte_carlo_options(
    N: int,
    K: float,
    t: float,
    vol: float,
    r: float,
    strike: int,
    steps: int,
    use_barrier: bool = True,
    B: float | None = None,
    ty: af.Dtype = af.float32,
) -> float:
    dt = t / (steps - 1)
    s = af.constant(strike, (N, 1), dtype=ty)

    randmat = af.randn((N, steps - 1), dtype=ty)
    randmat = af.exp((r - (vol * vol * 0.5)) * dt + vol * math.sqrt(dt) * randmat)

    S = af.product(af.join(1, s, randmat), axis=1)

    if use_barrier:
        if B is None:
            raise ValueError("Barrier value B must be provided if use_barrier is True.")
        S = S * af.all_true(S < B, 1)

    payoff = af.maxof(0, S - K)
    mean_payoff = cast(float, af.mean(payoff)) * math.exp(-r * t)

    return mean_payoff


def monte_carlo_simulate(N: int, use_barrier: bool, num_iter: int = 10) -> float:
    steps = 180
    stock_price = 100.0
    maturity = 0.5
    volatility = 0.3
    rate = 0.01
    strike = 100
    barrier = 115.0 if use_barrier else None

    total_time = time()
    for _ in range(num_iter):
        monte_carlo_options(N, stock_price, maturity, volatility, rate, strike, steps, use_barrier, barrier)
    average_time = (time() - total_time) / num_iter

    return average_time


def main() -> None:
    if len(sys.argv) > 1:
        device_id = int(sys.argv[1])
        af.set_device(device_id)
    af.info()

    # Initial simulation calls to test without and with barrier
    print("Simulation without barrier:", monte_carlo_simulate(1000, use_barrier=False))
    print("Simulation with barrier:", monte_carlo_simulate(1000, use_barrier=True))

    af.sync()  # Synchronize ArrayFire computations before timing analysis

    # Timing analysis for different numbers of paths
    for n in range(10000, 100001, 10000):
        time_vanilla = 1000 * monte_carlo_simulate(n, False, 100)
        time_barrier = 1000 * monte_carlo_simulate(n, True, 100)
        print(
            f"Time for {n:7d} paths - vanilla method: {time_vanilla:4.3f} ms, barrier method: {time_barrier:4.3f} ms"
        )


if __name__ == "__main__":
    main()
