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
import time
from typing import Tuple

import arrayfire as af


def initialize_parameters() -> Tuple[float, float, float, float, float, float, float, float]:
    """Initialize and return model parameters."""
    r = math.log(1.0319)  # risk-free rate
    rho = -0.82  # instantaneous correlation between Brownian motions
    sigmaV = 0.14  # variance of volatility
    kappa = 3.46  # mean reversion speed
    vBar = 0.008  # mean variance
    k = math.log(0.95)  # strike price, converted to log space
    x0 = 0  # initial log stock price
    v0 = 0.087**2  # initial volatility
    return r, rho, sigmaV, kappa, vBar, k, x0, v0


def simulate_heston_model(
    T: int, N: int, R: int, mu: float, kappa: float, vBar: float, sigmaV: float, rho: float, x0: float, v0: float
) -> Tuple[af.Array, af.Array]:
    """Simulate the Heston model for given parameters and return the resulting arrays."""
    deltaT = T / (N - 1)
    sqrtDeltaT = math.sqrt(deltaT)
    sqrtOneMinusRhoSquare = math.sqrt(1 - rho**2)

    m = af.constant(0, (2,))
    m[0] = rho
    m[1] = sqrtOneMinusRhoSquare
    zeroArray = af.constant(0, (R, 1))

    x = [af.constant(x0, (R,)) for _ in range(2)]
    v = [af.constant(v0, (R,)) for _ in range(2)]

    for t in range(1, N):
        t_previous = (t - 1) % 2
        t_current = t % 2

        dBt = af.randn((R, 2)) * sqrtDeltaT
        vLag = af.maxof(v[t_previous], zeroArray)
        sqrtVLag = af.sqrt(vLag)

        x[t_current] = x[t_previous] + (mu - 0.5 * vLag) * deltaT + sqrtVLag * dBt[:, 0]
        v[t_current] = vLag + kappa * (vBar - vLag) * deltaT + sigmaV * sqrtVLag * af.matmul(dBt, m)

    return x[t_current], af.maxof(v[t_current], zeroArray)


def main() -> None:
    T = 1
    nT = 20 * T
    R_first = 1000
    R = 5000000
    r, rho, sigmaV, kappa, vBar, k, x0, v0 = initialize_parameters()

    # Initial simulation
    simulate_heston_model(T, nT, R_first, r, kappa, vBar, sigmaV, rho, x0, v0)

    # Time the pricing of a vanilla call option
    tic = time.time()
    x, v = simulate_heston_model(T, nT, R, r, kappa, vBar, sigmaV, rho, x0, v0)
    af.sync()
    toc = time.time() - tic
    K = math.exp(k)
    C_CPU = math.exp(-r * T) * af.mean(af.maxof(af.exp(x) - K, af.constant(0, (R,))))
    print(f"Time elapsed = {toc:.3f} secs")
    print(f"Call price = {C_CPU:.6f}")
    print(f"Average final variance = {af.mean(v):.6f}")


if __name__ == "__main__":
    main()
