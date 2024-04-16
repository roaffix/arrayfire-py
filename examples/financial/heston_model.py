#!/usr/bin/env python

##############################################################################################
# Copyright (c) 2015, Michael Nowotny
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or other
# materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################################

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
        t_previous = (t + 1) % 2
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
