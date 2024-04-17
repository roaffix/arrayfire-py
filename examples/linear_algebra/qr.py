#!/usr/bin/env python

#######################################################
# Copyright (c) 2024, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

import arrayfire as af


def run_qr_inplace(array: af.Array) -> None:
    """Performs QR decomposition in place and prints the results."""
    print("Running QR InPlace")
    q_in = array.copy()
    tau = af.qr(q_in, inplace=True)
    print(q_in)
    print(tau)


def run_qr_factorization(array: af.Array) -> None:
    """Performs QR decomposition, extracting and printing Q and R matrices."""
    print("Running QR with Q and R factorization")
    q, r, tau = af.qr(array)
    print(q)
    print(r)
    print(tau)


def main() -> None:
    try:
        af.info()
        in_array = af.randu((5, 8))  # Random 5x8 matrix

        run_qr_inplace(in_array)
        run_qr_factorization(in_array)

    except Exception as e:
        print("Error: ", str(e))


if __name__ == "__main__":
    main()
