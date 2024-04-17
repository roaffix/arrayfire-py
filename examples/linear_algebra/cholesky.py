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


def generate_symmetric_positive_definite_matrix(n: int) -> af.Array:
    """Generates a symmetric positive definite matrix of size n x n."""
    t = af.randu((n, n))
    return af.matmul(t, t, rhs_opts=af.MatProp.TRANS) + af.identity((n, n)) * n


def run_cholesky_inplace(matrix: af.Array) -> None:
    """Performs Cholesky decomposition in place and prints the upper and lower triangular results."""
    print("Running Cholesky InPlace")
    cin_upper = matrix.copy()
    cin_lower = matrix.copy()

    af.cholesky(cin_upper, is_upper=True)
    af.cholesky(cin_lower, is_upper=False)

    print(cin_upper)
    print(cin_lower)


def run_cholesky_out_of_place(matrix: af.Array) -> None:
    """Performs Cholesky decomposition out of place and prints the results if successful."""
    print("Running Cholesky Out of place")

    out_upper, upper_success = af.cholesky(matrix, is_upper=True)
    out_lower, lower_success = af.cholesky(matrix, is_upper=False)

    if upper_success == 0:
        print("Upper triangular matrix:")
        print(out_upper)
    if lower_success == 0:
        print("Lower triangular matrix:")
        print(out_lower)


def main() -> None:
    try:
        af.info()
        n = 5
        spd_matrix = generate_symmetric_positive_definite_matrix(n)

        run_cholesky_inplace(spd_matrix)
        run_cholesky_out_of_place(spd_matrix)

    except Exception as e:
        print("Error: ", str(e))


if __name__ == "__main__":
    main()
