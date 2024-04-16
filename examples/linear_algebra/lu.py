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


def run_lu_inplace(array: af.Array) -> None:
    """Performs LU decomposition in place and prints the results."""
    print("Running LU InPlace")
    pivot = af.lu(array, inplace=True)
    print(array)
    print(pivot)


def run_lu_factorization(array: af.Array) -> None:
    """Performs LU decomposition, extracting and printing Lower and Upper matrices."""
    print("Running LU with Upper Lower Factorization")
    lower, upper, pivot = af.lu(array)
    print(lower)
    print(upper)
    print(pivot)


def main() -> None:
    try:
        af.info()  # Display ArrayFire library information
        in_array = af.randu((5, 8))  # Generate a random 5x8 matrix

        # Perform and print results of LU decomposition in place
        run_lu_inplace(in_array.copy())  # Use a copy to preserve the original matrix for the next function
        # Perform and print results of LU decomposition with L and U matrices
        run_lu_factorization(in_array)

    except Exception as e:
        print("Error: ", str(e))


if __name__ == "__main__":
    main()
