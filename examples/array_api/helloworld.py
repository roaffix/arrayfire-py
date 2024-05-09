#!/usr/bin/env python

#######################################################
# Copyright (c) 2024, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

import arrayfire.array_api as xp

try:
    print("Create a 5-by-3 matrix of random floats on the GPU\n")
    A = xp.asarray([5, 3, 1, 1])
    print(A)

    print("Element-wise arithmetic\n")
    B = xp.sin(A) + 1.5
    print(B)

    print("Matrix Multiplication")
    C = xp.multiply(A, B)
    print(C)

    print("Create a constant array")
    r = xp.full((16, 4, 1, 1), 2)
    print(r)

    print("Create 2-by-3 matrix from host data\n")
    d = [1, 2, 3, 4, 5, 6]
    D = xp.reshape(xp.asarray(d, dtype=xp.int32), (2, 3))
    print(D)

    print("Flip Vertically / Horizontally")
    print(A)
    print(xp.flip(A, axis=0))
    print(xp.flip(A, axis=1))

    print("Sum, Min, Max along row / columns")
    print(A)
    print(xp.min(A, axis=0))
    print(xp.max(A, axis=0))

    print(xp.sum(A, axis=0))
    print(xp.sum(A, axis=1))

except Exception as e:
    print("Error: " + str(e))
