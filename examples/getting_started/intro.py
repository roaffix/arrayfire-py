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
from array import array

import arrayfire as af

if __name__ == "__main__":
    if len(sys.argv) > 1:
        af.set_device(int(sys.argv[1]))
    af.info()

    print("\n---- Intro to ArrayFire using unsigned(s32) arrays ----\n")

    h_A = array("i", (1, 2, 4, -1, 2, 0, 4, 2, 3))
    h_B = array("i", (2, 3, 5, 6, 0, 10, -12, 0, 1))

    A = af.Array(obj=h_A, shape=(3, 3), dtype=af.int32)
    B = af.Array(obj=h_B, shape=(3, 3), dtype=af.int32)

    print("\n---- Sub referencing and sub assignment\n")
    print(A)
    print(A[0, :])
    print(A[:, 0])
    A[0, 0] = 11
    A[1] = 100
    print(A)
    print(B)
    A[1, :] = B[2, :]
    print(A)

    b_A = array("I", (1, 1, 1, 0, 1, 1, 0, 0, 0))
    b_B = array("I", (1, 0, 1, 0, 1, 0, 1, 0, 1))

    C = af.Array(obj=b_A, shape=(3, 3), dtype=af.uint32)
    D = af.Array(obj=b_B, shape=(3, 3), dtype=af.uint32)
    print(C)
    print(D)

    print("\n---- Bitwise operations\n")
    print(af.bitand(C, D))
    print(af.bitor(C, D))
    print(C, D)

    print("\n---- Transpose\n")
    print(A)
    print(af.transpose(A))

    print("\n---- Flip Vertically / Horizontally\n")
    print(A)
    print(af.flip(A, axis=0))
    print(af.flip(A, axis=1))

    print("\n---- Sum, Min, Max along row / columns\n")
    print(A)
    print(af.min(A, axis=0))  # type: ignore[arg-type]
    print(af.max(A, axis=0))  # type: ignore[arg-type]

    print(af.min(A, axis=1))  # type: ignore[arg-type]
    print(af.max(A, axis=1))  # type: ignore[arg-type]

    print(af.sum(A, axis=0))  # type: ignore[arg-type]
    print(af.sum(A, axis=1))  # type: ignore[arg-type]

    print("\n---- Get minimum with index\n")
    (min_val, min_idx) = af.imin(A, axis=0)
    print(min_val)  # type: ignore[arg-type]
    print(min_idx)  # type: ignore[arg-type]
