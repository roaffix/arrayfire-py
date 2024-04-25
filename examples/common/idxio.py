#!/usr/bin/env python


#######################################################
# Copyright (c) 2024, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################
def reverse_char(b: int) -> int:
    b = (b & 0xF0) >> 4 | (b & 0x0F) << 4
    b = (b & 0xCC) >> 2 | (b & 0x33) << 2
    b = (b & 0xAA) >> 1 | (b & 0x55) << 1
    return b


# http://stackoverflow.com/a/9144870/2192361
def reverse(x: int) -> int:
    x = ((x >> 1) & 0x55555555) | ((x & 0x55555555) << 1)
    x = ((x >> 2) & 0x33333333) | ((x & 0x33333333) << 2)
    x = ((x >> 4) & 0x0F0F0F0F) | ((x & 0x0F0F0F0F) << 4)
    x = ((x >> 8) & 0x00FF00FF) | ((x & 0x00FF00FF) << 8)
    x = ((x >> 16) & 0xFFFF) | ((x & 0xFFFF) << 16)
    return x


def read_idx(name: str) -> tuple[list[int], list[float]]:
    with open(name, "rb") as f:
        # In the C++ version, bytes the size of 4 chars are being read
        # May not work properly in machines where a char is not 1 byte
        bytes_read = f.read(4)
        bytes_read = bytearray(bytes_read)

        if bytes_read[2] != 8:
            raise RuntimeError("Unsupported data type")

        numdims = bytes_read[3]
        elemsize = 1

        # Read the dimensions
        elem = 1
        dims = [0] * numdims
        for i in range(numdims):
            bytes_read = bytearray(f.read(4))

            # Big endian to little endian
            for j in range(4):
                bytes_read[j] = reverse_char(bytes_read[j])
            bytes_read_int = int.from_bytes(bytes_read, "little")
            dim = reverse(bytes_read_int)

            elem = elem * dim
            dims[i] = dim

        # Read the data
        cdata = f.read(elem * elemsize)
        cdata_list = list(cdata)
        data = [float(cdata_elem) for cdata_elem in cdata_list]

        return (dims, data)


if __name__ == "__main__":
    # Example usage of reverse_char
    byte_value = 0b10101010
    reversed_byte = reverse_char(byte_value)
    print(f"Original byte: {byte_value:08b}, Reversed byte: {reversed_byte:08b}")

    # Example usage of reverse
    int_value = 0x12345678
    reversed_int = reverse(int_value)
    print(f"Original int: {int_value:032b}, Reversed int: {reversed_int:032b}")
