from arrayfire import (
    bool,
    complex64,
    complex128,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)
from arrayfire.dtypes import Dtype

all_dtypes = (
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float32,
    float64,
    complex64,
    complex128,
    bool,
)
boolean_dtypes = (bool,)
real_floating_dtypes = (float32, float64)
floating_dtypes = (float32, float64, complex64, complex128)
complex_floating_dtypes = (complex64, complex128)
integer_dtypes = (int8, int16, int32, int64, uint8, uint16, uint32, uint64)
signed_integer_dtypes = (int8, int16, int32, int64)
unsigned_integer_dtypes = (uint8, uint16, uint32, uint64)
integer_orboolean_dtypes = (
    bool,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)
real_numeric_dtypes = (
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)
numeric_dtypes = (
    float32,
    float64,
    complex64,
    complex128,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)

dtype_categories = {
    "all": all_dtypes,
    "real numeric": real_numeric_dtypes,
    "numeric": numeric_dtypes,
    "integer": integer_dtypes,
    "integer or boolean": integer_orboolean_dtypes,
    "boolean": boolean_dtypes,
    "real floating-point": floating_dtypes,
    "complex floating-point": complex_floating_dtypes,
    "floating-point": floating_dtypes,
}


# Note: the spec defines a restricted type promotion table compared to NumPy.
# In particular, cross-kind promotions like integer + float or boolean +
# integer are not allowed, even for functions that accept both kinds.
# Additionally, NumPy promotes signed integer + uint64 to float64, but this
# promotion is not allowed here. To be clear, Python scalar int objects are
# allowed to promote to floating-point dtypes, but only in array operators
# (see Array._promote_scalar) method in _array_object.py.
_promotion_table = {
    (int8, int8): int8,
    (int8, int16): int16,
    (int8, int32): int32,
    (int8, int64): int64,
    (int16, int8): int16,
    (int16, int16): int16,
    (int16, int32): int32,
    (int16, int64): int64,
    (int32, int8): int32,
    (int32, int16): int32,
    (int32, int32): int32,
    (int32, int64): int64,
    (int64, int8): int64,
    (int64, int16): int64,
    (int64, int32): int64,
    (int64, int64): int64,
    (uint8, uint8): uint8,
    (uint8, uint16): uint16,
    (uint8, uint32): uint32,
    (uint8, uint64): uint64,
    (uint16, uint8): uint16,
    (uint16, uint16): uint16,
    (uint16, uint32): uint32,
    (uint16, uint64): uint64,
    (uint32, uint8): uint32,
    (uint32, uint16): uint32,
    (uint32, uint32): uint32,
    (uint32, uint64): uint64,
    (uint64, uint8): uint64,
    (uint64, uint16): uint64,
    (uint64, uint32): uint64,
    (uint64, uint64): uint64,
    (int8, uint8): int16,
    (int8, uint16): int32,
    (int8, uint32): int64,
    (int16, uint8): int16,
    (int16, uint16): int32,
    (int16, uint32): int64,
    (int32, uint8): int32,
    (int32, uint16): int32,
    (int32, uint32): int64,
    (int64, uint8): int64,
    (int64, uint16): int64,
    (int64, uint32): int64,
    (uint8, int8): int16,
    (uint16, int8): int32,
    (uint32, int8): int64,
    (uint8, int16): int16,
    (uint16, int16): int32,
    (uint32, int16): int64,
    (uint8, int32): int32,
    (uint16, int32): int32,
    (uint32, int32): int64,
    (uint8, int64): int64,
    (uint16, int64): int64,
    (uint32, int64): int64,
    (float32, float32): float32,
    (float32, float64): float64,
    (float64, float32): float64,
    (float64, float64): float64,
    (complex64, complex64): complex64,
    (complex64, complex128): complex128,
    (complex128, complex64): complex128,
    (complex128, complex128): complex128,
    (float32, complex64): complex64,
    (float32, complex128): complex128,
    (float64, complex64): complex128,
    (float64, complex128): complex128,
    (complex64, float32): complex64,
    (complex64, float64): complex128,
    (complex128, float32): complex128,
    (complex128, float64): complex128,
    (bool, bool): bool,
}


def promote_types(type1: Dtype, type2: Dtype) -> Dtype:
    if (type1, type2) in _promotion_table:
        return _promotion_table[type1, type2]
    raise TypeError(f"{type1} and {type2} cannot be type promoted together")
