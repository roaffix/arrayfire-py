__all__ = [
    "Dtype",
    "b8",
    "bool",
    "c32",
    "c64",
    "complex32",
    "complex64",
    "f16",
    "f32",
    "f64",
    "float16",
    "float32",
    "float64",
    "int16",
    "int32",
    "int64",
    "s16",
    "s32",
    "s64",
    "u8",
    "u16",
    "u32",
    "u64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]


from typing import TypeAlias

py_bool: TypeAlias = bool

from arrayfire_wrapper import (  # noqa
    Dtype,
    b8,
    bool,
    c32,
    c64,
    complex32,
    complex64,
    f16,
    f32,
    f64,
    float16,
    float32,
    float64,
    int16,
    int32,
    int64,
    s16,
    s32,
    s64,
    u8,
    u16,
    u32,
    u64,
    uint8,
    uint16,
    uint32,
    uint64,
)

supported_dtypes = (
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
    complex64,
    complex32,
    bool,
    s16,
    s32,
    s64,
    u8,
    u16,
    u32,
    u64,
    f16,
    f32,
    f64,
    c32,
    c64,
    b8,
)


def implicit_dtype(number: int | float | py_bool | complex, array_dtype: Dtype) -> Dtype:
    if isinstance(number, py_bool):
        number_dtype = bool
    elif isinstance(number, int):
        number_dtype = int64
    elif isinstance(number, float):
        number_dtype = float64
    elif isinstance(number, complex):
        number_dtype = complex64
    else:
        raise TypeError(f"{type(number)} is not supported and can not be converted to af.Dtype.")

    if not (array_dtype == float32 or array_dtype == complex32):
        return number_dtype

    if number_dtype == float64:
        return float32

    if number_dtype == complex64:
        return complex32

    return number_dtype


def c_api_value_to_dtype(value: int) -> Dtype:
    for dtype in supported_dtypes:
        if value == dtype.c_api_value:
            return dtype

    raise TypeError("There is no supported dtype that matches passed dtype C API value.")


def str_to_dtype(value: str) -> Dtype:
    for dtype in supported_dtypes:
        if value == dtype.typecode or value == dtype.typename or value == dtype.name:
            return dtype

    raise TypeError("There is no supported dtype that matches passed dtype typecode.")
