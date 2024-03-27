from inspect import getfullargspec
from typing import TYPE_CHECKING, Callable, Iterator

import pytest

from array_api import _elementwise_functions, asarray
from array_api._array_object import Array
from array_api._dtypes import (
    boolean_dtypes,
    dtype_categories,
    floating_dtypes,
    int8,
    integer_dtypes,
    real_floating_dtypes,
)
from array_api._elementwise_functions import bitwise_left_shift, bitwise_right_shift


def nargs(func: Callable) -> int:
    return len(getfullargspec(func).args)


def test_function_types() -> None:
    # Test that every function accepts only the required input types. We only
    # test the negative cases here (error). The positive cases are tested in
    # the array API test suite.

    elementwise_function_input_types = {
        "abs": "numeric",
        "acos": "floating-point",
        "acosh": "floating-point",
        "add": "numeric",
        "asin": "floating-point",
        "asinh": "floating-point",
        "atan": "floating-point",
        "atan2": "real floating-point",
        "atanh": "floating-point",
        # "bitwise_and": "integer or boolean",
        # "bitwise_invert": "integer or boolean",
        # "bitwise_left_shift": "integer",
        # "bitwise_or": "integer or boolean",
        # "bitwise_right_shift": "integer",
        # "bitwise_xor": "integer or boolean",
        "ceil": "real numeric",
        # "conj": "complex floating-point",
        "cos": "floating-point",
        "cosh": "floating-point",
        "divide": "floating-point",
        "equal": "all",
        "exp": "floating-point",
        "expm1": "floating-point",
        "floor": "real numeric",
        "floor_divide": "real numeric",
        "greater": "real numeric",
        "greater_equal": "real numeric",
        # "imag": "complex floating-point",
        "isfinite": "numeric",
        "isinf": "numeric",
        "isnan": "numeric",
        "less": "real numeric",
        "less_equal": "real numeric",
        "log": "floating-point",
        "logaddexp": "real floating-point",
        "log10": "floating-point",
        "log1p": "floating-point",
        "log2": "floating-point",
        # "logical_and": "boolean",
        # "logical_not": "boolean",
        # "logical_or": "boolean",
        # "logical_xor": "boolean",
        "multiply": "numeric",
        "negative": "numeric",
        "not_equal": "all",
        "positive": "numeric",
        "pow": "numeric",
        # "real": "complex floating-point",
        "remainder": "real numeric",
        "round": "numeric",
        "sign": "numeric",
        "sin": "floating-point",
        "sinh": "floating-point",
        "sqrt": "floating-point",
        "square": "numeric",
        "subtract": "numeric",
        "tan": "floating-point",
        "tanh": "floating-point",
        "trunc": "real numeric",
    }

    def _array_vals() -> Iterator[Array]:
        for dt in integer_dtypes:
            if dt in {int8}:
                continue
            yield asarray([1], dtype=dt)
        # for d in boolean_dtypes:
        #     yield asarray(False, dtype=d)
        for dt in real_floating_dtypes:
            yield asarray([1.0], dtype=dt)

    for x in _array_vals():
        for func_name, types in elementwise_function_input_types.items():
            dtypes = dtype_categories[types]
            func = getattr(_elementwise_functions, func_name)
            if nargs(func) == 2:
                for y in _array_vals():
                    if x.dtype not in dtypes or y.dtype not in dtypes:
                        pytest.raises(TypeError, lambda: func(x, y))
            else:
                if x.dtype not in dtypes:
                    print(func)
                    pytest.raises(TypeError, lambda: func(x))


# def test_bitwise_shift_error() -> None:
#     # bitwise shift functions should raise when the second argument is negative
#     pytest.raises(ValueError, lambda: bitwise_left_shift(asarray([1, 1]), asarray([1, -1])))
#     pytest.raises(ValueError, lambda: bitwise_right_shift(asarray([1, 1]), asarray([1, -1])))
