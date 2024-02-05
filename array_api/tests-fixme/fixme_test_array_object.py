from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import pytest

from arrayfire import bool, int8, int16, int32, int64, uint64
from arrayfire.array_api._creation_function import asarray
from arrayfire.array_api._data_type_functions import result_type
from arrayfire.array_api._dtypes import (
    boolean_dtypes,
    complex_floating_dtypes,
    floating_dtypes,
    integer_dtypes,
    integer_or_boolean_dtypes,
    numeric_dtypes,
    real_floating_dtypes,
    real_numeric_dtypes,
)

if TYPE_CHECKING:
    from arrayfire.array_api._array_object import Array


def test_operators() -> None:
    # For every operator, we test that it works for the required type
    # combinations and raises TypeError otherwise
    binary_op_dtypes = {
        "__add__": "numeric",
        "__and__": "integer_or_boolean",
        "__eq__": "all",
        "__floordiv__": "real numeric",
        "__ge__": "real numeric",
        "__gt__": "real numeric",
        "__le__": "real numeric",
        "__lshift__": "integer",
        "__lt__": "real numeric",
        "__mod__": "real numeric",
        "__mul__": "numeric",
        "__ne__": "all",
        "__or__": "integer_or_boolean",
        "__pow__": "numeric",
        "__rshift__": "integer",
        "__sub__": "numeric",
        "__truediv__": "floating",
        "__xor__": "integer_or_boolean",
    }

    # Recompute each time because of in-place ops
    def _array_vals() -> Iterator[Array]:
        for d in integer_dtypes:
            yield asarray(1, dtype=d)
        for d in boolean_dtypes:
            yield asarray(False, dtype=d)
        for d in floating_dtypes:
            yield asarray(1.0, dtype=d)

    BIG_INT = int(1e30)
    for op, dtypes in binary_op_dtypes.items():
        ops = [op]
        if op not in ["__eq__", "__ne__", "__le__", "__ge__", "__lt__", "__gt__"]:
            rop = "__r" + op[2:]
            iop = "__i" + op[2:]
            ops += [rop, iop]
        for s in [1, 1.0, 1j, BIG_INT, False]:
            for _op in ops:
                for a in _array_vals():
                    # Test array op scalar. From the spec, the following combinations
                    # are supported:

                    # - Python bool for a bool array dtype,
                    # - a Python int within the bounds of the given dtype for integer array dtypes,
                    # - a Python int or float for real floating-point array dtypes
                    # - a Python int, float, or complex for complex floating-point array dtypes

                    if (
                        (
                            dtypes == "all"
                            or dtypes == "numeric"
                            and a.dtype in numeric_dtypes
                            or dtypes == "real numeric"
                            and a.dtype in real_numeric_dtypes
                            or dtypes == "integer"
                            and a.dtype in integer_dtypes
                            or dtypes == "integer_or_boolean"
                            and a.dtype in integer_or_boolean_dtypes
                            or dtypes == "boolean"
                            and a.dtype in boolean_dtypes
                            or dtypes == "floating"
                            and a.dtype in floating_dtypes
                        )
                        # bool is a subtype of int, which is why we avoid
                        # isinstance here.
                        and (
                            a.dtype in boolean_dtypes
                            and type(s) == bool
                            or a.dtype in integer_dtypes
                            and type(s) == int
                            or a.dtype in real_floating_dtypes
                            and type(s) in [float, int]
                            or a.dtype in complex_floating_dtypes
                            and type(s) in [complex, float, int]
                        )
                    ):
                        if a.dtype in integer_dtypes and s == BIG_INT:
                            pytest.raises(OverflowError, lambda: getattr(a, _op)(s))
                        else:
                            # ignore warnings from pow(BIG_INT)
                            pytest.raises(RuntimeWarning, getattr(a, _op)(s))
                            getattr(a, _op)(s)
                    else:
                        pytest.raises(TypeError, lambda: getattr(a, _op)(s))

                # Test array op array.
                for _op in ops:
                    for x in _array_vals():
                        for y in _array_vals():
                            # See the promotion table in NEP 47 or the array
                            # API spec page on type promotion. Mixed kind
                            # promotion is not defined.
                            if (
                                x.dtype == uint64
                                and y.dtype in [int8, int16, int32, int64]
                                or y.dtype == uint64
                                and x.dtype in [int8, int16, int32, int64]
                                or x.dtype in integer_dtypes
                                and y.dtype not in integer_dtypes
                                or y.dtype in integer_dtypes
                                and x.dtype not in integer_dtypes
                                or x.dtype in boolean_dtypes
                                and y.dtype not in boolean_dtypes
                                or y.dtype in boolean_dtypes
                                and x.dtype not in boolean_dtypes
                                or x.dtype in floating_dtypes
                                and y.dtype not in floating_dtypes
                                or y.dtype in floating_dtypes
                                and x.dtype not in floating_dtypes
                            ):
                                pytest.raises(TypeError, lambda: getattr(x, _op)(y))
                            # Ensure in-place operators only promote to the same dtype as the left operand.
                            elif _op.startswith("__i") and result_type(x.dtype, y.dtype) != x.dtype:
                                pytest.raises(TypeError, lambda: getattr(x, _op)(y))
                            # Ensure only those dtypes that are required for every operator are allowed.
                            elif (
                                dtypes == "all"
                                and (
                                    x.dtype in boolean_dtypes
                                    and y.dtype in boolean_dtypes
                                    or x.dtype in numeric_dtypes
                                    and y.dtype in numeric_dtypes
                                )
                                or (
                                    dtypes == "real numeric"
                                    and x.dtype in real_numeric_dtypes
                                    and y.dtype in real_numeric_dtypes
                                )
                                or (dtypes == "numeric" and x.dtype in numeric_dtypes and y.dtype in numeric_dtypes)
                                or dtypes == "integer"
                                and x.dtype in integer_dtypes
                                and y.dtype in integer_dtypes
                                or dtypes == "integer_or_boolean"
                                and (
                                    x.dtype in integer_dtypes
                                    and y.dtype in integer_dtypes
                                    or x.dtype in boolean_dtypes
                                    and y.dtype in boolean_dtypes
                                )
                                or dtypes == "boolean"
                                and x.dtype in boolean_dtypes
                                and y.dtype in boolean_dtypes
                                or dtypes == "floating"
                                and x.dtype in floating_dtypes
                                and y.dtype in floating_dtypes
                            ):
                                getattr(x, _op)(y)
                            else:
                                pytest.raises(TypeError, lambda: getattr(x, _op)(y))
