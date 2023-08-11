from typing import Union

from arrayfire import Array as AFArray
from arrayfire.array_api.array_object import Array
from arrayfire.array_api.dtypes import all_dtypes, promote_types
from arrayfire.dtypes import Dtype


def result_type(*arrays_and_dtypes: Union[Array, Dtype]) -> Dtype:
    """
    Array API compatible wrapper for :py:func:`np.result_type <numpy.result_type>`.

    See its docstring for more information.
    """
    # Note: we use a custom implementation that gives only the type promotions
    # required by the spec rather than using np.result_type. NumPy implements
    # too many extra type promotions like int64 + uint64 -> float64, and does
    # value-based casting on scalar arrays.
    A = []
    for a in arrays_and_dtypes:
        if isinstance(a, Array):
            a = a.dtype
        elif isinstance(a, AFArray) or a not in all_dtypes:
            raise TypeError("result_type() inputs must be array_api arrays or dtypes")
        A.append(a)

    if len(A) == 0:
        raise ValueError("at least one array or dtype is required")
    elif len(A) == 1:
        return A[0]
    else:
        t = A[0]
        for t2 in A[1:]:
            t = promote_types(t, t2)
        return t
