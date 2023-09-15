from dataclasses import dataclass
from typing import List, Tuple, Union

from arrayfire import Array as AFArray
from arrayfire.array_api._array_object import Array
from arrayfire.array_api._dtypes import all_dtypes, promote_types
from arrayfire.dtypes import Dtype


def astype(x: Array, dtype: Dtype, /, *, copy: bool = True) -> Array:
    return NotImplemented


def broadcast_arrays(*arrays: Array) -> List[Array]:
    return NotImplemented


def broadcast_to(x: Array, /, shape: Tuple[int, ...]) -> Array:
    return NotImplemented


def can_cast(from_: Union[Dtype, Array], to: Dtype, /) -> bool:
    return NotImplemented


@dataclass
class finfo_object:
    bits: int
    eps: float
    max: float
    min: float
    smallest_normal: float
    dtype: Dtype


@dataclass
class iinfo_object:
    bits: int
    max: int
    min: int
    dtype: Dtype


def finfo(type: Union[Dtype, Array], /) -> finfo_object:
    return NotImplemented


def iinfo(type: Union[Dtype, Array], /) -> iinfo_object:
    return NotImplemented


def isdtype(dtype: Dtype, kind: Union[Dtype, str, Tuple[Union[Dtype, str], ...]]) -> bool:
    return NotImplemented


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
