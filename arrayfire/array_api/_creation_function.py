from __future__ import annotations

from typing import List, Optional, Tuple, Union

from arrayfire import Array as AFArray
from arrayfire.array_api._array_object import Array
from arrayfire.array_api._constants import Device, NestedSequence, SupportsBufferProtocol
from arrayfire.array_api._dtypes import all_dtypes
from arrayfire.dtypes import Dtype
from arrayfire.library.device import PointerSource


def _check_valid_dtype(dtype: Optional[Dtype]) -> None:
    # Note: Only spelling dtypes as the dtype objects is supported.

    # We use this instead of "dtype in _all_dtypes" because the dtype objects
    # define equality with the sorts of things we want to disallow.
    for d in (None,) + all_dtypes:
        if dtype is d:
            return
    raise ValueError("dtype must be one of the supported dtypes")


def asarray(
    obj: Union[Array, bool, int, float, complex, NestedSequence, SupportsBufferProtocol],
    /,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    copy: Optional[bool] = None,
) -> Array:
    _check_valid_dtype(dtype)

    # if device not in supported_devices:
    #     raise ValueError(f"Unsupported device {device!r}")

    if dtype is None and isinstance(obj, int) and (obj > 2**64 or obj < -(2**63)):
        raise OverflowError("Integer out of bounds for array dtypes")

    if device == Device.cpu or device is None:
        to_device = False
    elif device == Device.gpu:
        to_device = True
    else:
        raise ValueError(f"Unsupported device {device!r}")

    if isinstance(obj, int | float):
        afarray = AFArray([obj], dtype=dtype, shape=(1,), to_device=to_device)
        return Array._new(afarray)

    afarray = AFArray(obj, dtype=dtype, to_device=to_device)
    return Array._new(afarray)


def arange(
    start: Union[int, float],
    /,
    stop: Optional[Union[int, float]] = None,
    step: Union[int, float] = 1,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    return NotImplemented


def empty(
    shape: Union[int, Tuple[int, ...]],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    return NotImplemented


def empty_like(x: Array, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None) -> Array:
    return NotImplemented


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    return NotImplemented


def full(
    shape: Union[int, Tuple[int, ...]],
    fill_value: Union[int, float],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    return NotImplemented


def full_like(
    x: Array,
    /,
    fill_value: Union[int, float],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    return NotImplemented


def linspace(
    start: Union[int, float],
    stop: Union[int, float],
    /,
    num: int,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    endpoint: bool = True,
) -> Array:
    return NotImplemented


def meshgrid(*arrays: Array, indexing: str = "xy") -> List[Array]:
    return NotImplemented


def ones(
    shape: Union[int, Tuple[int, ...]],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    return NotImplemented


def ones_like(x: Array, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None) -> Array:
    return NotImplemented


def tril(x: Array, /, *, k: int = 0) -> Array:
    return NotImplemented


def triu(x: Array, /, *, k: int = 0) -> Array:
    return NotImplemented


def zeros(
    shape: Union[int, Tuple[int, ...]],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    return NotImplemented


def zeros_like(x: Array, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None) -> Array:
    return NotImplemented
