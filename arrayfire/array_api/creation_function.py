from __future__ import annotations

from typing import Optional, Union

from arrayfire import Array as AFArray
from arrayfire.array_api.array_object import Array
from arrayfire.array_api.constants import Device, NestedSequence, SupportsBufferProtocol
from arrayfire.array_api.dtypes import all_dtypes
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
        pointer_source = PointerSource.host
    elif device == Device.gpu:
        pointer_source = PointerSource.device
    else:
        raise ValueError(f"Unsupported device {device!r}")

    if isinstance(obj, int | float):
        afarray = AFArray([obj], dtype=dtype, shape=(1,), pointer_source=pointer_source)
        return Array._new(afarray)

    afarray = AFArray(obj, dtype=dtype, pointer_source=pointer_source)
    return Array._new(afarray)
