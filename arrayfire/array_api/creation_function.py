from typing import Optional, Union

from arrayfire import Array as AFArray
from arrayfire.dtypes import Dtype, supported_dtypes
from arrayfire.library.device import supported_devices

from .array_object import Array
from .constants import Device, NestedSequence, SupportsBufferProtocol


def asarray(
    obj: Union[Array, bool, int, float, complex, NestedSequence, SupportsBufferProtocol],
    /,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    copy: Optional[bool] = None,
) -> Array:
    if dtype not in supported_dtypes:
        raise ValueError(f"Unsupported dtype {dtype!r}")

    # if device not in supported_devices:
    #     raise ValueError(f"Unsupported device {device!r}")

    if dtype is None and isinstance(obj, int) and (obj > 2**64 or obj < -(2**63)):
        raise OverflowError("Integer out of bounds for array dtypes")

    array = AFArray(obj, dtype=dtype, device=device)
    return Array._new(array)
