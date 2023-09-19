from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Any
from typing import cast as typing_cast

from arrayfire.backend._backend import _backend
from arrayfire.dtypes import CShape, CType, Dtype, c_dim_t, to_str
from arrayfire.library.device import PointerSource

from ._base import AFArrayType
from ._error_handler import safe_call

if TYPE_CHECKING:
    from ._base import _ArrayBuffer

# Array management


def create_handle(shape: tuple[int, ...], dtype: Dtype, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga3b8f5cf6fce69aa1574544bc2d44d7d0
    """
    out = AFArrayType.create_pointer()
    c_shape = CShape(*shape)

    safe_call(
        _backend.clib.af_create_handle(
            ctypes.pointer(out), c_shape.original_shape, ctypes.pointer(c_shape.c_array), dtype.c_api_value
        )
    )
    return out


def retain_array(arr: AFArrayType) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga7ed45b3f881c0f6c80c5cf2af886dbab
    """
    out = AFArrayType.create_pointer()

    safe_call(_backend.clib.af_retain_array(ctypes.pointer(out), arr))
    return out


def create_array(shape: tuple[int, ...], dtype: Dtype, array_buffer: _ArrayBuffer, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga834be32357616d8ab735087c6f681858
    """
    out = AFArrayType.create_pointer()
    c_shape = CShape(*shape)

    safe_call(
        _backend.clib.af_create_array(
            ctypes.pointer(out),
            ctypes.c_void_p(array_buffer.address),
            c_shape.original_shape,
            ctypes.pointer(c_shape.c_array),
            dtype.c_api_value,
        )
    )
    return out


def device_array(shape: tuple[int, ...], dtype: Dtype, array_buffer: _ArrayBuffer, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#gaad4fc77f872217e7337cb53bfb623cf5
    """
    out = AFArrayType.create_pointer()
    c_shape = CShape(*shape)

    safe_call(
        _backend.clib.af_device_array(
            ctypes.pointer(out),
            ctypes.c_void_p(array_buffer.address),
            c_shape.original_shape,
            ctypes.pointer(c_shape.c_array),
            dtype.c_api_value,
        )
    )
    return out


def create_strided_array(
    shape: tuple[int, ...],
    dtype: Dtype,
    array_buffer: _ArrayBuffer,
    offset: CType,
    strides: tuple[int, ...],
    pointer_source: PointerSource,
    /,
) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__internal__func__create.htm#gad31241a3437b7b8bc3cf49f85e5c4e0c
    """
    out = AFArrayType.create_pointer()
    c_shape = CShape(*shape)

    if offset is None:
        offset = c_dim_t(0)

    if strides is None:
        strides = (1, c_shape[0], c_shape[0] * c_shape[1], c_shape[0] * c_shape[1] * c_shape[2])

    if len(strides) < 4:
        strides += (strides[-1],) * (4 - len(strides))

    safe_call(
        _backend.clib.af_create_strided_array(
            ctypes.pointer(out),
            ctypes.c_void_p(array_buffer.address),
            offset,
            c_shape.original_shape,
            ctypes.pointer(c_shape.c_array),
            CShape(*strides).c_array,
            dtype.c_api_value,
            pointer_source.value,
        )
    )
    return out


def get_ctype(arr: AFArrayType) -> int:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga0dda6898e1c0d9a43efb56cd6a988c9b
    """
    out = ctypes.c_int(0)

    safe_call(_backend.clib.af_get_type(ctypes.pointer(out), arr))
    return out.value


def get_elements(arr: AFArrayType) -> int:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga6845bbe4385a60a606b88f8130252c1f
    """
    out = c_dim_t(0)

    safe_call(_backend.clib.af_get_elements(ctypes.pointer(out), arr))
    return out.value


def get_numdims(arr: AFArrayType) -> int:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#gaefa019d932ff58c2a829ce87edddd2a8
    """
    out = ctypes.c_uint(0)

    safe_call(_backend.clib.af_get_numdims(ctypes.pointer(out), arr))
    return out.value


def get_dims(arr: AFArrayType) -> tuple[int, ...]:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga8b90da50a532837d9763e301b2267348
    """
    d0 = c_dim_t(0)
    d1 = c_dim_t(0)
    d2 = c_dim_t(0)
    d3 = c_dim_t(0)

    safe_call(
        _backend.clib.af_get_dims(ctypes.pointer(d0), ctypes.pointer(d1), ctypes.pointer(d2), ctypes.pointer(d3), arr)
    )
    return (d0.value, d1.value, d2.value, d3.value)


def get_strides(arr: AFArrayType) -> tuple[int, ...]:
    """
    source: https://arrayfire.org/docs/group__internal__func__strides.htm#gaff91b376156ce0ad7180af6e68faab51
    """
    s0 = c_dim_t(0)
    s1 = c_dim_t(0)
    s2 = c_dim_t(0)
    s3 = c_dim_t(0)
    safe_call(
        _backend.clib.af_get_strides(
            ctypes.pointer(s0), ctypes.pointer(s1), ctypes.pointer(s2), ctypes.pointer(s3), arr
        )
    )
    return (s0.value, s1.value, s2.value, s3.value)


def get_scalar(arr: AFArrayType, dtype: Dtype, /) -> int | float | complex | bool | None:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#gaefe2e343a74a84bd43b588218ecc09a3
    """
    out = dtype.c_type()
    safe_call(_backend.clib.af_get_scalar(ctypes.pointer(out), arr))
    return typing_cast(int | float | complex | bool | None, out.value)


def is_empty(arr: AFArrayType) -> bool:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga19c749e95314e1c77d816ad9952fb680
    """
    out = ctypes.c_bool()
    safe_call(_backend.clib.af_is_empty(ctypes.pointer(out), arr))
    return out.value


def get_data_ptr(arr: AFArrayType, size: int, dtype: Dtype, /) -> ctypes.Array:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga6040dc6f0eb127402fbf62c1165f0b9d
    """
    c_shape = dtype.c_type * size
    ctypes_array = c_shape()
    safe_call(_backend.clib.af_get_data_ptr(ctypes.pointer(ctypes_array), arr))
    return ctypes_array


def copy_array(arr: AFArrayType) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga6040dc6f0eb127402fbf62c1165f0b9d
    """
    out = AFArrayType.create_pointer()
    safe_call(_backend.clib.af_copy_array(ctypes.pointer(out), arr))
    return out


def cast(arr: AFArrayType, dtype: Dtype, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__arith__func__cast.htm#gab0cb307d6f9019ac8cbbbe9b8a4d6b9b
    """
    out = AFArrayType.create_pointer()
    safe_call(_backend.clib.af_cast(ctypes.pointer(out), arr, dtype.c_api_value))
    return out


# Arrayfire Functions


def index_gen(
    arr: AFArrayType,
    ndims: int,
    indices: Any,
    /,
) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__index__func__index.htm#ga14a7d149dba0ed0b977335a3df9d91e6
    """
    out = AFArrayType.create_pointer()
    safe_call(_backend.clib.af_index_gen(ctypes.pointer(out), arr, c_dim_t(ndims), indices.pointer))
    return out


def transpose(arr: AFArrayType, conjugate: bool, /) -> AFArrayType:
    """
    https://arrayfire.org/docs/group__blas__func__transpose.htm#ga716b2b9bf190c8f8d0970aef2b57d8e7
    """
    out = AFArrayType.create_pointer()
    safe_call(_backend.clib.af_transpose(ctypes.pointer(out), arr, conjugate))
    return out


def reorder(arr: AFArrayType, ndims: int, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__manip__func__reorder.htm#ga57383f4d00a3a86eab08dddd52c3ad3d
    """
    out = AFArrayType.create_pointer()
    c_shape = CShape(*(tuple(reversed(range(ndims))) + tuple(range(ndims, 4))))
    safe_call(_backend.clib.af_reorder(ctypes.pointer(out), arr, *c_shape))
    return out


def array_as_str(arr: AFArrayType) -> str:
    """
    source:
    - https://arrayfire.org/docs/group__print__func__tostring.htm#ga01f32ef2420b5d4592c6e4b4964b863b
    - https://arrayfire.org/docs/group__device__func__free__host.htm#ga3f1149a837a7ebbe8002d5d2244e3370
    """
    arr_str = ctypes.c_char_p(0)
    safe_call(_backend.clib.af_array_to_string(ctypes.pointer(arr_str), "", arr, 4, True))
    py_str = to_str(arr_str)
    safe_call(_backend.clib.af_free_host(arr_str))
    return py_str


def where(arr: AFArrayType) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__scan__func__where.htm#gafda59a3d25d35238592dd09907be9d07
    """
    out = AFArrayType.create_pointer()
    safe_call(_backend.clib.af_where(ctypes.pointer(out), arr))
    return out


def af_range(shape: tuple[int, ...], axis: int, dtype: Dtype, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__data__func__range.htm#gadd6c9b479692454670a51e00ea5b26d5
    """
    out = AFArrayType.create_pointer()
    c_shape = CShape(*shape)
    safe_call(_backend.clib.af_range(ctypes.pointer(out), 4, c_shape.c_array, axis, dtype.c_api_value))
    return out


def identity(shape: tuple[int, ...], dtype: Dtype, /) -> AFArrayType:
    """
    source:
    """
    out = AFArrayType.create_pointer()
    c_shape = CShape(*shape)
    safe_call(_backend.clib.af_identity(ctypes.pointer(out), 4, c_shape.c_array, dtype.c_api_value))
    return out


def flat(arr: AFArrayType, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__manip__func__flat.htm#gac6dfb22cbd3b151ddffb9a4ddf74455e
    """
    out = AFArrayType.create_pointer()
    safe_call(_backend.clib.af_flat(ctypes.pointer(out), arr))
    return out


def assign_gen(lhs: AFArrayType, rhs: AFArrayType, ndims: int, indices: Any, /) -> AFArrayType:
    """
    source: https://arrayfire.org/docs/group__index__func__assign.htm#ga93cd5199c647dce0e3b823f063b352ae
    """
    out = AFArrayType.create_pointer()
    safe_call(_backend.clib.af_assign_gen(ctypes.pointer(out), lhs, ndims, indices.pointer, rhs))
    return out


def release_array(arr: AFArrayType, /) -> None:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#gad6c58648ed0db398e170dabf045e8309
    """
    safe_call(_backend.clib.af_release_array(arr))


def get_offset(arr: AFArrayType, /) -> int:
    """
    source: https://arrayfire.org/docs/group__internal__func__offset.htm#ga303cb334026bdb5cab86e038951d6a5a
    """
    out = c_dim_t(0)
    safe_call(_backend.clib.af_get_offset(ctypes.pointer(out), arr))
    return out.value


# Safe Call Wrapper


def get_last_error() -> ctypes.c_char_p:
    """
    source: https://arrayfire.org/docs/exception_8h.htm#a4f0227c17954d343021313f77e695c8e
    """
    out = ctypes.c_char_p(0)
    _backend.clib.af_get_last_error(ctypes.pointer(out), ctypes.pointer(c_dim_t(0)))
    return out


# Device


# FIXME
def sync(device_id: int) -> None:
    """
    source: https://arrayfire.org/docs/group__device__func__sync.htm#ga9dbc7f1e99d70170ad567c480b6ddbde
    """
    safe_call(_backend.clib.af_sync(device_id))


def get_device() -> int:
    """
    source: https://arrayfire.org/docs/group__device__func__set.htm#ga54120b126cfcb1b0b3ee25e0fc66b8a4
    """
    out = ctypes.c_int(0)
    safe_call(_backend.clib.af_get_device(ctypes.pointer(out)))
    return out.value


# Backend


def set_backend(backend_c_value: int, /) -> None:
    """
    source: https://arrayfire.org/docs/group__unified__func__setbackend.htm#ga6fde820e8802776b7fc823504b37f1b4
    """
    safe_call(_backend.clib.af_set_backend(backend_c_value))
    return None


def get_backend_count() -> int:
    """
    source: https://arrayfire.org/docs/group__unified__func__getbackendcount.htm#gad38c2dfedfdabfa264afa46d8664e9cd
    """
    out = ctypes.c_int(0)
    safe_call(_backend.clib.get().af_get_backend_count(ctypes.pointer(out)))
    return out.value


def get_device_id(arr: AFArrayType, /) -> int:
    """
    source: https://arrayfire.org/docs/group__unified__func__getdeviceid.htm#ga5d94b64dccd1c7cbc7a3a69fa64888c3
    """
    out = ctypes.c_int(0)
    safe_call(_backend.clib.get().af_get_device_id(ctypes.pointer(out), arr))
    return out.value


def get_size_of(dtype: Dtype, /) -> int:
    """
    source: https://arrayfire.org/docs/util_8h.htm#a8b72cffd10a92a7a2ee7f52dadda5216
    """
    out = ctypes.c_size_t(0)
    safe_call(_backend.clib.get().af_get_size_of(ctypes.pointer(out), dtype.c_api_value))
    return out.value


def get_backend_id(arr: AFArrayType, /) -> int:
    """
    source: https://arrayfire.org/docs/group__unified__func__getbackendid.htm#ga5fc39e209e1886cf250aec265c0d9079
    """
    out = ctypes.c_int(0)
    safe_call(_backend.clib.get().af_get_backend_id(ctypes.pointer(out), arr))
    return out.value


# Cuda specific


def get_stream(index: int) -> int:
    """
    source: https://arrayfire.org/docs/group__cuda__mat.htm#ga8323b850f80afe9878b099f647b0a7e5
    """
    out = AFArrayType.create_pointer()
    safe_call(_backend.clib.get().afcu_get_stream(ctypes.pointer(out), index))
    return out.value


def get_native_id(index: int) -> int:
    """
    source: https://arrayfire.org/docs/group__cuda__mat.htm#gaf38af1cbbf4be710cc8cbd95d20b24c4
    """
    out = ctypes.c_int(0)
    safe_call(_backend.clib.get().afcu_get_native_id(ctypes.pointer(out), index))
    return out.value


def set_native_id(index: int) -> None:
    """
    source: https://arrayfire.org/docs/group__cuda__mat.htm#ga966f4c6880e90ce91d9599c90c0db378
    """
    safe_call(_backend.clib.get().afcu_set_native_id(index))
    return None


def cublas_set_math_mode(mode: int) -> None:
    """
    source: https://arrayfire.org/docs/group__cuda__mat.htm#gac23ea38f0bff77a0e12555f27f47aa4f
    """
    safe_call(_backend.clib.get().afcu_cublasSetMathMode(mode))
    return None
