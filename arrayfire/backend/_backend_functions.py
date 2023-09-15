from __future__ import annotations

import warnings
from enum import Enum
from typing import TYPE_CHECKING

from ._backend import Backend, BackendType, get_backend
from ._clib_wrapper._unsorted import cublas_set_math_mode
from ._clib_wrapper._unsorted import get_backend_count as c_get_backend_count
from ._clib_wrapper._unsorted import get_backend_id as c_get_backend_id
from ._clib_wrapper._unsorted import get_device_id as c_get_device_id
from ._clib_wrapper._unsorted import get_native_id as c_get_native_id
from ._clib_wrapper._unsorted import get_size_of as c_get_size_of
from ._clib_wrapper._unsorted import get_stream as c_get_stream
from ._clib_wrapper._unsorted import set_backend as c_set_backend
from ._clib_wrapper._unsorted import set_native_id as c_set_native_id

if TYPE_CHECKING:
    from arrayfire import Array
    from arrayfire.dtypes import Dtype


class CublasMathMode(Enum):
    default = 0
    tensor_op = 1


def set_backend(backend_type: BackendType | str) -> None:
    """
    Set a specific backend by backend_type name.

    Parameters
    ----------
    backend_type : BackendType | str
        Name of the backend type to set.

    Raises
    ------
    ValueError
        If the given backend_type name is not a valid name for backend backend_type.
    TypeError
        If the given backend_type is not a valid type for backend backend_type.
    RuntimeError
        If the given backend_type is already the active backend backend_type.
    RuntimeError
        If the given backend_type could not be set as new backend backend_type.
    """
    backend = get_backend()
    current_active_backend_type = backend.backend_type

    if isinstance(backend_type, str):
        if backend_type not in [d.name for d in BackendType]:
            raise ValueError(f"{backend_type} is not a valid name for backend backend_type.")
        backend_type = BackendType[backend_type]

    if not isinstance(backend_type, BackendType):
        raise TypeError(f"{backend_type} is not a valid type for backend backend_type.")

    if current_active_backend_type == backend_type:
        raise RuntimeError(f"{backend_type} is already the active backend backend_type.")

    if backend.backend_type == BackendType.unified:
        c_set_backend(backend_type.value)

    # NOTE keep in mind that this operation works in-place
    # FIXME should not access private API
    backend._load_backend_lib(backend_type)

    if current_active_backend_type == backend.backend_type:
        raise RuntimeError(f"Could not set {backend_type} as new backend backend_type. Consider checking logs.")


def get_array_backend_name(array: Array) -> str:
    """
    Get the name of the backend on which the Array is located.

    Parameters
    ----------
    array : Array
        The Array to get the backend name of.

    Returns
    -------
    value : str
        Name of the backend on which the Array is located.
    """

    id_ = c_get_backend_id(array.arr)
    return BackendType(id_).name


def get_backend_id(array: Array) -> str:
    warnings.warn("Was renamed. Now get_array_backend_name() in main repo.", DeprecationWarning)
    return get_array_backend_name(array)


def get_backend_count() -> int:
    """
    Get a number of available backends.

    Returns
    -------

    value : int
        Number of available backends.
    """

    return c_get_backend_count()


def get_active_backend() -> Backend:
    """
    Get the current active backend.

    value : Backend
        Current active backend.
    """

    # TODO do not deprecate
    warnings.warn("A user has access explicitly only to the active backend.", DeprecationWarning)
    return get_backend()


def get_available_backends() -> Backend:
    """
    Get the list of available backends.

    Returns
    -------
    value : Backend
        Current active backend.
    """

    # TODO do not deprecate
    warnings.warn(
        "A user has access explicitly only to the active backend. Thus returning only active backend.",
        DeprecationWarning,
    )
    return get_active_backend()


def get_array_device_id(array: Array) -> int:
    """
    Get the id of the device on which the Array was created.

    Parameters
    ----------
    array : Array
        The Array to get the device id of.

    Returns
    -------
    value : int
        The id of the device on which the Array was created.
    """

    return c_get_device_id(array.arr)


def get_device_id(array: Array) -> int:
    warnings.warn("Was renamed due to unintuitive function name. Now get_array_device_id().", DeprecationWarning)
    return get_array_device_id(array)


def get_dtype_size(dtype: Dtype) -> int:
    """
    Get the size of the type represented by Dtype.

    Parameters
    ----------
    dtype : Dtype
        The type to get the size of.

    Returns
    -------
    value : int
        The size of the type in bytes.
    """

    return c_get_size_of(dtype)


def get_size_of(dtype: Dtype) -> int:
    warnings.warn("Was renamed due to unintuitive function name. Now get_dtype_size().", DeprecationWarning)
    return get_dtype_size(dtype)


# Previously module arrayfire.cuda


def _check_if_cuda_used() -> None:
    backend = get_backend()
    if backend.backend_type != BackendType.cuda:
        raise RuntimeError(
            f"Can not get the CUDA stream id because the other backend is in use: {backend.backend_type}."
        )


def get_stream(index: int) -> int:
    warnings.warn("Was renamed due to unintuitive function name. Now get_cuda_stream().", DeprecationWarning)
    return get_cuda_stream(index)


def get_cuda_stream(index: int) -> int:
    """
    Get the CUDA stream used for the device id by ArrayFire.

    Parameters
    ----------
    idx : int
        Specifies the index of the device.

    Returns
    -------
    value : int
        Denoting the stream id.

    Raises
    ------
    RuntimeError
        If the current backend type is not CUDA.
    """
    _check_if_cuda_used()

    return c_get_stream(index)


def get_native_id(index: int) -> int:
    warnings.warn("Was renamed due to unintuitive function name. Now get_native_cuda_id().", DeprecationWarning)
    return get_native_cuda_id(index)


def get_native_cuda_id(index: int) -> int:
    """
    Get native (unsorted) CUDA device id.

    Parameters
    ----------
    idx : int
        Specifies the (sorted) index of the device.

    Returns
    -------
    value : int
        Denoting the native cuda id.

    Raises
    ------
    RuntimeError
        If the current backend type is not CUDA.
    """
    _check_if_cuda_used()

    return c_get_native_id(index)


def set_native_id(index: int) -> None:
    warnings.warn("Was renamed due to unintuitive function name. Now get_native_cuda_id().", DeprecationWarning)
    return set_native_cuda_id(index)


def set_native_cuda_id(index: int) -> None:
    """
    Set native (unsorted) CUDA device id.

    Parameters
    ----------
    idx : int
        Specifies the (unsorted) native index of the device.

    Raises
    ------
    RuntimeError
        If the current backend type is not CUDA.
    """
    _check_if_cuda_used()

    return c_set_native_id(index)


def set_cublas_mode(mode: CublasMathMode | int = CublasMathMode.default) -> None:
    """
    Set cuBLAS math mode for CUDA backend. It enables the Tensor Core usage if available on CUDA backend GPUs.

    Parameters
    ----------
    mode : CublasMathMode | int
        Specify the mode available within CublasMathMode enum.

    Raises
    ------
    ValueError
        If the given math mode int value is not a valid value for cuBLAS math mode.
    RuntimeError
        If the current backend type is not CUDA.
    """
    if isinstance(mode, int):
        if mode not in [m.value for m in CublasMathMode]:
            raise ValueError(f"{mode} is not supported as cublas math mode.")
        mode = CublasMathMode(mode)

    _check_if_cuda_used()

    return cublas_set_math_mode(mode.value)
