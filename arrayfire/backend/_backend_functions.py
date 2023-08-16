from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Union

from ._backend import Backend, BackendType, get_backend
from ._clib_wrapper._unsorted import get_backend_count as c_get_backend_count
from ._clib_wrapper._unsorted import get_backend_id as c_get_backend_id
from ._clib_wrapper._unsorted import get_device_id as c_get_device_id
from ._clib_wrapper._unsorted import get_size_of as c_get_size_of
from ._clib_wrapper._unsorted import set_backend as c_set_backend

if TYPE_CHECKING:
    from arrayfire import Array
    from arrayfire.dtypes import Dtype


def set_backend(platform: Union[BackendType, str]) -> None:
    """
    Set a specific backend by platform name.

    Parameters
    ----------
    platform : Union[BackendType, str]
        Name of the backend platform to set.

    Raises
    ------
    ValueError
        If the given platform name is not a valid name for backend platform.
    TypeError
        If the given platform is not a valid type for backend platform.
    RuntimeError
        If the given platform is already the active backend platform.
    RuntimeError
        If the given platform could not be set as new backend platform.
    """
    backend = get_backend()
    current_active_platform = backend.platform

    if isinstance(platform, str):
        if platform not in [d.name for d in BackendType]:
            raise ValueError(f"{platform} is not a valid name for backend platform.")
        platform = BackendType[platform]

    if not isinstance(platform, BackendType):
        raise TypeError(f"{platform} is not a valid type for backend platform.")

    if current_active_platform == platform:
        raise RuntimeError(f"{platform} is already the active backend platform.")

    if backend.platform == BackendType.unified:
        c_set_backend(platform.value)

    # NOTE keep in mind that this operation works in-place
    # FIXME should not access private API
    backend._load_backend_lib(platform)

    if current_active_platform == backend.platform:
        raise RuntimeError(f"Could not set {platform} as new backend platform. Consider checking logs.")


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
    warnings.warn("Was renamed due to unintuitive function name. Now get_array_backend_name().", DeprecationWarning)
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
