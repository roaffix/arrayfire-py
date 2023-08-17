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


def set_backend(backend_type: Union[BackendType, str]) -> None:
    """
    Set a specific backend by backend_type name.

    Parameters
    ----------
    backend_type : Union[BackendType, str]
        Name of the backend backend_type to set.

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
