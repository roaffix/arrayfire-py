# flake8: noqa
from .version import ARRAYFIRE_VERSION, VERSION

__all__ = ["__version__"]
__version__ = VERSION

__all__ += ["__arrayfire_version__"]
__arrayfire_version__ = ARRAYFIRE_VERSION

__all__ += ["Array"]
from .library.array_object import Array

__all__ += [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "bool",
]
from .dtypes import (
    bool,
    complex64,
    complex128,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)

__all__ += [
    "get_backend",
    "get_active_backend",  # DeprecationWarning
    "get_array_backend_name",
    "get_array_device_id",
    "get_available_backends",  # DeprecationWarning
    "get_backend_count",
    "get_backend_id",  # DeprecationWarning
    "get_device_id",  # DeprecationWarning
    "get_dtype_size",
    "get_size_of",  # DeprecationWarning
    "set_backend",
]

from .backend import (
    get_backend,
    set_backend,
    get_available_backends,
    get_backend_count,
    get_backend_id,
    get_active_backend,
    get_array_backend_name,
    get_array_device_id,
    get_device_id,
    get_dtype_size,
    get_size_of,
)
