# flake8: noqa

__all__ = [
    "add",
    "sub",
    "mul",
    "div",
    "mod",
    "pow",
    "bitnot",
    "bitand",
    "bitor",
    "bitxor",
    "bitshiftl",
    "bitshiftr",
    "lt",
    "le",
    "gt",
    "ge",
    "eq",
    "neq",
]

from .operators import (
    add,
    bitand,
    bitnot,
    bitor,
    bitshiftl,
    bitshiftr,
    bitxor,
    div,
    eq,
    ge,
    gt,
    le,
    lt,
    mod,
    mul,
    neq,
    pow,
    sub,
)

__all__ += [
    "create_array",
    "create_handle",
    "create_strided_array",
    "device_array",
    "get_ctype",
    "get_elements",
    "get_numdims",
    "retain_array",
    "get_dims",
    "get_scalar",
    "is_empty",
    "get_data_ptr",
    "copy_array",
    "index_gen",
    "transpose",
    "reorder",
    "array_as_str",
    "where",
    "randu",
    "get_last_error",
    "set_backend",
    "get_backend_count",
    "get_device_id",
    "get_size_of",
    "get_backend_id",
]

from .unsorted import (
    array_as_str,
    copy_array,
    create_array,
    create_handle,
    create_strided_array,
    device_array,
    get_backend_count,
    get_backend_id,
    get_ctype,
    get_data_ptr,
    get_device_id,
    get_dims,
    get_elements,
    get_last_error,
    get_numdims,
    get_scalar,
    get_size_of,
    index_gen,
    is_empty,
    randu,
    reorder,
    retain_array,
    set_backend,
    transpose,
    where,
)

__all__ += ["safe_call"]

from .error_handler import safe_call

__all__ += ["count_all"]

from .reduction_operations import count_all

__all__ += ["create_constant_array"]

from .constant_array import create_constant_array
