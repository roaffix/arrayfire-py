__all__ = [
    "alloc_device",
    "alloc_host",
    "alloc_pinned",
    "device_gc",
    "device_info",
    "device_mem_info",
    "free_device",
    "free_host",
    "free_pinned",
    "get_dbl_support",
    "get_device",
    "get_device_count",
    "get_half_support",
    "get_kernel_cache_directory",
    "get_mem_step_size",
    "info",
    "info_string",
    "init",
    "print_mem_info",
    "set_device",
    "sync",
    "set_kernel_cache_directory",
    "set_mem_step_size",
]

import enum

from arrayfire_wrapper.lib import (
    alloc_device,
    alloc_host,
    device_gc,
    device_info,
    device_mem_info,
    get_kernel_cache_directory,
    get_mem_step_size,
    print_mem_info,
    set_kernel_cache_directory,
    set_mem_step_size,
    free_host,
    free_pinned,
    sync,
    free_device,
    get_device_count,
    info,
    init,
    info_string,
    get_dbl_support,
    get_half_support,
    alloc_pinned,
    get_device,
    set_device,
)


class PointerSource(enum.Enum):
    """
    Source of the pointer.
    """

    device = 0  # gpu
    host = 1  # cpu
