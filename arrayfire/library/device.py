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

from arrayfire_wrapper.lib import (
    alloc_device,
    alloc_host,
    alloc_pinned,
    device_gc,
    device_info,
    device_mem_info,
    free_device,
    free_host,
    free_pinned,
    get_dbl_support,
    get_device,
    get_device_count,
    get_half_support,
    get_kernel_cache_directory,
    get_mem_step_size,
    info,
    info_string,
    init,
    print_mem_info,
    set_device,
    set_kernel_cache_directory,
    set_mem_step_size,
)
from arrayfire_wrapper.lib import sync as wrapper_sync


def sync(device_id: int | None = None) -> None:
    """
    Blocks until all the functions on the specified device have completed execution.

    This function is used to synchronize the program execution with the operations
    being carried out on a GPU or other computation device, ensuring that all
    previously submitted operations are complete before the program proceeds.

    Parameters
    ----------
    device_id : int | None, optional
        The ID of the device on which to wait for all operations to complete.
        If None is provided, the current active device is used. Default is None.
    """
    if device_id is None:
        device_id = get_device()

    wrapper_sync(device_id)
