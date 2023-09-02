import enum

from arrayfire.backend import _clib_wrapper as wrapper


class PointerSource(enum.Enum):
    """
    Source of the pointer.
    """

    device = 0  # gpu
    host = 1  # cpu


def get_device() -> int:  # FIXME
    return wrapper.get_device()


def sync(device_id: int) -> None:  # FIXME
    return wrapper.sync(device_id)


supported_devices = []
