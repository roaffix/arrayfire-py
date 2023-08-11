import enum


class PointerSource(enum.Enum):
    """
    Source of the pointer.
    """

    device = 0  # gpu
    host = 1  # cpu


supported_devices = []
