"""
This file defines the types for type annotations.

These names aren't part of the module namespace, but they are used in the
annotations in the function signatures. The functions in the module are only
valid for inputs that match the given type annotations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import arrayfire as af

__all__ = [
    "Device",
    "SupportsDLPack",
    "SupportsBufferProtocol",
    "PyCapsule",
]

from typing import Any, Iterator, Protocol, TypeVar

_T_co = TypeVar("_T_co", covariant=True)


class NestedSequence(Protocol[_T_co]):
    def __getitem__(self, key: int, /) -> _T_co | NestedSequence[_T_co]: ...

    def __len__(self, /) -> int: ...

    def __iter__(self, /) -> Iterator[_T_co | NestedSequence[_T_co]]: ...


@dataclass
class Device:
    backend_type: af.BackendType
    device_id: int = 0

    @classmethod
    def use_default(cls) -> Device:
        _backend = af.get_backend()
        return cls(_backend.backend_type, af.get_device())

    def __post_init__(self) -> None:
        if not isinstance(self.backend_type, af.BackendType):
            raise ValueError("Bad backend type. Only support ones from af.BackendType.")

        if self.device_id < 0:
            raise ValueError("Device ID can not be lesser than 0")

        if self.device_id > af.get_device_count() - 1:
            raise ValueError("Device ID can not be higher than count of available devices.")

        if self.backend_type == af.BackendType.unified:
            raise ValueError(f"Uncompatible backend type '{self.backend_type.name}' with Array API.")

        if self.backend_type == af.BackendType.cpu and self.device_id != 0:
            raise ValueError(f"Device ID can not be greater than '{self.device_id}' with cpu backend.")


# TODO
# @dataclass
# class Device:
#     backend_type: BackendType
#     device_id: int

#     # TODO
#     def __post_init__(self) -> None:
#         # TODO
#         # Double check all unified mentions here and in wrapper and remove them completely
#         if self.backend_type == BackendType.unified:
#             raise ValueError("Unsupported backend type for Array API.")

#         if self.backend_type == BackendType.cpu and self.device_id != 0:
#             raise ValueError("Device ID cant not be greater than 0 for CPU.")

# Example 1:
# Device(BackendType.cuda, 1)

# Example 2:
# gpu_1 = Device("oneapi", 0)
# gpu_2 = Device("cuda", 1)
# cpu = Device("cpu", 0)

# aa = empty((2,3), dtype=float32, device=(Device.cuda, 1))
# bb = empty((2,3), dtype=float32, device=Device.CPU)
# aa + bb -> Error: bad devices


# class Device(enum.Enum):
#     CPU = enum.auto()
#     GPU = enum.auto()

#     def __repr__(self) -> str:
#         return str(self.value)

#     def __iter__(self) -> Iterator[Device]:
#         yield self


SupportsBufferProtocol = Any
PyCapsule = Any


class SupportsDLPack(Protocol):
    def __dlpack__(self, /, *, stream: None = ...) -> PyCapsule: ...


e = math.e
inf = math.inf
nan = math.nan
pi = math.pi
newaxis = None
