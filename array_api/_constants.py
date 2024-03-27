"""
This file defines the types for type annotations.

These names aren't part of the module namespace, but they are used in the
annotations in the function signatures. The functions in the module are only
valid for inputs that match the given type annotations.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from arrayfire_wrapper import BackendType

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


class Device(enum.Enum):
    CPU = enum.auto()
    GPU = enum.auto()

    def __repr__(self) -> str:
        return str(self.value)

    def __iter__(self) -> Iterator[Device]:
        yield self


SupportsBufferProtocol = Any
PyCapsule = Any


class SupportsDLPack(Protocol):
    def __dlpack__(self, /, *, stream: None = ...) -> PyCapsule: ...
