"""
This file defines the types for type annotations.

These names aren't part of the module namespace, but they are used in the
annotations in the function signatures. The functions in the module are only
valid for inputs that match the given type annotations.
"""

from __future__ import annotations

__all__ = [
    "Array",
    "Device",
    "SupportsDLPack",
    "SupportsBufferProtocol",
    "PyCapsule",
]

from typing import Any, Literal, Protocol, TypeVar
from .array_object import Array

_T_co = TypeVar("_T_co", covariant=True)


class NestedSequence(Protocol[_T_co]):
    def __getitem__(self, key: int, /) -> _T_co | NestedSequence[_T_co]:
        ...

    def __len__(self, /) -> int:
        ...


Device = Literal["cpu"]  # FIXME: add support for other devices
SupportsBufferProtocol = Any
PyCapsule = Any


class SupportsDLPack(Protocol):
    def __dlpack__(self, /, *, stream: None = ...) -> PyCapsule:
        ...
