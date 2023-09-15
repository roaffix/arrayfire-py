"""
This file defines the types for type annotations.

These names aren't part of the module namespace, but they are used in the
annotations in the function signatures. The functions in the module are only
valid for inputs that match the given type annotations.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "Device",
    "SupportsDLPack",
    "SupportsBufferProtocol",
    "PyCapsule",
]

from typing import Any, Iterator, Protocol, TypeVar

_T_co = TypeVar("_T_co", covariant=True)


class NestedSequence(Protocol[_T_co]):
    def __getitem__(self, key: int, /) -> _T_co | NestedSequence[_T_co]:
        ...

    def __len__(self, /) -> int:
        ...


class Device(Enum):
    cpu = "cpu"
    gpu = "gpu"

    def __iter__(self) -> Iterator[Device]:
        yield self


SupportsBufferProtocol = Any
PyCapsule = Any


class SupportsDLPack(Protocol):
    def __dlpack__(self, /, *, stream: None = ...) -> PyCapsule:
        ...
