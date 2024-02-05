from __future__ import annotations

from ._array_object import Array


def argmax(x: Array, /, *, axis: int | None = None, keepdims: bool = False) -> Array:
    return NotImplemented


def argmin(x: Array, /, *, axis: int | None = None, keepdims: bool = False) -> Array:
    return NotImplemented


def nonzero(x: Array, /) -> tuple[Array, ...]:
    return NotImplemented


def where(condition: Array, x1: Array, x2: Array, /) -> Array:
    return NotImplemented
