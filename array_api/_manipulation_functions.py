from __future__ import annotations

from ._array_object import Array


def concat(arrays: tuple[Array, ...] | list[Array], /, *, axis: int | None = 0) -> Array:
    return NotImplemented


def expand_dims(x: Array, /, *, axis: int) -> Array:
    return NotImplemented


def flip(x: Array, /, *, axis: int | tuple[int, ...] | None = None) -> Array:
    return NotImplemented


def permute_dims(x: Array, /, axes: tuple[int, ...]) -> Array:
    return NotImplemented


def reshape(x: Array, /, shape: tuple[int, ...], *, copy: bool | None = None) -> Array:
    return NotImplemented


def roll(
    x: Array,
    /,
    shift: int | tuple[int, ...],
    *,
    axis: int | tuple[int, ...] | None = None,
) -> Array:
    return NotImplemented


def squeeze(x: Array, /, axis: int | tuple[int, ...]) -> Array:
    return NotImplemented


def stack(arrays: tuple[Array, ...] | list[Array], /, *, axis: int = 0) -> Array:
    return NotImplemented
