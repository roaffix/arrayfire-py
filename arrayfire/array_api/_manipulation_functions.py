from __future__ import annotations

from typing import List, Optional, Tuple, Union

from ._array_object import Array


def concat(arrays: Union[Tuple[Array, ...], List[Array]], /, *, axis: Optional[int] = 0) -> Array:
    return NotImplemented


def expand_dims(x: Array, /, *, axis: int) -> Array:
    return NotImplemented


def flip(x: Array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Array:
    return NotImplemented


def permute_dims(x: Array, /, axes: Tuple[int, ...]) -> Array:
    return NotImplemented


def reshape(x: Array, /, shape: Tuple[int, ...], *, copy: Optional[bool] = None) -> Array:
    return NotImplemented


def roll(
    x: Array,
    /,
    shift: Union[int, Tuple[int, ...]],
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
) -> Array:
    return NotImplemented


def squeeze(x: Array, /, axis: Union[int, Tuple[int, ...]]) -> Array:
    return NotImplemented


def stack(arrays: Union[Tuple[Array, ...], List[Array]], /, *, axis: int = 0) -> Array:
    return NotImplemented
