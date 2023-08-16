from __future__ import annotations

from typing import Optional, Tuple, Union

from ._array_object import Array


def all(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    return NotImplemented


def any(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    return NotImplemented
