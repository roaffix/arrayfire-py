from __future__ import annotations

from typing import NamedTuple

from ._array_object import Array


class UniqueAllResult(NamedTuple):
    values: Array
    indices: Array
    inverse_indices: Array
    counts: Array


class UniqueCountsResult(NamedTuple):
    values: Array
    counts: Array


class UniqueInverseResult(NamedTuple):
    values: Array
    inverse_indices: Array


def unique_all(x: Array, /) -> UniqueAllResult:
    return NotImplemented


def unique_counts(x: Array, /) -> UniqueCountsResult:
    return NotImplemented


def unique_inverse(x: Array, /) -> UniqueInverseResult:
    return NotImplemented


def unique_values(x: Array, /) -> Array:
    return NotImplemented
