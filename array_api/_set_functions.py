from __future__ import annotations

from typing import NamedTuple

import arrayfire as af

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
    array_ = af.flat(x._array) if len(x.shape) != 1 else x._array

    sorted_array, original_indices = af.sort(array, is_ascending=True)
    unique_elements, unique_counts = af.set_unique(sorted_array, is_sorted=True)

    # Find indices of unique elements in the sorted array
    _, unique_indices_sorted = af.setunique(sorted_array, is_sorted=True, is_index=True)

    # Map indices in the sorted array back to original indices
    indices = af.gather(original_indices, unique_indices_sorted, 0)

    # Generate inverse indices
    _, inverse_indices = af.setunique(original_indices, is_sorted=False, is_index=True)

    # Counts of each unique element are directly obtained from setunique
    counts = unique_counts

    return UniqueAllResult(unique_elements, indices, inverse_indices, counts)


def unique_counts(x: Array, /) -> UniqueCountsResult:
    return NotImplemented


def unique_inverse(x: Array, /) -> UniqueInverseResult:
    return NotImplemented


def unique_values(x: Array, /) -> Array:
    return NotImplemented
