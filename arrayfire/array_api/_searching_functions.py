from __future__ import annotations

import arrayfire as af

from ._array_object import Array


def argmax(x: Array, /, *, axis: int | None = None, keepdims: bool = False) -> Array:
    # TODO
    # [] Add documentation
    # [] Figure out what to do with keepdims arg that is not actually works well with af case
    # [] Fix typings
    # source: https://data-apis.org/array-api/2022.12/API_specification/generated/array_api.argmax.html#argmax

    if axis is None:
        flat_array = af.flat(x._array)
        _, indices = af.imax(flat_array, axis=0)
    else:
        _, indices = af.imax(x._array, axis=axis)

    if keepdims:
        shape = tuple([1] * x.ndim)
        indices = af.moddims(indices, shape)  # type: ignore[arg-type]  # FIXME

    return Array._new(indices)


def argmin(x: Array, /, *, axis: int | None = None, keepdims: bool = False) -> Array:
    # TODO
    # [] Add documentation
    # [] Figure out what to do with keepdims arg that is not actually works well with af case
    # [] Fix typings
    # source: https://data-apis.org/array-api/2022.12/API_specification/generated/array_api.argmin.html#argmin

    if axis is None:
        flat_array = af.flat(x._array)
        _, indices = af.imin(flat_array, axis=0)
    else:
        _, indices = af.imin(x._array, axis=axis)

    if keepdims:
        shape = tuple([1] * x.ndim)
        indices = af.moddims(indices, shape)  # type: ignore[arg-type]  # FIXME

    return Array._new(indices)


def nonzero(x: Array, /) -> tuple[Array, ...]:
    # TODO
    # Add documentation
    # source: https://data-apis.org/array-api/2022.12/API_specification/generated/array_api.nonzero.html#nonzero
    flat_array = af.flat(x._array)

    non_zero_indices = af.where(flat_array != 0)

    if len(x.shape) == 1:
        return (Array._new(non_zero_indices),)
    else:
        idx = []
        for dim in reversed(x.shape):
            idx.append(Array._new(non_zero_indices % dim))
            non_zero_indices = non_zero_indices // dim

        return tuple(reversed(idx))


def where(condition: Array, x1: Array, x2: Array, /) -> Array:
    # TODO
    # Add documentation
    # source: https://data-apis.org/array-api/2022.12/API_specification/generated/array_api.where.html#where
    return Array._new(af.select(x1._array, x2._array, condition._array))
