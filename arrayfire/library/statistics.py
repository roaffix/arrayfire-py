__all__ = ["corrcoef", "cov", "mean", "median", "stdev", "topk", "var"]

from typing import cast, overload

import arrayfire_wrapper.lib as wrapper

from arrayfire import Array
from arrayfire.array_object import afarray_as_array
from arrayfire.library.constants import TopK, VarianceBias

# TODO
# Add missing documentation


def corrcoef(x: Array, y: Array, /) -> int | float | complex:
    return wrapper.corrcoef(x.arr, y.arr)


@afarray_as_array
def cov(x: Array, y: Array, /, *, bias: VarianceBias = VarianceBias.DEFAULT) -> Array:
    return cast(Array, wrapper.cov(x.arr, y.arr, bias))


@overload
def mean(x: Array, /, axis: None = None, *, weights: None = None) -> int | float | complex: ...


@overload
def mean(x: Array, /, axis: int, *, weights: None = None) -> Array: ...


@overload
def mean(x: Array, /, axis: None, *, weights: Array) -> int | float | complex: ...


@overload
def mean(x: Array, /, axis: int, *, weights: Array) -> Array: ...


def mean(x: Array, /, axis: None | int = None, *, weights: None | Array = None) -> int | float | complex | Array:
    if weights:
        if axis is None:
            return wrapper.mean_all_weighted(x.arr, weights.arr)

        return Array.from_afarray(wrapper.mean_weighted(x.arr, weights.arr, axis))

    if axis is None:
        return wrapper.mean_all(x.arr)

    return Array.from_afarray(wrapper.mean(x.arr, axis))


@overload
def median(x: Array, /, axis: None = None) -> int | float | complex: ...


@overload
def median(x: Array, /, axis: int) -> Array: ...


def median(x: Array, /, axis: None | int = None) -> int | float | complex | Array:
    if axis is None:
        return wrapper.median_all(x.arr)

    return Array.from_afarray(wrapper.median(x.arr, axis))


@overload
def stdev(x: Array, /, axis: None = None, *, bias: VarianceBias = VarianceBias.DEFAULT) -> int | float | complex: ...


@overload
def stdev(x: Array, /, axis: int, *, bias: VarianceBias = VarianceBias.DEFAULT) -> int | float | complex: ...


def stdev(
    x: Array, /, axis: None | int = None, *, bias: VarianceBias = VarianceBias.DEFAULT
) -> int | float | complex | Array:
    if axis is None:
        return wrapper.stdev_all(x.arr, bias)

    return Array.from_afarray(wrapper.stdev(x.arr, axis, bias))


def topk(x: Array, k: int, /, *, axis: int = 0, order: TopK = TopK.DEFAULT) -> tuple[Array, Array]:
    values, indices = wrapper.topk(x.arr, k, axis, order)
    return Array.from_afarray(values), Array.from_afarray(indices)


@overload
def var(
    x: Array, /, axis: None = None, *, weights: None = None, bias: VarianceBias = VarianceBias.DEFAULT
) -> int | float | complex: ...


@overload
def var(x: Array, /, axis: int, *, weights: None = None, bias: VarianceBias = VarianceBias.DEFAULT) -> Array: ...


@overload
def var(
    x: Array, /, axis: None, *, weights: Array, bias: VarianceBias = VarianceBias.DEFAULT
) -> int | float | complex: ...


@overload
def var(x: Array, /, axis: int, *, weights: Array, bias: VarianceBias = VarianceBias.DEFAULT) -> Array: ...


def var(
    x: Array,
    /,
    axis: None | int = None,
    *,
    weights: None | Array = None,
    bias: VarianceBias = VarianceBias.DEFAULT,
) -> int | float | complex | Array:
    if weights:
        if axis is None:
            return wrapper.var_all_weighted(x.arr, weights.arr)

        return Array.from_afarray(wrapper.var_weighted(x.arr, weights.arr, axis))

    if axis is None:
        return wrapper.var_all(x.arr, bias)

    return Array.from_afarray(wrapper.var(x.arr, axis, bias))
