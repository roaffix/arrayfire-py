import arrayfire as af

from ._array_object import Array


def take(x: Array, indices: Array, /, *, axis: int | None = None) -> Array:
    # TODO
    # Add documentation
    # source: https://data-apis.org/array-api/2022.12/API_specification/generated/array_api.take.html#take

    if axis is None:
        flat_array = af.flat(x._array)
        return Array._new(af.lookup(flat_array, indices._array))

    if axis != 0:
        shape = (x._array.size,)
        afarray = af.moddims(x._array, shape)

    return Array._new(af.lookup(afarray, indices._array, axis=axis))
