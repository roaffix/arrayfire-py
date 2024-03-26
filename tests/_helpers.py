import arrayfire as af


def round_to(list_: list[int | float | complex | bool], symbols: int = 3) -> list[int | float]:
    # HACK replace for e.g. abs(x1-x2) < 1e-6 ~ https://davidamos.dev/the-right-way-to-compare-floats-in-python/
    return [round(x, symbols) for x in list_]


def create_from_2d_nested(x1: float, x2: float, x3: float, x4: float, dtype: af.Dtype = af.float32) -> af.Array:
    array = af.randu((2, 2), dtype=dtype)
    array[0, 0] = x1
    array[0, 1] = x2
    array[1, 0] = x3
    array[1, 1] = x4
    return array
