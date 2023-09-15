def round_to(list_: list[int | float | complex | bool], symbols: int = 3) -> list[int | float]:
    # HACK replace for e.g. abs(x1-x2) < 1e-6 ~ https://davidamos.dev/the-right-way-to-compare-floats-in-python/
    return [round(x, symbols) for x in list_]
