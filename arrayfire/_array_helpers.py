from collections.abc import Callable

from typing_extensions import ParamSpec

from arrayfire import Array

P = ParamSpec("P")


def afarray_as_array(func: Callable[P, Array]) -> Callable[P, Array]:
    """
    Decorator that converts a function returning an array to return an ArrayFire Array.

    Parameters
    ----------
    func : Callable[P, Array]
        The original function that returns an array.

    Returns
    -------
    Callable[P, Array]
        A decorated function that returns an ArrayFire Array.
    """

    def decorated(*args: P.args, **kwargs: P.kwargs) -> Array:
        out = Array()
        result = func(*args, **kwargs)
        out.arr = result  # type: ignore[assignment]
        return out

    return decorated
