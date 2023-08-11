import pytest

from arrayfire.array_api import asarray
from arrayfire.array_api.array_object import Array
from arrayfire.array_api.constants import Device
from arrayfire.dtypes import float16


def test_asarray_errors() -> None:
    # Test various protections against incorrect usage
    pytest.raises(TypeError, lambda: Array([1]))
    pytest.raises(TypeError, lambda: asarray(["a"]))
    pytest.raises(ValueError, lambda: asarray([1.0], dtype=float16))
    pytest.raises(OverflowError, lambda: asarray(2**100))
    # pytest.raises(OverflowError, lambda: asarray([2**100]))  # FIXME
    asarray([1], device=Device.cpu)  # Doesn't error
    pytest.raises(ValueError, lambda: asarray([1], device="gpu"))  # type: ignore[arg-type]

    pytest.raises(ValueError, lambda: asarray([1], dtype=int))  # type: ignore[arg-type]
    pytest.raises(ValueError, lambda: asarray([1], dtype="i"))  # type: ignore[arg-type]
