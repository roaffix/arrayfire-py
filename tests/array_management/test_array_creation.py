import pytest

from arrayfire.dtypes import int64
from arrayfire.library.array_management.creation import constant
from arrayfire.library.array_management.creation import range as afrange

# Test cases for the constant function


def test_constant_1d() -> None:
    result = constant(42, (5,))
    assert result.shape == (5,)
    assert result.scalar() == 42


def test_constant_2d() -> None:
    result = constant(3.14, (3, 4))
    assert result.shape == (3, 4)
    assert round(result.scalar(), 2) == 3.14  # type: ignore[arg-type]


def test_constant_3d() -> None:
    result = constant(0, (2, 2, 2), dtype=int64)
    assert result.shape == (2, 2, 2)
    assert result.scalar() == 0
    assert result.dtype == int64


def test_constant_default_shape() -> None:
    result = constant(1.0)
    assert result.shape == (1,)
    assert result.scalar() == 1.0


# TODO add error handling
# def test_constant_invalid_dtype() -> None:
#     with pytest.raises(ValueError):
#         constant(42, (3, 3), dtype="invalid_dtype")


# Test cases for the range function


def test_range_1d() -> None:
    result = afrange((5,))
    assert result.shape == (5,)


def test_range_2d() -> None:
    result = afrange((3, 4))
    assert result.shape == (3, 4)


def test_range_3d() -> None:
    result = afrange((2, 2, 2))
    assert result.shape == (2, 2, 2)


def test_range_with_axis() -> None:
    result = afrange((3, 4), axis=1)
    assert result.shape == (3, 4)


def test_range_with_dtype() -> None:
    result = afrange((4, 3), dtype=int64)
    assert result.dtype == int64


def test_range_with_invalid_axis() -> None:
    with pytest.raises(ValueError):
        afrange((2, 3, 4), axis=4)
