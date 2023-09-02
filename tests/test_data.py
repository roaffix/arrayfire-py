import pytest

from arrayfire import Array
from arrayfire.dtypes import int64
from arrayfire.library import data, random
from arrayfire.library.vector_algorithms import all_true

# Test cases for the constant function


def test_constant_1d() -> None:
    result = data.constant(42, (5,))
    assert result.shape == (5,)
    assert result.scalar() == 42


def test_constant_2d() -> None:
    result = data.constant(3.14, (3, 4))
    assert result.shape == (3, 4)
    assert round(result.scalar(), 2) == 3.14  # type: ignore[arg-type]


def test_constant_3d() -> None:
    result = data.constant(0, (2, 2, 2), dtype=int64)
    assert result.shape == (2, 2, 2)
    assert result.scalar() == 0
    assert result.dtype == int64


def test_constant_default_shape() -> None:
    result = data.constant(1.0)
    assert result.shape == (1,)
    assert result.scalar() == 1.0


# TODO add error handling
# def test_constant_invalid_dtype() -> None:
#     with pytest.raises(ValueError):
#         data.constant(42, (3, 3), dtype="invalid_dtype")


# Test cases for the range function


def test_range_1d() -> None:
    result = data.range((5,))
    assert result.shape == (5,)


def test_range_2d() -> None:
    result = data.range((3, 4))
    assert result.shape == (3, 4)


def test_range_3d() -> None:
    result = data.range((2, 2, 2))
    assert result.shape == (2, 2, 2)


def test_range_with_axis() -> None:
    result = data.range((3, 4), axis=1)
    assert result.shape == (3, 4)


def test_range_with_dtype() -> None:
    result = data.range((4, 3), dtype=int64)
    assert result.dtype == int64


def test_range_with_invalid_axis() -> None:
    with pytest.raises(ValueError):
        data.range((2, 3, 4), axis=4)


# Test cases for the identity function

# BUG in to_list()
# def test_identity_2x2() -> None:
#     result = data.identity((2, 2))
#     expected = [[1, 0], [0, 1]]
#     assert result.to_list() == expected


# def test_identity_3x3() -> None:
#     result = data.identity((3, 3))
#     expected = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
#     assert result.to_list() == expected


# def test_identity_2x2x2() -> None:
#     result = data.identity((2, 2, 2))
#     expected = [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
#     assert result.to_list() == expected


# Test cases for the flat function


def test_flat_empty_array() -> None:
    arr = Array()
    flattened = data.flat(arr)
    assert flattened.shape == ()


def test_flat_1d() -> None:
    arr = random.randu((5,))
    flattened = data.flat(arr)
    assert flattened.shape == (5,)
    assert all_true(flattened == arr, 0)


def test_flat_2d() -> None:
    arr = random.randu((3, 2))
    flattened = data.flat(arr)
    assert flattened.shape == (6,)
    assert all_true(flattened == data.flat(arr), 0)


def test_flat_3d() -> None:
    arr = random.randu((3, 2, 4))
    flattened = data.flat(arr)
    assert flattened.shape == (24,)
    assert all_true(flattened == data.flat(arr), 0)


def test_flat_4d() -> None:
    arr = random.randu((3, 2, 4, 5))
    flattened = data.flat(arr)
    assert flattened.shape == (120,)
    assert all_true(flattened == data.flat(arr), 0)


def test_flat_large_array() -> None:
    arr = random.randu((1000, 1000))
    flattened = data.flat(arr)
    assert flattened.shape == (1000000,)
    assert all_true(flattened == data.flat(arr), 0)
