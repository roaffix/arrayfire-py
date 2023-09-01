import pytest

from arrayfire.dtypes import int64
from arrayfire.library import data
from arrayfire import Array


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


def _is_identity_matrix(arr: Array) -> bool:
    rows, cols = arr.shape
    if rows != cols:
        return False
    for i in range(rows):
        for j in range(cols):
            if i == j and arr[i, j] != 1:
                return False
            elif i != j and arr[i, j] != 0:
                return False
    return True


# Test cases for the identity function
def test_identity_2d() -> None:
    result = data.identity((3, 3))
    assert result.shape == (3, 3)
    assert _is_identity_matrix(result)


def test_identity_3d() -> None:
    result = data.identity((2, 2, 2))
    assert result.shape == (2, 2, 2)
    assert custom_all(result, lambda x: x == 1.0)


def test_identity_4d() -> None:
    result = data.identity((2, 2, 2, 2))
    assert result.shape == (2, 2, 2, 2)
    assert custom_all(result, lambda x: x == 1.0)


def test_identity_with_dtype() -> None:
    result = data.identity((3, 3), dtype=int64)
    assert result.shape == (3, 3)
    assert result.dtype == int64


def test_identity_invalid_shape() -> None:
    with pytest.raises(ValueError):
        data.identity((1,))


def test_identity_invalid_shape2() -> None:
    with pytest.raises(ValueError):
        data.identity((3,))


# Custom function to check if all elements in an array meet a condition
def custom_all(arr: Array) -> bool:
    for element in data.flat(arr):
        if not element:
            return False
    return True


# Test cases for the flat function
def test_flat_2d():
    input_array = af.randu(3, 2)  # Create a 3x2 random array
    result = data.flat(input_array)
    assert result.shape == (6,)  # Flattened shape should be 6 elements in 1D
    assert custom_all(result == input_array)


def test_flat_3d():
    input_array = af.randu(2, 2, 2)  # Create a 2x2x2 random array
    result = data.flat(input_array)
    assert result.shape == (8,)  # Flattened shape should be 8 elements in 1D
    assert custom_all(result == input_array)


def test_flat_empty_array():
    input_array = af.Array()  # Create an empty array
    result = data.flat(input_array)
    assert result.shape == (0,)  # Flattened shape of an empty array should be (0,)
