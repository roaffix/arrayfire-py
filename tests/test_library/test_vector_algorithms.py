import pytest

import arrayfire as af

# Test reduction operators


@pytest.fixture
def true_array() -> af.Array:
    return af.constant(1, (5, 5))


@pytest.fixture
def false_array() -> af.Array:
    arr = af.constant(1, (5, 5))
    arr[2, 2] = 0  # Set one element to False
    return arr


# BUG af.Array.to_list()
# def test_all_true_with_axis(true_array: af.Array) -> None:
#     result = va.all_true(true_array, axis=0)
#     assert result.to_list() == [True, True, True, True, True]


# Test cases for the sum function


@pytest.fixture
def sample_array() -> af.Array:
    return af.Array([1, 2, 3, 4])


def test_sum_no_axis_no_nan_value(sample_array: af.Array) -> None:
    result = af.sum(sample_array)
    assert result == 10  # Sum of all elements is 1 + 2 + 3 + 4 = 10


# from typing import Union
# import arrayfire as af
# import pytest

# # Fixture to create a sample ArrayFire array for testing
# @pytest.fixture
# def sample_array() -> af.Array:
#     return af.randu(4, 3)

# # Test cases
# def test_accum_default_axis(sample_array: af.Array) -> None:
#     result = af.accum(sample_array)
#     expected = af.moddims(af.scan(sample_array, af.BINARY_ADD, 0), sample_array.dims())
#     af.assert_allclose(result, expected)

# def test_accum_custom_axis(sample_array: af.Array) -> None:
#     result = af.accum(sample_array, axis=1)
#     expected = af.moddims(af.scan(sample_array, af.BINARY_ADD, 1), sample_array.dims())
#     af.assert_allclose(result, expected)

# def test_accum_custom_axis_2D_array() -> None:
#     # Test with a 2D array, specifying axis=1
#     array = af.Array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#     result = af.accum(array, axis=1)
#     expected = af.Array([[1, 3, 6], [4, 9, 15], [7, 15, 24]])
#     af.assert_allclose(result, expected)

# def test_accum_custom_axis_invalid_axis() -> None:
#     # Test with an invalid axis
#     array = af.randu(3, 4)
#     with pytest.raises(ValueError, match="Invalid axis"):
#         accum(array, axis=2)

# TODO add more test cases
