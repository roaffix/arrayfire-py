import pytest

from arrayfire import Array
from arrayfire.library import vector_algorithms as va
from arrayfire.library.array_management.creation import constant


@pytest.fixture
def true_array() -> Array:
    return constant(1, (5, 5))


@pytest.fixture
def false_array() -> Array:
    arr = constant(1, (5, 5))
    arr[2, 2] = 0  # Set one element to False
    return arr


# BUG Array.to_list()
# def test_all_true_with_axis(true_array: Array) -> None:
#     result = va.all_true(true_array, axis=0)
#     assert result.to_list() == [True, True, True, True, True]


# Test cases for the sum function


@pytest.fixture
def sample_array() -> Array:
    return Array([1, 2, 3, 4])


def test_sum_no_axis_no_nan_value(sample_array: Array) -> None:
    result = va.sum(sample_array)
    assert result == 10  # Sum of all elements is 1 + 2 + 3 + 4 = 10


# TODO add more test cases
