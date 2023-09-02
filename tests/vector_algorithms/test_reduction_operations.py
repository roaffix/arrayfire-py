from typing import TYPE_CHECKING

import pytest

from arrayfire import Array
from arrayfire.library import data
from arrayfire.library import vector_algorithms as va

# if TYPE_CHECKING:
#     from arrayfire import Array


@pytest.fixture
def true_array() -> Array:
    return data.constant(1, (5, 5))


# BUG Array.__setitem__
# @pytest.fixture
# def false_array() -> Array:
#     arr = data.constant(1, (5, 5))
#     arr[2, 2] = 0  # Set one element to False
#     return arr


# BUG Array.to_list()
# def test_all_true_with_axis(true_array: Array) -> None:
#     result = va.all_true(true_array, axis=0)
#     assert result.to_list() == [True, True, True, True, True]
