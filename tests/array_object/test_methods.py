from arrayfire.dtypes import float32, int32
from arrayfire.library.array_object import Array


def test_array_getitem_by_index() -> None:
    array = Array([1, 2, 3, 4, 5])

    int_item = array[2]
    assert array.dtype == int_item.dtype
    assert int_item.scalar() == 3


def test_array_getitem_by_slice() -> None:
    array = Array([1, 2, 3, 4, 5])

    slice_item = array[1:3]
    assert slice_item.to_list() == [2, 3]


def test_scalar() -> None:
    array = Array([1, 2, 3])
    assert array[1].scalar() == 2


def test_scalar_is_empty() -> None:
    array = Array()
    assert array.scalar() is None


def test_array_to_list() -> None:
    array = Array([1, 2, 3])
    assert array.to_list() == [1, 2, 3]


def test_array_to_list_is_empty() -> None:
    array = Array()
    assert array.to_list() == []


def test_array_to_list_comparison() -> None:
    array1 = Array([1, 2, 3])
    array2 = Array([1, 2, 3])
    assert array1 is not array2
    assert array1.to_list() == array2.to_list()


# BUG
# def test_copy_for_array_with_multiple_elements() -> None:
#     array = Array([1, 2, 3])
#     copy = array.copy()
#     assert array is not copy
#     assert array.to_list() == copy.to_list()
