from arrayfire import Array
from arrayfire.library import random
from arrayfire.library.array_management.modification import flat
from arrayfire.library.vector_algorithms import all_true

# Test cases for the flat function


def test_flat_empty_array() -> None:
    arr = Array()
    flattened = flat(arr)
    assert flattened.shape == ()


def test_flat_1d() -> None:
    arr = random.randu((5,))
    flattened = flat(arr)
    assert flattened.shape == (5,)
    assert all_true(flattened == arr, 0)


def test_flat_2d() -> None:
    arr = random.randu((3, 2))
    flattened = flat(arr)
    assert flattened.shape == (6,)
    assert all_true(flattened == flat(arr), 0)


def test_flat_3d() -> None:
    arr = random.randu((3, 2, 4))
    flattened = flat(arr)
    assert flattened.shape == (24,)
    assert all_true(flattened == flat(arr), 0)


def test_flat_4d() -> None:
    arr = random.randu((3, 2, 4, 5))
    flattened = flat(arr)
    assert flattened.shape == (120,)
    assert all_true(flattened == flat(arr), 0)


def test_flat_large_array() -> None:
    arr = random.randu((1000, 1000))
    flattened = flat(arr)
    assert flattened.shape == (1000000,)
    assert all_true(flattened == flat(arr), 0)
