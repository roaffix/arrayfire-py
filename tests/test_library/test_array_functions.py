import pytest

import arrayfire as af

# Test cases for the af.constant function


def test_constant_1d() -> None:
    result = af.constant(42, (5,))
    assert result.shape == (5,)
    assert result.scalar() == 42


def test_constant_2d() -> None:
    result = af.constant(3.14, (3, 4))
    assert result.shape == (3, 4)
    assert round(result.scalar(), 2) == 3.14  # type: ignore[arg-type]


def test_constant_3d() -> None:
    result = af.constant(0, (2, 2, 2), dtype=af.int64)
    assert result.shape == (2, 2, 2)
    assert result.scalar() == 0
    assert result.dtype == af.int64


def test_constant_default_shape() -> None:
    result = af.constant(1.0)
    assert result.shape == (1,)
    assert result.scalar() == 1.0


# TODO add error handling
# def test_constant_invalid_dtype() -> None:
#     with pytest.raises(ValueError):
#         af.constant(42, (3, 3), dtype="invalid_dtype")


# Test cases for the range function


def test_range_1d() -> None:
    result = af.range((5,))
    assert result.shape == (5,)


def test_range_2d() -> None:
    result = af.range((3, 4))
    assert result.shape == (3, 4)


def test_range_3d() -> None:
    result = af.range((2, 2, 2))
    assert result.shape == (2, 2, 2)


def test_range_with_axis() -> None:
    result = af.range((3, 4), axis=1)
    assert result.shape == (3, 4)


def test_range_with_dtype() -> None:
    result = af.range((4, 3), dtype=af.int64)
    assert result.dtype == af.int64


def test_range_with_invalid_axis() -> None:
    with pytest.raises(ValueError):
        af.range((2, 3, 4), axis=4)


# Test cases for the af.flat function


def test_flat_empty_array() -> None:
    arr = af.Array()
    flattened = af.flat(arr)
    assert flattened.shape == ()


def test_flat_1d() -> None:
    arr = af.randu((5,))
    flattened = af.flat(arr)
    assert flattened.shape == (5,)
    assert af.all_true(flattened == arr, 0)


def test_flat_2d() -> None:
    arr = af.randu((3, 2))
    flattened = af.flat(arr)
    assert flattened.shape == (6,)
    assert af.all_true(flattened == af.flat(arr), 0)


def test_flat_3d() -> None:
    arr = af.randu((3, 2, 4))
    flattened = af.flat(arr)
    assert flattened.shape == (24,)
    assert af.all_true(flattened == af.flat(arr), 0)


def test_flat_4d() -> None:
    arr = af.randu((3, 2, 4, 5))
    flattened = af.flat(arr)
    assert flattened.shape == (120,)
    assert af.all_true(flattened == af.flat(arr), 0)


def test_flat_large_array() -> None:
    arr = af.randu((1000, 1000))
    flattened = af.flat(arr)
    assert flattened.shape == (1000000,)
    assert af.all_true(flattened == af.flat(arr), 0)
