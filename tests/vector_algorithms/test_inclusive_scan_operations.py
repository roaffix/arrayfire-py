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
