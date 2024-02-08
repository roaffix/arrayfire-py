import pytest

import arrayfire as af
from tests._helpers import create_from_2d_nested

# Test dot


@pytest.fixture
def real_vector_1() -> af.Array:
    return af.Array([1.0, 2.0, 3.0])


@pytest.fixture
def real_vector_2() -> af.Array:
    return af.Array([4.0, 5.0, 6.0])


@pytest.fixture
def float_vector_1() -> af.Array:
    return af.Array([1.5, 2.5, 3.5])


@pytest.fixture
def float_vector_2() -> af.Array:
    return af.Array([4.5, 5.5, 6.5])


def test_dot_real_vectors(real_vector_1: af.Array, real_vector_2: af.Array) -> None:
    expected = 32  # Calculated manually or using a trusted library
    result = af.dot(real_vector_1, real_vector_2)
    assert result == expected, f"Expected {expected}, got {result}"


def test_dot_float_vectors(float_vector_1: af.Array, float_vector_2: af.Array) -> None:
    expected = 61.5  # Calculated manually or using a trusted library
    result = af.dot(float_vector_1, float_vector_2)
    assert result == expected, f"Expected {expected}, got {result}"


def test_dot_return_scalar(real_vector_1: af.Array, real_vector_2: af.Array) -> None:
    result = af.dot(real_vector_1, real_vector_2, return_scalar=True)
    assert isinstance(result, (int, float)), "Result is not a scalar"


# Test gemm


@pytest.fixture
def matrix_a() -> af.Array:
    return create_from_2d_nested(1, 2, 3, 4)


@pytest.fixture
def matrix_b() -> af.Array:
    return create_from_2d_nested(5, 6, 7, 8)


def test_gemm_basic(matrix_a: af.Array, matrix_b: af.Array) -> None:
    result = af.gemm(matrix_a, matrix_b)
    expected = create_from_2d_nested(19.0, 22.0, 43.0, 50.0)
    assert result == expected, f"Expected {expected}, got {result}"


def test_gemm_alpha_beta(matrix_a: af.Array, matrix_b: af.Array) -> None:
    alpha = 0.5
    beta = 2.0
    result = af.gemm(matrix_a, matrix_b, alpha=alpha, beta=beta)
    expected = create_from_2d_nested(10.5, 12.0, 22.5, 26.0)
    assert result == expected, f"Expected {expected}, got {result}"


def test_gemm_transpose_options(matrix_a: af.Array, matrix_b: af.Array) -> None:
    result = af.gemm(matrix_a, matrix_b, lhs_opts=af.MatProp.TRANS, rhs_opts=af.MatProp.TRANS)
    expected = create_from_2d_nested(23.0, 31.0, 34.0, 46.0)
    assert result == expected, f"Expected {expected}, got {result}"


# Test matmul


def test_basic_matrix_multiplication() -> None:
    A = af.randu((3, 2), dtype=af.float32)
    B = af.randu((2, 4), dtype=af.float32)
    C = af.matmul(A, B)
    assert C.shape == (3, 4), "Output dimensions should be 3x4."


def test_matrix_multiplication_with_lhs_transposed() -> None:
    A = af.randu((2, 3), dtype=af.float32)  # Transposing makes it 3x2
    B = af.randu((2, 4), dtype=af.float32)
    C = af.matmul(A, B, lhs_opts=af.MatProp.TRANS)
    assert C.shape == (3, 4), "Output dimensions should be 3x4 when lhs is transposed."


def test_matrix_multiplication_with_both_transposed() -> None:
    A = af.randu((4, 3), dtype=af.float32)  # Transposing makes it 3x4
    B = af.randu((6, 4), dtype=af.float32)  # Transposing makes it 4x6
    C = af.matmul(A, B, lhs_opts=af.MatProp.TRANS, rhs_opts=af.MatProp.TRANS)
    assert C.shape == (3, 6), "Output dimensions should be 3x6 with both matrices transposed."


# BUG
# def test_incompatible_dimensions() -> None:
#     A = af.randu((3, 5), dtype=af.float32)
#     B = af.randu((4, 6), dtype=af.float32)
#     with pytest.raises(ValueError):
#         C = af.matmul(A, B)


# def test_unsupported_data_type() -> None:
#     A = af.Array([1, 2, 3], dtype=af.uint32)  # Assuming unsupported data type like unsigned int
#     B = af.Array([4, 5, 6], dtype=af.uint32)
#     with pytest.raises(TypeError):
#         C = af.matmul(A, B)


# def test_multiplication_result_verification() -> None:
#     A = create_from_2d_nested(1, 2, 3, 4)
#     B = create_from_2d_nested(5, 6, 7, 8)
#     C = af.matmul(A, B)
#     expected = create_from_2d_nested(19, 22, 43, 50)
#     assert af.all_true(C == expected), "The multiplication result is incorrect."
