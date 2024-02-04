import math

import arrayfire as af
from tests._helpers import round_to


class TestArithmeticOperators:
    def setup_method(self) -> None:
        self.array1 = af.Array([1, 2, 3])
        self.array2 = af.Array([4, 5, 6])
        self.scalar = 2

    def test_add(self) -> None:
        res = af.add(self.array1, self.array2)
        res_sum = self.array1 + self.array2
        assert res.to_list() == res_sum.to_list() == [5, 7, 9]

    def test_sub(self) -> None:
        res = af.sub(self.array1, self.array2)
        res_sum = self.array1 - self.array2
        assert res.to_list() == res_sum.to_list() == [-3, -3, -3]

    def test_mul(self) -> None:
        res = af.mul(self.array1, self.array2)
        res_product = self.array1 * self.array2
        assert res.to_list() == res_product.to_list() == [4, 10, 18]

    def test_div(self) -> None:
        res = af.div(self.array1, self.array2)
        res_quotient = self.array1 / self.array2
        assert round_to(res.to_list()) == round_to(res_quotient.to_list()) == [0.25, 0.4, 0.5]

    def test_mod(self) -> None:
        res = af.mod(self.array1, self.array2)
        expected = [1 % 4, 2 % 5, 3 % 6]
        assert res.to_list() == expected

        res_scalar = af.mod(self.scalar, self.array1)
        expected_scalar = [2 % 1, 2 % 2, 2 % 3]
        assert res_scalar.to_list() == expected_scalar

    def test_pow(self) -> None:
        res = af.pow(self.array1, self.array2)
        expected = [1**4, 2**5, 3**6]
        assert res.to_list() == expected

    # BUG
    # OSError: exception: access violation reading 0xFFFFFFFFFFFFFFFF

    # def test_bitand(self) -> None:
    #     res_and = af.bitand(self.array1, self.array2)
    #     expected_and = [1 & 4, 2 & 5, 3 & 6]
    #     assert res_and.to_list() == expected_and

    # def test_bitor(self) -> None:
    #     res_or = af.bitor(self.array1, self.array2)
    #     expected_or = [1 | 4, 2 | 5, 3 | 6]
    #     assert res_or.to_list() == expected_or

    # def test_bitxor(self) -> None:
    #     res_xor = af.bitxor(self.array1, self.array2)
    #     expected_xor = [1 ^ 4, 2 ^ 5, 3 ^ 6]
    #     assert res_xor.to_list() == expected_xor

    # def test_bitshiftl(self) -> None:
    #     # bitshiftl and bitshiftr assume the second array is for shift amounts
    #     res_shiftl = af.bitshiftl(self.array1, af.Array([1, 1, 1]))
    #     expected_shiftl = [1 << 1, 2 << 1, 3 << 1]
    #     assert res_shiftl.to_list() == expected_shiftl

    # def test_bitshiftr(self) -> None:
    #     res_shiftr = af.bitshiftr(self.array1, af.Array([1, 1, 1]))
    #     expected_shiftr = [1 >> 1, 2 >> 1, 3 >> 1]
    #     assert res_shiftr.to_list() == expected_shiftr

    def test_le(self) -> None:
        # Less than or equal to
        res_le = af.le(self.array1, self.array2)
        expected_le = [1 <= 4, 2 <= 5, 3 <= 6]
        assert res_le.to_list() == expected_le

        # Scalar comparison
        res_scalar_le = af.le(self.array1, 2)
        expected_scalar_le = [x <= 2 for x in [1, 2, 3]]
        assert res_scalar_le.to_list() == expected_scalar_le

    def test_lt(self) -> None:
        # Less than
        res_lt = af.lt(self.array1, self.array2)
        expected_lt = [1 < 4, 2 < 5, 3 < 6]
        assert res_lt.to_list() == expected_lt

        # Scalar comparison
        res_scalar_lt = af.lt(self.array1, 2)
        expected_scalar_lt = [x < 2 for x in [1, 2, 3]]
        assert res_scalar_lt.to_list() == expected_scalar_lt

    def test_gt(self) -> None:
        # Greater than
        res_gt = af.gt(self.array1, self.array2)
        expected_gt = [1 > 4, 2 > 5, 3 > 6]
        assert res_gt.to_list() == expected_gt

        # Scalar comparison
        res_scalar_gt = af.gt(self.array1, 2)
        expected_scalar_gt = [x > 2 for x in [1, 2, 3]]
        assert res_scalar_gt.to_list() == expected_scalar_gt

    def test_ge(self) -> None:
        # Greater than or equal to
        res_ge = af.ge(self.array1, self.array2)
        expected_ge = [1 >= 4, 2 >= 5, 3 >= 6]
        assert res_ge.to_list() == expected_ge

        # Scalar comparison
        res_scalar_ge = af.ge(self.array1, 2)
        expected_scalar_ge = [x >= 2 for x in [1, 2, 3]]
        assert res_scalar_ge.to_list() == expected_scalar_ge

    def test_eq(self) -> None:
        # Equal to
        res_eq = af.eq(self.array1, self.array2)
        expected_eq = [1 == 4, 2 == 5, 3 == 6]
        assert res_eq.to_list() == expected_eq

        # Scalar comparison
        res_scalar_eq = af.eq(self.array1, 2)
        expected_scalar_eq = [x == 2 for x in [1, 2, 3]]
        assert res_scalar_eq.to_list() == expected_scalar_eq

    def test_neq(self) -> None:
        # Not equal to
        res_neq = af.neq(self.array1, self.array2)
        expected_neq = [1 != 4, 2 != 5, 3 != 6]
        assert res_neq.to_list() == expected_neq

        # Scalar comparison
        res_scalar_neq = af.neq(self.array1, 2)
        expected_scalar_neq = [x != 2 for x in [1, 2, 3]]
        assert res_scalar_neq.to_list() == expected_scalar_neq

    def test_min_max_operations(self) -> None:
        # minof
        res_min = af.minof(self.array1, self.array2)
        expected_min = [min(1, 4), min(2, 5), min(3, 6)]
        assert res_min.to_list() == expected_min

        # maxof
        res_max = af.maxof(self.array1, self.array2)
        expected_max = [max(1, 4), max(2, 5), max(3, 6)]
        assert res_max.to_list() == expected_max

    def test_rem(self) -> None:
        res = af.rem(self.array1, self.array2)
        expected = [1 % 4, 2 % 5, 3 % 6]  # Similar to mod but using rem function
        assert res.to_list() == expected

    def test_abs(self) -> None:
        negative_array = af.Array([-1, -2, -3])
        res = af.abs(negative_array)
        expected = [abs(-1), abs(-2), abs(-3)]
        assert res.to_list() == expected

    def test_sign(self) -> None:
        array = af.Array([-1, 0, 1])
        res = af.sign(array)
        expected = [1, 0, 0]
        assert res.to_list() == expected

    def test_round(self) -> None:
        array = af.Array([1.2, 2.5, 3.6])
        res = af.round(array)
        expected = [1, 3, 4]  # NOTE: Python's round may behave differently
        assert res.to_list() == expected

    def test_trunc(self) -> None:
        array = af.Array([1.9, -2.8, 3.7])
        res = af.trunc(array)
        expected = [1, -2, 3]
        assert res.to_list() == expected

    def test_floor(self) -> None:
        array = af.Array([1.2, -2.5, 3.8])
        res = af.floor(array)
        expected = [1, -3, 3]
        assert res.to_list() == expected

    def test_ceil(self) -> None:
        array = af.Array([1.2, -2.5, 3.8])
        res = af.ceil(array)
        expected = [2, -2, 4]
        assert res.to_list() == expected

    # BUG
    # def test_hypot(self) -> None:
    #     res = af.hypot(self.array1, self.array2)
    #     expected = [(1**2 + 4**2) ** 0.5, (2**2 + 5**2) ** 0.5, (3**2 + 6**2) ** 0.5]
    #     assert round_to(res.to_list()) == round_to(expected)

    def test_sin(self) -> None:
        # Sin
        res_sin = af.sin(af.Array([0, math.pi / 4, math.pi / 2]))
        expected_sin = [0, 1 / (2**0.5), 1]
        assert round_to(res_sin.to_list()) == round_to(expected_sin)  # type: ignore[arg-type]

    def test_cos(self) -> None:
        # Cos
        res_cos = af.cos(af.Array([0, math.pi / 4, math.pi / 2]))
        expected_cos = [1, 1 / (2**0.5), 0]
        assert round_to(res_cos.to_list()) == round_to(expected_cos)  # type: ignore[arg-type]

    def test_tan(self) -> None:
        # Tan
        res_tan = af.tan(af.Array([0, math.pi / 4, math.pi / 3]))
        expected_tan = [0, 1, 1.732]
        assert round_to(res_tan.to_list()) == round_to(expected_tan)  # type: ignore[arg-type]

    def test_asin(self) -> None:
        # ASin
        res_asin = af.asin(af.Array([0, 0.5, 1]))
        expected_asin = [0, math.pi / 6, math.pi / 2]
        assert round_to(res_asin.to_list()) == round_to(expected_asin)  # type: ignore[arg-type]

    def test_acos(self) -> None:
        # ACos
        res_acos = af.acos(af.Array([0, 0.5, 1]))
        expected_acos = [math.pi / 2, math.pi / 3, 0]
        assert round_to(res_acos.to_list()) == round_to(expected_acos)  # type: ignore[arg-type]

    def test_atan(self) -> None:
        # ATan
        res_atan = af.atan(af.Array([-1, 0, 1]))
        expected_atan = [-math.pi / 4, 0, math.pi / 4]  # Adjust expected values accordingly
        assert round_to(res_atan.to_list()) == round_to(expected_atan)  # type: ignore[arg-type]

    def test_atan2(self) -> None:
        res = af.atan2(self.array1, self.array2)
        expected = [math.atan2(1, 4), math.atan2(2, 5), math.atan2(3, 6)]
        assert round_to(res.to_list()) == round_to(expected)  # type: ignore[arg-type]

    def test_sinh(self) -> None:
        # Sinh
        array = af.Array([0, 1, -1])
        res_sinh = af.sinh(array)
        expected_sinh = [
            0,
            (math.e - 1 / math.e) / 2,
            -(math.e - 1 / math.e) / 2,
        ]
        assert round_to(res_sinh.to_list()) == round_to(expected_sinh)  # type: ignore[arg-type]

    def test_cosh(self) -> None:
        # Cosh
        array = af.Array([0, 1, -1])
        res_cosh = af.cosh(array)
        expected_cosh = [1, (math.e + 1 / math.e) / 2, (math.e + 1 / math.e) / 2]
        assert round_to(res_cosh.to_list()) == round_to(expected_cosh)  # type: ignore[arg-type]

    def test_tanh(self) -> None:
        # Tanh
        array = af.Array([0, 1, -1])
        res_tanh = af.tanh(array)
        expected_tanh = [0, (math.e**2 - 1) / (math.e**2 + 1), -(math.e**2 - 1) / (math.e**2 + 1)]
        assert round_to(res_tanh.to_list()) == round_to(expected_tanh)  # type: ignore[arg-type]

    # BUG
    # root(3, 2) != math.sqrt(3) == 3**0.5
    # def test_root(self) -> None:
    #     res = af.root(self.array1, 2)
    #     expected = [1**0.5, 2**0.5, 3**0.5]
    #     assert round_to(res.to_list()) == round_to(expected)  # type: ignore[arg-type]

    def test_pow2(self) -> None:
        res = af.pow2(self.array1)
        expected = [2**1, 2**2, 2**3]
        assert res.to_list() == expected

    def test_sigmoid(self) -> None:
        res = af.sigmoid(self.array1)
        expected = [1 / (1 + math.e**-x) for x in [1, 2, 3]]
        assert round_to(res.to_list()) == round_to(expected)  # type: ignore[arg-type]

    def test_exponential_functions(self) -> None:
        # Exp
        res_exp = af.exp(self.array1)
        expected_exp = [math.e**1, math.e**2, math.e**3]
        assert round_to(res_exp.to_list()) == round_to(expected_exp)  # type: ignore[arg-type]

        # Expm1
        res_expm1 = af.expm1(self.array1)
        expected_expm1 = [(math.e**x) - 1 for x in [1, 2, 3]]
        assert round_to(res_expm1.to_list()) == round_to(expected_expm1)  # type: ignore[arg-type]

    def test_error_functions(self) -> None:
        array_values = [1, 2, 3]  # from self.array1
        expected_erf = [math.erf(x) for x in array_values]
        expected_erfc = [math.erfc(x) for x in array_values]

        # Erf
        res_erf = af.erf(af.Array(array_values))  # type: ignore[arg-type]
        assert round_to(res_erf.to_list()) == round_to(expected_erf)  # type: ignore[arg-type]

        # Erfc
        res_erfc = af.erfc(af.Array(array_values))  # type: ignore[arg-type]
        assert round_to(res_erfc.to_list()) == round_to(expected_erfc)  # type: ignore[arg-type]

    def test_logarithmic_functions(self) -> None:
        # Log
        res_log = af.log(self.array1)
        expected_log = [math.log(x) for x in [1, 2, 3]]
        assert round_to(res_log.to_list()) == round_to(expected_log)  # type: ignore[arg-type]

        # Log1p
        res_log1p = af.log1p(self.array1)
        expected_log1p = [math.log(x + 1) for x in [1, 2, 3]]
        assert round_to(res_log1p.to_list()) == round_to(expected_log1p)  # type: ignore[arg-type]

        # Log10
        array = af.Array([1, 10, 100])
        res_log10 = af.log10(array)
        expected_log10 = [math.log10(x) for x in [1, 10, 100]]  # Using Python's log10 from math module
        assert round_to(res_log10.to_list()) == round_to(expected_log10)  # type: ignore[arg-type]

        # Log2
        array = af.Array([1, 2, 4])
        res_log2 = af.log2(array)
        expected_log2 = [math.log2(x) for x in [1, 2, 4]]  # Using Python's log2 from math module
        assert round_to(res_log2.to_list()) == round_to(expected_log2)  # type: ignore[arg-type]

    def test_sqrt(self) -> None:
        # Sqrt
        res_sqrt = af.sqrt(self.array1)
        expected_sqrt = [x**0.5 for x in [1, 2, 3]]
        assert round_to(res_sqrt.to_list()) == round_to(expected_sqrt)  # type: ignore[arg-type]

    def test_cbrt(self) -> None:
        # Cbrt
        res_cbrt = af.cbrt(self.array1)
        expected_cbrt = [x ** (1 / 3) for x in [1, 2, 3]]
        assert round_to(res_cbrt.to_list()) == round_to(expected_cbrt)  # type: ignore[arg-type]

    def test_factorial(self) -> None:
        # Factorial
        res_factorial = af.factorial(af.Array([0, 1, 2]))
        expected_factorial = [1, 1, 2]  # Using small numbers to avoid large outputs
        assert res_factorial.to_list() == expected_factorial

    def test_tgamma(self) -> None:
        # Tgamma
        array = af.Array([0.5, 1, 5])  # Including 0.5 to test the gamma function for non-integer
        res_tgamma = af.tgamma(array)
        expected_tgamma = [math.gamma(x) for x in [0.5, 1, 5]]
        assert round_to(res_tgamma.to_list()) == round_to(expected_tgamma)  # type: ignore[arg-type]

    def test_lgamma(self) -> None:
        # Lgamma
        array = af.Array([0.5, 1, 5])
        res_lgamma = af.lgamma(array)
        expected_lgamma = [math.lgamma(x) for x in [0.5, 1, 5]]
        assert round_to(res_lgamma.to_list()) == round_to(expected_lgamma)  # type: ignore[arg-type]

    def test_logical_and(self) -> None:
        # Logical And
        res_and = af.logical_and(self.array1, af.Array([0, 1, 1]))
        expected_and = [0 & 1, 1 & 1, 1 & 1]
        assert res_and.to_list() == expected_and

    def test_logical_or(self) -> None:
        # Logical OR array with array
        res_or = af.logical_or(self.array1, af.Array([0, 0, 1]))
        expected_or = [(x != 0) or (y != 0) for x, y in zip([1, 2, 3], [0, 0, 1])]
        assert res_or.to_list() == expected_or

        # Logical OR array with scalar
        res_scalar_or = af.logical_or(self.array1, 0)
        expected_scalar_or = [(x != 0) or (0 != 0) for x in [1, 2, 3]]
        assert res_scalar_or.to_list() == expected_scalar_or

    def test_logical_not(self) -> None:
        # Logical NOT
        res_not = af.logical_not(self.array1)
        expected_not = [not (x != 0) for x in [1, 2, 3]]
        assert res_not.to_list() == expected_not

    def test_negation(self) -> None:
        res_neg = af.neg(self.array1)
        expected_neg = [-1, -2, -3]
        assert res_neg.to_list() == expected_neg
