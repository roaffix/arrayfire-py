from arrayfire import Array
from arrayfire.library import operators
from tests._helpers import round_to


class TestArithmeticOperators:
    def setup_method(self) -> None:
        self.array1 = Array([1, 2, 3])
        self.array2 = Array([4, 5, 6])

    def test_add(self) -> None:
        res = operators.add(self.array1, self.array2)
        res_sum = self.array1 + self.array2
        assert res.to_list() == res_sum.to_list() == [5, 7, 9]

    def test_sub(self) -> None:
        res = operators.sub(self.array1, self.array2)
        res_sum = self.array1 - self.array2
        assert res.to_list() == res_sum.to_list() == [-3, -3, -3]

    def test_mul(self) -> None:
        res = operators.mul(self.array1, self.array2)
        res_product = self.array1 * self.array2
        assert res.to_list() == res_product.to_list() == [4, 10, 18]

    def test_div(self) -> None:
        res = operators.div(self.array1, self.array2)
        res_quotient = self.array1 / self.array2
        assert round_to(res.to_list()) == round_to(res_quotient.to_list()) == [0.25, 0.4, 0.5]
