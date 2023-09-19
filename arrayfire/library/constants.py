import math

from arrayfire.backend._clib_wrapper._error_handler import constant
from arrayfire.dtypes import float64

pi = constant(math.pi, (1,), float64)
