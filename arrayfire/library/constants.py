__all__ = ["pi"]

import math

import arrayfire_wrapper as afw

pi = afw.lib.constant(math.pi, (1,), afw.float64)
