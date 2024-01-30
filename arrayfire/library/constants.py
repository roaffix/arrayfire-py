__all__ = ["pi"]

import math

import arrayfire_wrapper.lib as wrapper

import arrayfire as af

pi = wrapper.constant(math.pi, (1,), af.float64)

# Typing constants

Scalar = int | float | complex | bool

# Wrapper constants

BinaryOperator = wrapper.BinaryOperator
