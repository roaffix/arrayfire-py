__all__ = ["pi"]

from arrayfire_wrapper.lib import BinaryOperator, ConvGradient, ImageFormat, Match, MatProp, Norm, TopK, VarianceBias

__all__ += ["Match", "MatProp", "BinaryOperator", "Norm", "ConvGradient", "VarianceBias", "TopK", "ImageFormat"]

import math

import arrayfire_wrapper.lib as wrapper

import arrayfire as af

pi = wrapper.constant(math.pi, (1,), af.float64)

# Typing constants

Scalar = int | float | complex | bool
