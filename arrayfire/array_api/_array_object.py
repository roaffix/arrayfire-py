from __future__ import annotations

import types
from typing import Any

import arrayfire as af

from ._constants import Device, NestedSequence, PyCapsule, SupportsBufferProtocol
from ._dtypes import (
    all_dtypes,
    boolean_dtypes,
    complex_floating_dtypes,
    dtype_categories,
    floating_dtypes,
    integer_dtypes,
    integer_or_boolean_dtypes,
    numeric_dtypes,
    promote_types,
)

# TODO
# - add check if two arrays are on the same device (both backend and device)
# - - include it in magic methods here
# - - include it in elementwise functions also


class Array:
    _array: af.Array

    def __new__(cls, *args: Any, **kwargs: Any) -> Array:
        raise TypeError(
            "The array_api Array object should not be instantiated directly. "
            "Use an array creation function, such as asarray(), instead."
        )

    def _check_allowed_dtypes(self, other: bool | int | float | Array, dtype_category: str, op: str) -> Array:
        """
        Helper function for operators to only allow specific input dtypes

        Use like

            other = self._check_allowed_dtypes(other, 'numeric', '__add__')
            if other is NotImplemented:
                return other
        """
        if self.dtype not in dtype_categories[dtype_category]:
            raise TypeError(f"Only {dtype_category} dtypes are allowed in {op}")
        if isinstance(other, int | complex | float | bool):
            other = self._promote_scalar(other)
        elif isinstance(other, Array):
            if other.dtype not in dtype_categories[dtype_category]:
                raise TypeError(f"Only {dtype_category} dtypes are allowed in {op}")
        else:
            return NotImplemented

        # This will raise TypeError for type combinations that are not allowed
        # to promote in the spec (even if the NumPy array operator would
        # promote them).
        res_dtype = promote_types(self.dtype, other.dtype)
        if op.startswith("__i"):
            # Note: NumPy will allow in-place operators in some cases where
            # the type promoted operator does not match the left-hand side
            # operand. For example,

            # >>> a = np.array(1, dtype=np.int8)
            # >>> a += np.array(1, dtype=np.int16)

            # The spec explicitly disallows this.
            if res_dtype != self.dtype:
                raise TypeError(f"Cannot perform {op} with dtypes {self.dtype} and {other.dtype}")

        return other

    def _promote_scalar(self, scalar: bool | int | float | complex) -> Array:
        """
        Returns a promoted version of a Python scalar appropriate for use with
        operations on self.

        This may raise an OverflowError in cases where the scalar is an
        integer that is too large to fit in a NumPy integer dtype, or
        TypeError when the scalar type is incompatible with the dtype of self.
        """
        from ._data_type_functions import iinfo

        # NOTE
        # Only Python scalar types that match the array dtype are allowed.
        if isinstance(scalar, bool):
            if self.dtype not in boolean_dtypes:
                raise TypeError("Python bool scalars can only be promoted with bool arrays")

        elif isinstance(scalar, int):
            if self.dtype in boolean_dtypes:
                raise TypeError("Python int scalars cannot be promoted with bool arrays")
            if self.dtype in integer_dtypes:
                info = iinfo(self.dtype)
                if not (info.min <= scalar <= info.max):
                    raise OverflowError("Python int scalars must be within the bounds of the dtype for integer arrays")

        elif isinstance(scalar, float):
            if self.dtype not in floating_dtypes:
                raise TypeError("Python float scalars can only be promoted with floating-point arrays.")

        elif isinstance(scalar, complex):
            if self.dtype not in complex_floating_dtypes:
                raise TypeError("Python complex scalars can only be promoted with complex floating-point arrays.")

        else:
            raise TypeError("'scalar' must be a Python scalar")

        # NOTE
        # Scalars are unconditionally cast to the same dtype as the array.

        # NOTE (numpy-specific rule)
        # The spec only specifies integer-dtype/int promotion behavior for integers within the bounds of the integer
        # dtype. Outside of those bounds we use the default NumPy behavior (either cast or raise OverflowError).
        return Array._new(af.constant(scalar, dtype=self.dtype, shape=(1,)))

    @staticmethod
    def _normalize_two_args(x1: Array, x2: Array) -> tuple[Array, Array]:
        # BUG, FIXME
        # """
        # Normalize inputs to two arg functions to fix type promotion rules

        # NumPy deviates from the spec type promotion rules in cases where one
        # argument is 0-dimensional and the other is not. For example:

        # >>> import numpy as np
        # >>> a = np.array([1.0], dtype=np.float32)
        # >>> b = np.array(1.0, dtype=np.float64)
        # >>> np.add(a, b) # The spec says this should be float64
        # array([2.], dtype=float32)

        # To fix this, we add a dimension to the 0-dimension array before passing it
        # through. This works because a dimension would be added anyway from
        # broadcasting, so the resulting shape is the same, but this prevents NumPy
        # from not promoting the dtype.
        # """
        # # Another option would be to use signature=(x1.dtype, x2.dtype, None),
        # # but that only works for ufuncs, so we would have to call the ufuncs
        # # directly in the operator methods. One should also note that this
        # # sort of trick wouldn't work for functions like searchsorted, which
        # # don't do normal broadcasting, but there aren't any functions like
        # # that in the array API namespace.
        # if x1.ndim == 0 and x2.ndim != 0:
        #     # The _array[None] workaround was chosen because it is relatively
        #     # performant. broadcast_to(x1._array, x2.shape) is much slower. We
        #     # could also manually type promote x2, but that is more complicated
        #     # and about the same performance as this.
        #     x1 = Array._new(x1._array[None])
        # elif x2.ndim == 0 and x1.ndim != 0:
        #     x2 = Array._new(x2._array[None])
        return (x1, x2)

    @classmethod
    def _new(cls, x: Array | bool | int | float | complex | NestedSequence | SupportsBufferProtocol, /) -> Array:
        """
        This is a private method for initializing the array API Array
        object.

        Functions outside of the array_api submodule should not use this
        method. Use one of the creation functions instead, such as
        ``asarray``.

        """
        obj = super().__new__(cls)
        # Note: The spec does not have array scalars, only 0-D arrays.
        if isinstance(x, bool | int | float | complex):
            # Convert the array scalar to a 0-D array
            x = af.constant(x)  # type: ignore[arg-type]
        if x.dtype not in all_dtypes:  # type: ignore[union-attr]
            raise TypeError(
                f"The array_api namespace does not support the dtype '{x.dtype}'"  # type: ignore[union-attr]
            )
        obj._array = x  # type: ignore[assignment]
        return obj

    def __str__(self: Array, /) -> str:
        """
        Performs the operation __str__.
        """
        return self._array.__str__()  # .replace("array", "Array")

    def __repr__(self: Array, /) -> str:
        """
        Performs the operation __repr__.
        """
        # TODO
        # Numpy representation:
        # suffix = f", dtype={self.dtype.name})"
        # if 0 in self.shape:
        #     prefix = "empty("
        #     mid = str(self.shape)
        # else:
        #     prefix = "Array("
        #     mid = np.array2string(self._array, separator=', ', prefix=prefix, suffix=suffix)
        # return prefix + mid + suffix
        return repr(self._array)

    def __abs__(self: Array, /) -> Array:
        """
        Performs the operation __abs__.
        """
        if self.dtype not in numeric_dtypes:
            raise TypeError("Only numeric dtypes are allowed in __abs__")
        res = self._array.__abs__()
        return self.__class__._new(res)

    def __add__(self: Array, other: int | float | Array, /) -> Array:
        """
        Performs the operation __add__.
        """
        other = self._check_allowed_dtypes(other, "numeric", "__add__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__add__(other._array)
        return self.__class__._new(res)

    def __and__(self: Array, other: int | bool | Array, /) -> Array:
        """
        Performs the operation __and__.
        """
        other = self._check_allowed_dtypes(other, "integer or boolean", "__and__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__and__(other._array)
        return self.__class__._new(res)

    def __array_namespace__(self: Array, /, *, api_version: str | None = None) -> types.ModuleType:
        if api_version is not None and not api_version.startswith("2021."):
            raise ValueError(f"Unrecognized array API version: {api_version!r}")

        from arrayfire import array_api

        return array_api

    def __bool__(self: Array, /) -> bool:
        """
        Performs the operation __bool__.
        """
        # Note: This is an error here.
        if self._array.ndim != 0:
            raise TypeError("bool is only allowed on arrays with 0 dimensions")
        return self._array.is_bool

    def __complex__(self: Array, /) -> complex:
        """
        Performs the operation __complex__.
        """
        # Note: This is an error here.
        if self._array.ndim != 0:
            raise TypeError("complex is only allowed on arrays with 0 dimensions")
        res = self._array.__complex__()
        return res

    def __dlpack__(self: Array, /, *, stream: None = None) -> PyCapsule:
        """
        Performs the operation __dlpack__.
        """
        return self._array.__dlpack__(stream=stream)

    # FIXME
    # def __dlpack_device__(self: Array, /) -> Tuple[IntEnum, int]:
    #     """
    #     Performs the operation __dlpack_device__.
    #     """
    #     # Note: device support is required for this
    #     return self._array.__dlpack_device__()

    def __eq__(self: Array, other: int | float | bool | Array, /) -> Array:  # type: ignore[override]
        """
        Performs the operation __eq__.
        """
        # Even though "all" dtypes are allowed, we still require them to be
        # promotable with each other.
        other = self._check_allowed_dtypes(other, "all", "__eq__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__eq__(other._array)
        return self.__class__._new(res)

    def __float__(self: Array, /) -> float:
        """
        Performs the operation __float__.
        """
        # Note: This is an error here.
        if self._array.ndim != 0:
            raise TypeError("float is only allowed on arrays with 0 dimensions")
        if self.dtype in complex_floating_dtypes:
            raise TypeError("float is not allowed on complex floating-point arrays")
        res = self._array.__float__()
        return res

    def __floordiv__(self: Array, other: int | float | Array, /) -> Array:
        """
        Performs the operation __floordiv__.
        """
        other = self._check_allowed_dtypes(other, "real numeric", "__floordiv__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__floordiv__(other._array)
        return self.__class__._new(res)

    def __ge__(self: Array, other: int | float | Array, /) -> Array:
        """
        Performs the operation __ge__.
        """
        other = self._check_allowed_dtypes(other, "real numeric", "__ge__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__ge__(other._array)
        return self.__class__._new(res)

    # def __getitem__(
    #     self: Array,
    #     key: Union[int, slice, ellipsis, Tuple[Union[int, slice, ellipsis], ...], Array],
    #     /,
    # ) -> Array:
    #     """
    #     Performs the operation __getitem__.
    #     """
    #     # Note: Only indices required by the spec are allowed. See the
    #     # docstring of _validate_index
    #     self._validate_index(key)
    #     if isinstance(key, Array):
    #         # Indexing self._array with array_api arrays can be erroneous
    #         key = key._array
    #     res = self._array.__getitem__(key)
    #     return self._new(res)

    def __gt__(self: Array, other: int | float | Array, /) -> Array:
        """
        Performs the operation __gt__.
        """
        other = self._check_allowed_dtypes(other, "real numeric", "__gt__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__gt__(other._array)
        return self.__class__._new(res)

    def __int__(self: Array, /) -> int:
        """
        Performs the operation __int__.
        """
        # Note: This is an error here.
        if self._array.ndim != 0:
            raise TypeError("int is only allowed on arrays with 0 dimensions")
        if self.dtype in complex_floating_dtypes:
            raise TypeError("int is not allowed on complex floating-point arrays")
        res = self._array.__int__()
        return res

    def __index__(self: Array, /) -> int:
        """
        Performs the operation __index__.
        """
        res = self._array.__index__()
        return res

    def __invert__(self: Array, /) -> Array:
        """
        Performs the operation __invert__.
        """
        if self.dtype not in integer_or_boolean_dtypes:
            raise TypeError("Only integer or boolean dtypes are allowed in __invert__")
        res = self._array.__invert__()
        return self.__class__._new(res)

    def __le__(self: Array, other: int | float | Array, /) -> Array:
        """
        Performs the operation __le__.
        """
        other = self._check_allowed_dtypes(other, "real numeric", "__le__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__le__(other._array)
        return self.__class__._new(res)

    def __lshift__(self: Array, other: int | Array, /) -> Array:
        """
        Performs the operation __lshift__.
        """
        other = self._check_allowed_dtypes(other, "integer", "__lshift__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__lshift__(other._array)
        return self.__class__._new(res)

    def __lt__(self: Array, other: int | float | Array, /) -> Array:
        """
        Performs the operation __lt__.
        """
        other = self._check_allowed_dtypes(other, "real numeric", "__lt__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__lt__(other._array)
        return self.__class__._new(res)

    def __matmul__(self: Array, other: Array, /) -> Array:
        """
        Performs the operation __matmul__.
        """
        # matmul is not defined for scalars, but without this, we may get
        # the wrong error message from asarray.
        other = self._check_allowed_dtypes(other, "numeric", "__matmul__")
        if other is NotImplemented:
            return other
        res = self._array.__matmul__(other._array)
        return self.__class__._new(res)

    def __mod__(self: Array, other: int | float | Array, /) -> Array:
        """
        Performs the operation __mod__.
        """
        other = self._check_allowed_dtypes(other, "real numeric", "__mod__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__mod__(other._array)
        return self.__class__._new(res)

    def __mul__(self: Array, other: int | float | Array, /) -> Array:
        """
        Performs the operation __mul__.
        """
        other = self._check_allowed_dtypes(other, "numeric", "__mul__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__mul__(other._array)
        return self.__class__._new(res)

    def __ne__(self: Array, other: int | float | bool | Array, /) -> Array:  # type: ignore[override]
        """
        Performs the operation __ne__.
        """
        other = self._check_allowed_dtypes(other, "all", "__ne__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__ne__(other._array)
        return self.__class__._new(res)

    def __neg__(self: Array, /) -> Array:
        """
        Performs the operation __neg__.
        """
        if self.dtype not in numeric_dtypes:
            raise TypeError("Only numeric dtypes are allowed in __neg__")
        res = self._array.__neg__()
        return self.__class__._new(res)

    def __or__(self: Array, other: int | bool | Array, /) -> Array:
        """
        Performs the operation __or__.
        """
        other = self._check_allowed_dtypes(other, "integer or boolean", "__or__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__or__(other._array)
        return self.__class__._new(res)

    def __pos__(self: Array, /) -> Array:
        """
        Performs the operation __pos__.
        """
        if self.dtype not in numeric_dtypes:
            raise TypeError("Only numeric dtypes are allowed in __pos__")
        res = self._array.__pos__()
        return self.__class__._new(res)

    # def __pow__(self: Array, other: int | float | Array, /) -> Array:
    #     """
    #     Performs the operation __pow__.
    #     """
    #     from ._elementwise_functions import pow

    #     other = self._check_allowed_dtypes(other, "numeric", "__pow__")
    #     if other is NotImplemented:
    #         return other
    #     # Note: NumPy's __pow__ does not follow type promotion rules for 0-d
    #     # arrays, so we use pow() here instead.
    #     return pow(self, other)

    def __rshift__(self: Array, other: int | Array, /) -> Array:
        """
        Performs the operation __rshift__.
        """
        other = self._check_allowed_dtypes(other, "integer", "__rshift__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__rshift__(other._array)
        return self.__class__._new(res)

    # def __setitem__(
    #     self,
    #     key: Union[int, slice, ellipsis, Tuple[Union[int, slice, ellipsis], ...], Array],
    #     value: int | float | bool | Array,
    #     /,
    # ) -> None:
    #     """
    #     Performs the operation __setitem__.
    #     """
    #     # Note: Only indices required by the spec are allowed. See the
    #     # docstring of _validate_index
    #     self._validate_index(key)
    #     if isinstance(key, Array):
    #         # Indexing self._array with array_api arrays can be erroneous
    #         key = key._array
    #     self._array.__setitem__(key, asarray(value)._array)

    def __sub__(self: Array, other: int | float | Array, /) -> Array:
        """
        Performs the operation __sub__.
        """
        other = self._check_allowed_dtypes(other, "numeric", "__sub__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__sub__(other._array)
        return self.__class__._new(res)

    # PEP 484 requires int to be a subtype of float, but __truediv__ should
    # not accept int.
    def __truediv__(self: Array, other: float | Array, /) -> Array:
        """
        Performs the operation __truediv__.
        """
        other = self._check_allowed_dtypes(other, "floating-point", "__truediv__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__truediv__(other._array)
        return self.__class__._new(res)

    def __xor__(self: Array, other: int | bool | Array, /) -> Array:
        """
        Performs the operation __xor__.
        """
        other = self._check_allowed_dtypes(other, "integer or boolean", "__xor__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__xor__(other._array)
        return self.__class__._new(res)

    def __iadd__(self: Array, other: int | float | Array, /) -> Array:
        """
        Performs the operation __iadd__.
        """
        other = self._check_allowed_dtypes(other, "numeric", "__iadd__")
        if other is NotImplemented:
            return other
        self._array.__iadd__(other._array)
        return self

    def __radd__(self: Array, other: int | float | Array, /) -> Array:
        """
        Performs the operation __radd__.
        """
        other = self._check_allowed_dtypes(other, "numeric", "__radd__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__radd__(other._array)
        return self.__class__._new(res)

    def __iand__(self: Array, other: int | bool | Array, /) -> Array:
        """
        Performs the operation __iand__.
        """
        other = self._check_allowed_dtypes(other, "integer or boolean", "__iand__")
        if other is NotImplemented:
            return other
        self._array.__iand__(other._array)
        return self

    def __rand__(self: Array, other: int | bool | Array, /) -> Array:
        """
        Performs the operation __rand__.
        """
        other = self._check_allowed_dtypes(other, "integer or boolean", "__rand__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__rand__(other._array)
        return self.__class__._new(res)

    def __ifloordiv__(self: Array, other: int | float | Array, /) -> Array:
        """
        Performs the operation __ifloordiv__.
        """
        other = self._check_allowed_dtypes(other, "real numeric", "__ifloordiv__")
        if other is NotImplemented:
            return other
        self._array.__ifloordiv__(other._array)
        return self

    def __rfloordiv__(self: Array, other: int | float | Array, /) -> Array:
        """
        Performs the operation __rfloordiv__.
        """
        other = self._check_allowed_dtypes(other, "real numeric", "__rfloordiv__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__rfloordiv__(other._array)
        return self.__class__._new(res)

    def __ilshift__(self: Array, other: int | Array, /) -> Array:
        """
        Performs the operation __ilshift__.
        """
        other = self._check_allowed_dtypes(other, "integer", "__ilshift__")
        if other is NotImplemented:
            return other
        self._array.__ilshift__(other._array)
        return self

    def __rlshift__(self: Array, other: int | Array, /) -> Array:
        """
        Performs the operation __rlshift__.
        """
        other = self._check_allowed_dtypes(other, "integer", "__rlshift__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__rlshift__(other._array)
        return self.__class__._new(res)

    def __imatmul__(self: Array, other: Array, /) -> Array:
        """
        Performs the operation __imatmul__.
        """
        # matmul is not defined for scalars, but without this, we may get
        # the wrong error message from asarray.
        other = self._check_allowed_dtypes(other, "numeric", "__imatmul__")
        if other is NotImplemented:
            return other
        res = self._array.__imatmul__(other._array)
        return self.__class__._new(res)

    def __rmatmul__(self: Array, other: Array, /) -> Array:
        """
        Performs the operation __rmatmul__.
        """
        # matmul is not defined for scalars, but without this, we may get
        # the wrong error message from asarray.
        other = self._check_allowed_dtypes(other, "numeric", "__rmatmul__")
        if other is NotImplemented:
            return other
        res = self._array.__rmatmul__(other._array)
        return self.__class__._new(res)

    def __imod__(self: Array, other: int | float | Array, /) -> Array:
        """
        Performs the operation __imod__.
        """
        other = self._check_allowed_dtypes(other, "real numeric", "__imod__")
        if other is NotImplemented:
            return other
        self._array.__imod__(other._array)
        return self

    def __rmod__(self: Array, other: int | float | Array, /) -> Array:
        """
        Performs the operation __rmod__.
        """
        other = self._check_allowed_dtypes(other, "real numeric", "__rmod__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__rmod__(other._array)
        return self.__class__._new(res)

    def __imul__(self: Array, other: int | float | Array, /) -> Array:
        """
        Performs the operation __imul__.
        """
        other = self._check_allowed_dtypes(other, "numeric", "__imul__")
        if other is NotImplemented:
            return other
        self._array.__imul__(other._array)
        return self

    def __rmul__(self: Array, other: int | float | Array, /) -> Array:
        """
        Performs the operation __rmul__.
        """
        other = self._check_allowed_dtypes(other, "numeric", "__rmul__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__rmul__(other._array)
        return self.__class__._new(res)

    def __ior__(self: Array, other: int | bool | Array, /) -> Array:
        """
        Performs the operation __ior__.
        """
        other = self._check_allowed_dtypes(other, "integer or boolean", "__ior__")
        if other is NotImplemented:
            return other
        self._array.__ior__(other._array)
        return self

    def __ror__(self: Array, other: int | bool | Array, /) -> Array:
        """
        Performs the operation __ror__.
        """
        other = self._check_allowed_dtypes(other, "integer or boolean", "__ror__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__ror__(other._array)
        return self.__class__._new(res)

    def __ipow__(self: Array, other: int | float | Array, /) -> Array:
        """
        Performs the operation __ipow__.
        """
        other = self._check_allowed_dtypes(other, "numeric", "__ipow__")
        if other is NotImplemented:
            return other
        self._array.__ipow__(other._array)
        return self

    def __rpow__(self: Array, other: int | float | Array, /) -> Array:
        """
        Performs the operation __rpow__.
        """
        other = self._check_allowed_dtypes(other, "numeric", "__rpow__")
        if other is NotImplemented:
            return other
        self._array.__rpow__(other._array)
        return self

    def __irshift__(self: Array, other: int | Array, /) -> Array:
        """
        Performs the operation __irshift__.
        """
        other = self._check_allowed_dtypes(other, "integer", "__irshift__")
        if other is NotImplemented:
            return other
        self._array.__irshift__(other._array)
        return self

    def __rrshift__(self: Array, other: int | Array, /) -> Array:
        """
        Performs the operation __rrshift__.
        """
        other = self._check_allowed_dtypes(other, "integer", "__rrshift__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__rrshift__(other._array)
        return self.__class__._new(res)

    def __isub__(self: Array, other: int | float | Array, /) -> Array:
        """
        Performs the operation __isub__.
        """
        other = self._check_allowed_dtypes(other, "numeric", "__isub__")
        if other is NotImplemented:
            return other
        self._array.__isub__(other._array)
        return self

    def __rsub__(self: Array, other: int | float | Array, /) -> Array:
        """
        Performs the operation __rsub__.
        """
        other = self._check_allowed_dtypes(other, "numeric", "__rsub__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__rsub__(other._array)
        return self.__class__._new(res)

    def __itruediv__(self: Array, other: float | Array, /) -> Array:
        """
        Performs the operation __itruediv__.
        """
        other = self._check_allowed_dtypes(other, "floating-point", "__itruediv__")
        if other is NotImplemented:
            return other
        self._array.__itruediv__(other._array)
        return self

    def __rtruediv__(self: Array, other: float | Array, /) -> Array:
        """
        Performs the operation __rtruediv__.
        """
        other = self._check_allowed_dtypes(other, "floating-point", "__rtruediv__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__rtruediv__(other._array)
        return self.__class__._new(res)

    def __ixor__(self: Array, other: int | bool | Array, /) -> Array:
        """
        Performs the operation __ixor__.
        """
        other = self._check_allowed_dtypes(other, "integer or boolean", "__ixor__")
        if other is NotImplemented:
            return other
        self._array.__ixor__(other._array)
        return self

    def __rxor__(self: Array, other: int | bool | Array, /) -> Array:
        """
        Performs the operation __rxor__.
        """
        other = self._check_allowed_dtypes(other, "integer or boolean", "__rxor__")
        if other is NotImplemented:
            return other
        self, other = self._normalize_two_args(self, other)
        res = self._array.__rxor__(other._array)
        return self.__class__._new(res)

    def to_device(self: Array, device: Device, /, stream: None = None) -> Array:
        # TODO
        # Pseudocode:
        # af_malloc_host(...)
        # af_write_array (void* allocated on host)
        # af_create_array(host_pointer, new_device)
        # af_free_host(host_pointer)

        if stream is not None:
            raise ValueError("The stream argument to to_device() is not supported")

        return NotImplemented

    @property
    def dtype(self) -> af.Dtype:
        """
        Array API compatible wrapper for :py:meth:`np.ndarray.dtype <numpy.ndarray.dtype>`.

        See its docstring for more information.
        """
        return self._array.dtype

    @property
    def device(self) -> Device:
        # TODO
        # Pseudocode:
        # return Device(af.get_backend.get_arrays_active_backend(self), af.get_arrays_device(self))
        return NotImplemented

    @property
    def mT(self) -> Array:
        # TODO
        return NotImplemented

    @property
    def ndim(self) -> int:
        """
        Array API compatible wrapper for :py:meth:`np.ndarray.ndim <numpy.ndarray.ndim>`.

        See its docstring for more information.
        """
        return self._array.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Array API compatible wrapper for :py:meth:`np.ndarray.shape <numpy.ndarray.shape>`.

        See its docstring for more information.
        """
        return self._array.shape

    @property
    def size(self) -> int:
        """
        Array API compatible wrapper for :py:meth:`np.ndarray.size <numpy.ndarray.size>`.

        See its docstring for more information.
        """
        return self._array.size

    @property
    def T(self) -> Array:
        """
        Array API compatible wrapper for :py:meth:`np.ndarray.T <numpy.ndarray.T>`.

        See its docstring for more information.
        """
        # NOTE: T only works on 2-dimensional arrays. See the corresponding
        # note in the specification:
        # https://data-apis.org/array-api/latest/API_specification/array_object.html#t
        if self.ndim != 2:
            raise ValueError(
                "x.T requires x to have 2 dimensions. "
                "Use x.mT to transpose stacks of matrices and permute_dims() to permute dimensions."
            )
        return self.__class__._new(self._array.T)
