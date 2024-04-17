from __future__ import annotations

__all__ = ["Array"]

import array as _pyarray
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, ParamSpec, cast

import arrayfire_wrapper.lib as wrapper
from arrayfire_wrapper.defines import AFArray, ArrayBuffer, CType

from .dtypes import Dtype
from .dtypes import bool as afbool
from .dtypes import c_api_value_to_dtype, float32, str_to_dtype

if TYPE_CHECKING:
    from ctypes import Array as CArray
    from enum import Enum

P = ParamSpec("P")


def afarray_as_array(func: Callable[P, Array]) -> Callable[P, Array]:
    """
    Decorator that converts a function returning an array to return an ArrayFire Array.

    Parameters
    ----------
    func : Callable[P, Array]
        The original function that returns an array.

    Returns
    -------
    Callable[P, Array]
        A decorated function that returns an ArrayFire Array.
    """

    @wraps(func)
    def decorated(*args: P.args, **kwargs: P.kwargs) -> Array:
        result = func(*args, **kwargs)
        return Array.from_afarray(result)  # type: ignore[arg-type]  # FIXME
    return decorated


class Array:
    def __init__(
        self,
        obj: None | Array | _pyarray.array | int | AFArray | list[bool | int | float] = None,
        dtype: None | Dtype | str = None,
        shape: tuple[int, ...] = (),
        to_device: bool = False,
        offset: CType | None = None,
        strides: tuple[int, ...] | None = None,
    ) -> None:
        self._arr = AFArray.create_null_pointer()
        _no_initial_dtype = False  # HACK, FIXME

        if len(shape) > 4:
            raise ValueError("Can not create 5 or more -dimensional arrays.")

        if isinstance(dtype, str):
            dtype = str_to_dtype(dtype)  # type: ignore[arg-type]

        if dtype is None:
            _no_initial_dtype = True
            dtype = float32

        if obj is None:
            if not shape:  # shape is None or empty tuple
                self._arr = wrapper.create_handle((), dtype)
                return

            self._arr = wrapper.create_handle(shape, dtype)
            return

        if isinstance(obj, Array):
            self._arr = wrapper.retain_array(obj.arr)
            return

        if isinstance(obj, _pyarray.array):
            _type_char: str = obj.typecode
            _array_buffer = ArrayBuffer(*obj.buffer_info())

        elif isinstance(obj, list):
            # TODO fix an issue when Array can not be created from float values to complex
            if _no_initial_dtype:
                arr_typecode = "f"
            elif dtype.typecode in _pyarray.typecodes:
                arr_typecode = dtype.typecode
            else:
                raise TypeError(f"Unsupported typecode. Can not create a python array from '{repr(dtype)}'")

            _array = _pyarray.array(arr_typecode, obj)
            _type_char = _array.typecode
            _array_buffer = ArrayBuffer(*_array.buffer_info())

        elif isinstance(obj, int) or isinstance(obj, AFArray):
            _array_buffer = ArrayBuffer(obj if not isinstance(obj, AFArray) else obj.value)  # type: ignore[arg-type]

            if not shape:
                raise TypeError("Expected to receive the initial shape due to the obj being a data pointer.")

            if _no_initial_dtype:
                raise TypeError("Expected to receive the initial dtype due to the obj being a data pointer.")

            _type_char = dtype.typecode

        else:
            raise TypeError("Passed object obj is an object of unsupported class.")

        if not shape:
            if _array_buffer.length != 0:
                shape = (_array_buffer.length,)
            else:
                RuntimeError("Shape and buffer length are size invalid.")

        if not _no_initial_dtype and dtype.typecode != _type_char:
            raise TypeError("Can not create array of requested type from input data type")

        if not (offset or strides):
            if not to_device:
                self._arr = wrapper.create_array(shape, dtype, _array_buffer)
                return

            self._arr = wrapper.device_array(shape, dtype, _array_buffer)
            return

        self._arr = wrapper.create_strided_array(
            shape, dtype, _array_buffer, offset, strides, wrapper.PointerSource(to_device)  # type: ignore[arg-type]
        )

    # Arithmetic Operators

    def __pos__(self) -> Array:
        """
        Evaluates +self_i for each element of an array instance.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.

        Returns
        -------
        out : Array
            An array containing the evaluated result for each element. The returned array must have the same data type
            as self.
        """
        return self

    def __neg__(self) -> Array:
        """
        Evaluates +self_i for each element of an array instance.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.

        Returns
        -------
        out : Array
            An array containing the evaluated result for each element in self. The returned array must have a data type
            determined by Type Promotion Rules.

        """
        return process_c_function(0, self, wrapper.sub)

    def __add__(self, other: int | float | Array, /) -> Array:
        """
        Calculates the sum for each element of an array instance with the respective element of the array other.

        Parameters
        ----------
        self : Array
            Array instance (augend array). Should have a numeric data type.
        other: int | float | Array
            Addend array. Must be compatible with self (see Broadcasting). Should have a numeric data type.

        Returns
        -------
        out : Array
            An array containing the element-wise sums. The returned array must have a data type determined
            by Type Promotion Rules.
        """
        return process_c_function(self, other, wrapper.add)

    def __sub__(self, other: int | float | Array, /) -> Array:
        """
        Calculates the difference for each element of an array instance with the respective element of the array other.

        The result of self_i - other_i must be the same as self_i + (-other_i) and must be governed by the same
        floating-point rules as addition (see array.__add__()).

        Parameters
        ----------
        self : Array
            Array instance (minuend array). Should have a numeric data type.
        other: int | float | Array
            Subtrahend array. Must be compatible with self (see Broadcasting). Should have a numeric data type.

        Returns
        -------
        out : Array
            An array containing the element-wise differences. The returned array must have a data type determined
            by Type Promotion Rules.
        """
        return process_c_function(self, other, wrapper.sub)

    def __mul__(self, other: int | float | Array, /) -> Array:
        """
        Calculates the product for each element of an array instance with the respective element of the array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: int | float | Array
            Other array. Must be compatible with self (see Broadcasting). Should have a numeric data type.

        Returns
        -------
        out : Array
            An array containing the element-wise products. The returned array must have a data type determined
            by Type Promotion Rules.
        """
        return process_c_function(self, other, wrapper.mul)

    def __truediv__(self, other: int | float | Array, /) -> Array:
        """
        Evaluates self_i / other_i for each element of an array instance with the respective element of the
        array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: int | float | Array
            Other array. Must be compatible with self (see Broadcasting). Should have a numeric data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array should have a floating-point data type
            determined by Type Promotion Rules.

        Note
        ----
        - If one or both of self and other have integer data types, the result is implementation-dependent, as type
        promotion between data type “kinds” (e.g., integer versus floating-point) is unspecified.
        Specification-compliant libraries may choose to raise an error or return an array containing the element-wise
        results. If an array is returned, the array must have a real-valued floating-point data type.
        """
        return process_c_function(self, other, wrapper.div)

    def __floordiv__(self, other: int | float | Array, /) -> Array:
        # TODO
        return NotImplemented

    def __mod__(self, other: int | float | Array, /) -> Array:
        """
        Evaluates self_i % other_i for each element of an array instance with the respective element of the
        array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a real-valued data type.
        other: int | float | Array
            Other array. Must be compatible with self (see Broadcasting). Should have a real-valued data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. Each element-wise result must have the same sign as the
            respective element other_i. The returned array must have a real-valued floating-point data type determined
            by Type Promotion Rules.

        Note
        ----
        - For input arrays which promote to an integer data type, the result of division by zero is unspecified and
        thus implementation-defined.
        """
        return process_c_function(self, other, wrapper.mod)

    def __pow__(self, other: int | float | Array, /) -> Array:
        """
        Calculates an implementation-dependent approximation of exponentiation by raising each element (the base) of
        an array instance to the power of other_i (the exponent), where other_i is the corresponding element of the
        array other.

        Parameters
        ----------
        self : Array
            Array instance whose elements correspond to the exponentiation base. Should have a numeric data type.
        other: int | float | Array
            Other array whose elements correspond to the exponentiation exponent. Must be compatible with self
            (see Broadcasting). Should have a numeric data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type determined
            by Type Promotion Rules.
        """
        return process_c_function(self, other, wrapper.pow)

    # Array Operators

    def __matmul__(self, other: Array, /) -> Array:
        # TODO get from blas - make vanilla version and not copy af.matmul as is
        return NotImplemented

    # Bitwise Operators

    def __invert__(self) -> Array:
        """
        Evaluates ~self_i for each element of an array instance.

        Parameters
        ----------
        self : Array
            Array instance. Should have an integer or boolean data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have the same data type as self.
        """
        return Array.from_afarray(wrapper.bitnot(self._arr))

    def __and__(self, other: int | bool | Array, /) -> Array:
        """
        Evaluates self_i & other_i for each element of an array instance with the respective element of the
        array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: int | bool | Array
            Other array. Must be compatible with self (see Broadcasting). Should have a numeric data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type determined
            by Type Promotion Rules.
        """
        return process_c_function(self, other, wrapper.bitand)

    def __or__(self, other: int | bool | Array, /) -> Array:
        """
        Evaluates self_i | other_i for each element of an array instance with the respective element of the
        array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: int | bool | Array
            Other array. Must be compatible with self (see Broadcasting). Should have a numeric data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type determined
            by Type Promotion Rules.
        """
        return process_c_function(self, other, wrapper.bitor)

    def __xor__(self, other: int | bool | Array, /) -> Array:
        """
        Evaluates self_i ^ other_i for each element of an array instance with the respective element of the
        array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: int | bool | Array
            Other array. Must be compatible with self (see Broadcasting). Should have a numeric data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type determined
            by Type Promotion Rules.
        """
        return process_c_function(self, other, wrapper.bitxor)

    def __lshift__(self, other: int | Array, /) -> Array:
        """
        Evaluates self_i << other_i for each element of an array instance with the respective element of the
        array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: int | Array
            Other array. Must be compatible with self (see Broadcasting). Should have a numeric data type.
            Each element must be greater than or equal to 0.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have the same data type as self.
        """
        return process_c_function(self, other, wrapper.bitshiftl)

    def __rshift__(self, other: int | Array, /) -> Array:
        """
        Evaluates self_i >> other_i for each element of an array instance with the respective element of the
        array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: int | Array
            Other array. Must be compatible with self (see Broadcasting). Should have a numeric data type.
            Each element must be greater than or equal to 0.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have the same data type as self.
        """
        return process_c_function(self, other, wrapper.bitshiftr)

    # Comparison Operators

    def __lt__(self, other: int | float | Array, /) -> Array:
        """
        Computes the truth value of self_i < other_i for each element of an array instance with the respective
        element of the array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: int | float | Array
            Other array. Must be compatible with self (see Broadcasting). Should have a real-valued data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type of bool.
        """
        return process_c_function(self, other, wrapper.lt)

    def __le__(self, other: int | float | Array, /) -> Array:
        """
        Computes the truth value of self_i <= other_i for each element of an array instance with the respective
        element of the array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: int | float | Array
            Other array. Must be compatible with self (see Broadcasting). Should have a real-valued data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type of bool.
        """
        return process_c_function(self, other, wrapper.le)

    def __gt__(self, other: int | float | Array, /) -> Array:
        """
        Computes the truth value of self_i > other_i for each element of an array instance with the respective
        element of the array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: int | float | Array
            Other array. Must be compatible with self (see Broadcasting). Should have a real-valued data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type of bool.
        """
        return process_c_function(self, other, wrapper.gt)

    def __ge__(self, other: int | float | Array, /) -> Array:
        """
        Computes the truth value of self_i >= other_i for each element of an array instance with the respective
        element of the array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: int | float | Array
            Other array. Must be compatible with self (see Broadcasting). Should have a real-valued data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type of bool.
        """
        return process_c_function(self, other, wrapper.ge)

    def __eq__(self, other: int | float | bool | Array, /) -> Array:  # type: ignore[override]
        """
        Computes the truth value of self_i == other_i for each element of an array instance with the respective
        element of the array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: int | float | bool | Array
            Other array. Must be compatible with self (see Broadcasting). May have any data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type of bool.
        """
        return process_c_function(self, other, wrapper.eq)

    def __ne__(self, other: int | float | bool | Array, /) -> Array:  # type: ignore[override]
        """
        Computes the truth value of self_i != other_i for each element of an array instance with the respective
        element of the array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: int | float | bool | Array
            Other array. Must be compatible with self (see Broadcasting). May have any data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type of bool.
        """
        return process_c_function(self, other, wrapper.neq)

    # Reflected Arithmetic Operators

    def __radd__(self, other: int | float | Array, /) -> Array:
        """
        Return other + self.
        """
        return process_c_function(other, self, wrapper.add)

    def __rsub__(self, other: int | float | Array, /) -> Array:
        """
        Return other - self.
        """
        return process_c_function(other, self, wrapper.sub)

    def __rmul__(self, other: int | float | Array, /) -> Array:
        """
        Return other * self.
        """
        return process_c_function(other, self, wrapper.mul)

    def __rtruediv__(self, other: int | float | Array, /) -> Array:
        """
        Return other / self.
        """
        return process_c_function(other, self, wrapper.div)

    def __rfloordiv__(self, other: int | float | Array, /) -> Array:
        # TODO
        return NotImplemented

    def __rmod__(self, other: int | float | Array, /) -> Array:
        """
        Return other % self.
        """
        return process_c_function(other, self, wrapper.mod)

    def __rpow__(self, other: int | float | Array, /) -> Array:
        """
        Return other ** self.
        """
        return process_c_function(other, self, wrapper.pow)

    # Reflected Array Operators

    def __rmatmul__(self, other: Array, /) -> Array:
        # TODO
        return NotImplemented

    # Reflected Bitwise Operators

    def __rand__(self, other: int | bool | Array, /) -> Array:
        """
        Return other & self.
        """
        return process_c_function(other, self, wrapper.bitand)

    def __ror__(self, other: int | bool | Array, /) -> Array:
        """
        Return other | self.
        """
        return process_c_function(other, self, wrapper.bitor)

    def __rxor__(self, other: int | bool | Array, /) -> Array:
        """
        Return other ^ self.
        """
        return process_c_function(other, self, wrapper.bitxor)

    def __rlshift__(self, other: int | Array, /) -> Array:
        """
        Return other << self.
        """
        return process_c_function(other, self, wrapper.bitshiftl)

    def __rrshift__(self, other: int | Array, /) -> Array:
        """
        Return other >> self.
        """
        return process_c_function(other, self, wrapper.bitshiftr)

    # In-place Arithmetic Operators

    def __iadd__(self, other: int | float | Array, /) -> Array:
        # TODO discuss either we need to support complex and bool as other input type
        """
        Return self += other.
        """
        return process_c_function(self, other, wrapper.add)

    def __isub__(self, other: int | float | Array, /) -> Array:
        """
        Return self -= other.
        """
        return process_c_function(self, other, wrapper.sub)

    def __imul__(self, other: int | float | Array, /) -> Array:
        """
        Return self *= other.
        """
        return process_c_function(self, other, wrapper.mul)

    def __itruediv__(self, other: int | float | Array, /) -> Array:
        """
        Return self /= other.
        """
        return process_c_function(self, other, wrapper.div)

    def __ifloordiv__(self, other: int | float | Array, /) -> Array:
        # TODO
        return NotImplemented

    def __imod__(self, other: int | float | Array, /) -> Array:
        """
        Return self %= other.
        """
        return process_c_function(self, other, wrapper.mod)

    def __ipow__(self, other: int | float | Array, /) -> Array:
        """
        Return self **= other.
        """
        return process_c_function(self, other, wrapper.pow)

    # In-place Array Operators

    def __imatmul__(self, other: Array, /) -> Array:
        # TODO
        return NotImplemented

    # In-place Bitwise Operators

    def __iand__(self, other: int | bool | Array, /) -> Array:
        """
        Return self &= other.
        """
        return process_c_function(self, other, wrapper.bitand)

    def __ior__(self, other: int | bool | Array, /) -> Array:
        """
        Return self |= other.
        """
        return process_c_function(self, other, wrapper.bitor)

    def __ixor__(self, other: int | bool | Array, /) -> Array:
        """
        Return self ^= other.
        """
        return process_c_function(self, other, wrapper.bitxor)

    def __ilshift__(self, other: int | Array, /) -> Array:
        """
        Return self <<= other.
        """
        return process_c_function(self, other, wrapper.bitshiftl)

    def __irshift__(self, other: int | Array, /) -> Array:
        """
        Return self >>= other.
        """
        return process_c_function(self, other, wrapper.bitshiftr)

    # Methods

    def __abs__(self) -> Array:
        # TODO
        return NotImplemented

    def __array_namespace__(self, *, api_version: str | None = None) -> Any:
        # TODO
        return NotImplemented

    # def __bool__(self) -> bool:
    #     # TODO consider using scalar() and is_scalar()
    #     return NotImplemented

    def __complex__(self) -> complex:
        # TODO
        return NotImplemented

    def __dlpack__(self, *, stream: int | Any | None = None):  # type: ignore[no-untyped-def]
        # TODO implementation and expected return type -> PyCapsule
        return NotImplemented

    def __dlpack_device__(self) -> tuple[Enum, int]:
        # TODO
        return NotImplemented

    def __float__(self) -> float:
        # TODO
        return NotImplemented

    def __getitem__(self, key: IndexKey, /) -> Array:
        """
        Returns self[key].

        Parameters
        ----------
        self : Array
            Array instance.
        key : int | slice | tuple[int | slice, ...] | Array
            Index key.

        Returns
        -------
        out : Array
            An array containing the accessed value(s). The returned array must have the same data type as self.
        """
        # TODO
        # API Specification - key: Union[int, slice, ellipsis, tuple[Union[int, slice, ellipsis], ...], array].
        # consider using af.span to replace ellipsis during refactoring
        out = Array()
        ndims = self.ndim

        if isinstance(key, Array) and key == afbool.c_api_value:
            ndims = 1
            if wrapper.count_all(key.arr) == 0:  # HACK was count() method before
                return out

        # HACK known issue
        out._arr = wrapper.index_gen(self._arr, ndims, wrapper.get_indices(key))  # type: ignore[arg-type]
        return out

    def __index__(self) -> int:
        # TODO
        return NotImplemented

    def __int__(self) -> int:
        # TODO
        return NotImplemented

    def __len__(self) -> int:
        return self.shape[0] if self.shape else 0

    def __setitem__(self, key: IndexKey, value: int | float | bool | Array, /) -> None:
        ndims = self.ndim

        is_array_with_bool = isinstance(key, Array) and type(key) is afbool

        if is_array_with_bool:
            ndims = 1
            num = wrapper.count_all(key.arr)  # type: ignore[union-attr]
            if num == 0:
                return

        if isinstance(value, int | float | complex | bool):
            dims = _get_processed_index(key, self.shape)
            if is_array_with_bool:
                ndims = 1
                other_arr = wrapper.create_constant_array(value, (int(num.real),), self.dtype)
            else:
                other_arr = wrapper.create_constant_array(value, dims, self.dtype)
            del_other = True
        else:
            other_arr = value.arr
            del_other = False

        indices = wrapper.get_indices(key)  # type: ignore[arg-type]  # FIXME
        out = wrapper.assign_gen(self._arr, other_arr, ndims, indices)

        wrapper.release_array(self._arr)
        if del_other:
            wrapper.release_array(other_arr)
        self._arr = out

    def __str__(self) -> str:
        # TODO change the look of array str. E.g., like np.array
        # if not _in_display_dims_limit(self.shape):
        #     return _metadata_string(self.dtype, self.shape)
        return _metadata_string(self.dtype) + _array_as_str(self)

    def __repr__(self) -> str:
        # return _metadata_string(self.dtype, self.shape)
        # TODO change the look of array representation. E.g., like np.array
        return _array_as_str(self)

    def __del__(self) -> None:
        if not hasattr(self._arr, "value"):
            return

        if self._arr.value == 0:
            return

        wrapper.release_array(self._arr)
        self._arr.value = 0

    def to_device(self, device: Any, /, *, stream: int | Any = None) -> Array:
        # TODO implementation and change device type from Any to Device
        return NotImplemented

    # Attributes

    @property
    def dtype(self) -> Dtype:
        """
        Data type of the array elements.

        Returns
        -------
        out : Dtype
            Array data type.
        """
        return c_api_value_to_dtype(wrapper.get_type(self._arr))

    @property
    def device(self) -> Any:
        # TODO
        return NotImplemented

    @property
    @afarray_as_array
    def T(self) -> Array:
        """
        Transpose of the array.

        Returns
        -------
        Array
            Two-dimensional array whose first and last dimensions (axes) are permuted in reverse order relative to
            original array. The returned array must have the same data type as the original array.

        Note
        ----
        - The array instance must be two-dimensional. If the array instance is not two-dimensional, an error
        should be raised.
        """
        if self.ndim < 2:
            raise TypeError(f"Array should be at least 2-dimensional. Got {self.ndim}-dimensional array")

        # TODO add check if out.dtype == self.dtype
        return cast(Array, wrapper.transpose(self._arr, False))

    @property
    @afarray_as_array
    def H(self) -> Array:
        return cast(Array, wrapper.transpose(self._arr, True))

    @property
    def size(self) -> int:
        """
        Number of elements in an array.

        Returns
        -------
        int
            Number of elements in an array

        Note
        ----
        - This must equal the product of the array's dimensions.
        """
        # NOTE previously - elements()
        return wrapper.get_elements(self._arr)

    @property
    def ndim(self) -> int:
        """
        Number of array dimensions (axes).

        int
            Number of array dimensions (axes).
        """
        return wrapper.get_numdims(self._arr)

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Array dimensions.

        Returns
        -------
        tuple[int, ...]
            Array dimensions.
        """
        # NOTE skipping passing any None values
        return wrapper.get_dims(self._arr)[: self.ndim]

    @property
    def offset(self) -> int:
        """
        Return the offset of the first element relative to the raw pointer.

        Returns
        -------
        int
            The offset in number of elements.
        """
        return wrapper.get_offset(self._arr)

    @property
    def strides(self) -> tuple[int, ...]:
        """
        Return the distance in bytes between consecutive elements for each dimension.

        Returns
        -------
        tuple[int, ...]
            The strides for each dimension.
        """
        return wrapper.get_strides(self._arr)[: self.ndim]

    # TODO rename front_to_host or smth. Extend doc: move first element of array from gpu to cpu
    def scalar(self) -> int | float | bool | complex | None:  # FIXME
        """
        Return the first element of the array
        """
        # TODO change the logic of this method
        if self.is_empty():
            return None

        return wrapper.get_scalar(self._arr, self.dtype)

    def is_empty(self) -> bool:
        """
        Check if the array is empty i.e. it has no elements.
        """
        return wrapper.is_empty(self._arr)

    def to_list(self, row_major: bool = False) -> list[int | float | bool | complex]:
        if self.is_empty():
            return []

        array = _reorder(self) if row_major else self
        ctypes_array = wrapper.get_data_ptr(array.arr, array.size, array.dtype)

        if array.ndim == 1:
            return cast(list, ctypes_array[:])  # HACK

        out = []
        for i in range(array.size):
            idx = i
            sub_list = []
            for j in range(array.ndim):
                div = array.shape[j]
                sub_list.append(idx % div)
                idx //= div
            out.append(ctypes_array[tuple(sub_list)])  # type: ignore[call-overload]  # FIXME
        return out

    def to_ctype_array(self, row_major: bool = False) -> CArray:
        if self.is_empty():
            raise RuntimeError("Can not convert an empty array to ctype.")

        array = _reorder(self) if row_major else self
        return wrapper.get_data_ptr(array.arr, array.size, array.dtype)

    @afarray_as_array
    def copy(self) -> Array:
        """
        Performs a deep copy of the array.

        Returns
        -------
        out: af.Array()
             An identical copy of self.
        """

        return cast(Array, wrapper.copy_array(self._arr))

    @property
    def arr(self) -> AFArray:
        return self._arr

    @classmethod
    def from_afarray(cls, arr: AFArray) -> Array:
        """
        Creates an instance of Array from an AFArray object.

        Parameters
        ----------
        array: AFArray
            The array object to wrap in the Array instance.

        Returns
        -------
        Array
            An instance of Array wrapping the given array.
        """
        out = cls()
        out._arr = arr
        return out

    @property
    def is_linear(self) -> bool:
        return wrapper.is_linear(self._arr)

    @property
    def is_owner(self) -> bool:
        return wrapper.is_owner(self._arr)

    @property
    def is_bool(self) -> bool:
        return wrapper.is_bool(self._arr)

    @property
    def is_column(self) -> bool:
        return wrapper.is_column(self._arr)

    @property
    def is_row(self) -> bool:
        return wrapper.is_row(self._arr)

    @property
    def is_complex(self) -> bool:
        return wrapper.is_complex(self._arr)

    @property
    def is_double(self) -> bool:
        return wrapper.is_double(self._arr)

    @property
    def is_floating(self) -> bool:
        return wrapper.is_floating(self._arr)

    @property
    def is_half(self) -> bool:
        return wrapper.is_half(self._arr)

    @property
    def is_integer(self) -> bool:
        return wrapper.is_integer(self._arr)

    @property
    def is_real(self) -> bool:
        return wrapper.is_real(self._arr)

    @property
    def is_real_floating(self) -> bool:
        return wrapper.is_realfloating(self._arr)

    @property
    def is_single(self) -> bool:
        return wrapper.is_single(self._arr)

    @property
    def is_sparse(self) -> bool:
        return wrapper.is_sparse(self._arr)

    @property
    def is_vector(self) -> bool:
        return wrapper.is_vector(self._arr)

    @property
    def device_pointer(self) -> int:
        return wrapper.get_device_ptr(self._arr)

    @property
    def is_locked_array(self) -> bool:
        return wrapper.is_locked_array(self._arr)

    def lock_array(self) -> None:
        return wrapper.lock_array(self._arr)

    def unlock_array(self) -> None:
        return wrapper.unlock_array(self._arr)


IndexKey = int | float | complex | bool | wrapper.ParallelRange | slice | tuple[int | slice, ...] | Array


def _reorder(array: Array) -> Array:
    """
    Returns a reordered array to help interoperate with row major formats.
    """
    if array.ndim == 1:
        return array

    return Array(wrapper.reorder(array.arr, array.ndim))


def _metadata_string(dtype: Dtype, dims: tuple[int, ...] | None = None) -> str:
    return "arrayfire.Array()\n" f"Type: {dtype.name}\n" f"Dims: {str(dims) if dims else ''}"


@afarray_as_array
def process_c_function(lhs: int | float | Array, rhs: int | float | Array, c_function: Any) -> Array:
    if isinstance(lhs, Array) and isinstance(rhs, Array):
        lhs_array = lhs.arr
        rhs_array = rhs.arr

    elif isinstance(lhs, Array) and isinstance(rhs, int | float):
        lhs_array = lhs.arr
        rhs_array = wrapper.create_constant_array(rhs, lhs.shape, lhs.dtype)

    elif isinstance(lhs, int | float) and isinstance(rhs, Array):
        lhs_array = wrapper.create_constant_array(lhs, rhs.shape, rhs.dtype)
        rhs_array = rhs.arr

    else:
        raise TypeError(f"{type(rhs)} is not supported and can not be passed to C binary function.")

    return cast(Array, c_function(lhs_array, rhs_array))


def _get_processed_index(key: IndexKey, shape: tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(key, tuple):
        return tuple(_index_to_afindex(key[i], shape[i]) for i in range(len(key)))

    return (_index_to_afindex(key, shape[0]),) + shape[1:]


def _index_to_afindex(key: int | float | complex | bool | slice | wrapper.ParallelRange | Array, axis: int) -> int:
    if isinstance(key, int | float | complex | bool):
        out = 1
    elif isinstance(key, slice):
        out = _slice_to_length(key, axis)
    elif isinstance(key, wrapper.ParallelRange):
        out = _slice_to_length(key.S, axis)
    elif isinstance(key, Array):
        if key.dtype == afbool:
            from arrayfire.library.vector_algorithms import sum as af_sum

            out = int(af_sum(key))  # type: ignore[arg-type]
        else:
            out = key.size
    else:
        raise IndexError(f"Invalid key type {type(key)}.")

    return out


def _slice_to_length(key: slice, axis: int) -> int:
    # print(key, axis)

    start = key.start
    stop = key.stop
    step = key.step

    if key.start is None:
        start = 0
    elif key.start < 0:
        start = axis - key.start

    if key.stop is None:
        stop = axis
    elif key.stop < 0:
        stop = axis - key.stop

    if key.step is None:
        step = 1

    return int(((stop - start - 1) / step) + 1)


def _array_as_str(array: Array) -> str:
    return wrapper.array_to_string("", array.arr, 4, True)
