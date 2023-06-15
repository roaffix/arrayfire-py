from __future__ import annotations

import array as py_array
import ctypes
import enum
from typing import Any, List, Optional, Tuple, Union

from .. import backend
from ..backend import ArrayBuffer
from ..backend.wrapped import everything
from ..backend.wrapped.constant_array import create_constant_array
from ..backend.wrapped.indexing import CIndexStructure, IndexStructure
from ..backend.wrapped.reduction_operations import count_all
from ..dtypes import CType
from ..dtypes import bool as af_bool
from ..dtypes import float32 as af_float32
from ..dtypes.helpers import Dtype, c_api_value_to_dtype, str_to_dtype
from .device import PointerSource

# TODO use int | float in operators -> remove bool | complex support


class Array:
    def __init__(
            self, x: Union[None, Array, py_array.array, int, ctypes.c_void_p, List[Union[int, float]]] = None,
            dtype: Union[None, Dtype, str] = None, shape: Tuple[int, ...] = (),
            pointer_source: PointerSource = PointerSource.host, offset: Optional[CType] = None,
            strides: Optional[Tuple[int, ...]] = None) -> None:
        _no_initial_dtype = False  # HACK, FIXME

        if isinstance(dtype, str):
            dtype = str_to_dtype(dtype)  # type: ignore[arg-type]

        if dtype is None:
            _no_initial_dtype = True
            dtype = af_float32

        if x is None:
            if not shape:  # shape is None or empty tuple
                self.arr = everything.create_handle((), dtype)
                return

            self.arr = everything.create_handle(shape, dtype)
            return

        if isinstance(x, Array):
            self.arr = everything.retain_array(x.arr)
            return

        if isinstance(x, py_array.array):
            _type_char: str = x.typecode
            _array_buffer = ArrayBuffer(*x.buffer_info())

        elif isinstance(x, list):
            _array = py_array.array("f", x)  # BUG [True, False] -> dtype: f32   # TODO add int and float
            _type_char = _array.typecode
            _array_buffer = ArrayBuffer(*_array.buffer_info())

        elif isinstance(x, int) or isinstance(x, ctypes.c_void_p):  # TODO
            _array_buffer = ArrayBuffer(x if not isinstance(x, ctypes.c_void_p) else x.value)  # type: ignore[arg-type]

            if not shape:
                raise TypeError("Expected to receive the initial shape due to the x being a data pointer.")

            if _no_initial_dtype:
                raise TypeError("Expected to receive the initial dtype due to the x being a data pointer.")

            _type_char = dtype.typecode

        else:
            raise TypeError("Passed object x is an object of unsupported class.")

        if not shape:
            if _array_buffer.length != 0:
                shape = (_array_buffer.length, )
            else:
                RuntimeError("Shape and buffer length are size invalid.")

        if not _no_initial_dtype and dtype.typecode != _type_char:
            raise TypeError("Can not create array of requested type from input data type")

        if not (offset or strides):
            if pointer_source == PointerSource.host:
                self.arr = everything.create_array(shape, dtype, _array_buffer)
                return

            self.arr = everything.device_array(shape, dtype, _array_buffer)
            return

        self.arr = everything.create_strided_array(
            shape, dtype, _array_buffer, offset, strides, pointer_source)  # type: ignore[arg-type]

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
        return _process_c_function(0, self, backend.sub)

    def __add__(self, other: Union[int, float, Array], /) -> Array:
        """
        Calculates the sum for each element of an array instance with the respective element of the array other.

        Parameters
        ----------
        self : Array
            Array instance (augend array). Should have a numeric data type.
        other: Union[int, float, Array]
            Addend array. Must be compatible with self (see Broadcasting). Should have a numeric data type.

        Returns
        -------
        out : Array
            An array containing the element-wise sums. The returned array must have a data type determined
            by Type Promotion Rules.
        """
        return _process_c_function(self, other, backend.add)

    def __sub__(self, other: Union[int, float, Array], /) -> Array:
        """
        Calculates the difference for each element of an array instance with the respective element of the array other.

        The result of self_i - other_i must be the same as self_i + (-other_i) and must be governed by the same
        floating-point rules as addition (see array.__add__()).

        Parameters
        ----------
        self : Array
            Array instance (minuend array). Should have a numeric data type.
        other: Union[int, float, Array]
            Subtrahend array. Must be compatible with self (see Broadcasting). Should have a numeric data type.

        Returns
        -------
        out : Array
            An array containing the element-wise differences. The returned array must have a data type determined
            by Type Promotion Rules.
        """
        return _process_c_function(self, other, backend.sub)

    def __mul__(self, other: Union[int, float, Array], /) -> Array:
        """
        Calculates the product for each element of an array instance with the respective element of the array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: Union[int, float, Array]
            Other array. Must be compatible with self (see Broadcasting). Should have a numeric data type.

        Returns
        -------
        out : Array
            An array containing the element-wise products. The returned array must have a data type determined
            by Type Promotion Rules.
        """
        return _process_c_function(self, other, backend.mul)

    def __truediv__(self, other: Union[int, float, Array], /) -> Array:
        """
        Evaluates self_i / other_i for each element of an array instance with the respective element of the
        array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: Union[int, float, Array]
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
        return _process_c_function(self, other, backend.div)

    def __floordiv__(self, other: Union[int, float, Array], /) -> Array:
        # TODO
        return NotImplemented

    def __mod__(self, other: Union[int, float, Array], /) -> Array:
        """
        Evaluates self_i % other_i for each element of an array instance with the respective element of the
        array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a real-valued data type.
        other: Union[int, float, Array]
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
        return _process_c_function(self, other, backend.mod)

    def __pow__(self, other: Union[int, float, Array], /) -> Array:
        """
        Calculates an implementation-dependent approximation of exponentiation by raising each element (the base) of
        an array instance to the power of other_i (the exponent), where other_i is the corresponding element of the
        array other.

        Parameters
        ----------
        self : Array
            Array instance whose elements correspond to the exponentiation base. Should have a numeric data type.
        other: Union[int, float, Array]
            Other array whose elements correspond to the exponentiation exponent. Must be compatible with self
            (see Broadcasting). Should have a numeric data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type determined
            by Type Promotion Rules.
        """
        return _process_c_function(self, other, backend.pow)

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
        # FIXME
        out = Array()
        out.arr = backend.bitnot(self.arr)
        return out

    def __and__(self, other: Union[int, bool, Array], /) -> Array:
        """
        Evaluates self_i & other_i for each element of an array instance with the respective element of the
        array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: Union[int, bool, Array]
            Other array. Must be compatible with self (see Broadcasting). Should have a numeric data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type determined
            by Type Promotion Rules.
        """
        return _process_c_function(self, other, backend.bitand)

    def __or__(self, other: Union[int, bool, Array], /) -> Array:
        """
        Evaluates self_i | other_i for each element of an array instance with the respective element of the
        array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: Union[int, bool, Array]
            Other array. Must be compatible with self (see Broadcasting). Should have a numeric data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type determined
            by Type Promotion Rules.
        """
        return _process_c_function(self, other, backend.bitor)

    def __xor__(self, other: Union[int, bool, Array], /) -> Array:
        """
        Evaluates self_i ^ other_i for each element of an array instance with the respective element of the
        array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: Union[int, bool, Array]
            Other array. Must be compatible with self (see Broadcasting). Should have a numeric data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type determined
            by Type Promotion Rules.
        """
        return _process_c_function(self, other, backend.bitxor)

    def __lshift__(self, other: Union[int, Array], /) -> Array:
        """
        Evaluates self_i << other_i for each element of an array instance with the respective element of the
        array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: Union[int, Array]
            Other array. Must be compatible with self (see Broadcasting). Should have a numeric data type.
            Each element must be greater than or equal to 0.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have the same data type as self.
        """
        return _process_c_function(self, other, backend.bitshiftl)

    def __rshift__(self, other: Union[int, Array], /) -> Array:
        """
        Evaluates self_i >> other_i for each element of an array instance with the respective element of the
        array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: Union[int, Array]
            Other array. Must be compatible with self (see Broadcasting). Should have a numeric data type.
            Each element must be greater than or equal to 0.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have the same data type as self.
        """
        return _process_c_function(self, other, backend.bitshiftr)

    # Comparison Operators

    def __lt__(self, other: Union[int, float, Array], /) -> Array:
        """
        Computes the truth value of self_i < other_i for each element of an array instance with the respective
        element of the array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: Union[int, float, Array]
            Other array. Must be compatible with self (see Broadcasting). Should have a real-valued data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type of bool.
        """
        return _process_c_function(self, other, backend.lt)

    def __le__(self, other: Union[int, float, Array], /) -> Array:
        """
        Computes the truth value of self_i <= other_i for each element of an array instance with the respective
        element of the array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: Union[int, float, Array]
            Other array. Must be compatible with self (see Broadcasting). Should have a real-valued data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type of bool.
        """
        return _process_c_function(self, other, backend.le)

    def __gt__(self, other: Union[int, float, Array], /) -> Array:
        """
        Computes the truth value of self_i > other_i for each element of an array instance with the respective
        element of the array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: Union[int, float, Array]
            Other array. Must be compatible with self (see Broadcasting). Should have a real-valued data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type of bool.
        """
        return _process_c_function(self, other, backend.gt)

    def __ge__(self, other: Union[int, float, Array], /) -> Array:
        """
        Computes the truth value of self_i >= other_i for each element of an array instance with the respective
        element of the array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: Union[int, float, Array]
            Other array. Must be compatible with self (see Broadcasting). Should have a real-valued data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type of bool.
        """
        return _process_c_function(self, other, backend.ge)

    def __eq__(self, other: Union[int, float, bool, Array], /) -> Array:  # type: ignore[override]
        """
        Computes the truth value of self_i == other_i for each element of an array instance with the respective
        element of the array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: Union[int, float, bool, Array]
            Other array. Must be compatible with self (see Broadcasting). May have any data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type of bool.
        """
        return _process_c_function(self, other, backend.eq)

    def __ne__(self, other: Union[int, float, bool, Array], /) -> Array:  # type: ignore[override]
        """
        Computes the truth value of self_i != other_i for each element of an array instance with the respective
        element of the array other.

        Parameters
        ----------
        self : Array
            Array instance. Should have a numeric data type.
        other: Union[int, float, bool, Array]
            Other array. Must be compatible with self (see Broadcasting). May have any data type.

        Returns
        -------
        out : Array
            An array containing the element-wise results. The returned array must have a data type of bool.
        """
        return _process_c_function(self, other, backend.neq)

    # Reflected Arithmetic Operators

    def __radd__(self, other: Array, /) -> Array:
        """
        Return other + self.
        """
        return _process_c_function(other, self, backend.add)

    def __rsub__(self, other: Array, /) -> Array:
        """
        Return other - self.
        """
        return _process_c_function(other, self, backend.sub)

    def __rmul__(self, other: Array, /) -> Array:
        """
        Return other * self.
        """
        return _process_c_function(other, self, backend.mul)

    def __rtruediv__(self, other: Array, /) -> Array:
        """
        Return other / self.
        """
        return _process_c_function(other, self, backend.div)

    def __rfloordiv__(self, other:  Array, /) -> Array:
        # TODO
        return NotImplemented

    def __rmod__(self, other: Array, /) -> Array:
        """
        Return other % self.
        """
        return _process_c_function(other, self, backend.mod)

    def __rpow__(self, other: Array, /) -> Array:
        """
        Return other ** self.
        """
        return _process_c_function(other, self, backend.pow)

    # Reflected Array Operators

    def __rmatmul__(self, other: Array, /) -> Array:
        # TODO
        return NotImplemented

    # Reflected Bitwise Operators

    def __rand__(self, other: Array, /) -> Array:
        """
        Return other & self.
        """
        return _process_c_function(other, self, backend.bitand)

    def __ror__(self, other: Array, /) -> Array:
        """
        Return other | self.
        """
        return _process_c_function(other, self, backend.bitor)

    def __rxor__(self, other: Array, /) -> Array:
        """
        Return other ^ self.
        """
        return _process_c_function(other, self, backend.bitxor)

    def __rlshift__(self, other: Array, /) -> Array:
        """
        Return other << self.
        """
        return _process_c_function(other, self, backend.bitshiftl)

    def __rrshift__(self, other: Array, /) -> Array:
        """
        Return other >> self.
        """
        return _process_c_function(other, self, backend.bitshiftr)

    # In-place Arithmetic Operators

    def __iadd__(self, other: Union[int, float, Array], /) -> Array:
        # TODO discuss either we need to support complex and bool as other input type
        """
        Return self += other.
        """
        return _process_c_function(self, other, backend.add)

    def __isub__(self, other: Union[int, float, Array], /) -> Array:
        """
        Return self -= other.
        """
        return _process_c_function(self, other, backend.sub)

    def __imul__(self, other: Union[int, float, Array], /) -> Array:
        """
        Return self *= other.
        """
        return _process_c_function(self, other, backend.mul)

    def __itruediv__(self, other: Union[int, float, Array], /) -> Array:
        """
        Return self /= other.
        """
        return _process_c_function(self, other, backend.div)

    def __ifloordiv__(self, other: Union[int, float, Array], /) -> Array:
        # TODO
        return NotImplemented

    def __imod__(self, other: Union[int, float, Array], /) -> Array:
        """
        Return self %= other.
        """
        return _process_c_function(self, other, backend.mod)

    def __ipow__(self, other: Union[int, float, Array], /) -> Array:
        """
        Return self **= other.
        """
        return _process_c_function(self, other, backend.pow)

    # In-place Array Operators

    def __imatmul__(self, other: Array, /) -> Array:
        # TODO
        return NotImplemented

    # In-place Bitwise Operators

    def __iand__(self, other: Union[int, bool, Array], /) -> Array:
        """
        Return self &= other.
        """
        return _process_c_function(self, other, backend.bitand)

    def __ior__(self, other: Union[int, bool, Array], /) -> Array:
        """
        Return self |= other.
        """
        return _process_c_function(self, other, backend.bitor)

    def __ixor__(self, other: Union[int, bool, Array], /) -> Array:
        """
        Return self ^= other.
        """
        return _process_c_function(self, other, backend.bitxor)

    def __ilshift__(self, other: Union[int, Array], /) -> Array:
        """
        Return self <<= other.
        """
        return _process_c_function(self, other, backend.bitshiftl)

    def __irshift__(self, other: Union[int, Array], /) -> Array:
        """
        Return self >>= other.
        """
        return _process_c_function(self, other, backend.bitshiftr)

    # Methods

    def __abs__(self) -> Array:
        # TODO
        return NotImplemented

    def __array_namespace__(self, *, api_version: Optional[str] = None) -> Any:
        # TODO
        return NotImplemented

    def __bool__(self) -> bool:
        # TODO consider using scalar() and is_scalar()
        return NotImplemented

    def __complex__(self) -> complex:
        # TODO
        return NotImplemented

    def __dlpack__(self, *, stream: Union[None, int, Any] = None):  # type: ignore[no-untyped-def]
        # TODO implementation and expected return type -> PyCapsule
        return NotImplemented

    def __dlpack_device__(self) -> Tuple[enum.Enum, int]:
        # TODO
        return NotImplemented

    def __float__(self) -> float:
        # TODO
        return NotImplemented

    def __getitem__(self, key: Union[int, slice, Tuple[Union[int, slice, ], ...], Array], /) -> Array:
        """
        Returns self[key].

        Parameters
        ----------
        self : Array
            Array instance.
        key : Union[int, slice, Tuple[Union[int, slice, ], ...], Array]
            Index key.

        Returns
        -------
        out : Array
            An array containing the accessed value(s). The returned array must have the same data type as self.
        """
        # TODO
        # API Specification - key: Union[int, slice, ellipsis, Tuple[Union[int, slice, ellipsis], ...], array].
        # consider using af.span to replace ellipsis during refactoring
        out = Array()
        ndims = self.ndim

        if isinstance(key, Array) and key == af_bool.c_api_value:
            ndims = 1
            if count_all(key.arr) == 0:  # HACK was count() method before
                return out

        # HACK known issue
        out.arr = everything.index_gen(self.arr, ndims, key, _get_indices(key))  # type: ignore[arg-type]
        return out

    def __index__(self) -> int:
        # TODO
        return NotImplemented

    def __int__(self) -> int:
        # TODO
        return NotImplemented

    def __len__(self) -> int:
        return self.shape[0] if self.shape else 0

    def __setitem__(
            self, key: Union[int, slice, Tuple[Union[int, slice, ], ...], Array],
            value: Union[int, float, bool, Array], /) -> None:
        # TODO
        return NotImplemented  # type: ignore[return-value]  # FIXME

    def __str__(self) -> str:
        # TODO change the look of array str. E.g., like np.array
        # if not _in_display_dims_limit(self.shape):
        #     return _metadata_string(self.dtype, self.shape)
        return _metadata_string(self.dtype) + everything.array_as_str(self.arr)

    def __repr__(self) -> str:
        # return _metadata_string(self.dtype, self.shape)
        # TODO change the look of array representation. E.g., like np.array
        return everything.array_as_str(self.arr)

    def to_device(self, device: Any, /, *, stream: Union[int, Any] = None) -> Array:
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
        return c_api_value_to_dtype(everything.get_ctype(self.arr))

    @property
    def device(self) -> Any:
        # TODO
        return NotImplemented

    @property
    def mT(self) -> Array:
        # TODO
        return NotImplemented

    @property
    def T(self) -> Array:
        """
        Transpose of the array.

        Returns
        -------
        out : Array
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
        out = Array()
        out.arr = everything.transpose(self.arr, False)
        return out

    @property
    def size(self) -> int:
        """
        Number of elements in an array.

        Returns
        -------
        out : int
            Number of elements in an array

        Note
        ----
        - This must equal the product of the array's dimensions.
        """
        # NOTE previously - elements()
        return everything.get_elements(self.arr)

    @property
    def ndim(self) -> int:
        """
        Number of array dimensions (axes).

        out : int
            Number of array dimensions (axes).
        """
        return everything.get_numdims(self.arr)

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Array dimensions.

        Returns
        -------
        out : tuple[int, ...]
            Array dimensions.
        """
        # NOTE skipping passing any None values
        return everything.get_dims(self.arr)[:self.ndim]

    def scalar(self) -> Union[None, int, float, bool, complex]:
        """
        Return the first element of the array
        """
        # TODO change the logic of this method
        if self.is_empty():
            return None

        return everything.get_scalar(self.arr, self.dtype)

    def is_empty(self) -> bool:
        """
        Check if the array is empty i.e. it has no elements.
        """
        return everything.is_empty(self.arr)

    def to_list(self, row_major: bool = False) -> List[Union[None, int, float, bool, complex]]:
        if self.is_empty():
            return []

        array = _reorder(self) if row_major else self
        ctypes_array = everything.get_data_ptr(array.arr, array.size, array.dtype)

        if array.ndim == 1:
            return list(ctypes_array)

        out = []
        for i in range(array.size):
            idx = i
            sub_list = []
            for j in range(array.ndim):
                div = array.shape[j]
                sub_list.append(idx % div)
                idx //= div
            out.append(ctypes_array[sub_list[::-1]])  # type: ignore[call-overload]  # FIXME
        return out

    def to_ctype_array(self, row_major: bool = False) -> ctypes.Array:
        if self.is_empty():
            raise RuntimeError("Can not convert an empty array to ctype.")

        array = _reorder(self) if row_major else self
        return everything.get_data_ptr(array.arr, array.size, array.dtype)


def _reorder(array: Array) -> Array:
    """
    Returns a reordered array to help interoperate with row major formats.
    """
    if array.ndim == 1:
        return array

    out = Array()
    out.arr = everything.reorder(array.arr, array.ndim)
    return out


def _metadata_string(dtype: Dtype, dims: Optional[Tuple[int, ...]] = None) -> str:
    return (
        "arrayfire.Array()\n"
        f"Type: {dtype.typename}\n"
        f"Dims: {str(dims) if dims else ''}")


def _process_c_function(lhs: Union[int, float, Array], rhs: Union[int, float, Array], c_function: Any) -> Array:
    out = Array()

    if isinstance(lhs, Array) and isinstance(rhs, Array):
        lhs_array = lhs.arr
        rhs_array = rhs.arr

    elif isinstance(lhs, Array) and isinstance(rhs, (int, float)):
        lhs_array = lhs.arr
        rhs_array = create_constant_array(rhs, lhs.shape, lhs.dtype)

    elif isinstance(lhs, (int, float)) and isinstance(rhs, Array):
        lhs_array = create_constant_array(lhs, rhs.shape, rhs.dtype)
        rhs_array = rhs.arr

    else:
        raise TypeError(f"{type(rhs)} is not supported and can not be passed to C binary function.")

    out.arr = c_function(lhs_array, rhs_array)
    return out


def _get_indices(key: Union[int, slice, Tuple[Union[int, slice, ], ...], Array]) -> CIndexStructure:
    indices = CIndexStructure()

    if isinstance(key, tuple):
        for n in range(len(key)):
            indices[n] = IndexStructure(key[n])
    else:
        indices[0] = IndexStructure(key)

    return indices