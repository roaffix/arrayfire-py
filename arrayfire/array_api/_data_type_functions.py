from dataclasses import dataclass

import arrayfire as af

from ._array_object import Array
from ._dtypes import (
    all_dtypes,
    boolean_dtypes,
    complex_floating_dtypes,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    integer_dtypes,
    numeric_dtypes,
    promote_types,
    real_floating_dtypes,
    signed_integer_dtypes,
    unsigned_integer_dtypes,
)


def astype(x: Array, dtype: af.Dtype, /, *, copy: bool = True) -> Array:
    """
    Casts an array to a specified data type, respecting or overriding type promotion rules.

    This function copies the input array `x` to a new array with the specified data type `dtype`. It does not follow
    standard type promotion rules and allows for explicit data type conversions, which might include downcasting or
    converting between incompatible types such as complex to real numbers.

    Parameters
    ----------
    x : Array
        The input array to cast.
    dtype : af.Dtype
        The desired data type for the new array.
    copy : bool, optional
        Specifies whether to always return a new array or return the original array when the new data type matches
        the original data type of `x`. Defaults to True, which means a new array is always created.

    Returns
    -------
    Array
        An array with the specified data type. The shape of the returned array is identical to that of the input
        array `x`.

    Notes
    -----
    - Casting from complex to real data types is not permitted by this function to avoid data loss and ambiguity.
    Instead, users must explicitly decide how to handle complex numbers (e.g., taking real or imaginary parts) before
    casting.
    - When casting from boolean to numeric types, `True` is treated as `1` and `False` as `0`.
    - When casting from numeric types to boolean, `0` is treated as `False` and any non-zero value as `True`.
    - For complex to boolean conversions, `0 + 0j` is `False`, and all other complex values are considered `True`.
    - The behavior of casting NaN and infinity values to integral types is undefined and depends on the implementation.

    Examples
    --------
    >>> a = asarray([1, 2, 3], dtype=int32)
    >>> astype(a, dtype=float32)
    Array([1.0, 2.0, 3.0])

    >>> c = asarray([True, False, True], dtype=bool)
    >>> astype(c, dtype=int32)
    Array([1, 0, 1])

    >>> d = asarray([0, -1, 1], dtype=int32)
    >>> astype(d, dtype=bool)
    Array([False, True, True])

    Raises
    ------
    ValueError
        If attempting to cast a complex array directly to a real or integral data type.

    Implementation Details
    -----------------------
    This function ensures that the data conversion respects the intended data type and copy specifications, making it
    suitable for both memory management and type control in computational tasks.
    """
    if x.dtype in complex_floating_dtypes:
        raise ValueError("Casting is not allowed from complex dtypes.")

    afarray = af.copy_array(x._array) if copy else x._array
    return Array._new(af.cast(afarray, dtype))


def broadcast_arrays(*arrays: Array) -> list[Array]:
    return NotImplemented


def broadcast_to(x: Array, /, shape: tuple[int, ...]) -> Array:
    return NotImplemented


def can_cast(from_: af.Dtype | Array, to: af.Dtype, /) -> bool:
    return NotImplemented


@dataclass
class finfo_object:
    bits: int
    eps: float
    max: float
    min: float
    smallest_normal: float
    dtype: af.Dtype


@dataclass
class iinfo_object:
    bits: int
    max: int
    min: int
    dtype: af.Dtype


def finfo(type: af.Dtype | Array, /) -> finfo_object:
    """
    Returns information about the machine limits for floating-point data types.

    This function provides detailed attributes of floating-point data types such as precision, minimum and maximum
    values, and epsilon. If a complex data type is provided, it will return information about the real component of
    the complex type, as both real and imaginary parts of complex numbers share the same precision.

    Parameters
    ----------
    type : af.Dtype | Array
        The floating-point data type or an array from which to derive the data type. If a complex data type is used,
        the information pertains to its real-valued component.

    Returns
    -------
    finfo_object
        An object with detailed attributes of the floating-point data type:
        - bits (int): Number of bits occupied by the real-valued data type.
        - eps (float): Smallest representable positive number such that 1.0 + eps != 1.0.
        - max (float): Largest representable real-valued number.
        - min (float): Smallest representable real-valued number.
        - smallest_normal (float): Smallest positive normal number that can be represented accurately.
        - dtype (af.Dtype): The data type queried.

    Notes
    -----
    - As of version 2022.12, support for complex data types was added. Information provided for complex types relates
      to the properties of the real and imaginary components, which are identical.
    - This function is useful for understanding the characteristics of different floating-point types, especially
      when precision and range are critical to the application.

    Examples
    --------
    >>> finfo(float32)
    finfo_object(bits=32, eps=1.19209290e-7, max=3.4028234e38, min=-3.4028234e38, smallest_normal=1.1754943e-38, dtype=float32)

    >>> finfo(array([1.0, 2.0], dtype=float64))
    finfo_object(bits=64, eps=2.2204460492503131e-16, max=1.7976931348623157e308, min=-1.7976931348623157e308, smallest_normal=2.2250738585072014e-308, dtype=float64)

    Raises
    ------
    ValueError
        If the `type` argument is neither a recognized floating-point `af.Dtype` nor an `Array` containing a supported
        floating-point data type.

    """  # noqa
    if isinstance(type, af.Dtype):
        dtype = type
    elif isinstance(type, Array):
        dtype = Array.dtype  # type: ignore[assignment]
    else:
        raise ValueError("Unsupported dtype.")

    if dtype == float32:
        return finfo_object(32, 1.19209290e-7, 3.4028234e38, -3.4028234e38, 1.1754943e-38, float32)
    if dtype == float64:
        return finfo_object(
            64,
            2.2204460492503131e-16,
            1.7976931348623157e308,
            -1.7976931348623157e308,
            2.2250738585072014e-308,
            float64,
        )
    if dtype == float16:
        return finfo_object(16, 0.00097656, 65504, -65504, 0.00006103515625, float16)

    raise ValueError("Unsupported dtype.")


def iinfo(type: af.Dtype | Array, /) -> iinfo_object:
    """
    Returns information about the machine limits for integer data types.

    This function provides attributes of integer data types such as the number of bits and the range of representable
    values (minimum and maximum). It can accept either an integer data type directly or an array from which the integer
    data type is inferred.

    Parameters
    ----------
    type : af.Dtype | Array
        The integer data type or an array from which to derive the data type.

    Returns
    -------
    iinfo_object
        An object with attributes detailing the properties of the integer data type:
        - bits (int): Number of bits occupied by the integer data type.
        - max (int): Largest representable integer.
        - min (int): Smallest representable integer.
        - dtype (af.Dtype): The data type queried.

    Notes
    -----
    - This function is essential for understanding the storage and range limitations of various integer data types
      within a machine's architecture, especially useful when precision and overflow issues are a concern.

    Examples
    --------
    >>> iinfo(int32)
    iinfo_object(bits=32, max=2147483647, min=-2147483648, dtype=int32)

    >>> iinfo(array([1, 2, 3], dtype=int64))
    iinfo_object(bits=64, max=9223372036854775807, min=-9223372036854775808, dtype=int64)

    Raises
    ------
    ValueError
        If the `type` argument is neither a recognized integer `af.Dtype` nor an `Array` containing a supported
        integer data type.

    """
    if isinstance(type, af.Dtype):
        dtype = type
    elif isinstance(type, Array):
        dtype = Array.dtype  # type: ignore[assignment]
    else:
        raise ValueError("Unsupported dtype.")

    if dtype == int32:
        return iinfo_object(32, 2147483648, -2147483647, int32)
    if dtype == int16:
        return iinfo_object(16, 32767, -32768, int16)
    if dtype == int8:
        return iinfo_object(8, 127, -128, int8)
    if dtype == int64:
        return iinfo_object(64, 9223372036854775807, -9223372036854775808, int64)

    raise ValueError("Unsupported dtype.")


def isdtype(dtype: af.Dtype, kind: af.Dtype | str | tuple[af.Dtype | str, ...]) -> bool:
    """
    Determines if a provided dtype matches a specified data type kind.

    This function checks if the input dtype matches the specified kind. It supports checking against single data type
    identifiers (both dtype objects and string representations) as well as combinations of data types specified in a
    tuple.

    Parameters
    ----------
    dtype : af.Dtype
        The input data type to check.
    kind : af.Dtype | str | tuple[af.Dtype | str, ...]
        The kind against which to check the dtype. This can be a single data type descriptor (dtype or string), or a
        tuple containing multiple data type descriptors. Supported string identifiers include:
        - 'bool': For boolean data types.
        - 'signed integer': For signed integer data types (e.g., int8, int16, int32, int64).
        - 'unsigned integer': For unsigned integer data types (e.g., uint8, uint16, uint32, uint64).
        - 'integral': Shorthand for all integer data types (signed and unsigned).
        - 'real floating': For real-valued floating-point data types (e.g., float32, float64).
        - 'complex floating': For complex floating-point data types (e.g., complex64, complex128).
        - 'numeric': Shorthand for all numeric data types (integral, real floating, and complex floating).

    Returns
    -------
    bool
        True if the input dtype matches the specified kind; False otherwise.

    Notes
    -----
    - This function is designed to be flexible and can handle extensions to the supported data types, as long as the
      extensions remain consistent within their respective categories (e.g., only integer types in 'integral').
    - The flexibility allows for the inclusion of additional data types such as float16 or bfloat16 under the
      'real floating' category or int128 under 'signed integer', if they are supported by the implementation.

    Examples
    --------
    >>> isdtype(float32, 'real floating')
    True

    >>> isdtype(int32, 'numeric')
    True

    >>> isdtype(complex64, ('complex floating', 'real floating'))
    True

    >>> isdtype(uint32, ('signed integer', 'unsigned integer'))
    True

    """
    dtype_kinds = {
        "bool": boolean_dtypes,
        "signed integer": signed_integer_dtypes,
        "unsigned integer": unsigned_integer_dtypes,
        "integral": integer_dtypes,
        "real floating": real_floating_dtypes,
        "complex floating": complex_floating_dtypes,
        "numeric": numeric_dtypes,
    }

    if isinstance(kind, tuple):
        return any(isdtype(dtype, single_kind) for single_kind in kind)

    elif isinstance(kind, af.Dtype):
        return dtype == kind

    elif isinstance(kind, str):
        if kind in dtype_kinds:
            return dtype in dtype_kinds[kind]

        raise ValueError(f"Unsupported kind: {kind}")

    raise ValueError("Kind must be a dtype, a string identifier, or a tuple of identifiers.")


def result_type(*arrays_and_dtypes: Array | af.Dtype) -> af.Dtype:
    """
    Array API compatible wrapper for :py:func:`np.result_type <numpy.result_type>`.

    See its docstring for more information.
    """
    # FIXME
    # Code duplicate from numpy src

    # Note: we use a custom implementation that gives only the type promotions
    # required by the spec rather than using np.result_type. NumPy implements
    # too many extra type promotions like int64 + uint64 -> float64, and does
    # value-based casting on scalar arrays.
    A = []
    for a in arrays_and_dtypes:
        if isinstance(a, Array):
            a = a.dtype
        elif isinstance(a, af.Array) or a not in all_dtypes:
            raise TypeError("result_type() inputs must be array_api arrays or dtypes")
        A.append(a)

    if len(A) == 0:
        raise ValueError("at least one array or dtype is required")
    elif len(A) == 1:
        return A[0]
    else:
        t = A[0]
        for t2 in A[1:]:
            t = promote_types(t, t2)
        return t
