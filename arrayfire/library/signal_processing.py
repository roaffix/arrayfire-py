__all__ = [
    "fft",
    "fft2",
    "fft2_c2r",
    "fft2_r2c",
    "fft3",
    "fft3_c2r",
    "fft3_r2c",
    "fft_c2r",
    "fft_r2c",
    "fft_convolve1",
    "fft_convolve2",
    "fft_convolve3",
    "convolve2"
    "ifft",
    "ifft2",
    "ifft3",
    "set_fft_plan_cache_size",
    "fir",
    "iir",
    "approx1",
    "approx1_uniform",
    "approx2",
    "approx2_uniform",
    "convolve1",
    "convolve2",
    "convolve2_nn",
    "convolve2_separable",
    "convolve3",
]

from typing import cast

import arrayfire_wrapper.lib as wrapper
from arrayfire_wrapper.lib import set_fft_plan_cache_size

from arrayfire.array_object import Array, afarray_as_array
from arrayfire.library.constants import ConvDomain, ConvMode, Interp

# Convolutions


@afarray_as_array
def convolve1(
    signal: Array, kernel: Array, /, *, mode: ConvMode = ConvMode.DEFAULT, domain: ConvDomain = ConvDomain.AUTO
) -> Array:
    return cast(Array, wrapper.convolve1(signal.arr, kernel.arr, mode, domain))


@afarray_as_array
def fft_convolve1(signal: Array, kernel: Array, /, *, mode: ConvMode = ConvMode.DEFAULT) -> Array:
    return cast(Array, wrapper.fft_convolve1(signal.arr, kernel.arr, mode))


@afarray_as_array
def convolve2(
    signal: Array, kernel: Array, /, *, mode: ConvMode = ConvMode.DEFAULT, domain: ConvDomain = ConvDomain.AUTO
) -> Array:
    return cast(Array, wrapper.convolve2(signal.arr, kernel.arr, mode, domain))


@afarray_as_array
def fft_convolve2(signal: Array, kernel: Array, /, *, mode: ConvMode = ConvMode.DEFAULT) -> Array:
    return cast(Array, wrapper.fft_convolve2(signal.arr, kernel.arr, mode))


@afarray_as_array
def convolve2_nn(
    signal: Array,
    kernel: Array,
    /,
    *,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
) -> Array:
    return cast(Array, wrapper.convolve2_nn(signal.arr, kernel.arr, stride, padding, dilation))


@afarray_as_array
def convolve2_separable(
    column_kernel: Array, row_kernel: Array, signal: Array, /, *, mode: ConvMode = ConvMode.DEFAULT
) -> Array:
    return cast(Array, wrapper.convolve2_sep(column_kernel.arr, row_kernel.arr, signal.arr, mode))


@afarray_as_array
def convolve3(
    signal: Array, kernel: Array, /, *, mode: ConvMode = ConvMode.DEFAULT, domain: ConvDomain = ConvDomain.AUTO
) -> Array:
    return cast(Array, wrapper.convolve3(signal.arr, kernel.arr, mode, domain))


@afarray_as_array
def fft_convolve3(signal: Array, kernel: Array, /, *, mode: ConvMode = ConvMode.DEFAULT) -> Array:
    return cast(Array, wrapper.fft_convolve3(signal.arr, kernel.arr, mode))


# Fast Fourier Transformations


def fft(signal: Array, /, *, scale: float | None = None, output_size: int = 0, inplace: bool = False) -> Array:
    if inplace:
        scale = scale or 1.0 / signal.shape[0]
        wrapper.fft_inplace(signal.arr, scale)
        return signal

    scale = 1.0 if not scale else scale
    return Array.from_afarray(wrapper.fft(signal.arr, scale, output_size))


def fft2(
    signal: Array, /, *, scale: float | None = None, output_size: tuple[int, int] = (0, 0), inplace: bool = False
) -> Array:
    if inplace:
        shape = signal.shape
        scale = scale or 1.0 / (shape[0] * shape[1])
        wrapper.fft2_inplace(signal.arr, scale)
        return signal

    scale = 1.0 if not scale else scale
    return Array.from_afarray(wrapper.fft2(signal.arr, scale, output_size[0], output_size[1]))


def fft3(
    signal: Array,
    /,
    *,
    scale: float | None = None,
    output_size: tuple[int, int, int] = (0, 0, 0),
    inplace: bool = False,
) -> Array:
    if inplace:
        shape = signal.shape
        scale = scale or 1.0 / (shape[0] * shape[1] * shape[2])
        wrapper.fft3_inplace(signal.arr, scale)
        return signal

    scale = 1.0 if not scale else scale
    return Array.from_afarray(wrapper.fft3(signal.arr, scale, output_size[0], output_size[1], output_size[2]))


def _convert_to_c2r_dim(axis: int, is_odd: bool, /) -> int:
    return 2 * (axis - 1) + int(is_odd)


@afarray_as_array
def fft_c2r(signal: Array, /, *, scale: float | None = None, is_odd: bool = False) -> Array:
    if not scale:
        shape = signal.shape
        output_size = _convert_to_c2r_dim(shape[0], is_odd)
        scale = 1.0 / output_size

    return cast(Array, wrapper.fft_c2r(signal.arr, scale, is_odd))


@afarray_as_array
def fft2_c2r(signal: Array, /, *, scale: float | None = None, is_odd: bool = False) -> Array:
    if not scale:
        shape = signal.shape
        output_size = _convert_to_c2r_dim(shape[0], is_odd), shape[1]
        scale = 1.0 / (output_size[0] * output_size[1])

    return cast(Array, wrapper.fft2_c2r(signal.arr, scale, is_odd))


@afarray_as_array
def fft3_c2r(signal: Array, /, *, scale: float | None = None, is_odd: bool = False) -> Array:
    if not scale:
        shape = signal.shape
        output_size = _convert_to_c2r_dim(shape[0], is_odd), shape[1], shape[2]
        scale = 1.0 / (output_size[0] * output_size[1] * output_size[2])

    return cast(Array, wrapper.fft3_c2r(signal.arr, scale, is_odd))


@afarray_as_array
def fft_r2c(signal: Array, /, *, scale: float | None = None, output_size: int = 0) -> Array:
    scale = scale or 1.0
    return cast(Array, wrapper.fft_r2c(signal.arr, scale, output_size))


@afarray_as_array
def fft2_r2c(signal: Array, /, *, scale: float | None = None, output_size: tuple[int, int] = (0, 0)) -> Array:
    scale = scale or 1.0
    return cast(Array, wrapper.fft2_r2c(signal.arr, scale, output_size[0], output_size[1]))


@afarray_as_array
def fft3_r2c(signal: Array, /, *, scale: float | None = None, output_size: tuple[int, int, int] = (0, 0, 0)) -> Array:
    scale = scale or 1.0
    return cast(Array, wrapper.fft3_r2c(signal.arr, scale, output_size[0], output_size[1], output_size[2]))


def ifft(
    signal: Array, /, *, scale: float | None = None, output_size: int | None = None, inplace: bool = False
) -> Array:
    output_size_ = output_size if output_size and not inplace else signal.shape[:1][0]
    scale = scale or 1.0 / output_size_

    if inplace:
        wrapper.ifft_inplace(signal.arr, scale)
        return signal

    return Array.from_afarray(wrapper.ifft(signal.arr, scale, output_size_))


def ifft2(
    signal: Array, /, *, scale: float | None = None, output_size: tuple[int, int] | None = None, inplace: bool = False
) -> Array:
    output_size_ = output_size if output_size and not inplace else signal.shape[:2]
    scale = scale or 1.0 / (output_size_[0] * output_size_[1])

    if inplace:
        wrapper.ifft2_inplace(signal.arr, scale)
        return signal

    return cast(Array, wrapper.ifft2(signal.arr, scale, output_size_[0], output_size_[1]))


def ifft3(
    signal: Array, /, *, scale: float | None = None, output_size: tuple[int, int] | None = None, inplace: bool = False
) -> Array:
    output_size_ = output_size if output_size and not inplace else signal.shape[:3]
    scale = scale or 1.0 / (output_size_[0] * output_size_[1] * output_size_[2])

    if inplace:
        wrapper.ifft3_inplace(signal.arr, scale)
        return signal

    return Array.from_afarray(wrapper.ifft3(signal.arr, scale, output_size_[0], output_size_[1], output_size_[2]))


# Filter


@afarray_as_array
def fir(b: Array, x: Array, /) -> Array:
    return cast(Array, wrapper.fir(b.arr, x.arr))


@afarray_as_array
def iir(b: Array, a: Array, x: Array, /) -> Array:
    return cast(Array, wrapper.iir(b.arr, a.arr, x.arr))


# Interpolation and approximation


def _scale_position_by_axis0(x_current: Array, x_original: Array) -> Array:
    x0 = x_original[0, 0, 0, 0]
    dx = x_original[1, 0, 0, 0] - x0
    return (x_current - x0) / dx


def _scale_position_by_axis1(y_current: Array, y_original: Array) -> Array:
    y0 = y_original[0, 0, 0, 0]
    dy = y_original[0, 1, 0, 0] - y0
    return (y_current - y0) / dy


# TODO
# double check either the approx1_v2 is needed.
@afarray_as_array
def approx1(
    signal: Array,
    positions: Array,
    /,
    *,
    method: Interp = Interp.LINEAR,
    off_grid: float = 0.0,
    original_positions: Array | None = None,
) -> Array:
    x_position = positions if not original_positions else _scale_position_by_axis0(positions, original_positions)

    return cast(Array, wrapper.approx1(signal.arr, x_position.arr, method, off_grid))


@afarray_as_array
def approx1_uniform(
    signal: Array,
    positions: Array,
    interp_axis: int,
    start_index: int,
    step_index: int,
    /,
    *,
    method: Interp = Interp.LINEAR,
    off_grid: float = 0.0,
) -> Array:
    return cast(
        Array,
        wrapper.approx1_uniform(signal.arr, positions.arr, interp_axis, start_index, step_index, method, off_grid),
    )


@afarray_as_array
def approx2(
    signal: Array,
    positions: tuple[Array, Array],
    /,
    *,
    method: Interp = Interp.LINEAR,
    off_grid: float = 0.0,
    original_positions: tuple[Array | None, Array | None] = (None, None),
) -> Array:
    x_position = (
        positions[0] if not original_positions[0] else _scale_position_by_axis0(positions[0], original_positions[0])
    )
    y_position = (
        positions[1] if not original_positions[1] else _scale_position_by_axis1(positions[1], original_positions[1])
    )

    return cast(Array, wrapper.approx2(signal.arr, x_position.arr, y_position.arr, method, off_grid))


@afarray_as_array
def approx2_uniform(
    signal: Array,
    positions: tuple[Array, Array],
    interp_axis: tuple[int, int],
    start_index: tuple[int, int],
    step_index: tuple[int, int],
    /,
    *,
    method: Interp = Interp.LINEAR,
    off_grid: float = 0.0,
) -> Array:
    return cast(
        Array,
        wrapper.approx2_uniform(
            signal.arr,
            positions[0].arr,
            interp_axis[0],
            start_index[0],
            step_index[0],
            positions[1].arr,
            interp_axis[1],
            start_index[1],
            step_index[1],
            method,
            off_grid,
        ),
    )
