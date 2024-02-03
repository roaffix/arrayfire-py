__all__ = [
    "color_space",
    "gray2rgb",
    "hsv2rgb",
    "rgb2gray",
    "rgb2hsv",
    "rgb2ycbcr",
    "anisotropic_diffusion",
    "bilateral",
    "canny",
    "inverse_deconv",
    "iterative_deconv",
    "maxfilt",
    "mean_shift",
    "medfilt",
    "medfilt1",
    "medfilt2",
    "minfilt",
    "sat",
    "sobel_operator",
    "gaussian_kernel",
    "hist_equal",
    "histogram",
    "resize",
    "rotate",
    "scale",
    "skew",
    "transform",
    "transform_coordinates",
    "translate",
    "confidence_cc",
    "regions",
    "dilate",
    "erode",
    "wrap",
    "unwrap",
]


from typing import cast

import arrayfire_wrapper.lib as wrapper
from dtypes import Dtype, float32

from arrayfire import Array
from arrayfire.array_object import afarray_as_array
from arrayfire.library.constants import (
    CannyThreshold,
    Connectivity,
    CSpace,
    Diffusion,
    Flux,
    Interp,
    IterativeDeconv,
    Pad,
    YCCStd,
)

from .array_management.creation import constant
from .vector_algorithms import max as afmax
from .vector_algorithms import min as afmin

# Colorspace conversions


@afarray_as_array
def color_space(image: Array, to_type: CSpace, from_type: CSpace, /) -> Array:
    return cast(Array, wrapper.color_space(image.arr, to_type, from_type))


@afarray_as_array
def gray2rgb(image: Array, /, *, r_factor: float = 1.0, g_factor: float = 1.0, b_factor: float = 1.0) -> Array:
    return cast(Array, wrapper.gray2rgb(image.arr, r_factor, g_factor, b_factor))


@afarray_as_array
def hsv2rgb(image: Array, /) -> Array:
    return cast(Array, wrapper.hsv2rgb(image.arr))


@afarray_as_array
def rgb2gray(
    image: Array, /, *, r_factor: float = 0.2126, g_factor: float = 0.7152, b_factor: float = 0.0722
) -> Array:
    return cast(Array, wrapper.rgb2gray(image.arr, r_factor, g_factor, b_factor))


@afarray_as_array
def rgb2hsv(image: Array, /) -> Array:
    return cast(Array, wrapper.rgb2hsv(image.arr))


@afarray_as_array
def rgb2ycbcr(image: Array, /, *, standard: YCCStd = YCCStd.YCC_601) -> Array:
    return cast(Array, wrapper.rgb2ycbcr(image.arr, standard))


@afarray_as_array
def ycbcr2rgb(image: Array, /, *, standard: YCCStd = YCCStd.YCC_601) -> Array:
    return cast(Array, wrapper.ycbcr2rgb(image.arr, standard))


# Filters


@afarray_as_array
def anisotropic_diffusion(
    image: Array,
    timestep: float,
    conductance: float,
    iterations: int,
    /,
    *,
    fftype: Flux = Flux.QUADRATIC,
    diffusion_kind: Diffusion = Diffusion.GRAD,
) -> Array:
    return cast(
        Array, wrapper.anisotropic_diffusion(image.arr, timestep, conductance, iterations, fftype, diffusion_kind)
    )


@afarray_as_array
def bilateral(image: Array, spatial_sigma: float, chromatic_sigma: float, /, *, is_color: bool = False) -> Array:
    return cast(Array, wrapper.bilateral(image.arr, spatial_sigma, chromatic_sigma, is_color))


@afarray_as_array
def canny(
    image: Array,
    low_threshold: float,
    high_threshold: float,
    /,
    *,
    threshold_type: CannyThreshold = CannyThreshold.MANUAL,
    sobel_window: int = 3,
    is_fast: bool = False,
) -> Array:
    return cast(Array, wrapper.canny(image.arr, threshold_type, low_threshold, high_threshold, sobel_window, is_fast))


@afarray_as_array
def inverse_deconv(
    image: Array, psf: Array, gamma: float, /, *, algo: IterativeDeconv = IterativeDeconv.DEFAULT
) -> Array:
    return cast(Array, wrapper.inverse_deconv(image.arr, psf.arr, gamma, algo))


@afarray_as_array
def iterative_deconv(
    image: Array,
    psf: Array,
    iterations: int,
    relax_factor: float,
    /,
    *,
    algo: IterativeDeconv = IterativeDeconv.DEFAULT,
) -> Array:
    return cast(Array, wrapper.iterative_deconv(image.arr, psf.arr, iterations, relax_factor, algo))


@afarray_as_array
def maxfilt(image: Array, /, *, wind_length: int = 3, wind_width: int = 3, edge_pad: Pad = Pad.ZERO) -> Array:
    return cast(Array, wrapper.maxfilt(image.arr, wind_length, wind_width, edge_pad))


@afarray_as_array
def mean_shift(
    image: Array, spatial_sigma: float, chromatic_sigma: float, iterations: int, /, *, is_color: bool = False
) -> Array:
    return cast(Array, wrapper.mean_shift(image.arr, spatial_sigma, chromatic_sigma, iterations, is_color))


@afarray_as_array
def medfilt(image: Array, /, *, wind_length: int = 3, wind_width: int = 3, edge_pad: Pad = Pad.ZERO) -> Array:
    return cast(Array, wrapper.medfilt(image.arr, wind_length, wind_width, edge_pad))


@afarray_as_array
def medfilt1(image: Array, /, *, wind_width: int = 3, edge_pad: Pad = Pad.ZERO) -> Array:
    return cast(Array, wrapper.medfilt1(image.arr, wind_width, edge_pad))


@afarray_as_array
def medfilt2(image: Array, /, *, wind_length: int = 3, wind_width: int = 3, edge_pad: Pad = Pad.ZERO) -> Array:
    return cast(Array, wrapper.medfilt2(image.arr, wind_length, wind_width, edge_pad))


@afarray_as_array
def minfilt(image: Array, /, *, wind_length: int = 3, wind_width: int = 3, edge_pad: Pad = Pad.ZERO) -> Array:
    return cast(Array, wrapper.minfilt(image.arr, wind_length, wind_width, edge_pad))


@afarray_as_array
def sat(image: Array, /) -> Array:
    return cast(Array, wrapper.sat(image.arr))


def sobel_operator(image: Array, /, *, kernel_size: int = 3) -> tuple[Array, Array]:
    dx, dy = wrapper.sobel_operator(image.arr, kernel_size)
    return Array.from_afarray(dx), Array.from_afarray(dy)


# Gaussian kernel


def gaussian_kernel(
    rows: int, columns: int, /, *, rows_sigma: None | float = None, columns_sigma: None | float = None
) -> Array:
    rs = 0.25 * rows + 0.75 if not rows_sigma else rows_sigma
    cs = 0.25 * columns + 0.75 if not columns_sigma else columns_sigma

    return cast(Array, wrapper.gaussian_kernel(rows, columns, rs, cs))


# Histograms


@afarray_as_array
def hist_equal(image: Array, histogram: Array, /) -> Array:
    return cast(Array, wrapper.hist_equal(image.arr, histogram.arr))


@afarray_as_array
def histogram(
    image: Array, n_bins: int, /, *, min_value: None | float = None, max_value: None | float = None
) -> Array:
    min_v = afmin(image) if not min_value else min_value
    max_v = afmax(image) if not max_value else max_value

    return cast(Array, wrapper.histogram(image.arr, n_bins, min_v, max_v))  # type: ignore[arg-type]


# Image transformation


@afarray_as_array
def resize(image: Array, size_scale: float | tuple[int, int], /, *, method: Interp = Interp.NEAREST) -> Array:
    if isinstance(size_scale, float):
        dims = (int(size_scale * image.shape[0]), int(size_scale * image.shape[1]))
    elif isinstance(size_scale, tuple):
        dims = size_scale
    else:
        raise TypeError(
            "`size_scale` should be specified either as scale of original image, or the specific output size."
        )

    return cast(Array, wrapper.resize(image.arr, dims[0], dims[1], method))


@afarray_as_array
def rotate(image: Array, theta: float, /, *, is_crop: bool = True, method: Interp = Interp.NEAREST) -> Array:
    return cast(Array, wrapper.rotate(image.arr, theta, is_crop, method))


@afarray_as_array
def scale(
    image: Array,
    scale: tuple[float, float],
    /,
    *,
    output_size: tuple[int, int] = (0, 0),
    method: Interp = Interp.NEAREST,
) -> Array:
    return cast(Array, wrapper.scale(image.arr, scale[0], scale[1], output_size[0], output_size[1], method))


@afarray_as_array
def skew(
    image: Array,
    skew: tuple[float, float],
    /,
    *,
    output_size: tuple[int, int] = (0, 0),
    method: Interp = Interp.NEAREST,
    is_inverse: bool = True,
) -> Array:
    return cast(Array, wrapper.skew(image.arr, skew[0], skew[1], output_size[0], output_size[1], method, is_inverse))


@afarray_as_array
def transform(
    image: Array,
    transform_matrix: Array,
    /,
    *,
    output_size: tuple[int, int] = (0, 0),
    method: Interp = Interp.NEAREST,
    is_inverse: bool = True,
) -> Array:
    return cast(
        Array, wrapper.transform(image.arr, transform_matrix.arr, output_size[0], output_size[1], method, is_inverse)
    )


@afarray_as_array
def transform_coordinates(image: Array, coordinates: tuple[float, float], /) -> Array:
    return cast(Array, wrapper.transform_coordinates(image.arr, coordinates[0], coordinates[1]))


@afarray_as_array
def translate(
    image: Array, translation: tuple[float, float], /, *, output_size: tuple[int, int], method: Interp = Interp.NEAREST
) -> Array:
    return cast(
        Array, wrapper.translate(image.arr, translation[0], translation[1], output_size[0], output_size[1], method)
    )


# Labeling


@afarray_as_array
def confidence_cc(
    image: Array,
    seed: tuple[Array, Array],
    radius: int,
    multiplier: int,
    iterations: int,
    segmented_value: float,
    /,
) -> Array:
    return cast(
        Array,
        wrapper.confidence_cc(image.arr, seed[0].arr, seed[1].arr, radius, multiplier, iterations, segmented_value),
    )


@afarray_as_array
def regions(
    image: Array, /, *, connectivity: Connectivity = Connectivity.FOUR, output_dtype: Dtype = float32
) -> Array:
    return cast(Array, wrapper.regions(image.arr, connectivity, output_dtype))


# Morphological operations


# TODO
# Split back to two separate functions
def dilate(image: Array, /, *, mask: Array | None = None) -> Array:
    if image.ndim == 2:
        mask_ = constant(1, (3, 3), float32) if not mask else mask
        return Array.from_afarray(wrapper.dilate(image.arr, mask_.arr))

    if image.ndim == 3:
        mask_ = constant(1, (3, 3, 3), float32) if not mask else mask
        return Array.from_afarray(wrapper.dilate3(image.arr, mask_.arr))

    raise ValueError("Image should be either 2D or 3D.")


# TODO
# Split back to two separate functions
def erode(image: Array, /, *, mask: Array | None = None) -> Array:
    if image.ndim == 2:
        mask_ = constant(1, (3, 3), float32) if not mask else mask
        return Array.from_afarray(wrapper.erode(image.arr, mask_.arr))

    if image.ndim == 3:
        mask_ = constant(1, (3, 3, 3), float32) if not mask else mask
        return Array.from_afarray(wrapper.erode3(image.arr, mask_.arr))

    raise ValueError("Image should be either 2D or 3D.")


# Wrapping


@afarray_as_array
def wrap(
    image: Array,
    output_size: tuple[int, int],
    window: tuple[int, int],
    stride: tuple[int, int],
    /,
    *,
    padding: tuple[int, int],
    is_column: bool = True,
) -> Array:
    return cast(
        Array,
        wrapper.wrap(
            image.arr,
            output_size[0],
            output_size[1],
            window[0],
            window[1],
            stride[0],
            stride[1],
            padding[0],
            padding[1],
            is_column,
        ),
    )


@afarray_as_array
def unwrap(
    image: Array,
    output_size: tuple[int, int],
    window: tuple[int, int],
    stride: tuple[int, int],
    /,
    *,
    padding: tuple[int, int],
    is_column: bool = True,
) -> Array:
    return cast(
        Array,
        wrapper.unwrap(
            image.arr,
            output_size[0],
            output_size[1],
            window[0],
            window[1],
            stride[0],
            stride[1],
            padding[0],
            padding[1],
            is_column,
        ),
    )
