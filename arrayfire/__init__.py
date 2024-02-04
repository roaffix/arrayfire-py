# flake8: noqa
from .version import VERSION

__all__ = ["__version__"]
__version__ = VERSION

# TODO
# add __arrayfire_version__

__all__ += ["Array"]
from .array_object import Array

__all__ += [
    "Dtype",
    "b8",
    "bool",
    "c32",
    "c64",
    "complex32",
    "complex64",
    "f16",
    "f32",
    "f64",
    "float16",
    "float32",
    "float64",
    "int16",
    "int32",
    "int64",
    "s16",
    "s32",
    "s64",
    "u8",
    "u16",
    "u32",
    "u64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]

from .dtypes import (
    Dtype,
    b8,
    bool,
    c32,
    c64,
    complex32,
    complex64,
    f16,
    f32,
    f64,
    float16,
    float32,
    float64,
    int16,
    int32,
    int64,
    s16,
    s32,
    s64,
    u8,
    u16,
    u32,
    u64,
    uint8,
    uint16,
    uint32,
    uint64,
)

__all__ += [
    "constant",
    "diag",
    "identity",
    "iota",
    "lower",
    "upper",
    "pad",
    "range",
    "isinf",
    "isnan",
    "iszero",
    "set_manual_eval_flag",
    "eval",
    "copy_array",
    "flat",
    "flip",
    "join",
    "moddims",
    "reorder",
    "replace",
    "select",
    "shift",
    "tile",
    "transpose",
]

from arrayfire.library.array_functions import (
    constant,
    copy_array,
    diag,
    eval,
    flat,
    flip,
    identity,
    iota,
    isinf,
    isnan,
    iszero,
    join,
    lower,
    moddims,
    pad,
    range,
    reorder,
    replace,
    select,
    set_manual_eval_flag,
    shift,
    tile,
    transpose,
    upper,
)

__all__ += ["gloh", "orb", "sift", "dog", "fast", "harris", "susan", "hamming_matcher", "nearest_neighbour"]

from arrayfire.library.computer_vision import (
    dog,
    fast,
    gloh,
    hamming_matcher,
    harris,
    nearest_neighbour,
    orb,
    sift,
    susan,
)

__all__ += [
    "Match",
    "MatProp",
    "BinaryOperator",
    "Norm",
    "ConvGradient",
    "VarianceBias",
    "TopK",
    "ImageFormat",
    "CSpace",
    "YCCStd",
    "Flux",
    "Diffusion",
    "CannyThreshold",
    "Connectivity",
    "ConvDomain",
    "ConvMode",
    "Interp",
    "IterativeDeconv",
    "Pad",
    "pi",
]

from arrayfire.library.constants import (
    BinaryOperator,
    CannyThreshold,
    Connectivity,
    ConvDomain,
    ConvGradient,
    ConvMode,
    CSpace,
    Diffusion,
    Flux,
    ImageFormat,
    Interp,
    IterativeDeconv,
    Match,
    MatProp,
    Norm,
    Pad,
    TopK,
    VarianceBias,
    YCCStd,
    pi,
)

__all__ += [
    "alloc_device",
    "alloc_host",
    "alloc_pinned",
    "device_gc",
    "device_info",
    "device_mem_info",
    "free_device",
    "free_host",
    "free_pinned",
    "get_dbl_support",
    "get_device",
    "get_device_count",
    "get_half_support",
    "get_kernel_cache_directory",
    "get_mem_step_size",
    "info",
    "info_string",
    "init",
    "print_mem_info",
    "set_device",
    "sync",
    "set_kernel_cache_directory",
    "set_mem_step_size",
]

from arrayfire.library.device import (
    alloc_device,
    alloc_host,
    alloc_pinned,
    device_gc,
    device_info,
    device_mem_info,
    free_device,
    free_host,
    free_pinned,
    get_dbl_support,
    get_device,
    get_device_count,
    get_half_support,
    get_kernel_cache_directory,
    get_mem_step_size,
    info,
    info_string,
    init,
    print_mem_info,
    set_device,
    set_kernel_cache_directory,
    set_mem_step_size,
    sync,
)

__all__ += [
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

from arrayfire.library.image_processing import (
    anisotropic_diffusion,
    bilateral,
    canny,
    color_space,
    confidence_cc,
    dilate,
    erode,
    gaussian_kernel,
    gray2rgb,
    hist_equal,
    histogram,
    hsv2rgb,
    inverse_deconv,
    iterative_deconv,
    maxfilt,
    mean_shift,
    medfilt,
    medfilt1,
    medfilt2,
    minfilt,
    regions,
    resize,
    rgb2gray,
    rgb2hsv,
    rgb2ycbcr,
    rotate,
    sat,
    scale,
    skew,
    sobel_operator,
    transform,
    transform_coordinates,
    translate,
    unwrap,
    wrap,
)

__all__ += [
    "is_image_io_available",
    "read_array",
    "save_array",
    "load_image",
    "load_image_native",
    "load_image_memory",
    "delete_image_memory",
    "save_image",
    "save_image_native",
    "save_image_memory",
]

from arrayfire.library.input_and_output import (
    delete_image_memory,
    is_image_io_available,
    load_image,
    load_image_memory,
    load_image_native,
    read_array,
    save_array,
    save_image,
    save_image_memory,
    save_image_native,
)

__all__ += ["cublas_set_math_mode", "get_native_id", "get_stream", "set_native_id"]

from arrayfire.library.interface_functions import cublas_set_math_mode, get_native_id, get_stream, set_native_id

__all__ += [
    "dot",
    "gemm",
    "matmul",
    "is_lapack_available",
    "cholesky",
    "lu",
    "qr",
    "svd",
    "det",
    "inverse",
    "norm",
    "pinverse",
    "rank",
    "solve",
]

from arrayfire.library.linear_algebra import (
    cholesky,
    det,
    dot,
    gemm,
    inverse,
    is_lapack_available,
    lu,
    matmul,
    norm,
    pinverse,
    qr,
    rank,
    solve,
    svd,
)

__all__ += ["convolve2_gradient_nn"]

from arrayfire.library.machine_learning import convolve2_gradient_nn

__all__ += [
    "add",
    "sub",
    "mul",
    "div",
    "mod",
    "pow",
    "bitnot",
    "bitand",
    "bitor",
    "bitxor",
    "bitshiftl",
    "bitshiftr",
    "lt",
    "le",
    "gt",
    "ge",
    "eq",
    "neq",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "exp",
    "expm1",
    "log",
    "log1p",
    "log2",
    "log10",
    "sqrt",
    "cbrt",
    "hypot",
    "erf",
    "erfc",
    "tgamma",
    "lgamma",
    "pow2",
    "sign",
    "abs",
    "ceil",
    "floor",
    "round",
    "trunc",
    "isinf",
    "isnan",
    "iszero",
    "isinf",
    "isnan",
    "iszero",
    "isinf",
    "isnan",
    "clamp",
    "arg",
    "conjg",
    "cplx",
    "imag",
    "factorial",
    "maxof",
    "minof",
    "real",
    "rem",
    "root",
    "rsqrt",
    "sigmoid",
    "logical_and",
    "logical_or",
    "logical_not",
    "neg",
]


from .library.mathematical_functions import (
    abs,
    acos,
    acosh,
    add,
    arg,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    bitand,
    bitnot,
    bitor,
    bitshiftl,
    bitshiftr,
    bitxor,
    cbrt,
    ceil,
    conjg,
    cos,
    cosh,
    cplx,
    div,
    eq,
    erf,
    erfc,
    exp,
    expm1,
    factorial,
    floor,
    ge,
    gt,
    hypot,
    imag,
    isinf,
    isnan,
    iszero,
    le,
    lgamma,
    log,
    log1p,
    log2,
    log10,
    logical_and,
    logical_not,
    logical_or,
    lt,
    maxof,
    minof,
    mod,
    mul,
    neg,
    neq,
    pow,
    pow2,
    real,
    rem,
    root,
    round,
    rsqrt,
    sigmoid,
    sign,
    sin,
    sinh,
    sqrt,
    sub,
    tan,
    tanh,
    tgamma,
    trunc,
)

__all__ += ["randu"]

from arrayfire.library.random import randu

__all__ += [
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
]

from arrayfire.library.signal_processing import (
    approx1,
    approx1_uniform,
    approx2,
    approx2_uniform,
    fft,
    fft2,
    fft2_c2r,
    fft2_r2c,
    fft3,
    fft3_c2r,
    fft3_r2c,
    fft_c2r,
    fft_convolve1,
    fft_convolve2,
    fft_convolve3,
    fft_r2c,
    fir,
    ifft,
    ifft2,
    ifft3,
    iir,
    set_fft_plan_cache_size,
)

__all__ += ["corrcoef", "cov", "mean", "median", "stdev", "topk", "var"]

from arrayfire.library.statistics import corrcoef, cov, mean, median, stdev, topk, var

__all__ += [
    "get_active_backend",
    "get_available_backends",
    "get_backend_count",
    "get_backend_id",
    "get_device_id",
    "set_backend",
]

from arrayfire.library.unified_api_functions import (
    get_active_backend,
    get_available_backends,
    get_backend_count,
    get_backend_id,
    get_device_id,
    set_backend,
)

__all__ += [
    "accum",
    "scan",
    "where",
    "all_true",
    "any_true",
    "sum",
    "product",
    "count",
    "imax",
    "max",
    "imin",
    "min",
    "diff1",
    "diff2",
    "gradient",
    "set_intersect",
    "set_union",
    "set_unique",
    "sort",
]

from arrayfire.library.vector_algorithms import (
    accum,
    all_true,
    any_true,
    count,
    diff1,
    diff2,
    gradient,
    imax,
    imin,
    max,
    min,
    product,
    scan,
    set_intersect,
    set_union,
    set_unique,
    sort,
    sum,
    where,
)

__all__ += ["cast"]

from arrayfire.library.utils import cast
