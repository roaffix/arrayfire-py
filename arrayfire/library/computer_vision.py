from typing import cast

import arrayfire_wrapper.lib as wrapper

from arrayfire.array_object import Array, afarray_as_array
from arrayfire.library.constants import Match
from arrayfire.library.features import Features


def gloh(
    image: Array,
    /,
    n_layers: int = 3,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10.0,
    initial_sigma: float = 1.6,
    dobule_input: bool = True,
    intensity_scale: float = 1.0 / 255,
    feature_ratio: float = 0.05,
) -> tuple[Features, Array]:
    features, descriptors = wrapper.gloh(
        image.arr,
        n_layers,
        contrast_threshold,
        edge_threshold,
        initial_sigma,
        dobule_input,
        intensity_scale,
        feature_ratio,
    )
    return Features.from_affeatures(features), Array.from_afarray(descriptors)


def orb(
    image: Array,
    /,
    fast_threshold: float = 20.0,
    max_features: int = 400,
    scale_factor: float = 1.5,
    n_levels: int = 4,
    blur_image: bool = False,
) -> tuple[Features, Array]:
    features, descriptors = wrapper.orb(image.arr, fast_threshold, max_features, scale_factor, n_levels, blur_image)
    return Features.from_affeatures(features), Array.from_afarray(descriptors)


def sift(
    image: Array,
    /,
    n_layers: int = 3,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10.0,
    initial_sigma: float = 1.6,
    dobule_input: bool = True,
    intensity_scale: float = 1.0 / 255,
    feature_ratio: float = 0.05,
) -> tuple[Features, Array]:
    features, descriptors = wrapper.sift(
        image.arr,
        n_layers,
        contrast_threshold,
        edge_threshold,
        initial_sigma,
        dobule_input,
        intensity_scale,
        feature_ratio,
    )
    return Features.from_affeatures(features), Array.from_afarray(descriptors)


@afarray_as_array
def dog(image: Array, radius1: int, radius2: int, /) -> Array:
    return cast(Array, wrapper.dog(image.arr, radius1, radius2))


def fast(
    image: Array,
    /,
    fast_threshold: float = 20.0,
    arc_length: int = 9,
    non_max: bool = True,
    feature_ratio: float = 0.05,
    edge: int = 3,
) -> Features:
    return Features.from_affeatures(wrapper.fast(image.arr, fast_threshold, arc_length, non_max, feature_ratio, edge))


def harris(
    image: Array,
    /,
    max_corners: int = 500,
    min_response: float = 1e5,
    sigma: float = 1.0,
    block_size: int = 0,
    k_threshold: float = 0,
) -> Features:
    return Features.from_affeatures(
        wrapper.harris(image.arr, max_corners, min_response, sigma, block_size, k_threshold)
    )


def susan(
    image: Array,
    /,
    radius: int = 500,
    diff_threshold: float = 1e5,
    geom_threshold: float = 1.0,
    feature_ratio: float = 0.05,
    edge: int = 3,
) -> Features:
    return Features.from_affeatures(
        wrapper.susan(image.arr, radius, diff_threshold, geom_threshold, feature_ratio, edge)
    )


def hamming_matcher(query: Array, train: Array, /, axis: int = 0, n_nearest: int = 1) -> tuple[Array, Array]:
    indices, distance = wrapper.hamming_matcher(query.arr, train.arr, axis, n_nearest)
    return Array.from_afarray(indices), Array.from_afarray(distance)


def nearest_neighbour(
    query: Array, train: Array, /, axis: int = 0, n_nearest: int = 1, match_type: Match = Match.SSD
) -> tuple[Array, Array]:
    indices, distance = wrapper.nearest_neighbour(query.arr, train.arr, axis, n_nearest, match_type)
    return Array.from_afarray(indices), Array.from_afarray(distance)
