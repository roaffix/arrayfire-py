__all__ = ["gloh", "orb", "sift", "dog", "fast", "harris", "susan", "hamming_matcher", "nearest_neighbour", "match_template"]

from typing import cast

import arrayfire_wrapper.lib as wrapper

from arrayfire.array_object import Array, afarray_as_array
from arrayfire.library.constants import Match
from arrayfire.library.features import Features

# TODO
# Add examples to docs


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
    """
    Implements the GLOH (Gradient Location and Orientation Histogram) feature detection and descriptor extraction for
    images.

    Parameters
    ----------
    image : Array
        A 2D ArrayFire array representing the input image.
    n_layers : int, default: 3
        The number of layers per octave. The number of octaves is calculated based on the image dimensions and
        `initial_sigma`.
    contrast_threshold : float, default: 0.04
        The contrast threshold used to filter out weak features in low-contrast regions of the image.
    edge_threshold : float, default: 10.0
        The edge threshold used to filter out edge-like features to ensure feature points are from corners.
    initial_sigma : float, default: 1.6
        The initial sigma (scale) for the Gaussian blur applied to the image at the first layer.
    dobule_input : bool, default: True
        If True, the image size is doubled before processing to detect features at higher scales.
    intensity_scale : float, default: 1.0 / 255
        The scale factor applied to the image intensities, typically used to normalize the pixel values.
    feature_ratio : float, default: 0.05
        The ratio of the total number of pixels in the image used to limit the number of features detected.

    Returns
    -------
    tuple[Features, Array]
        A tuple containing two elements:
        - `Features`: An object holding the detected features, including their locations and scales.
        - `Array`: An ArrayFire array containing the GLOH descriptors for the detected features, with each descriptor
          having 272 elements.

    Note
    ----
    - The `gloh` function is particularly effective for object recognition and image matching tasks.
    - The choice of parameters can significantly impact the number and quality of features detected and described.
    """
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
    """
    Extracts ORB features and their descriptors from an image.

    Parameters
    ----------
    image : Array
        The input image as a 2D ArrayFire array.

    fast_threshold : float, default=20.0
        The FAST keypoint detection threshold.

    max_features : int, default=400
        The maximum number of keypoints to detect.

    scale_factor : float, default=1.5
        The scale factor between levels in the image pyramid.

    n_levels : int, default=4
        The number of levels in the image pyramid.

    blur_image : bool, default=False
        If True, the image is blurred before keypoint detection.

    Returns
    -------
    tuple[Features, Array]
        A tuple containing:
        - An ArrayFire Features object with detected keypoints.
        - An ArrayFire Array with corresponding descriptors.
    """
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
    """
    Extracts SIFT features and their descriptors from an image using the ArrayFire library.

    Parameters
    ----------
    image : Array
        The input image as a 2D ArrayFire array on which SIFT features are to be detected.

    n_layers : int, default=3
        The number of layers per octave in the SIFT algorithm. The number of octaves is calculated based on the image
        size.

    contrast_threshold : float, default=0.04
        The contrast threshold used to filter out weak features in low-contrast regions of the image.

    edge_threshold : float, default=10.0
        The edge threshold used to filter out edge-like features. Higher values mean fewer features rejected based on
        edge-like characteristics.

    initial_sigma : float, default=1.6
        The initial sigma (standard deviation) for the Gaussian blur applied to the image before feature detection.

    dobule_input : bool, default=True
        If True, the input image will be upscaled by a factor of 2 before processing, which helps in detecting features
        at larger scales.

    intensity_scale : float, default=1.0 / 255
        The scale factor applied to the image intensities. Typically used to normalize the image intensities.

    feature_ratio : float, default=0.05
        The maximum number of features to be detected, expressed as a ratio of the total number of image pixels.

    Returns
    -------
    tuple[Features, Array]
        A tuple containing:
        - An ArrayFire Features object encapsulating the detected keypoints.
        - An ArrayFire Array containing the corresponding descriptors for each keypoint. The descriptors are
          128-dimensional vectors describing the local appearance around each keypoint.

    Note
    ----
    The SIFT algorithm is a patented technique, and its use in commercial applications may require licensing. In
    academic and research settings, it remains a popular choice due to its robustness and reliability.
    """
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
    """
    Performs the Difference of Gaussians (DoG) operation on an image. This operation is a band-pass filter that
    highlights regions of an image with high spatial frequency, which correspond to edges. Typically used in edge
    detection and as a preprocessing step in feature extraction algorithms like SIFT.

    Parameters
    ----------
    image : Array
        The input image as a 2D ArrayFire array.

    radius1 : int
        The radius of the first Gaussian blur kernel. This parameter indirectly controls the sigma (standard deviation)
        of the Gaussian function, with a larger radius resulting in a more significant blur.

    radius2 : int
        The radius of the second Gaussian blur kernel. As with `radius1`, this parameter controls the degree of blur,
        but typically `radius2` > `radius1` to ensure a broader range of spatial frequencies are captured.

    Returns
    -------
    Array
        An ArrayFire array containing the result of the DoG operation. The output array highlights edges and
        transitions in the input image, with higher intensity values corresponding to stronger edges.

    Note
    ----
    The effective sigma values for the Gaussian blurs are calculated as 0.25 * radius, where the radius is the
    parameter passed to the function. The DoG operation is sensitive to the choice of radius parameters, which should
    be chosen based on the specific requirements of the application and the characteristics of the input image.
    """
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
    """
    Detects corners and interest points in an image using the FAST (Features from Accelerated Segment Test) algorithm.

    Parameters
    ----------
    image : Array
        The input image as a 2D ArrayFire array. The image should be grayscale.

    fast_threshold : float, default=20.0
        The intensity threshold for considering a pixel to be brighter or darker than the circular set of pixels
        around the candidate pixel. This value determines the sensitivity of the feature detection: higher values
        result in fewer features being detected.

    arc_length : int, default=9
        The minimum number of contiguous edge pixels (in the circle around the candidate pixel) required for the
        candidate pixel to be considered as a corner. The maximum length should be 16.

    non_max : bool, default=True
        If True, non-maximal suppression is applied to the detected features, ensuring that only the strongest
        features are retained.

    feature_ratio : float, default=0.05
        Specifies the maximum ratio of features to pixels in the image, controlling the density of features detected.

    edge : int, default=3
        The number of pixels to ignore along the edge of the image. This parameter helps in excluding features that
        are too close to the edge of the image, which may not be reliable.

    Returns
    -------
    Features
        An ArrayFire Features object containing the detected points. The features include locations and scores,
        while orientations and sizes are not computed by the FAST algorithm.

    Note
    ----
    The FAST algorithm is particularly well-suited for real-time applications due to its computational efficiency.
    However, it is sensitive to the choice of `fast_threshold` and `arc_length` parameters, which should be tuned
    based on the specific requirements of the application and the characteristics of the input images.
    """
    return Features.from_affeatures(wrapper.fast(image.arr, fast_threshold, arc_length, non_max, feature_ratio, edge))


def harris(
    image: Array,
    /,
    max_corners: int = 500,
    min_response: float = 1e5,
    sigma: float = 1.0,
    block_size: int = 0,
    k_threshold: float = 0.04,
) -> Features:
    """
    Detects corners in an image using the Harris corner detection algorithm.

    Parameters
    ----------
    image : Array
        The input image as a 2D ArrayFire array. The image should be grayscale for optimal results.

    max_corners : int, default=500
        The maximum number of corners to return. If there are more corners than `max_corners`, only the strongest
        ones (as determined by the Harris response) are returned.

    min_response : float, default=1e5
        The minimum response value for a corner to be considered. This value helps to filter out weak corners.

    sigma : float, default=1.0
        The standard deviation of the Gaussian filter applied to the input image. This parameter is used only
        when `block_size` is 0. Valid ranges are 0.5 to 5.0.

    block_size : int, default=0
        The size of the neighborhood considered for corner detection. A larger value considers a larger neighborhood.
        If set to 0, a circular window based on `sigma` is used instead.

    k_threshold : float, default=0.04
        The Harris detector free parameter in the equation. Common values are between 0.04 to 0.06.

    Returns
    -------
    Features
        An ArrayFire Features object containing the detected corners' locations and their Harris response scores.
        Orientation and size are not computed.

    Note
    ----
    The Harris corner detector is particularly sensitive to `sigma`, `block_size`, and `k_threshold` parameters,
    which should be chosen based on the specific requirements of the application and the characteristics of the input
    images. It's recommended to adjust these parameters to balance detection sensitivity and computational efficiency.
    """
    return Features.from_affeatures(
        wrapper.harris(image.arr, max_corners, min_response, sigma, block_size, k_threshold)
    )


def susan(
    image: Array,
    /,
    radius: int = 3,
    diff_threshold: float = 32.0,
    geom_threshold: float = 10.0,
    feature_ratio: float = 0.05,
    edge: int = 3,
) -> Features:
    """
    Detects corners and edges in an image using the SUSAN corner detection algorithm.

    Parameters
    ----------
    image : Array
        The input image as a 2D ArrayFire array. The image should be grayscale.

    radius : int, default=3
        The radius of the circular mask applied to each pixel to determine if it's a corner. A smaller radius
        will detect finer features, while a larger radius will detect broader features.

    diff_threshold : float, default=32.0
        The intensity difference threshold. This value determines how much the intensity of neighboring pixels
        can differ from the nucleus (central pixel) to be considered part of the univalue segment.

    geom_threshold : float, default=10.0
        The geometric threshold. This value determines the minimum number of contiguous pixels within the
        circular mask that need to be similar to the nucleus for a pixel to be considered a corner.

    feature_ratio : float, default=0.05
        Specifies the maximum number of features to detect as a ratio of total image pixels. This helps to control
        the density of features detected in the image.

    edge : int, default=3
        Specifies the number of pixels to ignore along the edge of the image. This is useful for excluding features
        that are too close to the edges and may not be reliable.

    Returns
    -------
    Features
        An ArrayFire Features object containing the detected corners and edges' locations. Orientation and size are
        not computed for SUSAN features.

    Note
    ----
    The SUSAN algorithm is sensitive to the choice of `radius`, `diff_threshold`, and `geom_threshold` parameters.
    These should be carefully chosen based on the specific requirements of the application and the characteristics
    of the input image. Adjusting these parameters can help balance the sensitivity to corners and computational
    efficiency.
    """
    return Features.from_affeatures(
        wrapper.susan(image.arr, radius, diff_threshold, geom_threshold, feature_ratio, edge)
    )


def hamming_matcher(query: Array, train: Array, /, axis: int = 0, n_nearest: int = 1) -> tuple[Array, Array]:
    """
    Finds the nearest neighbors for each descriptor in a query set from a training set, based on the Hamming distance.

    Parameters
    ----------
    query : Array
        The query set of feature descriptors as an ArrayFire array. Each descriptor should be a row in a 2D array
        or along the specified `axis` in higher dimensions.

    train : Array
        The training set of feature descriptors as an ArrayFire array. This serves as the database from which the
        closest matches to the query descriptors are found.

    axis : int, default=0
        The dimension along which the feature descriptors are aligned. Typically, descriptors are arranged as rows
        in a 2D array (axis=0).

    n_nearest : int, default=1
        The number of nearest neighbors to find for each query descriptor. Setting `n_nearest` greater than 1 enables
        finding multiple close matches.

    Returns
    -------
    tuple[Array, Array]
        A tuple containing two ArrayFire arrays:
        - The first array contains the indices of the closest matches in the training set for each query descriptor.
        - The second array contains the Hamming distances of these matches.

    Note
    ----
    The Hamming matcher is particularly effective for binary feature descriptors and is widely used in computer vision
    tasks such as object recognition and tracking. When using `n_nearest` > 1, the function returns multiple matches
    per query descriptor, which can be useful for robust matching strategies.
    """
    indices, distance = wrapper.hamming_matcher(query.arr, train.arr, axis, n_nearest)
    return Array.from_afarray(indices), Array.from_afarray(distance)


def nearest_neighbour(
    query: Array, train: Array, /, axis: int = 0, n_nearest: int = 1, match_type: Match = Match.SSD
) -> tuple[Array, Array]:
    """
    Finds the nearest neighbors for each descriptor in a query set from a training set based on a specified metric.

    Parameters
    ----------
    query : Array
        The query set of feature descriptors as an ArrayFire array. Each descriptor should be aligned along the
        specified `axis` in a multidimensional array.

    train : Array
        The training set of feature descriptors as an ArrayFire array. This serves as the database from which the
        closest matches to the query descriptors are found.

    axis : int, default=0
        The dimension along which the feature descriptors are aligned. For a 2D array of descriptors, this is
        typically 0, indicating that each descriptor is a row.

    n_nearest : int, default=1
        The number of nearest neighbors to find for each query descriptor. Allows for finding multiple matches per
        query.

    match_type : Match, default=Match.SSD
        The matching metric to use for finding nearest neighbors. `Match.SSD` uses the sum of squared differences,
        suitable for floating-point descriptors. Other metrics can be specified if supported.

    Returns
    -------
    tuple[Array, Array]
        A tuple containing two ArrayFire arrays:
        - The first array contains the indices of the closest matches in the training set for each query descriptor.
        - The second array contains the distances of these matches according to the specified metric.

    Note
    ----
    The `nearest_neighbour` function is versatile, supporting various descriptor types and matching metrics. It is
    particularly useful in computer vision tasks such as object recognition, where matching feature descriptors between
    images is essential.
    """
    indices, distance = wrapper.nearest_neighbour(query.arr, train.arr, axis, n_nearest, match_type)
    return Array.from_afarray(indices), Array.from_afarray(distance)

def match_template(search_image: Array, template_image: Array, / ,match_type: Match = Match.SAD) -> Array:
    template = wrapper.match_template(search_image, template_image, match_type)
    return Array.from_afarray(template)
